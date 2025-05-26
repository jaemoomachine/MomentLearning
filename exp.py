import os
import math
import random
import copy
import argparse

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from models.Moment_TE import MomentTE
from models.Moment_TE_NoSampling import MomentTE_NoSampling

from baselines.Frozen_TE import FrozenTE
from baselines.Standard_TE import StandardTE
from baselines.Switch_TE import SwitchTE
from baselines.Sparse_TE import SparseTE
from utils.metrics import regression_metrics


model_dict = {
    'MomentTE' : MomentTE,
    'StandardTE' : StandardTE,
    'FrozenTE': FrozenTE,
    'SwitchTE' : SwitchTE,
    'SparseTE' : SparseTE,
    'MomentTE_NoSampling':MomentTE_NoSampling
    
}

class TimeSeriesDataset(Dataset):
    def __init__(self, series, seq_len, pred_len, target_col_index):
        self.series = series
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.tgt_idx = target_col_index

    def __len__(self):
        return len(self.series) - self.seq_len - self.pred_len + 1

    def __getitem__(self, idx):
        x = self.series[idx: idx + self.seq_len]            
        y = self.series[idx + self.seq_len: idx + self.seq_len + self.pred_len, self.tgt_idx]  
        return torch.from_numpy(x).float(), torch.from_numpy(y).float()


def split_series(data, train_ratio, val_ratio):
    n = len(data)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    return data[:train_end], data[train_end:val_end], data[val_end:]

def get_scaler(train_data, mode):
    if mode == 'minmax':
        scaler = MinMaxScaler()
    elif mode == 'zscore':
        scaler = StandardScaler()
    else:
        raise ValueError("Unsupported mode")
    scaler.fit(train_data)
    return scaler

def extract_gamma_matrix(moment_ffn: MomentFFN):
    gammas = []
    for layer in moment_ffn.layers:
        if isinstance(layer, MomentLinear):
            gammas.append(layer.gamma[1:].unsqueeze(0)) 
    return torch.cat(gammas, dim=0)  

def diversity_l2(matrices):
    loss = 0.0
    M = len(matrices)
    for i in range(M):
        for j in range(i+1, M):
            diff = matrices[i] - matrices[j]
            loss = loss - torch.norm(diff, p='fro')**2
    return loss

def diversity_cos(matrices):
    loss = 0.0
    M = len(matrices)
    vecs = [m.view(-1) for m in matrices]
    for i in range(M):
        for j in range(i+1, M):
            num = torch.dot(vecs[i], vecs[j])
            denom = vecs[i].norm() * vecs[j].norm() + 1e-8
            loss = loss + num / denom
    return loss

def main(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available() : torch.cuda.set_device(args.gpu)
    
    args.data_path = f"{args.base_path}/dataset/{args.data_name}.csv"
    df = pd.read_csv(args.data_path)
    if 'date' in df.columns:
        df = df.drop(columns=['date'])

    if 'OT' not in df.columns:
        raise ValueError("The target column 'OT' was not found in the dataset.")

    features = df.columns.tolist()
    target_idx = features.index('OT')

    data_vals = df.values.astype(np.float32)
    train_vals, val_vals, test_vals = split_series(data_vals, args.train_ratio, args.val_ratio)

    scaler = get_scaler(train_vals, args.scale_mode)
    train_p = scaler.transform(train_vals)
    val_p = scaler.transform(val_vals)
    test_p = scaler.transform(test_vals)

    train_ds = TimeSeriesDataset(train_p, args.seq_len, args.pred_len, target_col_index=target_idx)
    val_ds = TimeSeriesDataset(val_p, args.seq_len, args.pred_len, target_col_index=target_idx)
    test_ds = TimeSeriesDataset(test_p, args.seq_len, args.pred_len, target_col_index=target_idx)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    cfg = argparse.Namespace(model=args.model, data_name = args.data_name, 
                             c_in=train_p.shape[1], d_model=args.d_model, dropout=args.dropout,
                             d_ff=args.d_ff, nhead=args.nhead, bias=True, L_sub=args.L_sub,
                             K=args.K, beta=args.beta, ffn_type=args.ffn_type,
                             num_experts=args.num_experts, topk=args.topk,
                             sparse_gating=args.sparse_gating, expert_depth=args.expert_depth,
                             num_layers=args.num_layers, c_out=args.pred_len, ff_init=args.ff_init)

    model = model_dict[args.model](cfg).to(device)
    
    if isinstance(model, MomentTE) :
        module_names = []
        gamma_dims   = []
        for name, module in model.named_modules():
            if hasattr(module, 'gamma'):
                module_names.append(name)
                gamma_dims.append(module.gamma.numel())

        col_names = []
        for name, dim in zip(module_names, gamma_dims):
            for k in range(dim):
                col_names.append(f"{name}_gamma{k}")
        Gamma_epoch, Gamma_batch = [], []
    
    model_id = get_model_id(args)
    print(f"::: {model_id} :::")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params}")
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable_params}")    
    
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()
    best_val_loss = float('inf')
    best_path = f"{args.base_path}/Checkpoint/{model_id}.pt"
    if not os.path.exists(f"{args.base_path}/Checkpoint"):
        os.makedirs(f"{args.base_path}/Checkpoint")
        
    lambda_l2  = args.lambda_l2
    lambda_cos = args.lambda_cos
        
    for ep in range(1, args.epochs+1):
        model.train()
        train_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(x)
            
            mse = F.mse_loss(pred, y)
            if args.regularizer != 'no' and isinstance(model, MoME):
                gamma_matrices = [extract_gamma_matrix(exp) for exp in model.experts]
            else:
                gamma_matrices = []

            if args.epoch_freeze >= ep and gamma_matrices:
                if args.regularizer == 'no': loss = mse
                elif args.regularizer == 'l2' : loss = mse + lambda_l2 * diversity_l2(gamma_matrices) 
                elif args.regularizer == 'cos' : loss = mse + lambda_cos * diversity_cos(gamma_matrices)
                elif args.regularizer == 'l2cos' : loss = mse + lambda_l2 * diversity_l2(gamma_matrices) + lambda_cos * diversity_cos(gamma_matrices)
                    
            else:
                loss = mse            
            
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * x.size(0)
            
            if isinstance(model, MomentTE):
                gamma_batch = []    
                for name, module in model.named_modules():
                    if hasattr(module, 'gamma'):
                        gamma_batch.append(module.gamma.clone().detach().cpu().numpy().copy())
                Gamma_batch.append(gamma_batch)

        if isinstance(model, MomentTE):
            print(f"\n=== Epoch {ep} Gamma ===")
            total = 0
            frozen = 0
            for m in model.modules():
                if hasattr(m, 'mask'):
                    n = m.mask.numel()
                    f = int(m.mask.sum().item())
                    total += n
                    frozen += f
            sampled = total - frozen
            print(f"[Epoch {ep}] sampled_weights={sampled}, frozen_weights={frozen}")
            if ep == 1: total_sampled = total
            gamma_epoch = []
            for name, module in model.named_modules():
                if hasattr(module, 'gamma'):
                    print(f"[Epoch {ep}]] {name}.gamma = {module.gamma.clone().detach().cpu().numpy().copy()}")
                    gamma_epoch.append(module.gamma.clone().detach().cpu().numpy().copy())
            Gamma_epoch.append(gamma_epoch)
            
            
        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x)
                val_loss += criterion(pred, y).item() * x.size(0)
        val_loss /= len(val_loader.dataset)
        
        if ep >= 0 and val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_path)

        if isinstance(model, MomentTE):
            if args.strategy == 'beta': model.partial_freeze()
            elif args.strategy == 'distinct' and args.epoch_freeze == ep: model.full_freeze()

        print(f"Epoch {ep}: Train Loss={train_loss:.6f}, Val Loss={val_loss:.6f}")

        
    if isinstance(model, MomentTE):
        epoch_matrix = []
        for epoch_list in Gamma_epoch:
            epoch_matrix.append(np.concatenate(epoch_list))
        df_epoch = pd.DataFrame(epoch_matrix, columns=col_names)

        batch_matrix = []
        for batch_list in Gamma_batch:
            batch_matrix.append(np.concatenate(batch_list))
        df_batch = pd.DataFrame(batch_matrix, columns=col_names)
    else: pass
    
    model.eval()
    if isinstance(model, MomentTE): model.full_freeze()
        
    test_loss = 0.0
    preds, trues = [], []
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            test_loss += criterion(pred, y).item() * x.size(0)
            preds.append(pred.detach().cpu().numpy())
            trues.append(y.detach().cpu().numpy())
    test_loss /= len(test_loader.dataset)
    print(f"Test Loss={test_loss:.6f}")

    preds = np.concatenate(preds, axis = 0)
    trues = np.concatenate(trues, axis = 0)
    preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
    trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
    print('test shape:', preds.shape, trues.shape)

    def inverse_transform_target(data, scaler, target_idx):
        temp = np.zeros((data.shape[0], scaler.n_features_in_))
        temp[:, target_idx] = data[:, 0] if data.ndim == 2 else data
        return scaler.inverse_transform(temp)[:, target_idx]

    preds_inv = inverse_transform_target(preds.reshape(-1, 1), scaler, target_idx).reshape(preds.shape)
    trues_inv = inverse_transform_target(trues.reshape(-1, 1), scaler, target_idx).reshape(trues.shape)

    metrics = regression_metrics(trues_inv.reshape(-1), preds_inv.reshape(-1))
    metrics['total_params'] = total_params
    metrics['trainable_params'] = trainable_params
    if isinstance(model, MomentTE): metrics['total_sampled'] = total_sampled

    print(model_id, ':')    
    print(metrics)
    folder_path = f'{args.base_path}/Result/'
    os.makedirs(folder_path, exist_ok=True)

    pd.DataFrame(metrics, index=[0]).to_csv(f"{folder_path}Metrics_{model_id}.csv", index=False)
    if isinstance(model, MomentTE):
        df_epoch.to_csv(f"{folder_path}Gamma_epoch_{model_id}.csv", index=False)
        df_batch.to_csv(f"{folder_path}Gamma_batch_{model_id}.csv", index=False)
    np.save(folder_path + f'pred_{model_id}.npy', preds_inv)
    np.save(folder_path + f'true_{model_id}.npy', trues_inv)

    torch.cuda.empty_cache()
