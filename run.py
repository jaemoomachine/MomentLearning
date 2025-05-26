import argparse
from exp import main

parser = argparse.ArgumentParser("Moment Learning")

parser.add_argument("--no", type=int, default=1)
parser.add_argument("--model", type=str, default='MomentTE')
parser.add_argument("--strategy", type=str, choices=['beta','distinct'], default='beta')
parser.add_argument("--epoch_freeze", type=int, default=10)
parser.add_argument("--data_name", type=str, choices=['fBM', 'Levy','fBM_Levy'], default='fBM')

parser.add_argument("--base_path", type=str, default='.')
parser.add_argument("--data_path", type=str, default=None) #required=True)
parser.add_argument("--seq_len",    type=int, default=96)
parser.add_argument("--pred_len",   type=int, default=96)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--train_ratio", type=float, default=0.8)
parser.add_argument("--val_ratio",   type=float, default=0.1)
parser.add_argument("--scale_mode", type=str, choices=["minmax", "zscore"], default="minmax")
parser.add_argument("--lr",         type=float, default=1e-3)
parser.add_argument("--epochs",     type=int, default=30)
parser.add_argument("--ff_init", type=str, choices=["normal", "uniform"], default="uniform", help='for FrozenTE')

# model-specific args
parser.add_argument("--d_model",    type=int, default=64)
parser.add_argument("--d_ff",       type=int, default=256)
parser.add_argument("--nhead",      type=int, default=4)
parser.add_argument("--dropout",    type=float, default=0.0)
parser.add_argument("--ffn_type",   type=str, choices=["moment_ffn","mome"], default="moment_ffn")
parser.add_argument("--L_sub",      type=int, default=2)
parser.add_argument("--K",          type=int, default=4)
parser.add_argument("--beta",       type=float, default=1/30)
parser.add_argument("--num_experts",type=int, default=4)
parser.add_argument("--topk",       type=int, default=1)
parser.add_argument("--sparse_gating", type=bool, default=False)
parser.add_argument("--expert_depth",  type=int, default=2)
parser.add_argument("--num_layers",    type=int, default=2)
parser.add_argument("--c_out",      type=int, default=1)

parser.add_argument("--regularizer",     type=str, default='l2cos')
parser.add_argument("--lambda_l2",       type=float, default=1e-3)
parser.add_argument("--lambda_cos",       type=float, default=1e-3)

parser.add_argument("--gpu",        type=int, default=0)
parser.add_argument("--seed",       type=int, default=2025)

args, _ = parser.parse_known_args()

if __name__ == '__main__':
    main(args)