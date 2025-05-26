def get_model_id(args):
    if args.model == 'MomentTE':
        model_id = '{}_{}_{}_{}_K{}_Lsub{}_M{}_H{}_s{}_p{}_d{}_ff{}_head{}_l{}_{}_l2{}_cos{}'.format(
                args.no,
                args.data_name,
                args.model,
                args.ffn_type,
                args.K,
                args.L_sub,
                args.num_experts,
                args.expert_depth,    
                args.seq_len,
                args.pred_len,
                args.d_model,
                args.d_ff,
                args.nhead,
                args.num_layers,
                args.regularizer,
                str(args.lambda_l2).replace('0.','_'),
                str(args.lambda_cos).replace('0.','_')
        )            
        if args.sparse_gating == True : model_id += '_sparse'
        if args.strategy == 'distinct' : model_id += '_distinct'
  

    if args.model == 'MomentTE_NoSampling':
        model_id = '{}_{}_{}_{}_K{}_Lsub{}_M{}_H{}_s{}_p{}_d{}_ff{}_head{}_l{}_{}_l2{}_cos{}'.format(
                args.no,
                args.data_name,
                args.model,
                args.ffn_type,
                args.K,
                args.L_sub,
                args.num_experts,
                args.expert_depth,    
                args.seq_len,
                args.pred_len,
                args.d_model,
                args.d_ff,
                args.nhead,
                args.num_layers,
                args.regularizer,
                str(args.lambda_l2).replace('0.','_'),
                str(args.lambda_cos).replace('0.','_')
        )            
        if args.sparse_gating == True : model_id += '_sparse'
        if args.strategy == 'distinct' : model_id += '_distinct'
    

    elif args.model == 'StandardTE':
        model_id = '{}_{}_{}_s{}_p{}_d{}_ff{}_head{}_l{}'.format(
                args.no,
                args.data_name,
                args.model,  
                args.seq_len,
                args.pred_len,
                args.d_model,
                args.d_ff,
                args.nhead,
                args.num_layers
        )                    
        
    elif args.model == 'SwitchTE':
        model_id = '{}_{}_{}_M{}_s{}_p{}_d{}_ff{}_head{}_l{}'.format(
                args.no,
                args.data_name,
                args.model,
                args.num_experts,
                args.seq_len,
                args.pred_len,
                args.d_model,
                args.d_ff,
                args.nhead,
                args.num_layers
        )            
        
        if args.sparse_gating == True : model_id += '_sparse'
        
    elif args.model == 'FrozenTE':
        model_id = '{}_{}_{}_{}_s{}_p{}_d{}_ff{}_head{}_l{}'.format(
                args.no,
                args.data_name,
                args.model,
                args.ff_init, 
                args.seq_len,
                args.pred_len,
                args.d_model,
                args.d_ff,
                args.nhead,
                args.num_layers
        )               
        

    elif args.model == 'SparseTE':
        model_id = '{}_{}_{}_s{}_p{}_d{}_ff{}_head{}_l{}'.format(
                args.no,
                args.data_name,
                args.model,  
                args.seq_len,
                args.pred_len,
                args.d_model,
                args.d_ff,
                args.nhead,
                args.num_layers
        )                   
                  
    return model_id