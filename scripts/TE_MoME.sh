python -u run.py --model MomentTE --ffn_type mome --strategy beta --data_name fBM --L_sub 2 --K 4 --num_experts 4 gpu 0 &
python -u run.py --model MomentTE --ffn_type mome --strategy beta --data_name Levy --L_sub 2 --K 4 --num_experts 4 gpu 0 &
python -u run.py --model MomentTE --ffn_type mome --strategy beta --data_name fBM_Levy --L_sub 2 --K 4 --num_experts 4 gpu 0


