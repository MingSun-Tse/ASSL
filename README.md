# Ablation Study
EDSR baseline (16 blocks).
```
# PR=0.1
CUDA_VISIBLE_DEVICES=3 nohup python main.py --model EDSR --scale 2 --patch_size 96 --ext sep --dir_data /home/yulun/data/SR/RGB/BIX2X3X4/pt_bin --data_train DF2K --data_test DF2K --data_range 1-3450/3551-3555 --epochs 200 --chop --save_results --save_models --n_resblocks 16 --n_feats 64 --method ASSL --wn --stage_pr [0-1000:0.1] --skip_layers *mean*,*tail* --same_pruned_wg_layers model.head.0,model.body.16,*body.2 --reg_upper_limit 0.5 --reg_granularity_prune 0.0001 --update_reg_interval 20 --stabilize_reg_interval 43150 --pre_train ../pretrain_model/edsr_baseline_x2-1bc95232.pt --save Ablation/EDSR_F64R16BIX2_DF2K_ASSL0.1_RGP0.0001_RUL0.5_Pretrain > /dev/null & 

# PR=0.3
CUDA_VISIBLE_DEVICES=0 nohup python main.py --model EDSR --scale 2 --patch_size 96 --ext sep --dir_data /home/yulun/data/SR/RGB/BIX2X3X4/pt_bin --data_train DF2K --data_test DF2K --data_range 1-3450/3551-3555 --epochs 200 --chop --save_results --save_models --n_resblocks 16 --n_feats 64 --method ASSL --wn --stage_pr [0-1000:0.3] --skip_layers *mean*,*tail* --same_pruned_wg_layers model.head.0,model.body.16,*body.2 --reg_upper_limit 0.5 --reg_granularity_prune 0.0001 --update_reg_interval 20 --stabilize_reg_interval 43150 --pre_train ../pretrain_model/edsr_baseline_x2-1bc95232.pt --save Ablation/EDSR_F64R16BIX2_DF2K_ASSL0.3_RGP0.0001_RUL0.5_Pretrain > /dev/null & 

# PR=0.5
CUDA_VISIBLE_DEVICES=1 nohup python main.py --model EDSR --scale 2 --patch_size 96 --ext sep --dir_data /home/yulun/data/SR/RGB/BIX2X3X4/pt_bin --data_train DF2K --data_test DF2K --data_range 1-3450/3551-3555 --epochs 200 --chop --save_results --save_models --n_resblocks 16 --n_feats 64 --method ASSL --wn --stage_pr [0-1000:0.5] --skip_layers *mean*,*tail* --same_pruned_wg_layers model.head.0,model.body.16,*body.2 --reg_upper_limit 0.5 --reg_granularity_prune 0.0001 --update_reg_interval 20 --stabilize_reg_interval 43150 --pre_train ../pretrain_model/edsr_baseline_x2-1bc95232.pt --save Ablation/EDSR_F64R16BIX2_DF2K_ASSL0.5_RGP0.0001_RUL0.5_Pretrain > /dev/null & 

# PR=0.7
CUDA_VISIBLE_DEVICES=2 nohup python main.py --model EDSR --scale 2 --patch_size 96 --ext sep --dir_data /home/yulun/data/SR/RGB/BIX2X3X4/pt_bin --data_train DF2K --data_test DF2K --data_range 1-3450/3551-3555 --epochs 200 --chop --save_results --save_models --n_resblocks 16 --n_feats 64 --method ASSL --wn --stage_pr [0-1000:0.7] --skip_layers *mean*,*tail* --same_pruned_wg_layers model.head.0,model.body.16,*body.2 --reg_upper_limit 0.5 --reg_granularity_prune 0.0001 --update_reg_interval 20 --stabilize_reg_interval 43150 --pre_train ../pretrain_model/edsr_baseline_x2-1bc95232.pt --save Ablation/EDSR_F64R16BIX2_DF2K_ASSL0.7_RGP0.0001_RUL0.5_Pretrain > /dev/null & 

# PR=0.9
CUDA_VISIBLE_DEVICES=3 nohup python main.py --model EDSR --scale 2 --patch_size 96 --ext sep --dir_data /home/yulun/data/SR/RGB/BIX2X3X4/pt_bin --data_train DF2K --data_test DF2K --data_range 1-3450/3551-3555 --epochs 200 --chop --save_results --save_models --n_resblocks 16 --n_feats 64 --method ASSL --wn --stage_pr [0-1000:0.9] --skip_layers *mean*,*tail* --same_pruned_wg_layers model.head.0,model.body.16,*body.2 --reg_upper_limit 0.5 --reg_granularity_prune 0.0001 --update_reg_interval 20 --stabilize_reg_interval 43150 --pre_train ../pretrain_model/edsr_baseline_x2-1bc95232.pt --save Ablation/EDSR_F64R16BIX2_DF2K_ASSL0.9_RGP0.0001_RUL0.5_Pretrain > /dev/null &
```
# Main Benchmark
LEDSR baseline (16 blocks).
--method ASSL --wn
--stage_pr [0-1000:0.1]
--save Ablation/EDSR_F64R16BIX2_DF2K_ASSL0.1_RGP0.0001_RUL0.5_Pretrain

```
# Prune from 256 to 49, pr=0.80859375, x2
CUDA_VISIBLE_DEVICES=1 python main.py --model LEDSR --scale 2 --patch_size 96 --ext sep --dir_data /home/yulun/data/SR/RGB/BIX2X3X4/pt_bin --data_train DF2K --data_test DF2K --data_range 1-3550/3551-3555 --chop --save_results --n_resblocks 16 --n_feats 256 --method ASSL --wn --stage_pr [0-1000:0.80859375] --skip_layers *mean*,*tail* --same_pruned_wg_layers model.head.0,model.body.16,*body.2 --reg_upper_limit 0.5 --reg_granularity_prune 0.0001 --update_reg_interval 20 --stabilize_reg_interval 43150 --pre_train ../pretrain_model/LEDSR_F256R16BIX2_DF2K_M311.pt --save main/SR/LEDSR_F256R16BIX2_DF2K_ASSL0.80859375_RGP0.0001_RUL0.5_Pretrain

# Prune from 256 to 49, pr=0.80859375, x3
CUDA_VISIBLE_DEVICES=2 python main.py --model LEDSR --scale 3 --patch_size 144 --ext sep --dir_data /home/yulun/data/SR/RGB/BIX2X3X4/pt_bin --data_train DF2K --data_test DF2K --data_range 1-3550/3551-3555 --chop --save_results --n_resblocks 16 --n_feats 256 --method ASSL --wn --stage_pr [0-1000:0.80859375] --skip_layers *mean*,*tail* --same_pruned_wg_layers model.head.0,model.body.16,*body.2 --reg_upper_limit 0.5 --reg_granularity_prune 0.0001 --update_reg_interval 20 --stabilize_reg_interval 43150 --pre_train ../pretrain_model/LEDSR_F256R16BIX3_DF2K_M230.pt --save main/SR/LEDSR_F256R16BIX3_DF2K_ASSL0.80859375_RGP0.0001_RUL0.5_Pretrain

# Prune from 256 to 49, pr=0.80859375, x4
CUDA_VISIBLE_DEVICES=3 python main.py --model LEDSR --scale 4 --patch_size 192 --ext sep --dir_data /home/yulun/data/SR/RGB/BIX2X3X4/pt_bin --data_train DF2K --data_test DF2K --data_range 1-3550/3551-3555 --chop --save_results --n_resblocks 16 --n_feats 256 --method ASSL --wn --stage_pr [0-1000:0.80859375] --skip_layers *mean*,*tail* --same_pruned_wg_layers model.head.0,model.body.16,*body.2 --reg_upper_limit 0.5 --reg_granularity_prune 0.0001 --update_reg_interval 20 --stabilize_reg_interval 43150 --pre_train ../pretrain_model/LEDSR_F256R16BIX4_DF2K_M231.pt --save main/SR/LEDSR_F256R16BIX4_DF2K_ASSL0.80859375_RGP0.0001_RUL0.5_Pretrain
```
