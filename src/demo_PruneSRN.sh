################################################## EDSR, F64B16, pr0.5
# Note: 863 batchs per epoch, 0.5/0.0001*20/863 = 116 epochs for reg, 50 epochs for stabilize_reg. Then prune. Then finetune for 300 epochs.
# pr0.30: 2.01x speedup, 2.01x compression
# pr0.50: 3.56x speedup, 3.59x compression

# Prune (base scheme) pr=0.5
CUDA_VISIBLE_DEVICES=2 python main.py --model EDSR --scale 2 --patch_size 96 --ext sep --dir_data /home/yulun/data/SR/RGB/BIX2X3X4/pt_bin --data_train DF2K --data_test DF2K --data_range 1-3450/3551-3555 --epochs 300 --chop --save_results --save_models --n_resblocks 16 --n_feats 64 --method GReg-1 --stage_pr [0-1000:0.5] --skip_layers *mean*,*tail* --same_pruned_wg_layers model.head.0,model.body.16,*body.2 --reg_upper_limit 0.5 --reg_granularity_prune 0.0001 --update_reg_interval 20 --stabilize_reg_interval 43150 --pre_train ../pretrain_model/edsr_baseline_x2-1bc95232.pt --save EDSR_F64R16BIX2_DF2K_Prune0.5_RGP0.0001_RUL0.5_Pretrain 

# Scratch (same epochs)
CUDA_VISIBLE_DEVICES=2 python main.py --model EDSR --scale 2 --patch_size 96 --ext sep --dir_data /home/yulun/data/SR/RGB/BIX2X3X4/pt_bin --data_train DF2K --data_test DF2K --data_range 1-3450/3551-3555 --chop --save_results --save_models --n_resblocks 16 --n_feats 64 --decay_type multiple_step --decay 366 --epochs 466 --method L1 --stage_pr [0-1000:0.5] --skip_layers *mean*,*tail* --save EDSR_F64R16BIX2_DF2K_Scratch0.5_SameEpochs

# ------------------------------------------------------
# Ablation study: Ours, pr=0.1, 0.3, 0.7, 0.9
CUDA_VISIBLE_DEVICES=0 python main.py --model EDSR --scale 2 --patch_size 96 --ext sep --dir_data /home/yulun/data/SR/RGB/BIX2X3X4/pt_bin --data_train DF2K --data_test DF2K --data_range 1-3450/3551-3555 --epochs 200 --chop --save_results --save_models --n_resblocks 16 --n_feats 64 --method GReg-1 --stage_pr [0-1000:0.1] --skip_layers *mean*,*tail* --same_pruned_wg_layers model.head.0,model.body.16,*body.2 --reg_upper_limit 0.5 --reg_granularity_prune 0.0001 --update_reg_interval 20 --stabilize_reg_interval 43150 --pre_train ../pretrain_model/edsr_baseline_x2-1bc95232.pt --save Ablation/EDSR_F64R16BIX2_DF2K_Prune0.1_RGP0.0001_RUL0.5_Pretrain

CUDA_VISIBLE_DEVICES=1 python main.py --model EDSR --scale 2 --patch_size 96 --ext sep --dir_data /home/yulun/data/SR/RGB/BIX2X3X4/pt_bin --data_train DF2K --data_test DF2K --data_range 1-3450/3551-3555 --epochs 200 --chop --save_results --save_models --n_resblocks 16 --n_feats 64 --method GReg-1 --stage_pr [0-1000:0.3] --skip_layers *mean*,*tail* --same_pruned_wg_layers model.head.0,model.body.16,*body.2 --reg_upper_limit 0.5 --reg_granularity_prune 0.0001 --update_reg_interval 20 --stabilize_reg_interval 43150 --pre_train ../pretrain_model/edsr_baseline_x2-1bc95232.pt --save Ablation/EDSR_F64R16BIX2_DF2K_Prune0.3_RGP0.0001_RUL0.5_Pretrain

CUDA_VISIBLE_DEVICES=2 python main.py --model EDSR --scale 2 --patch_size 96 --ext sep --dir_data /home/yulun/data/SR/RGB/BIX2X3X4/pt_bin --data_train DF2K --data_test DF2K --data_range 1-3450/3551-3555 --epochs 200 --chop --save_results --save_models --n_resblocks 16 --n_feats 64 --method GReg-1 --stage_pr [0-1000:0.7] --skip_layers *mean*,*tail* --same_pruned_wg_layers model.head.0,model.body.16,*body.2 --reg_upper_limit 0.5 --reg_granularity_prune 0.0001 --update_reg_interval 20 --stabilize_reg_interval 43150 --pre_train ../pretrain_model/edsr_baseline_x2-1bc95232.pt --save Ablation/EDSR_F64R16BIX2_DF2K_Prune0.7_RGP0.0001_RUL0.5_Pretrain

CUDA_VISIBLE_DEVICES=3 python main.py --model EDSR --scale 2 --patch_size 96 --ext sep --dir_data /home/yulun/data/SR/RGB/BIX2X3X4/pt_bin --data_train DF2K --data_test DF2K --data_range 1-3450/3551-3555 --epochs 200 --chop --save_results --save_models --n_resblocks 16 --n_feats 64 --method GReg-1 --stage_pr [0-1000:0.9] --skip_layers *mean*,*tail* --same_pruned_wg_layers model.head.0,model.body.16,*body.2 --reg_upper_limit 0.5 --reg_granularity_prune 0.0001 --update_reg_interval 20 --stabilize_reg_interval 43150 --pre_train ../pretrain_model/edsr_baseline_x2-1bc95232.pt --save Ablation/EDSR_F64R16BIX2_DF2K_Prune0.9_RGP0.0001_RUL0.5_Pretrain

# Ablation study: L1, pr=0.1, 0.3, 0.7, 0.9
CUDA_VISIBLE_DEVICES=2 python main.py --model EDSR --scale 2 --patch_size 96 --ext sep --dir_data /home/yulun/data/SR/RGB/BIX2X3X4/pt_bin --data_train DF2K --data_test DF2K --data_range 1-3450/3551-3555 --chop --save_results --save_models --n_resblocks 16 --n_feats 64 --epochs 200 --method L1 --stage_pr [0-1000:0.1] --skip_layers *mean*,*tail* --pre_train ../pretrain_model/edsr_baseline_x2-1bc95232.pt --save Ablation/EDSR_F64R16BIX2_DF2K_L10.1_Pretrain

CUDA_VISIBLE_DEVICES=2 python main.py --model EDSR --scale 2 --patch_size 96 --ext sep --dir_data /home/yulun/data/SR/RGB/BIX2X3X4/pt_bin --data_train DF2K --data_test DF2K --data_range 1-3450/3551-3555 --chop --save_results --save_models --n_resblocks 16 --n_feats 64 --epochs 200 --method L1 --stage_pr [0-1000:0.3] --skip_layers *mean*,*tail* --pre_train ../pretrain_model/edsr_baseline_x2-1bc95232.pt --save Ablation/EDSR_F64R16BIX2_DF2K_L10.3_Pretrain

CUDA_VISIBLE_DEVICES=2 python main.py --model EDSR --scale 2 --patch_size 96 --ext sep --dir_data /home/yulun/data/SR/RGB/BIX2X3X4/pt_bin --data_train DF2K --data_test DF2K --data_range 1-3450/3551-3555 --chop --save_results --save_models --n_resblocks 16 --n_feats 64 --epochs 200 --method L1 --stage_pr [0-1000:0.7] --skip_layers *mean*,*tail* --pre_train ../pretrain_model/edsr_baseline_x2-1bc95232.pt --save Ablation/EDSR_F64R16BIX2_DF2K_L10.7_Pretrain

CUDA_VISIBLE_DEVICES=2 python main.py --model EDSR --scale 2 --patch_size 96 --ext sep --dir_data /home/yulun/data/SR/RGB/BIX2X3X4/pt_bin --data_train DF2K --data_test DF2K --data_range 1-3450/3551-3555 --chop --save_results --save_models --n_resblocks 16 --n_feats 64 --epochs 200 --method L1 --stage_pr [0-1000:0.9] --skip_layers *mean*,*tail* --pre_train ../pretrain_model/edsr_baseline_x2-1bc95232.pt --save Ablation/EDSR_F64R16BIX2_DF2K_L10.9_Pretrain

# Ablation study: Scratch, pr=0.1, 0.3, 0.7, 0.9
CUDA_VISIBLE_DEVICES=2 python main.py --model EDSR --scale 2 --patch_size 96 --ext sep --dir_data /home/yulun/data/SR/RGB/BIX2X3X4/pt_bin --data_train DF2K --data_test DF2K --data_range 1-3450/3551-3555 --chop --save_results --save_models --n_resblocks 16 --n_feats 64 --epochs 200 --method L1 --stage_pr [0-1000:0.1] --skip_layers *mean*,*tail* --save Ablation/EDSR_F64R16BIX2_DF2K_Scratch0.1

CUDA_VISIBLE_DEVICES=2 python main.py --model EDSR --scale 2 --patch_size 96 --ext sep --dir_data /home/yulun/data/SR/RGB/BIX2X3X4/pt_bin --data_train DF2K --data_test DF2K --data_range 1-3450/3551-3555 --chop --save_results --save_models --n_resblocks 16 --n_feats 64 --epochs 200 --method L1 --stage_pr [0-1000:0.3] --skip_layers *mean*,*tail* --save Ablation/EDSR_F64R16BIX2_DF2K_Scratch0.3

CUDA_VISIBLE_DEVICES=2 python main.py --model EDSR --scale 2 --patch_size 96 --ext sep --dir_data /home/yulun/data/SR/RGB/BIX2X3X4/pt_bin --data_train DF2K --data_test DF2K --data_range 1-3450/3551-3555 --chop --save_results --save_models --n_resblocks 16 --n_feats 64 --epochs 200 --method L1 --stage_pr [0-1000:0.7] --skip_layers *mean*,*tail* --save Ablation/EDSR_F64R16BIX2_DF2K_Scratch0.7

CUDA_VISIBLE_DEVICES=2 python main.py --model EDSR --scale 2 --patch_size 96 --ext sep --dir_data /home/yulun/data/SR/RGB/BIX2X3X4/pt_bin --data_train DF2K --data_test DF2K --data_range 1-3450/3551-3555 --chop --save_results --save_models --n_resblocks 16 --n_feats 64 --epochs 200 --method L1 --stage_pr [0-1000:0.9] --skip_layers *mean*,*tail* --save Ablation/EDSR_F64R16BIX2_DF2K_Scratch0.9


# Ablation study: from F256/F128 to F32
CUDA_VISIBLE_DEVICES=1 python main.py --model EDSR --scale 2 --patch_size 96 --ext sep --dir_data /home/yulun/data/SR/RGB/BIX2X3X4/pt_bin --data_train DF2K --data_test DF2K --data_range 1-3450/3551-3555 --epochs 200 --chop --save_results --save_models --n_resblocks 16 --n_feats 128 --method GReg-1 --stage_pr [0-1000:0.75] --skip_layers *mean*,*tail* --same_pruned_wg_layers model.head.0,model.body.16,*body.2 --reg_upper_limit 0.5 --reg_granularity_prune 0.0001 --update_reg_interval 20 --stabilize_reg_interval 43150 --pre_train ../pretrain_model/EDSR_F128R16P48B16BIX2M1277.pt --save Ablation/EDSR_F128R16BIX2_DF2K_Prune0.75_RGP0.0001_RUL0.5_Pretrain

CUDA_VISIBLE_DEVICES=2 python main.py --model EDSR --scale 2 --patch_size 96 --ext sep --dir_data /home/yulun/data/SR/RGB/BIX2X3X4/pt_bin --data_train DF2K --data_test DF2K --data_range 1-3450/3551-3555 --epochs 200 --chop --save_results --save_models --n_resblocks 16 --n_feats 256 --method GReg-1 --stage_pr [0-1000:0.875] --skip_layers *mean*,*tail* --same_pruned_wg_layers model.head.0,model.body.16,*body.2 --reg_upper_limit 0.5 --reg_granularity_prune 0.0001 --update_reg_interval 20 --stabilize_reg_interval 43150 --pre_train ../pretrain_model/EDSR_F256R16P48B16BIX2M2299.pt --save Ablation/EDSR_F256R16BIX2_DF2K_Prune0.875_RGP0.0001_RUL0.5_Pretrain
# ------------------------------------------------------



# LEDSR_F256R16BIX2_DF2K_M311.pt
# LEDSR_F256R16BIX3_DF2K_M230.pt
# LEDSR_F256R16BIX4_DF2K_M231.pt
################## Small model
# Prune from 256 to 45, pr=0.82421875, x2
CUDA_VISIBLE_DEVICES=1 python main.py --model LEDSR --scale 2 --patch_size 96 --ext sep --dir_data /home/yulun/data/SR/RGB/BIX2X3X4/pt_bin --data_train DF2K --data_test DF2K --data_range 1-3550/3551-3555 --chop --save_results --n_resblocks 16 --n_feats 256 --method GReg-1 --stage_pr [0-1000:0.82421875] --skip_layers *mean*,*tail* --same_pruned_wg_layers model.head.0,model.body.16,*body.2 --reg_upper_limit 0.5 --reg_granularity_prune 0.0001 --update_reg_interval 20 --stabilize_reg_interval 43150 --pre_train ../pretrain_model/LEDSR_F256R16BIX2_DF2K_M311.pt --save LEDSR_F256R16BIX2_DF2K_Prune0.82421875_RGP0.0001_RUL0.5_Pretrain

CUDA_VISIBLE_DEVICES=3 python main.py --model LEDSR --scale 2 --patch_size 96 --ext sep --dir_data /home/yulun/data/SR/RGB/BIX2X3X4/pt_bin --data_train DF2K --data_test DF2K --data_range 1-3550/3551-3555 --chop --save_results --n_resblocks 16 --n_feats 256 --pre_train ../pretrain_model/LEDSR_F256R16BIX2_DF2K_M311.pt --save LEDSR_F256R16BIX2_DF2K_Prune0.82421875_RGP0.0001_RUL0.5_Pretrain


# Prune from 256 to 45, pr=0.82421875, x3
CUDA_VISIBLE_DEVICES=2 python main.py --model LEDSR --scale 3 --patch_size 144 --ext sep --dir_data /home/yulun/data/SR/RGB/BIX2X3X4/pt_bin --data_train DF2K --data_test DF2K --data_range 1-3550/3551-3555 --chop --save_results --n_resblocks 16 --n_feats 256 --method GReg-1 --stage_pr [0-1000:0.82421875] --skip_layers *mean*,*tail* --same_pruned_wg_layers model.head.0,model.body.16,*body.2 --reg_upper_limit 0.5 --reg_granularity_prune 0.0001 --update_reg_interval 20 --stabilize_reg_interval 43150 --pre_train ../pretrain_model/LEDSR_F256R16BIX3_DF2K_M230.pt --save LEDSR_F256R16BIX3_DF2K_Prune0.82421875_RGP0.0001_RUL0.5_Pretrain

# Prune from 256 to 45, pr=0.82421875, x4
CUDA_VISIBLE_DEVICES=3 python main.py --model LEDSR --scale 4 --patch_size 192 --ext sep --dir_data /home/yulun/data/SR/RGB/BIX2X3X4/pt_bin --data_train DF2K --data_test DF2K --data_range 1-3550/3551-3555 --chop --save_results --n_resblocks 16 --n_feats 256 --method GReg-1 --stage_pr [0-1000:0.82421875] --skip_layers *mean*,*tail* --same_pruned_wg_layers model.head.0,model.body.16,*body.2 --reg_upper_limit 0.5 --reg_granularity_prune 0.0001 --update_reg_interval 20 --stabilize_reg_interval 43150 --pre_train ../pretrain_model/LEDSR_F256R16BIX4_DF2K_M231.pt --save LEDSR_F256R16BIX4_DF2K_Prune0.82421875_RGP0.0001_RUL0.5_Pretrain

##
# Prune from 256 to 49, pr=0.80859375, x2
CUDA_VISIBLE_DEVICES=1 python main.py --model LEDSR --scale 2 --patch_size 96 --ext sep --dir_data /home/yulun/data/SR/RGB/BIX2X3X4/pt_bin --data_train DF2K --data_test DF2K --data_range 1-3550/3551-3555 --chop --save_results --n_resblocks 16 --n_feats 256 --method GReg-1 --stage_pr [0-1000:0.80859375] --skip_layers *mean*,*tail* --same_pruned_wg_layers model.head.0,model.body.16,*body.2 --reg_upper_limit 0.5 --reg_granularity_prune 0.0001 --update_reg_interval 20 --stabilize_reg_interval 43150 --pre_train ../pretrain_model/LEDSR_F256R16BIX2_DF2K_M311.pt --save LEDSR_F256R16BIX2_DF2K_Prune0.80859375_RGP0.0001_RUL0.5_Pretrain
# Prune from 256 to 49, pr=0.80859375, x3
CUDA_VISIBLE_DEVICES=2 python main.py --model LEDSR --scale 3 --patch_size 144 --ext sep --dir_data /home/yulun/data/SR/RGB/BIX2X3X4/pt_bin --data_train DF2K --data_test DF2K --data_range 1-3550/3551-3555 --chop --save_results --n_resblocks 16 --n_feats 256 --method GReg-1 --stage_pr [0-1000:0.80859375] --skip_layers *mean*,*tail* --same_pruned_wg_layers model.head.0,model.body.16,*body.2 --reg_upper_limit 0.5 --reg_granularity_prune 0.0001 --update_reg_interval 20 --stabilize_reg_interval 43150 --pre_train ../pretrain_model/LEDSR_F256R16BIX3_DF2K_M230.pt --save LEDSR_F256R16BIX3_DF2K_Prune0.80859375_RGP0.0001_RUL0.5_Pretrain

# Prune from 256 to 49, pr=0.80859375, x4
CUDA_VISIBLE_DEVICES=3 python main.py --model LEDSR --scale 4 --patch_size 192 --ext sep --dir_data /home/yulun/data/SR/RGB/BIX2X3X4/pt_bin --data_train DF2K --data_test DF2K --data_range 1-3550/3551-3555 --chop --save_results --n_resblocks 16 --n_feats 256 --method GReg-1 --stage_pr [0-1000:0.80859375] --skip_layers *mean*,*tail* --same_pruned_wg_layers model.head.0,model.body.16,*body.2 --reg_upper_limit 0.5 --reg_granularity_prune 0.0001 --update_reg_interval 20 --stabilize_reg_interval 43150 --pre_train ../pretrain_model/LEDSR_F256R16BIX4_DF2K_M231.pt --save LEDSR_F256R16BIX4_DF2K_Prune0.80859375_RGP0.0001_RUL0.5_Pretrain

##
# Prune from 64 to 49, pr=0.234375, x2
CUDA_VISIBLE_DEVICES=1 python main.py --model LEDSR --scale 2 --patch_size 96 --ext sep --dir_data /home/yulun/data/SR/RGB/BIX2X3X4/pt_bin --data_train DF2K --data_test DF2K --data_range 1-3550/3551-3555 --chop --save_results --n_resblocks 16 --n_feats 64 --method GReg-1 --stage_pr [0-1000:0.234375] --skip_layers *mean*,*tail* --same_pruned_wg_layers model.head.0,model.body.16,*body.2 --reg_upper_limit 0.5 --reg_granularity_prune 0.0001 --update_reg_interval 20 --stabilize_reg_interval 43150 --pre_train ../pretrain_model/LEDSR_F64R16B16BIX2M840.pt --save LEDSR_T49_F64R16BIX2_Prune0.234375_RGP0.0001_RUL0.5_Pretrain
# Prune from 64 to 49, pr=0.234375, x3
CUDA_VISIBLE_DEVICES=2 python main.py --model LEDSR --scale 3 --patch_size 144 --ext sep --dir_data /home/yulun/data/SR/RGB/BIX2X3X4/pt_bin --data_train DF2K --data_test DF2K --data_range 1-3550/3551-3555 --chop --save_results --n_resblocks 16 --n_feats 64 --method GReg-1 --stage_pr [0-1000:0.234375] --skip_layers *mean*,*tail* --same_pruned_wg_layers model.head.0,model.body.16,*body.2 --reg_upper_limit 0.5 --reg_granularity_prune 0.0001 --update_reg_interval 20 --stabilize_reg_interval 43150 --pre_train ../pretrain_model/LEDSR_F64R16B16BIX3M645.pt --save LEDSR_T49_F64R16BIX3_Prune0.234375_RGP0.0001_RUL0.5_Pretrain

# Prune from 64 to 49, pr=0.234375, x4
CUDA_VISIBLE_DEVICES=3 python main.py --model LEDSR --scale 4 --patch_size 192 --ext sep --dir_data /home/yulun/data/SR/RGB/BIX2X3X4/pt_bin --data_train DF2K --data_test DF2K --data_range 1-3550/3551-3555 --chop --save_results --n_resblocks 16 --n_feats 64 --method GReg-1 --stage_pr [0-1000:0.234375] --skip_layers *mean*,*tail* --same_pruned_wg_layers model.head.0,model.body.16,*body.2 --reg_upper_limit 0.5 --reg_granularity_prune 0.0001 --update_reg_interval 20 --stabilize_reg_interval 43150 --pre_train ../pretrain_model/LEDSR_F64R16B16BIX4M642.pt --save LEDSR_T49_F64R16BIX4_Prune0.234375_RGP0.0001_RUL0.5_Pretrain




################## Large model 
# Prune RIRSRX2
CUDA_VISIBLE_DEVICES=2 python main.py --model RIRSR --scale 2 --patch_size 96 --chop --save_results --n_resgroups 10 --n_resblocks 20 --n_feats 96  --method GReg-1 --stage_pr [0-1000:0.34] --skip_layers *mean*,*tail* --same_pruned_wg_layers model.head.0,model.body.10,*body.2,*body.20 --reg_upper_limit 0.5 --reg_granularity_prune 0.0001 --update_reg_interval 20 --stabilize_reg_interval 43150 --save RIRSR_BIX2_G10R20F96P48_Prune0.34_RGP0.0001_RUL0.5_Pretrain --pre_train ../pretrain_model/RIRSR_BIX2_G10R20F96P48M793.pt 

#CUDA_VISIBLE_DEVICES=2 python main.py --model RIRSR --scale 2 --patch_size 96 --chop --save_results --n_resgroups 10 --n_resblocks 20 --n_feats 96  --method GReg-1 --stage_pr [0-1000:0.34] --skip_layers *mean*,*tail* --same_pruned_wg_layers model.head.0,model.body.10,*body.2,*body.20 --reg_upper_limit 1e-3 --reg_granularity_prune 0.0001 --update_reg_interval 20 --stabilize_reg_interval 50 --save RIRSR_BIX2_G10R20F96P48_Prune0.34_RGP0.0001_RUL0.5_Pretrain --pre_train ../pretrain_model/RIRSR_BIX2_G10R20F96P48M793.pt 

# Prune RIRSRX3
CUDA_VISIBLE_DEVICES=2 python main.py --model RIRSR --scale 3 --patch_size 144 --chop --save_results --n_resgroups 10 --n_resblocks 20 --n_feats 96  --method GReg-1 --stage_pr [0-1000:0.34] --skip_layers *mean*,*tail* --same_pruned_wg_layers model.head.0,model.body.10,*body.2,*body.20 --reg_upper_limit 0.5 --reg_granularity_prune 0.0001 --update_reg_interval 20 --stabilize_reg_interval 43150 --save RIRSR_BIX3_G10R20F96P48_Prune0.34_RGP0.0001_RUL0.5_Pretrain --pre_train ../pretrain_model/RIRSR_BIX3_G10R20F96P48M589.pt 

# Prune RIRSRX4
CUDA_VISIBLE_DEVICES=3 python main.py --model RIRSR --scale 4 --patch_size 192 --chop --save_results --n_resgroups 10 --n_resblocks 20 --n_feats 96  --method GReg-1 --stage_pr [0-1000:0.34] --skip_layers *mean*,*tail* --same_pruned_wg_layers model.head.0,model.body.10,*body.2,*body.20 --reg_upper_limit 0.5 --reg_granularity_prune 0.0001 --update_reg_interval 20 --stabilize_reg_interval 43150 --save RIRSR_BIX4_G10R20F96P48_Prune0.34_RGP0.0001_RUL0.5_Pretrain --pre_train ../pretrain_model/RIRSR_BIX4_G10R20F96P48M88.pt 




################################################## Resume (the same as before)
# For example, if we want to resume experiment <Experiment Name>:
CUDA_VISIBLE_DEVICES=0 python main.py --model EDSR --scale 2 --patch_size 96 --ext sep --dir_data /home/yulun/data/SR/RGB/BIX2X3X4/pt_bin --data_train DF2K --data_test DF2K  --data_range 1-3450/3551-3555 --epochs 300 --chop --save_results --n_resblocks 16 --n_feats 64 --load <Experiment Name> --resume -1


################################################## Test (the same as before)
CUDA_VISIBLE_DEVICES=0 python main.py --data_test Demo --model EDSR --scale 2 --n_resblocks 16 --n_feats 64  --reset --dir_demo /media/yulun/10THD1/data/super-resolution/LRBI/Set5/x2  --test_only --save_results --pre_train ../experiment/EDSR_F64R16BIX2_DF2K__GReg-1_pr0.5_granularity0.0001_limit0.5/model/model_latest.pt --save EDSR_F64R16BIX2_DF2K__GReg-1_pr0.5_granularity0.0001_limit0.5__Test






################################################## FLOPs and Params Summary
EDSR, RF256R16, x2
!python cal_modelsize.py --model EDSR --scale 2 --n_resblocks 16 --n_feats 256
------------------------------------------------------------------------------------------
                             Totals
Total params             21.847067M
Trainable params         21.847043M
Non-trainable params           24.0
Mult-Adds             5.0360942592T
==========================================================================================

IMDN: 694K，Flops=158.8G


################################################## Below are the scripts of ASSL method (NIPS'21) ##################################################
# Ablation study: Ours, pr=0.1, 0.3, 0.5, 0.7, 0.9
CUDA_VISIBLE_DEVICES=3 nohup python main.py --model EDSR --scale 2 --patch_size 96 --ext sep --dir_data /home/yulun/data/SR/RGB/BIX2X3X4/pt_bin --data_train DF2K --data_test DF2K --data_range 1-3450/3551-3555 --epochs 200 --chop --save_results --save_models --n_resblocks 16 --n_feats 64 --method ASSL --wn --stage_pr [0-1000:0.1] --skip_layers *mean*,*tail* --same_pruned_wg_layers model.head.0,model.body.16,*body.2 --reg_upper_limit 0.5 --reg_granularity_prune 0.0001 --update_reg_interval 20 --stabilize_reg_interval 43150 --pre_train ../pretrain_model/edsr_baseline_x2-1bc95232.pt --same_pruned_wg_criterion reg --save Ablation/EDSR_F64R16BIX2_DF2K_ASSL0.1_RGP0.0001_RUL0.5_Pretrain > /dev/null & 

CUDA_VISIBLE_DEVICES=0 nohup python main.py --model EDSR --scale 2 --patch_size 96 --ext sep --dir_data /home/yulun/data/SR/RGB/BIX2X3X4/pt_bin --data_train DF2K --data_test DF2K --data_range 1-3450/3551-3555 --epochs 200 --chop --save_results --save_models --n_resblocks 16 --n_feats 64 --method ASSL --wn --stage_pr [0-1000:0.3] --skip_layers *mean*,*tail* --same_pruned_wg_layers model.head.0,model.body.16,*body.2 --reg_upper_limit 0.5 --reg_granularity_prune 0.0001 --update_reg_interval 20 --stabilize_reg_interval 43150 --pre_train ../pretrain_model/edsr_baseline_x2-1bc95232.pt --same_pruned_wg_criterion reg --save Ablation/EDSR_F64R16BIX2_DF2K_ASSL0.3_RGP0.0001_RUL0.5_Pretrain > /dev/null & 

CUDA_VISIBLE_DEVICES=1 nohup python main.py --model EDSR --scale 2 --patch_size 96 --ext sep --dir_data /home/yulun/data/SR/RGB/BIX2X3X4/pt_bin --data_train DF2K --data_test DF2K --data_range 1-3450/3551-3555 --epochs 200 --chop --save_results --save_models --n_resblocks 16 --n_feats 64 --method ASSL --wn --stage_pr [0-1000:0.5] --skip_layers *mean*,*tail* --same_pruned_wg_layers model.head.0,model.body.16,*body.2 --reg_upper_limit 0.5 --reg_granularity_prune 0.0001 --update_reg_interval 20 --stabilize_reg_interval 43150 --pre_train ../pretrain_model/edsr_baseline_x2-1bc95232.pt --same_pruned_wg_criterion reg --save Ablation/EDSR_F64R16BIX2_DF2K_ASSL0.5_RGP0.0001_RUL0.5_Pretrain > /dev/null & 

CUDA_VISIBLE_DEVICES=2 nohup python main.py --model EDSR --scale 2 --patch_size 96 --ext sep --dir_data /home/yulun/data/SR/RGB/BIX2X3X4/pt_bin --data_train DF2K --data_test DF2K --data_range 1-3450/3551-3555 --epochs 200 --chop --save_results --save_models --n_resblocks 16 --n_feats 64 --method ASSL --wn --stage_pr [0-1000:0.7] --skip_layers *mean*,*tail* --same_pruned_wg_layers model.head.0,model.body.16,*body.2 --reg_upper_limit 0.5 --reg_granularity_prune 0.0001 --update_reg_interval 20 --stabilize_reg_interval 43150 --pre_train ../pretrain_model/edsr_baseline_x2-1bc95232.pt --same_pruned_wg_criterion reg --save Ablation/EDSR_F64R16BIX2_DF2K_ASSL0.7_RGP0.0001_RUL0.5_Pretrain > /dev/null & 

CUDA_VISIBLE_DEVICES=3 nohup python main.py --model EDSR --scale 2 --patch_size 96 --ext sep --dir_data /home/yulun/data/SR/RGB/BIX2X3X4/pt_bin --data_train DF2K --data_test DF2K --data_range 1-3450/3551-3555 --epochs 200 --chop --save_results --save_models --n_resblocks 16 --n_feats 64 --method ASSL --wn --stage_pr [0-1000:0.9] --skip_layers *mean*,*tail* --same_pruned_wg_layers model.head.0,model.body.16,*body.2 --reg_upper_limit 0.5 --reg_granularity_prune 0.0001 --update_reg_interval 20 --stabilize_reg_interval 43150 --pre_train ../pretrain_model/edsr_baseline_x2-1bc95232.pt --same_pruned_wg_criterion reg --save Ablation/EDSR_F64R16BIX2_DF2K_ASSL0.9_RGP0.0001_RUL0.5_Pretrain > /dev/null & 


# main 
# Prune from 256 to 49, pr=0.80859375, x2
CUDA_VISIBLE_DEVICES=1 python main.py --model LEDSR --scale 2 --patch_size 96 --ext sep --dir_data /home/yulun/data/SR/RGB/BIX2X3X4/pt_bin --data_train DF2K --data_test DF2K --data_range 1-3550/3551-3555 --chop --save_results --n_resblocks 16 --n_feats 256 --method ASSL --wn --stage_pr [0-1000:0.80859375] --skip_layers *mean*,*tail* --same_pruned_wg_layers model.head.0,model.body.16,*body.2 --reg_upper_limit 0.5 --reg_granularity_prune 0.0001 --update_reg_interval 20 --stabilize_reg_interval 43150 --pre_train ../pretrain_model/LEDSR_F256R16BIX2_DF2K_M311.pt --same_pruned_wg_criterion reg --save main/SR/LEDSR_F256R16BIX2_DF2K_ASSL0.80859375_RGP0.0001_RUL0.5_Pretrain

# Prune from 256 to 49, pr=0.80859375, x3
CUDA_VISIBLE_DEVICES=2 python main.py --model LEDSR --scale 3 --patch_size 144 --ext sep --dir_data /home/yulun/data/SR/RGB/BIX2X3X4/pt_bin --data_train DF2K --data_test DF2K --data_range 1-3550/3551-3555 --chop --save_results --n_resblocks 16 --n_feats 256 --method ASSL --wn --stage_pr [0-1000:0.80859375] --skip_layers *mean*,*tail* --same_pruned_wg_layers model.head.0,model.body.16,*body.2 --reg_upper_limit 0.5 --reg_granularity_prune 0.0001 --update_reg_interval 20 --stabilize_reg_interval 43150 --pre_train ../pretrain_model/LEDSR_F256R16BIX3_DF2K_M230.pt --same_pruned_wg_criterion reg --save main/SR/LEDSR_F256R16BIX3_DF2K_ASSL0.80859375_RGP0.0001_RUL0.5_Pretrain

# Prune from 256 to 49, pr=0.80859375, x4
CUDA_VISIBLE_DEVICES=3 python main.py --model LEDSR --scale 4 --patch_size 192 --ext sep --dir_data /home/yulun/data/SR/RGB/BIX2X3X4/pt_bin --data_train DF2K --data_test DF2K --data_range 1-3550/3551-3555 --chop --save_results --n_resblocks 16 --n_feats 256 --method ASSL --wn --stage_pr [0-1000:0.80859375] --skip_layers *mean*,*tail* --same_pruned_wg_layers model.head.0,model.body.16,*body.2 --reg_upper_limit 0.5 --reg_granularity_prune 0.0001 --update_reg_interval 20 --stabilize_reg_interval 43150 --pre_train ../pretrain_model/LEDSR_F256R16BIX4_DF2K_M231.pt --same_pruned_wg_criterion reg --save main/SR/LEDSR_F256R16BIX4_DF2K_ASSL0.80859375_RGP0.0001_RUL0.5_Pretrain


