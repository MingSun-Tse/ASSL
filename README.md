# Ablation Study
EDSR baseline (16 blocks).
```python
# PR=0.1
CUDA_VISIBLE_DEVICES=3 nohup python main.py --model EDSR --scale 2 --patch_size 96 --ext sep --dir_data /home/yulun/data/SR/RGB/BIX2X3X4/pt_bin --data_train DF2K --data_test DF2K --data_range 1-3450/3551-3555 --epochs 200 --chop --save_results --save_models --n_resblocks 16 --n_feats 64 --method ASSL --wn --stage_pr [0-1000:0.1] --skip_layers *mean*,*tail* --same_pruned_wg_layers model.head.0,model.body.16,*body.2 --reg_upper_limit 0.5 --reg_granularity_prune 0.0001 --update_reg_interval 20 --stabilize_reg_interval 43150 --pre_train ../pretrain_model/edsr_baseline_x2-1bc95232.pt --same_pruned_wg_criterion reg --save Ablation/EDSR_F64R16BIX2_DF2K_ASSL0.1_RGP0.0001_RUL0.5_Pretrain > /dev/null & 

# PR=0.3
CUDA_VISIBLE_DEVICES=0 nohup python main.py --model EDSR --scale 2 --patch_size 96 --ext sep --dir_data /home/yulun/data/SR/RGB/BIX2X3X4/pt_bin --data_train DF2K --data_test DF2K --data_range 1-3450/3551-3555 --epochs 200 --chop --save_results --save_models --n_resblocks 16 --n_feats 64 --method ASSL --wn --stage_pr [0-1000:0.3] --skip_layers *mean*,*tail* --same_pruned_wg_layers model.head.0,model.body.16,*body.2 --reg_upper_limit 0.5 --reg_granularity_prune 0.0001 --update_reg_interval 20 --stabilize_reg_interval 43150 --pre_train ../pretrain_model/edsr_baseline_x2-1bc95232.pt --same_pruned_wg_criterion reg --save Ablation/EDSR_F64R16BIX2_DF2K_ASSL0.3_RGP0.0001_RUL0.5_Pretrain > /dev/null & 

# PR=0.5
CUDA_VISIBLE_DEVICES=1 nohup python main.py --model EDSR --scale 2 --patch_size 96 --ext sep --dir_data /home/yulun/data/SR/RGB/BIX2X3X4/pt_bin --data_train DF2K --data_test DF2K --data_range 1-3450/3551-3555 --epochs 200 --chop --save_results --save_models --n_resblocks 16 --n_feats 64 --method ASSL --wn --stage_pr [0-1000:0.5] --skip_layers *mean*,*tail* --same_pruned_wg_layers model.head.0,model.body.16,*body.2 --reg_upper_limit 0.5 --reg_granularity_prune 0.0001 --update_reg_interval 20 --stabilize_reg_interval 43150 --pre_train ../pretrain_model/edsr_baseline_x2-1bc95232.pt --same_pruned_wg_criterion reg --save Ablation/EDSR_F64R16BIX2_DF2K_ASSL0.5_RGP0.0001_RUL0.5_Pretrain > /dev/null & 

# PR=0.7
CUDA_VISIBLE_DEVICES=2 nohup python main.py --model EDSR --scale 2 --patch_size 96 --ext sep --dir_data /home/yulun/data/SR/RGB/BIX2X3X4/pt_bin --data_train DF2K --data_test DF2K --data_range 1-3450/3551-3555 --epochs 200 --chop --save_results --save_models --n_resblocks 16 --n_feats 64 --method ASSL --wn --stage_pr [0-1000:0.7] --skip_layers *mean*,*tail* --same_pruned_wg_layers model.head.0,model.body.16,*body.2 --reg_upper_limit 0.5 --reg_granularity_prune 0.0001 --update_reg_interval 20 --stabilize_reg_interval 43150 --pre_train ../pretrain_model/edsr_baseline_x2-1bc95232.pt --same_pruned_wg_criterion reg --save Ablation/EDSR_F64R16BIX2_DF2K_ASSL0.7_RGP0.0001_RUL0.5_Pretrain > /dev/null & 

# PR=0.9
CUDA_VISIBLE_DEVICES=3 nohup python main.py --model EDSR --scale 2 --patch_size 96 --ext sep --dir_data /home/yulun/data/SR/RGB/BIX2X3X4/pt_bin --data_train DF2K --data_test DF2K --data_range 1-3450/3551-3555 --epochs 200 --chop --save_results --save_models --n_resblocks 16 --n_feats 64 --method ASSL --wn --stage_pr [0-1000:0.9] --skip_layers *mean*,*tail* --same_pruned_wg_layers model.head.0,model.body.16,*body.2 --reg_upper_limit 0.5 --reg_granularity_prune 0.0001 --update_reg_interval 20 --stabilize_reg_interval 43150 --pre_train ../pretrain_model/edsr_baseline_x2-1bc95232.pt --same_pruned_wg_criterion reg --save Ablation/EDSR_F64R16BIX2_DF2K_ASSL0.9_RGP0.0001_RUL0.5_Pretrain > /dev/null &
```
# Main Benchmark
LEDSR baseline (16 blocks).
--method ASSL --wn
--stage_pr [0-1000:0.1]
--same_pruned_wg_criterion reg
--save Ablation/EDSR_F64R16BIX2_DF2K_ASSL0.1_RGP0.0001_RUL0.5_Pretrain

```python
# Prune from 256 to 49, pr=0.80859375, x2
CUDA_VISIBLE_DEVICES=1 python main.py --model LEDSR --scale 2 --patch_size 96 --ext sep --dir_data /home/yulun/data/SR/RGB/BIX2X3X4/pt_bin --data_train DF2K --data_test DF2K --data_range 1-3550/3551-3555 --chop --save_results --n_resblocks 16 --n_feats 256 --method ASSL --wn --stage_pr [0-1000:0.80859375] --skip_layers *mean*,*tail* --same_pruned_wg_layers model.head.0,model.body.16,*body.2 --reg_upper_limit 0.5 --reg_granularity_prune 0.0001 --update_reg_interval 20 --stabilize_reg_interval 43150 --pre_train ../pretrain_model/LEDSR_F256R16BIX2_DF2K_M311.pt --same_pruned_wg_criterion reg --save main/SR/LEDSR_F256R16BIX2_DF2K_ASSL0.80859375_RGP0.0001_RUL0.5_Pretrain

# Prune from 256 to 49, pr=0.80859375, x3
CUDA_VISIBLE_DEVICES=2 python main.py --model LEDSR --scale 3 --patch_size 144 --ext sep --dir_data /home/yulun/data/SR/RGB/BIX2X3X4/pt_bin --data_train DF2K --data_test DF2K --data_range 1-3550/3551-3555 --chop --save_results --n_resblocks 16 --n_feats 256 --method ASSL --wn --stage_pr [0-1000:0.80859375] --skip_layers *mean*,*tail* --same_pruned_wg_layers model.head.0,model.body.16,*body.2 --reg_upper_limit 0.5 --reg_granularity_prune 0.0001 --update_reg_interval 20 --stabilize_reg_interval 43150 --pre_train ../pretrain_model/LEDSR_F256R16BIX3_DF2K_M230.pt --same_pruned_wg_criterion reg --save main/SR/LEDSR_F256R16BIX3_DF2K_ASSL0.80859375_RGP0.0001_RUL0.5_Pretrain

# Prune from 256 to 49, pr=0.80859375, x4
CUDA_VISIBLE_DEVICES=3 python main.py --model LEDSR --scale 4 --patch_size 192 --ext sep --dir_data /home/yulun/data/SR/RGB/BIX2X3X4/pt_bin --data_train DF2K --data_test DF2K --data_range 1-3550/3551-3555 --chop --save_results --n_resblocks 16 --n_feats 256 --method ASSL --wn --stage_pr [0-1000:0.80859375] --skip_layers *mean*,*tail* --same_pruned_wg_layers model.head.0,model.body.16,*body.2 --reg_upper_limit 0.5 --reg_granularity_prune 0.0001 --update_reg_interval 20 --stabilize_reg_interval 43150 --pre_train ../pretrain_model/LEDSR_F256R16BIX4_DF2K_M231.pt --same_pruned_wg_criterion reg --save main/SR/LEDSR_F256R16BIX4_DF2K_ASSL0.80859375_RGP0.0001_RUL0.5_Pretrain
```

---
# TPAMI Extension
## Ablation Study
```python
# Step 1: note --reg_upper_limit 0.05
CUDA_VISIBLE_DEVICES=0 python main.py --model EDSR --scale 2 --patch_size 96 --ext sep --dir_data /home/yulun/data/SR/RGB/BIX2X3X4/pt_bin --data_train DF2K --data_test DF2K --data_range 1-3450/3551-3555 --epochs 200 --chop --save_results --save_models --n_resblocks 16 --n_feats 64 --method ASSL --wn --stage_pr 0.113 --skip_layers *mean*,*tail* --greg_mode all --compare_mode global --same_pruned_wg_layers model.head.0,model.body.16,*body.2 --reg_upper_limit 0.05 --reg_granularity_prune 0.0001 --update_reg_interval 20 --stabilize_reg_interval 1 --pre_train ../pretrain_model/edsr_baseline_x2-1bc95232.pt --save Ablation/EDSR_F64R16BIX2_DF2K_ASSL0.113_RGP0.0001_RUL0.05_Pretrain_RegSelect

CUDA_VISIBLE_DEVICES=1 python main.py --model EDSR --scale 2 --patch_size 96 --ext sep --dir_data /home/yulun/data/SR/RGB/BIX2X3X4/pt_bin --data_train DF2K --data_test DF2K --data_range 1-3450/3551-3555 --epochs 200 --chop --save_results --save_models --n_resblocks 16 --n_feats 64 --method ASSL --wn --stage_pr 0.311 --skip_layers *mean*,*tail* --greg_mode all --compare_mode global --same_pruned_wg_layers model.head.0,model.body.16,*body.2 --reg_upper_limit 0.05 --reg_granularity_prune 0.0001 --update_reg_interval 20 --stabilize_reg_interval 1 --pre_train ../pretrain_model/edsr_baseline_x2-1bc95232.pt --save Ablation/EDSR_F64R16BIX2_DF2K_ASSL0.311_RGP0.0001_RUL0.05_Pretrain_RegSelect

CUDA_VISIBLE_DEVICES=2 python main.py --model EDSR --scale 2 --patch_size 96 --ext sep --dir_data /home/yulun/data/SR/RGB/BIX2X3X4/pt_bin --data_train DF2K --data_test DF2K --data_range 1-3450/3551-3555 --epochs 200 --chop --save_results --save_models --n_resblocks 16 --n_feats 64 --method ASSL --wn --stage_pr 0.71 --skip_layers *mean*,*tail* --greg_mode all --compare_mode global --same_pruned_wg_layers model.head.0,model.body.16,*body.2 --reg_upper_limit 0.05 --reg_granularity_prune 0.0001 --update_reg_interval 20 --stabilize_reg_interval 1 --pre_train ../pretrain_model/edsr_baseline_x2-1bc95232.pt --save Ablation/EDSR_F64R16BIX2_DF2K_ASSL0.71_RGP0.0001_RUL0.05_Pretrain_RegSelect

CUDA_VISIBLE_DEVICES=3 python main.py --model EDSR --scale 2 --patch_size 96 --ext sep --dir_data /home/yulun/data/SR/RGB/BIX2X3X4/pt_bin --data_train DF2K --data_test DF2K --data_range 1-3450/3551-3555 --epochs 200 --chop --save_results --save_models --n_resblocks 16 --n_feats 64 --method ASSL --wn --stage_pr 0.9 --skip_layers *mean*,*tail* --greg_mode all --compare_mode global --same_pruned_wg_layers model.head.0,model.body.16,*body.2 --reg_upper_limit 0.05 --reg_granularity_prune 0.0001 --update_reg_interval 20 --stabilize_reg_interval 1 --pre_train ../pretrain_model/edsr_baseline_x2-1bc95232.pt --save Ablation/EDSR_F64R16BIX2_DF2K_ASSL0.9_RGP0.0001_RUL0.05_Pretrain_RegSelect


# Step 2: note --reg_upper_limit 0.5
CUDA_VISIBLE_DEVICES=3 python main.py --model EDSR --scale 2 --patch_size 96 --ext sep --dir_data /home/yulun/data/SR/RGB/BIX2X3X4/pt_bin --data_train DF2K --data_test DF2K --data_range 1-3450/3551-3555 --epochs 200 --chop --save_results --save_models --n_resblocks 16 --n_feats 64 --method ASSL --wn --stage_pr ../experiment/Ablation/EDSR_F64R16BIX2_DF2K_ASSL0.113_RGP0.0001_RUL0.05_Pretrain_RegSelect/model/model_just_finished_prune.pt --skip_layers *mean*,*tail* --greg_mode part --compare_mode local --same_pruned_wg_layers model.head.0,model.body.16,*body.2 --same_pruned_wg_criterion reg --reg_upper_limit 0.5 --reg_granularity_prune 0.0001 --update_reg_interval 30 --stabilize_reg_interval 86300 --pre_train ../pretrain_model/edsr_baseline_x2-1bc95232.pt --save Ablation/EDSR_F64R16BIX2_DF2K_ASSL0.113_RGP0.0001_RUL0.5_Stabilize86300_URI30_Pretrain_RegPrune_SelectRUL0.05

CUDA_VISIBLE_DEVICES=1 python main.py --model EDSR --scale 2 --patch_size 96 --ext sep --dir_data /home/yulun/data/SR/RGB/BIX2X3X4/pt_bin --data_train DF2K --data_test DF2K --data_range 1-3450/3551-3555 --epochs 200 --chop --save_results --save_models --n_resblocks 16 --n_feats 64 --method ASSL --wn --stage_pr ../experiment/Ablation/EDSR_F64R16BIX2_DF2K_ASSL0.32_RGP0.0001_RUL0.05_Pretrain_RegSelect/model/model_just_finished_prune.pt --skip_layers *mean*,*tail* --greg_mode part --compare_mode local --same_pruned_wg_layers model.head.0,model.body.16,*body.2 --same_pruned_wg_criterion reg --reg_upper_limit 0.5 --reg_granularity_prune 0.0001 --update_reg_interval 30 --stabilize_reg_interval 86300 --pre_train ../pretrain_model/edsr_baseline_x2-1bc95232.pt --save Ablation/EDSR_F64R16BIX2_DF2K_ASSL0.32_RGP0.0001_RUL0.5_Stabilize86300_URI30_Pretrain_RegPrune_SelectRUL0.05

CUDA_VISIBLE_DEVICES=2 python main.py --model EDSR --scale 2 --patch_size 96 --ext sep --dir_data /home/yulun/data/SR/RGB/BIX2X3X4/pt_bin --data_train DF2K --data_test DF2K --data_range 1-3450/3551-3555 --epochs 200 --chop --save_results --save_models --n_resblocks 16 --n_feats 64 --method ASSL --wn --stage_pr ../experiment/Ablation/EDSR_F64R16BIX2_DF2K_ASSL0.71_RGP0.0001_RUL0.05_Pretrain_RegSelect/model/model_just_finished_prune.pt --skip_layers *mean*,*tail* --greg_mode part --compare_mode local --same_pruned_wg_layers model.head.0,model.body.16,*body.2 --same_pruned_wg_criterion reg --reg_upper_limit 0.5 --reg_granularity_prune 0.0001 --update_reg_interval 30 --stabilize_reg_interval 86300 --pre_train ../pretrain_model/edsr_baseline_x2-1bc95232.pt --save Ablation/EDSR_F64R16BIX2_DF2K_ASSL0.71_RGP0.0001_RUL0.5_Stabilize86300_URI30_Pretrain_RegPrune_SelectRUL0.05

CUDA_VISIBLE_DEVICES=3 python main.py --model EDSR --scale 2 --patch_size 96 --ext sep --dir_data /home/yulun/data/SR/RGB/BIX2X3X4/pt_bin --data_train DF2K --data_test DF2K --data_range 1-3450/3551-3555 --epochs 200 --chop --save_results --save_models --n_resblocks 16 --n_feats 64 --method ASSL --wn --stage_pr ../experiment/Ablation/EDSR_F64R16BIX2_DF2K_ASSL0.9_RGP0.0001_RUL0.05_Pretrain_RegSelect/model/model_just_finished_prune.pt --skip_layers *mean*,*tail* --greg_mode part --compare_mode local --same_pruned_wg_layers model.head.0,model.body.16,*body.2 --same_pruned_wg_criterion reg --reg_upper_limit 0.5 --reg_granularity_prune 0.0001 --update_reg_interval 30 --stabilize_reg_interval 86300 --pre_train ../pretrain_model/edsr_baseline_x2-1bc95232.pt --save Ablation/EDSR_F64R16BIX2_DF2K_ASSL0.9_RGP0.0001_RUL0.5_Stabilize86300_URI30_Pretrain_RegPrune_SelectRUL0.05

```

# Main Benchmark
LEDSR baseline (16 blocks).
--method ASSL --wn
--stage_pr [0-1000:0.1]
--same_pruned_wg_criterion reg
--save Ablation/EDSR_F64R16BIX2_DF2K_ASSL0.1_RGP0.0001_RUL0.5_Pretrain

```python
# x2, RegSelect
CUDA_VISIBLE_DEVICES=0 python main.py --model LEDSR --scale 2 --patch_size 96 --ext sep --dir_data /home/yulun/data/SR/RGB/BIX2X3X4/pt_bin --data_train DF2K --data_test DF2K --data_range 1-3550/3551-3555 --chop --save_results --n_resblocks 16 --n_feats 256 --method ASSL --wn --stage_pr 0.701 --skip_layers *mean*,*tail* --greg_mode all --compare_mode global --same_pruned_wg_layers model.head.0,model.body.16,*body.2 --reg_upper_limit 0.05 --reg_granularity_prune 0.0001 --update_reg_interval 20 --stabilize_reg_interval 1 --pre_train ../pretrain_model/LEDSR_F256R16BIX2_DF2K_M311.pt --save main/SR/LEDSR_F256R16BIX2_DF2K_ASSL0.701_Pretrain_RegSelect_RGP0.0001_RUL0.05

CUDA_VISIBLE_DEVICES=2 python main.py --model LEDSR --scale 2 --patch_size 96 --ext sep --dir_data /home/yulun/data/SR/RGB/BIX2X3X4/pt_bin --data_train DF2K --data_test DF2K --data_range 1-3550/3551-3555 --chop --save_results --n_resblocks 16 --n_feats 256 --method ASSL --wn --stage_pr 0.702 --skip_layers *mean*,*tail* --greg_mode all --compare_mode global --same_pruned_wg_layers model.head.0,model.body.16,*body.2 --reg_upper_limit 0.05 --reg_granularity_prune 0.0001 --update_reg_interval 20 --stabilize_reg_interval 1 --pre_train ../pretrain_model/LEDSR_F256R16BIX2_DF2K_M311.pt --save main/SR/LEDSR_F256R16BIX2_DF2K_ASSL0.702_Pretrain_RegSelect_RGP0.0001_RUL0.05

CUDA_VISIBLE_DEVICES=0 python main.py --model LEDSR --scale 2 --patch_size 96 --ext sep --dir_data /home/yulun/data/SR/RGB/BIX2X3X4/pt_bin --data_train DF2K --data_test DF2K --data_range 1-3550/3551-3555 --chop --save_results --n_resblocks 16 --n_feats 256 --method ASSL --wn --stage_pr 0.703 --skip_layers *mean*,*tail* --greg_mode all --compare_mode global --same_pruned_wg_layers model.head.0,model.body.16,*body.2 --reg_upper_limit 0.05 --reg_granularity_prune 0.0001 --update_reg_interval 20 --stabilize_reg_interval 1 --pre_train ../pretrain_model/LEDSR_F256R16BIX2_DF2K_M311.pt --save main/SR/LEDSR_F256R16BIX2_DF2K_ASSL0.703_Pretrain_RegSelect_RGP0.0001_RUL0.05


# x3, RegSelect
CUDA_VISIBLE_DEVICES=2 python main.py --model LEDSR --scale 3 --patch_size 144 --ext sep --dir_data /home/yulun/data/SR/RGB/BIX2X3X4/pt_bin --data_train DF2K --data_test DF2K --data_range 1-3550/3551-3555 --chop --save_results --n_resblocks 16 --n_feats 256 --method ASSL --wn --stage_pr 0.702 --skip_layers *mean*,*tail* --greg_mode all --compare_mode global --same_pruned_wg_layers model.head.0,model.body.16,*body.2 --reg_upper_limit 0.05 --reg_granularity_prune 0.0001 --update_reg_interval 20 --stabilize_reg_interval 1 --pre_train ../pretrain_model/LEDSR_F256R16BIX2_DF2K_M311.pt --save main/SR/LEDSR_F256R16BIX3_DF2K_ASSL0.702_Pretrain_RegSelect_RGP0.0001_RUL0.05


# x2, RegPrune (To Be Updated)
CUDA_VISIBLE_DEVICES=1 python main.py --model LEDSR --scale 2 --patch_size 96 --ext sep --dir_data /home/yulun/data/SR/RGB/BIX2X3X4/pt_bin --data_train DF2K --data_test DF2K --data_range 1-3550/3551-3555 --chop --save_results --n_resblocks 16 --n_feats 256 --method ASSL --wn --stage_pr ../experiment/main/SR/LEDSR_F256R16BIX2_DF2K_ASSL0.80859375_Pretrain_RegSelect_RGP0.0001_RUL0.05/model/model_just_finished_prune.pt --skip_layers *mean*,*tail* --greg_mode part --compare_mode local --same_pruned_wg_layers model.head.0,model.body.16,*body.2 --reg_upper_limit 0.5 --reg_granularity_prune 0.0001 --update_reg_interval 30 --stabilize_reg_interval 86300 --pre_train ../pretrain_model/LEDSR_F256R16BIX2_DF2K_M311.pt --save main/SR/LEDSR_F256R16BIX2_DF2K_ASSL0.80859375_Pretrain_RegPrune_RGP0.0001_RUL0.5_Stabilize86300_RUI30


```

