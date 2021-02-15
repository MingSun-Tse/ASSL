# Pytorch 1.2.0
# conda activate pt1.2.0
#########
# train with bin
#CUDA_VISIBLE_DEVICES=3 python main.py --model EDSR --scale 2 --patch_size 96 --save EDSR_F64R16BIX2_DF2K_bin --reset --ext bin --dir_data /media/yulun/10THD1/data/PrepareDataSSD/SR/RGB/BIX2X3X4/pt_bin --data_train DF2K --data_test DF2K --data_range 1-3450/3451-3460 --epochs 300 --chop --save_results
#CUDA_VISIBLE_DEVICES=3 python main.py --model EDSR --scale 3 --patch_size 144 --save EDSR_F64R16BIX3_DF2K_bin --reset --ext bin --dir_data /media/yulun/10THD1/data/PrepareDataSSD/SR/RGB/BIX2X3X4/pt_bin --data_train DF2K --data_test DF2K --data_range 1-3450/3451-3460 --epochs 300 --chop --save_results
#CUDA_VISIBLE_DEVICES=3 python main.py --model EDSR --scale 4 --patch_size 192 --save EDSR_F64R16BIX4_DF2K_bin --reset --ext bin --dir_data /media/yulun/10THD1/data/PrepareDataSSD/SR/RGB/BIX2X3X4/pt_bin --data_train DF2K --data_test DF2K --data_range 1-3450/3451-3460 --epochs 300 --chop --save_results

#CUDA_VISIBLE_DEVICES=1 python main.py --model EDSR --scale 2 --patch_size 96 --save EDSR_F64R16BIX2_DF2K_bin --reset --ext bin --dir_data /media/yulun/10THD1/data/PrepareDataSSD/SR/RGB/BIX2X3X4/pt_bin --data_train DF2K --data_test DF2K --data_range 1-3550/3551-3555 --epochs 300 --chop --save_results

# 2021/02/09
# train with pt
# conda activate pt1.2.0
# some baselines
# EDSR baseline 
CUDA_VISIBLE_DEVICES=3 python main.py --model EDSR --scale 2 --patch_size 96 --save EDSR_F64R16BIX2_DF2K --n_resblocks 16 --n_feats 64 --ext sep --dir_data /home/yulun/data/SR/RGB/BIX2X3X4/pt_bin --data_train DF2K --data_test DF2K --data_range 1-3450/3451-3460 --epochs 300 --chop --save_results --reset
# EDSR_F256R32 
CUDA_VISIBLE_DEVICES=3 python main.py --model EDSR --scale 2 --patch_size 96 --save EDSR_F256R32BIX2_DF2K --n_resblocks 32 --n_feats 256 --res_scale 0.1 --ext sep --dir_data /home/yulun/data/SR/RGB/BIX2X3X4/pt_bin --data_train DF2K --data_test DF2K --data_range 1-3450/3451-3460 --epochs 300 --chop --save_results --reset
# EDSR_F64R80 
CUDA_VISIBLE_DEVICES=2 python main.py --model EDSR --scale 2 --patch_size 96 --save EDSR_F64R80BIX2_DF2K --n_resblocks 80 --n_feats 64 --ext sep --dir_data /home/yulun/data/SR/RGB/BIX2X3X4/pt_bin --data_train DF2K --data_test DF2K --data_range 1-3450/3451-3460 --epochs 300 --chop --save_results --reset
# RCAN
CUDA_VISIBLE_DEVICES=1 python main.py --model RCAN --scale 2 --patch_size 96 --save RCAN_BIX2_G10R20P48 --scale 2 --n_resgroups 10 --n_resblocks 20 --n_feats 64 --ext sep --dir_data /home/yulun/data/SR/RGB/BIX2X3X4/pt_bin --data_train DF2K --data_test DF2K --data_range 1-3450/3451-3460 --epochs 300 --chop --save_results --reset
