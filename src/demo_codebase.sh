# Pytorch 1.2.0
#
#########
# train with bin
CUDA_VISIBLE_DEVICES=3 python main.py --model EDSR --scale 2 --patch_size 96 --save EDSR_F64R16BIX2_DF2K_bin --reset --ext bin --dir_data /media/yulun/10THD1/data/PrepareDataSSD/SR/RGB/BIX2X3X4/pt_bin --data_train DF2K --data_test DF2K --data_range 1-3450/3451-3460 --epochs 300 --chop --save_results
CUDA_VISIBLE_DEVICES=3 python main.py --model EDSR --scale 3 --patch_size 144 --save EDSR_F64R16BIX3_DF2K_bin --reset --ext bin --dir_data /media/yulun/10THD1/data/PrepareDataSSD/SR/RGB/BIX2X3X4/pt_bin --data_train DF2K --data_test DF2K --data_range 1-3450/3451-3460 --epochs 300 --chop --save_results
CUDA_VISIBLE_DEVICES=3 python main.py --model EDSR --scale 4 --patch_size 192 --save EDSR_F64R16BIX4_DF2K_bin --reset --ext bin --dir_data /media/yulun/10THD1/data/PrepareDataSSD/SR/RGB/BIX2X3X4/pt_bin --data_train DF2K --data_test DF2K --data_range 1-3450/3451-3460 --epochs 300 --chop --save_results

CUDA_VISIBLE_DEVICES=1 python main.py --model EDSR --scale 2 --patch_size 96 --save EDSR_F64R16BIX2_DF2K_bin --reset --ext bin --dir_data /media/yulun/10THD1/data/PrepareDataSSD/SR/RGB/BIX2X3X4/pt_bin --data_train DF2K --data_test DF2K --data_range 1-3550/3551-3555 --epochs 300 --chop --save_results
