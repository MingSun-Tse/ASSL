# ASSL (NeurIPS'21 Spotlight)

<div align="left">
    <a><img src="figs/smile.png"  height="70px" ></a>
    <a><img src="figs/neu.png"  height="70px" ></a>
</div>

This repository is for a new network pruning method (`Aligned Structured Sparsity Learning, ASSL`) for efficient single image super-resolution (SR), introduced in our NeurIPS 2021 **Spotlight** paper:
> **Aligned Structured Sparsity Learning for Efficient Image Super-Resolution [[Camera Ready](https://papers.nips.cc/paper/2021/file/15de21c670ae7c3f6f3f1f37029303c9-Paper.pdf)] [[Visual Results](https://github.com/MingSun-Tse/ASSL/releases)]** \
> [Yulun Zhang*](http://yulunzhang.com/), [Huan Wang*](http://huanwang.tech/), [Can Qin](http://canqin.tech/), and [Yun Fu](http://www1.ece.neu.edu/~yunfu/) (*equal contribution) \
> Northeastern University, Boston, MA, USA


## Introduction
<div align="center">
  <img src="figs/NIPS21_ASSL.png" width="650px">
</div>
Lightweight image super-resolution (SR) networks have obtained promising results with moderate model size. Many SR methods have focused on designing lightweight architectures, which neglect to further reduce the redundancy of network parameters. On the other hand, model compression techniques, like neural architecture search and knowledge distillation, typically consume considerable memory and computation resources. In contrast, network pruning is a cheap and effective model compression technique. However, it is hard to be applied to SR networks directly, because filter pruning for residual blocks is well-known tricky. To address the above issues, we propose aligned structured sparsity learning (ASSL), which introduces a weight normalization layer and applies L2 regularization to the scale parameters for sparsity. To align the pruned filter locations across different layers, we propose a sparsity structure alignment penalty term, which minimizes the norm of soft mask gram matrix. We apply aligned structured sparsity learning strategy to train efficient image SR network, named as ASSLN, with smaller model size and lower computation than state-of-the-art methods. We conduct extensive comparisons with lightweight SR networks. Our ASSLN achieves superior performance gains over recent methods quantitatively and visually.

## Install
```python
git clone git@github.com:mingsun-tse/ASSL.git -b master
cd ASSL/src

# install dependencies (PyTorch 1.2.0 used), Anaconda is strongly recommended
pip install -r requirements.txt
```


## Train
### Prepare training data 

1. Download DIV2K training data (800 training + 100 validtion images) from [DIV2K dataset](https://data.vision.ee.ethz.ch/cvl/DIV2K/) or [SNU_CVLab](https://cv.snu.ac.kr/research/EDSR/DIV2K.tar) and [Flickr2K dataset](https://cv.snu.ac.kr/research/EDSR/Flickr2K.tar) from SNU_CVLab.

2. Specify '--dir_data' based on the HR and LR images path. In option.py, '--ext' is set as 'sep_reset', which first convert .png to .npy. If all the training images (.png) are converted to .npy files, then set '--ext sep' to skip converting files.

For more informaiton, please refer to [EDSR(PyTorch)](https://github.com/thstkdgus35/EDSR-PyTorch).

### Prepare pretrained dense model
Neural network pruning is typically conducted on a *pretrained* model. Our method also follows this common practice. Before we run the pruning scripts next, here we set up the pretrained dense models. Download the `pretrained_models.zip` from our [releases](https://github.com/MingSun-Tse/ASSL/releases), and unzip it as follows:
```python
wget https://github.com/MingSun-Tse/ASSL/releases/download/v0.1/pretrained_models.zip
unzip pretrained_models.zip
mv pretrained_models ..
```

### Run
```python
# Prune from 256 to 49, pr=0.80859375, x2
python main.py --model LEDSR --scale 2 --patch_size 96 --ext sep --dir_data <your_data_path> --data_train DF2K --data_test DF2K --data_range 1-3550/3551-3555 --chop --save_results --n_resblocks 16 --n_feats 256 --method ASSL --wn --stage_pr [0-1000:0.80859375] --skip_layers *mean*,*tail* --same_pruned_wg_layers model.head.0,model.body.16,*body.2 --reg_upper_limit 0.5 --reg_granularity_prune 0.0001 --update_reg_interval 20 --stabilize_reg_interval 43150 --pre_train ../pretrained_models/LEDSR_F256R16BIX2_DF2K_M311.pt --same_pruned_wg_criterion reg --save main/SR/LEDSR_F256R16BIX2_DF2K_ASSL0.80859375_RGP0.0001_RUL0.5_Pretrain

# Prune from 256 to 49, pr=0.80859375, x3
python main.py --model LEDSR --scale 3 --patch_size 144 --ext sep --dir_data <your_data_path> --data_train DF2K --data_test DF2K --data_range 1-3550/3551-3555 --chop --save_results --n_resblocks 16 --n_feats 256 --method ASSL --wn --stage_pr [0-1000:0.80859375] --skip_layers *mean*,*tail* --same_pruned_wg_layers model.head.0,model.body.16,*body.2 --reg_upper_limit 0.5 --reg_granularity_prune 0.0001 --update_reg_interval 20 --stabilize_reg_interval 43150 --pre_train ../pretrained_models/LEDSR_F256R16BIX3_DF2K_M230.pt --same_pruned_wg_criterion reg --save main/SR/LEDSR_F256R16BIX3_DF2K_ASSL0.80859375_RGP0.0001_RUL0.5_Pretrain

# Prune from 256 to 49, pr=0.80859375, x4
python main.py --model LEDSR --scale 4 --patch_size 192 --ext sep --dir_data <your_data_path> --data_train DF2K --data_test DF2K --data_range 1-3550/3551-3555 --chop --save_results --n_resblocks 16 --n_feats 256 --method ASSL --wn --stage_pr [0-1000:0.80859375] --skip_layers *mean*,*tail* --same_pruned_wg_layers model.head.0,model.body.16,*body.2 --reg_upper_limit 0.5 --reg_granularity_prune 0.0001 --update_reg_interval 20 --stabilize_reg_interval 43150 --pre_train ../pretrained_models/LEDSR_F256R16BIX4_DF2K_M231.pt --same_pruned_wg_criterion reg --save main/SR/LEDSR_F256R16BIX4_DF2K_ASSL0.80859375_RGP0.0001_RUL0.5_Pretrain
```
where `<your_data_path>` refers to the data directory path. One example on our PC is: `/home/yulun/data/SR/RGB/BIX2X3X4/pt_bin`.


## Test
After training, to use the trained models to generate HR images, you may use the following snippet. Currectly, you can use our [final models](https://github.com/MingSun-Tse/ASSL/releases) to test first:
```
wget https://github.com/MingSun-Tse/ASSL/releases/download/v0.1/final_models.zip
unzip final_models.zip
mv final_models ..
python main.py --data_test Demo --scale 4 --dir_demo <your_test_data_path> --test_only --save_results --pre_train ../final_models/ASSLN_F49_X4.pt --save Test_ASSLN_F49_X4
```
where `<your_test_data_path>` refers to the test data path on your computer. One example on our PC is: `/media/yulun/10THD1/data/super-resolution/LRBI/Set5/x4`.


## Results
### Quantitative Results
PSNR/SSIM comparison on popular SR benchmark datasets is shown below (best in red, second best in blue).
<div align="center">
  <img src="figs/psnr_ssim.png" width="800px">
</div>

### Visual Results
Visual comparison (x4) among lightweight SR approaches on the Urban100 dataset is shown below. Please see our [releases](https://github.com/MingSun-Tse/ASSL/releases) for the complete visual results on Set5/Set14/B100/Urban100/Manga109.
<div align="center">
  <img src="figs/visual_urban100_x4.png" width="800px">
</div>

## Citation
If you find the code helpful in your resarch or work, please cite the following papers.
```
@InProceedings{Lim_2017_CVPR_Workshops,
  author = {Lim, Bee and Son, Sanghyun and Kim, Heewon and Nah, Seungjun and Lee, Kyoung Mu},
  title = {Enhanced Deep Residual Networks for Single Image Super-Resolution},
  booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
  month = {July},
  year = {2017}
}

@inproceedings{zhang2021aligned,
    title={Aligned Structured Sparsity Learning for Efficient Image Super-Resolution},
    author={Zhang, Yulun and Wang, Huan and Qin, Can and Fu, Yun},
    booktitle={NeurIPS},
    year={2021}
}
```

## Acknowledgements
We refer to the following implementations when we develop this code: [EDSR-PyTorch](https://github.com/thstkdgus35/EDSR-PyTorch), [RCAN](https://github.com/yulunzhang/RCAN), [Regularization-Pruning](https://github.com/MingSun-Tse/Regularization-Pruning). Great thanks to them!
