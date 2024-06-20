# Efficient document image binarization

> [Efficient generative adversarial networks for document image binarization based on discrete wavelet transform](https://arxiv.org/abs/)

<p align="center">
  <img src="readme_fig/fig_intro.jpg" width="1024" title="intro">
</p>

## Citation
If you find our paper useful in your research, please consider citing:
```

```
    
## Method
<p align="center">
  <img src="readme_fig/fig_network.jpg" width="640" title="network">
</p>

## Datasets
* Our Training Set (143) includes:

  [DIBCO 2009](http://users.iit.demokritos.gr/~bgat/DIBCO2009/benchmark/) (10), [H-DIBCO 2010](http://users.iit.demokritos.gr/~bgat/H-DIBCO2010/benchmark/) (10), [H-DIBCO 2012](http://utopia.duth.gr/~ipratika/HDIBCO2012/benchmark/) (14), [Bickley Diary](https://github.com/vqnhat/DSN-Binarization/files/2793688/original_gt_labeled.zip) (7), [PHIBD](http://www.iapr-tc11.org/mediawiki/index.php/Persian_Heritage_Image_Binarization_Dataset_(PHIBD_2012)) (15), [SMADI](https://tc11.cvc.uab.es/datasets/SMADI_1) (87).
  
* Our Testing Set (102) includes:

  [DIBCO 2011](http://utopia.duth.gr/~ipratika/DIBCO2011/benchmark/) (16), [DIBCO 2013](http://utopia.duth.gr/~ipratika/DIBCO2013/benchmark/) (16), [H-DIBCO 2014](http://users.iit.demokritos.gr/~bgat/HDIBCO2014/benchmark/) (10), [H-DIBCO 2016](http://vc.ee.duth.gr/h-dibco2016/benchmark/) (10), [DIBCO 2017](http://vc.ee.duth.gr/dibco2017/benchmark/) (20), [H-DIBCO 2018](https://vc.ee.duth.gr/h-dibco2018/benchmark/) (10), [DIBCO 2019](https://vc.ee.duth.gr/dibco2019/benchmark/) (20).

## Environment
* NVIDIA GPU + CUDA CuDNN
* Python >= 3.6 (Pytorch)
```
  pip install -r requirements.txt
```

## UNet & EfficientNetV2-S
### Train
```
  cd unet_effnetv2
  python image_to_256.py
  python image_to_512.py
  python train_stage2_unet.py --epochs 10 --lambda_loss 30 --base_model_name tu-efficientnetv2_rw_s --batch_size 64
  python predict_for_stage3_unet.py --base_model_name tu-efficientnetv2_rw_s --lambda_loss 30
  python train_stage3_unet.py --epochs 10 --lambda_loss 30 --base_model_name tu-efficientnetv2_rw_s --batch_size 64
  python train_stage3_unet_resize.py --epochs 150 --lambda_loss 30 --base_model_name tu-efficientnetv2_rw_s --batch_size 16
```
### Test
```
  python eval_stage3_all_unet.py --lambda_loss 30 --base_model_name tu-efficientnetv2_rw_s --batch_size 64
```

## UNet++ & EfficientNetV2-S
### Train
```
  cd unetplusplus_effnetv2
  python image_to_256.py
  python image_to_512.py
  python train_stage2.py --epochs 10 --lambda_loss 30 --base_model_name tu-efficientnetv2_rw_s --batch_size 64
  python predict_for_stage3.py --base_model_name tu-efficientnetv2_rw_s --lambda_loss 30
  python train_stage3.py --epochs 10 --lambda_loss 30 --base_model_name tu-efficientnetv2_rw_s --batch_size 64
  python train_stage3_resize.py --epochs 150 --lambda_loss 30 --base_model_name tu-efficientnetv2_rw_s --batch_size 16
```

### Test
```
  python eval_stage3_all.py --lambda_loss 30 --base_model_name tu-efficientnetv2_rw_s --batch_size 64
```
