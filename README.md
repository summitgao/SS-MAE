## 📖 SS-MAE: Spatial-Spectral Masked Auto-Encoder for Mulit-Source Remote Sensing Image Classification (IEEE TGRS 2023)

[![ARXIV](https://img.shields.io/badge/Paper-ARIXV-blue)](https://arxiv.org/abs/2311.04442)
[![IEEE](https://img.shields.io/badge/Paper-IEEE%20TGRS-blue)](https://ieeexplore.ieee.org/document/10314566/)

This code is for our paper "SS-MAE: Spatial-Spectral Masked Auto-Encoder for Mulit-Source Remote Sensing Image Classification (IEEE TGRS 2023)".

🔥 We hope SS-MAE is helpful for your work. Thanks a lot for your attention.🔥

If you have any questions, please contact us. Email: linjyan00@163.com, gaofeng@ouc.edu.cn

## Dataset
https://drive.google.com/file/d/1iZEIAVhlt2QJb_RECp0bHFVN7C8po8ag/view?usp=sharing

## Usage
### Pretraining (Berlin)
python main.py  --is_pretrain 1 --is_train 0 --dataset Berlin --num_classes 8 --pca_num 30 --mask_ratio 0.3 --pretrain_num 200000 --channel_num 248 --batch_size 128 --device cuda:0 --lr 0.0001 --is_load_pretrain 0 --depth 2  --head 8  --dim  256 --epoch 300
### Training (Berlin)
python main.py  --is_pretrain 0 --is_train 1 --dataset Berlin --num_classes 8 --pca_num 30 --mask_ratio 0.3 --pretrain_num 200000 --channel_num 248 --batch_size 128 --device cuda:0 --lr 0.0001 --is_load_pretrain 1 --depth 2  --head 8  --dim  256 --epoch 300




