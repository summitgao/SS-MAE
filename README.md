## SS-MAE: Spatial-Spectral Masked Auto-Encoder for Mulit-Source Remote Sensing Image Classification

## Dataset
https://drive.google.com/file/d/1iZEIAVhlt2QJb_RECp0bHFVN7C8po8ag/view?usp=sharing

## Usage
### Pretraining (Berlin)
python main.py  --is_pretrain 1 --is_train 0 --dataset Berlin --num_classes 8 --pca_num 30 --mask_ratio 0.3 --pretrain_num 200000 --channel_num 248 --batch_size 128 --device cuda:0 --lr 0.0001 --is_load_pretrain 0 --depth 2  --head 8  --dim  256 --epoch 300
### Training (Berlin)
python main.py  --is_pretrain 0 --is_train 1 --dataset Berlin --num_classes 8 --pca_num 30 --mask_ratio 0.3 --pretrain_num 200000 --channel_num 248 --batch_size 128 --device cuda:0 --lr 0.0001 --is_load_pretrain 1 --depth 2  --head 8  --dim  256 --epoch 300




