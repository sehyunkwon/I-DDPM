# Importace Weighted Denoising Diffusion Probabilistic Models

This repository is for 'Importance Weighted Denoising Diffusion Probabilistic Models' which is conducted in 2023 PGM lecture in SNU. 


## Paper
TBA

## Requirements
- Python 3.6
- Packages
    Upgrade pip for installing latest tensorboard
    ```
    pip install -U pip setuptools
    pip install -r requirements.txt
    ```
- Download precalculated statistic for dataset:

    [cifar10.train.npz](https://drive.google.com/drive/folders/1UBdzl6GtNMwNQ5U-4ESlIer43tNjiGJC?usp=sharing)

    Create folder `stats` for `cifar10.train.npz`.
    ```
    stats
    └── cifar10.train.npz
    ```

## Train I-DDPM From Scratch
- Take CIFAR10 for example:
    ```
    python train_IDDPM.py --train \
        --flagfile ./config/CIFAR10.txt
    ```
- [Optional] Overwrite arguments
    ```
    python train_IDDPM.py --train \
        --flagfile ./config/CIFAR10.txt \
        --batch_size 64 \
        --logdir ./path/to/logdir
    ```
- [Optional] Select GPU IDs
    ```
    CUDA_VISIBLE_DEVICES=1 python train_IDDPM.py --train \
        --flagfile ./config/CIFAR10.txt
    ```
- [Optional] Multi-GPU training
    ```
    CUDA_VISIBLE_DEVICES=0,1,2,3 python train_IDDPM.py --train \
        --flagfile ./config/CIFAR10.txt \
        --parallel
    ```

## Train DDPM From Scratch
- Take CIFAR10 for example:
    ```
    python train_DDPM.py --train \
        --flagfile ./config/CIFAR10.txt
    ```
- [Optional] Overwrite arguments
    ```
    python train_DDPM.py --train \
        --flagfile ./config/CIFAR10.txt \
        --batch_size 64 \
        --logdir ./path/to/logdir
    ```
- [Optional] Select GPU IDs
    ```
    CUDA_VISIBLE_DEVICES=1 python train_DDPM.py --train \
        --flagfile ./config/CIFAR10.txt
    ```
- [Optional] Multi-GPU training
    ```
    CUDA_VISIBLE_DEVICES=0,1,2,3 python train_DDPM.py --train \
        --flagfile ./config/CIFAR10.txt \
        --parallel
    ```

## Evaluate
- A `flagfile.txt` is autosaved to your log directory. The default logdir for `config/CIFAR10.txt` is `./logs/DDPM_CIFAR10_EPS`
- Start evaluation
    ```
    python main.py \
        --flagfile ./logs/DDPM_CIFAR10_EPS/flagfile.txt \
        --notrain \
        --eval
    ```
- [Optional] Multi-GPU evaluation
    ```
    CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py \
        --flagfile ./logs/DDPM_CIFAR10_EPS/flagfile.txt \
        --notrain \
        --eval \
        --parallel
    ```


## Reproducing Experiment

### CIFAR10
TBA

## Reference

[1] [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239)

[2] [Unofficial PyTorch implementation](https://github.com/w86763777/pytorch-ddpm)

[3] [Official TensorFlow implementation](https://github.com/hojonathanho/diffusion)
