# EasyTSF

The implement of paper "CDFNet: Collaborative Decomposition and Forecasting Network for Time Series" (under reviewed)

## Usage

### Environment

Step by Step with Conda:
```shell
conda create -n CDFNet python=3.10
conda activate CDFNet
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia
python -m pip install lightning
```

or you can just:
```shell
pip install -r requirements.txt
```

### Code and Data
ETTh1 and ETTm1 can be downloaded within this project, and other datasets can be downloaded from [Baidu Drive](https://pan.baidu.com/s/18NKge4dsMIuGQFom7n2S2w?pwd=zumh) or 
[Google Drive](https://drive.google.com/file/d/17JYLHDPIdLv9haLiDF9G_eGewhmMhGbq/view?usp=sharing).

### Running
```shell
python train.py -c config/ETTh1.py
```
