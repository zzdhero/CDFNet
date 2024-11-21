# EasyTSF

The implement of paper "CDFNet: Collaborative Decomposition and Forecasting Network for Time Series" (under reviewed)

  Time series forecasting is a long-standing research area, and related methods gain wider applications with the support of deep learning techniques. However, the complexity of real-world time series limits these methods capabilities. To address this issue, we propose the Adaptive Functional Decomposition module (AFD) inspired by the Kolmogorov-Arnold Network (KAN), where KAN is a novel function-fitting network with learnable parameters. AFD integrates multiple KAN layers to capture different parts of temporal patterns and uses reconstruction loss to encourage accurate decomposition in the training. Then, we propose a simple yet effective Collaborative
Decomposition and Forecasting Network (CDFNet), which uses AFD to decompose the input time series into several sub-series and adopts a homogenous functional forecasting module to predict the future state of each sub-series. Finally, we conduct a comparison with six baselines on seven datasets, CDFNet gets 54 times rank-1 over 70 metrics, which confirms the effectiveness of our proposed method.

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
