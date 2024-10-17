# Powerformer (In Submission)

### This is an offical implementation of Powerformer: [Powerformer: A Transformer with Weighted Causal Attention for Time-series Forecasting]. 

## Key Designs

Powerformer is built on top of [PatchTST](https://github.com/yuqinie98/PatchTST). Here we highlight key changes to the traditional [Transformer](https://proceedings.neurips.cc/paper_files/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf) architecture by both PatchTST and Powerformer.

### PatchTST
:star2: **Patching**: segmentation of time series into subseries-level patches which are served as input tokens to Transformer.

:star2: **Channel-independence**: each channel contains a single univariate time series that shares the same embedding and Transformer weights across all the series.

### Powerformer

:star2: **Weighted Causal Attention**: impart a causal and local bias into attention by introducing a causal mask and a local mask that decays the attention weights in a consistent fashion with common power law decays of time-series pairwise corellations.

:star2: **Transformer Time-Series Implicit Bias**: replacing all traditional multihead self-attention with weighted causal attention imposes a causal and local implicit bias in a way that matches generally observed pairwise correlation distributions.

## Results

### Supervised Learning

Powerformer outperform other state-of-the-art Transformer-based models. Compared to previous models, Powerformer has 29 of the 40 MSE and MAE evaluations over 5 real-world datasets. This is almost twice as much as the second best model, which outperforms on 15 tasks.

![alt text](https://github.com/yuqinie98/PatchTST/blob/main/pic/table1.png)

## Getting Started

1. Install requirements. ```pip install -r requirements.txt```

2. Download data. You can download all the datasets from [Autoformer](https://drive.google.com/drive/folders/1ZOYpTUa82_jCcxIdTmyr0LXQfvaM9vIy). Create a seperate folder ```./dataset``` and put all the csv files in the directory.

3. Training. All the scripts are in the directory ```./scripts/Powerformer```. For example, if you want to get the multivariate forecasting results for weather dataset, just run the following command, and you can open ```./result.txt``` to see the results once the training is done:
```
sh ./scripts/PatchTST/weather.sh
```

You can adjust the hyperparameters based on your needs (e.g. different patch length, different look-back windows and prediction lengths.). We also provide codes for the baseline models.



## Acknowledgement

We acknowledge and thank the PatchTST authors for the use of their repository:

https://github.com/yuqinie98/PatchTST
