# Spatial-Separated Attention Module (S²AM)

This repo contains the PyTorch implement of the paper:

[Regional Attentive Skip-Connection for Image Harmonization]()<br>
[Xiaodong Cun](), [Chi-Man Pun]()<br>
University of Macau<br>
Technical Report,2019

[Improving the Harmony of the Composite Image by Spatial-Separated Attention Module]()<br>
[Xiaodong Cun](), [Chi-Man Pun]()<br>
University of Macau<br>
Submitted to IEEE TVCG, 2019

## Requirements

```
git clone https://github.com/vinthony/Spatial-Separated-Attention-Module.git
cd Spatial-Separated-Attention-Module
pip install -r requirements
```

## Datasets

```
bash download_dataset.sh
```

## Train

All the options of the training process can be found in `options.py`

```
chmod +x ./example/train.sh && ./example/train.sh
```


## Visualization

We use `TensorboardX`  to monitor the training process, just install it by the [introduction](https://github.com/lanpa/tensorboardX) of tensorboardX.

run the watching commond as :
```
tensorboard --logdir ./checkpoint
```
## Demo
Download the pre-trained model from our synthesied datasets and run the nootbook in `notebook/visualize.ipynb`

## the Application of Spatial-Separated Attention Module (S²AM) without mask

### Image Classification

We evaluate our method with the baseline attention module: [CBAM](https://arxiv.org/abs/1807.06521) and original ResNet in CIFAR-10 with the default setting of code in [pytorch_resnet_cifar10](https://github.com/akamaster/pytorch_resnet_cifar10)

| method | Test err (Orginal) | Test err (w/ CBAM) | **Test err (w/ S²AM)**|
| -- | -- | -- | -- |
| ResNet20 | 8.45% | 7.91% | **7.60%** |
| ResNet32 | 7.40% | 7.07% | **7.06%** |
| ResNet44 | 6.96% | 6.92% | **6.58%** |
| ResNet56 | 6.47% | 6.43% | **6.41%** |


## Acknowledgements


