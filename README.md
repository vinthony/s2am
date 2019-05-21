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

### Requirements

```
pip install -r requirements
```

### Datasets

```
bash download_dataset.sh
```

### Train

```
./example/train.sh
```

### Demo
Download the pre-trained model from our synthesied datasets and run the nootbook in `notebook/vis.ipynb`

### Applications

##### Image Classification

We evaluate our method with the baseline attention module: [CBAM](https://arxiv.org/abs/1807.06521) and original ResNet in CIFAR-10 with the default setting of code in [pytorch_resnet_cifar10](https://github.com/akamaster/pytorch_resnet_cifar10)

| method | Test err (Ours run) | 
| -- | -- |
| ResNet20 | 8.45% |
| ResNet20 + CBAM | 7.91% |
| **ResNet20 + R²AM (Ours)** |  **7.6%** |
| ResNet32 |  7.4% |
| ResNet32 + CBAM | 7.07% |
| **ResNet32 + R²AM (Ours)** | **7.06%** |
| ResNet44 | 6.96% |
| ResNet44 + CBAM | 6.92% |
| **ResNet44 + R²AM (Ours)** | **6.85%** |






