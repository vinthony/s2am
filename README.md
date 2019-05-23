# Spatial-Separated Attention Module (S²AM)

This repo contains the PyTorch implement of the following paper:

&nbsp;&nbsp;&nbsp;&nbsp;[Regional Attentive Skip-Connection for Image Harmonization]()<br>
&nbsp;&nbsp;&nbsp;&nbsp;[Xiaodong Cun](https://vinthony.github.io/academicpages.github.io/), [Chi-Man Pun](http://www.cis.umac.mo/~cmpun/)<br>
&nbsp;&nbsp;&nbsp;&nbsp;University of Macau<br>
&nbsp;&nbsp;&nbsp;&nbsp;Technical Report,2019

&nbsp;&nbsp;&nbsp;&nbsp;[Improving the Harmony of the Composite Image by Spatial-Separated Attention Module]()<br>
&nbsp;&nbsp;&nbsp;&nbsp;[Xiaodong Cun](https://vinthony.github.io/academicpages.github.io/), [Chi-Man Pun](http://www.cis.umac.mo/~cmpun/)<br>
&nbsp;&nbsp;&nbsp;&nbsp;University of Macau<br>
&nbsp;&nbsp;&nbsp;&nbsp;Submitted to IEEE TVCG, 2019

## Requirements
The code is tested on the python 3.6 and PyTorch v0.4+ under Ubuntu 18.04 OS.</br>
You need to install all the requirements from `pip`.</br>
`Anaconda` is highly recommendation for install the dependences.</br> 
```
git clone https://github.com/vinthony/s2am.git
cd s2am
pip install -r requirements.txt
```

## Datasets
We train the network under the synthesized datasets.<br>
`SCOCO` dataset contains `40k` images for training and `1.7k` images for testing.<br>
`S-Adobe5k` dataset contains `32k` images form training and `2k` images for testing. <br>
The following command will automatically download the dataset and unzip the synthesized dataset to the cooresponding dataset folders.

```
bash download_dataset.sh scoco
bash download_dataset.sh sadobe5k
```

## Train

All the options of the training can be found in `options.py`

```
# train the S2AD methods from our TVCG paper.
chmod +x ./example/train_harmorization_s2ad.sh && ./example/train_harmorization_s2ad.sh

# train the S2ASC methods from the skip-connection paper.
chmod +x ./example/train_harmorization_s2asc.sh && ./example/train_harmorization_s2asc.sh

# train the image harmonization w/o mask task from our TVCG paper.
chmod +x ./example/train_harmorization_wo_mask.sh && ./example/train_harmorization_wo_mask.sh
```

## Visualization

We use `TensorboardX`  to monitor the training process, just install it by the [introduction](https://github.com/lanpa/tensorboardX) of tensorboardX.

run the watching commond as :
```
tensorboard --logdir ./checkpoint
```
## Demo
Download the pre-trained model from our synthesied datasets and run the nootbook in `notebook/visualize.ipynb`

## The Application of Spatial-Separated Attention Module (S²AM) w/o mask

#### Image Classification

We evaluate our method with the baseline attention module: [CBAM](https://arxiv.org/abs/1807.06521) and original ResNet in CIFAR-10 with the default setting of code in [pytorch_resnet_cifar10](https://github.com/akamaster/pytorch_resnet_cifar10)

| method | Test err (Orginal) | Test err (w/ CBAM) | **Test err (w/ S²AM)**|
| -- | -- | -- | -- |
| ResNet20 | 8.45% | 7.91% | **7.60%** |
| ResNet32 | 7.40% | 7.07% | **7.06%** |
| ResNet44 | 6.96% | 6.92% | **6.58%** |
| ResNet56 | 6.47% | 6.43% | **6.41%** |



## Acknowledgements

