# Spatial-Separated Attention Module (S²AM)
[Arxiv](https://arxiv.org/abs/1907.06406) | [Project](https://vinthony.github.io/academicpages.github.io/projects/rasc.html) | [Dataset](TBD)

This repo contains the PyTorch implement of the following paper:

&nbsp;&nbsp;&nbsp;&nbsp;[Improving the Harmony of the Composite Image by Spatial-Separated Attention Module](https://arxiv.org/abs/1907.06406)<br>
&nbsp;&nbsp;&nbsp;&nbsp;[Xiaodong Cun](https://vinthony.github.io/academicpages.github.io/), [Chi-Man Pun](http://www.cis.umac.mo/~cmpun/)<br>
&nbsp;&nbsp;&nbsp;&nbsp;University of Macau<br>
&nbsp;&nbsp;&nbsp;&nbsp;Trans. on Image Processing, 2020, In Press.

> This repo. is still work-in-progress. More details of the pre-trained models, training details and the datasets will be released in the future.

## **Citation**

If you find our work useful in your research, please consider citing:
```
@misc{cun2019improving,
    title={Improving the Harmony of the Composite Image by Spatial-Separated Attention Module},
    author={Xiaodong Cun and Chi-Man Pun},
    year={2019},
    eprint={1907.06406},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```

## Abstract

Image composition is one of the most important applications in image processing. However, the inharmonious appearance between the spliced region and background degrade the quality of image. Thus, we address the problem of Image Harmonization: Given a spliced image and the mask of the spliced region, we try to harmonize the ''style'' of the pasted region with the background (non-spliced region). Previous approaches have been focusing on learning directly by the neural network.
In this work, we start from an empirical observation: the differences can only be found in the spliced region between the spliced image and the harmonized result while they share the same semantic information and the appearance in the non-spliced region. Thus, in order to learn the feature map in the masked region and the others individually, we propose a novel attention module named Spatial-Separated Attention Module (S²AM). Furthermore, we design a novel image harmonization framework by inserting the S²AM in the coarser low level features of the Unet structure by two different ways. Besides image harmonization, we make a big step for harmonizing the composite image without the specific mask under previous observation. The experiments show that the proposed S²AM performs better than other state-of-the-art attention modules in our task.  Moreover, we demonstrate the advantages of our model against other state-of-the-art image harmonization methods via criteria from multiple points of view.

## Some Results

![results](https://user-images.githubusercontent.com/4397546/61209516-931c0f00-a72c-11e9-84ef-c7b7bc794c0e.png)
![sample](https://user-images.githubusercontent.com/4397546/61209520-93b4a580-a72c-11e9-881f-40de42c3a4f7.png)


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
[SCOCO dataset(5G)](https://uofmacau-my.sharepoint.com/:u:/g/personal/yb87432_umac_mo/EZ4-Qpk-iHFLrTfZKIzAivoBn_Jm8mMkZ6EiqAsz4M0Uhg?e=uqShf3) contains `40k` images for training and `1.7k` images for testing.<br>
`S-Adobe5k` dataset contains `32k` images form training and `2k` images for testing. <br>
The following command will automatically download the dataset and unzip the synthesized dataset to the cooresponding dataset folders.

```
TBD
```

## Train

All the options of the training can be found in `options.py`

```
# train the S2AD methods 
chmod +x ./example/train_harmorization_s2ad.sh && ./example/train_harmorization_s2ad.sh

# train the S2ASC methods .
chmod +x ./example/train_harmorization_s2asc.sh && ./example/train_harmorization_s2asc.sh

# train the image harmonization w/o mask task from our paper.
chmod +x ./example/train_harmorization_wo_mask.sh && ./example/train_harmorization_wo_mask.sh
```

## Visualization

We use `TensorboardX`  to monitor the training process, just install it by the [introduction](https://github.com/lanpa/tensorboardX) of tensorboardX.

run the watching commond as :
```
tensorboard --logdir ./checkpoint
```
## Demo
```
# download the pretrain models
TBD

# create the jupyter lab server 
jupyter lab
```
run the nootbook in `notebook/visualize.ipynb` and `notebook/evaluation.ipynb`

## The Application of Spatial-Separated Attention Module (S²AM) w/o mask

#### Image Classification

We evaluate our method with the baseline attention module: [CBAM](https://arxiv.org/abs/1807.06521) and original ResNet in CIFAR-10 with the default setting of code in [pytorch_resnet_cifar10](https://github.com/akamaster/pytorch_resnet_cifar10)

| method | Test err (Orginal) | Test err (w/ CBAM) | **Test err (w/ S²AM)**|
| -- | -- | -- | -- |
| ResNet20 | 8.45% | 7.91% | **7.60%** |
| ResNet32 | 7.40% | 7.07% | **7.06%** |
| ResNet44 | 6.96% | 6.92% | **6.58%** |
| ResNet56 | 6.47% | 6.43% | **6.41%** |

#### Interactive Wartmark Removal from a region.
By regard a region as mask, Our method can use to remove the visible wartmark from the image. We generate the datasets from VOC as image and 100 famous logo as watermark region. The network trains on 70 of them and testing on the rest of them, here are some random results:
![1511](https://user-images.githubusercontent.com/4397546/61209289-e80b5580-a72b-11e9-9608-6da743935cb0.png)
![1582](https://user-images.githubusercontent.com/4397546/61209290-e80b5580-a72b-11e9-862a-24f71217b43d.png)
![1654](https://user-images.githubusercontent.com/4397546/61209291-e8a3ec00-a72b-11e9-8372-ed45e26d18e4.png)
![1728](https://user-images.githubusercontent.com/4397546/61209292-e8a3ec00-a72b-11e9-875b-ed7bf9027af9.png)

