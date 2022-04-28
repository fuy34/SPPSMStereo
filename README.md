#SPPSMNet: Spixel Pyramid Stereo Matching Network

This is a PyTorch implementation of the SPPSMNet proposed in our CVPR-20 paper:

[Superpixel Segmentation with Fully Convolutional Network](https://arxiv.org/abs/2003.12929)

[Fengting Yang](http://personal.psu.edu/fuy34/), [Qian Sun](https://www.linkedin.com/in/qiansuun), [Hailin Jin](https://research.adobe.com/person/hailin-jin/), and [Zihan Zhou](https://faculty.ist.psu.edu/zzhou/Home.html)

The code for the superpixel segmentation method (SpixelFCN) is available in [this repository](https://github.com/fuy34/superpixel_fcn).
Please contact Fengting Yang (fuy34bkup@gmail.com) if you have any questions.

## Prerequisites
The training code was mainly developed and tested with python 2.7, PyTorch 0.4.1, CUDA 9, and Ubuntu 16.04.

Prior to running this code, we need a pre-trained SpixelFCN model to initialize the SpixelFCN part. Two pre-trained model with
grid size 4 and 16 have been provided in ```preTrain_spixel_model``` folder. Please refer to [SpixelFCN](https://github.com/fuy34/superpixel_fcn) 
to pre-train a model with different grid size.  

## Data preparation
We train/fine-tune our model on three public dataset: [SceneFlow](https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html),
[HRVS](https://drive.google.com/file/d/1SgEIrH_IQTKJOToUwR1rx4-237sThUqX/view), and [Middlebury-v2](https://vision.middlebury.edu/stereo/data/scenes2014/). 

* For Sceneflow, we follow the official training and test set.
* For HRVS, we have listed the training and validation split in ```dataloader/HRVS_*.txt```.
* For Middlebury-v2, we fine-tuned on both the training and the additional dataset for our submission. 

Please follow the offical instruction of each dataset to download them. 

## Training
We first train the model on SceneFlow dataset. For joint training, run
```
python main.py --datapath <SCENEFLOW_PATH> --savemodel <SF_DUMP_PATH> --m_w 30
```
For SpixelFCN fixed training, run
```
python main_fixed.py --datapath <SCENEFLOW_PATH> --savemodel <SF_FIX_DUMP_PATH> 
```
The superpixel grid size is equal to 4 for the SceneFlow training, and the corresponding SpixelFCN checkpoint will be loaded 
automatically from ```preTrain_spixel_model```.

## Fine-tuning 
Next, we can fine-tune the pre-trained SceneFlow model on HR_VS, by running
```
python finetune_HRVS_sp16.py --datapath <HRVS_PATH> --loadmodel <SF_CKPT> --m_w 30
```
Similarly, we also provide the code to fine-tune a fixed model in ```finetune_HRVS_sp16_fixed.py```. 

To fine-tune on Middlebury-v2, run
```
python finetune_mb_sp16.py --dataset <MB_PATH> --loadmodel <SF_CKPT> --m_w 60
```
The superpixel grid size is equal to 16 for these two datasets. The corresponding SpixelFCN checkpoint will be automatically loaded and overwrite 
the weight in ```<SF_CKPT>```.

## Test
We have provided our pre-trained [SceneFlow model](https://drive.google.com/file/d/11cXW21MU_a66bYXZ5S-0ZPzn7TI2vyRG/view?usp=sharing)
and [HR_VS model](https://drive.google.com/file/d/1rdf1qrtUN3R2eZWpsbuk7pAE-gJmCK9t/view?usp=sharing) online. To test on SceneFlow, run
```
python infer_sceneflow.py --dataset <SCENEFLOW_PATH> --loadmodel <SF_CKPT>
```
To test on HRVS, run 
```
python infer_sceneflow.py --dataset <HRVS_PATH> --loadmodel <HRVS_CKPT>
```
The EPE score will be shown after the test.

We can also use ```submission_mb.py``` to create the submission package for the Middlebury-v2 evaluation. 

## Acknowledgement
This code is developed based on the [PSMNet](https://github.com/JiaRenChang/PSMNet) and [SpixelFCN](https://github.com/fuy34/superpixel_fcn). 
