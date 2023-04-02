# Amazon-inventory-Reconciliation
# Amazon-Inventory-Reconciliation-Using-AI
The project consists of using a bin image dataset to count the number of items in each bin, to detect variance from recorded inventory.

The Amazon Bin Image Dataset contains images and metadata from bins of a pod in an operating Amazon Fulfillment Center. The bin images in this dataset are captured as robot units carry pods as part of normal Amazon Fulfillment Center operations. You can download and find the details at [here](https://aws.amazon.com/ko/public-datasets/amazon-bin-images/).  We propose 3 different tasks, which would be practically useful when we want the bins double-checked. These tasks are sometimes very challenging because of heavy occlusions and a large number of object categories. We would like to open a new challenge in order to attract talented researchers in both academia and industry for these tasks. As a starting point, we provide baseline methods and pre-trained models for two tasks, counting and object verification tasks.

## 1. Tasks

### 1.1. Counting
This is a simple task that you are supposed to count every object instances in the bin. You count individual instances separately, which means if there are two same objects in the bin, you count them as two. 
![counting](figs/counting.png)

### 1.2. Object verification 
This is a task for verifying the presence of the object in the bin. You will be given an image and question pair. The question contains object category and it's presence, e.g. 'Is there a toothbrush in the bin?'. Your program should be able to give an answer 'yes' or 'no'.
![obj_verification](figs/obj_verification.png)

### 1.3. Object quantity verification 
This is a task for verifying the quantity of the object in the bin. You will be given an image and question pair. The question contains the quantity of the object, e.g. 'are there 2 toothbrush in the bin?', your program should be able to give an answer 'yes' or 'no'. In this task, the image in question contains at least one object in question.

![obj_quant_verification](figs/obj_quant_verification.png)


### Evaluation metrics
For counting task, you will be evaluated by two standrad metrics, accuracy(precision) and root mean square error(RMSE). 1 is indicator function, and p and g is prediction and ground truth respectively.

For both verification tasks, you will be evaluated by accuracy.

## 2. Dataset

These are some typical images in the dataset. A bin contains multiple object categories and various number of instances. The corresponding metadata exist for each bin image and it includes the object category identification(Amazon Standard Identification Number, ASIN), quantity, size of objects, weights, and so on. The size of bins are various depending on the size of objects in it. The tapes in front of the bins are for preventing the items from falling out of the bins and sometimes it might make the objects unclear. Objects are sometimes heavily occluded by other objects or limited viewpoint of the images.

## Prerequisites
1. [PyTorch](https://github.com/pytorch/pytorch)
2. [torch-vision](https://github.com/pytorch/vision)

## 3. Data preparation
Directory includes useful codes for dataset preparation and development kits.

### 3.1 Training and validation split
You can make your own training/validation split.

### 3.2 Task specific metadata.
Once you have train/val split and metadata files, now you can generate task specific metadata files. This will be used when you train the baseline methods.

### 3.3 Resizing images
For baseline methods, we resized all image into 224x224 for convinient training purpose. You will have new directory $(data)/bin-images-resize that contain resized images

### 3.4 Moderate and hard task
We divide the each task into two levels of difficulty(moderate and hard). For moderate difficulty, you will be tested over the bin images that contain upto 5 objects. For hard task, you will be tested over all bin images. You can submit your results whatever you are interseted (both, or one of them). As baseline methods, we provide ones for moderate difficulty.

## 4. Deep Convolutional Classification Network for Counting
It is a simple classification network for counting task. The deep CNN will classify the image as one of 6 categories(0-5, for moderate difficulty). We used resnet 34 layer architecture and trained from the scratch. 

### 4.1 Training

### 4.2 Evaluation on validataion sets

## 5. Deep Convolutional Siamese Network for Object Verification
We provide one baseline method for object verification task. Since the number of object categories are huge, we propose to learn how to compare the images instead of modelling all individual object categories. For example, we pick positive pair, which both images should contain at least one common object(no common object for negative pair), and train the network to classify the pair as positive or negative. Siamese network would be proper architectural choice for this purpose.

When testing, we are given one image and the name of object category(asin) in question. From training images, we pick all images that contain the object category in question, and make pairs with the given image. And, we will make final decision as a majority votes based on the results from all pairs. More formally,
