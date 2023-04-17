
# TEXTURE EXTRACTION TECHNIQUES FOR CLASSIFICATION


In this paper, we explored and proved a hypothesis with help of a framework which suggests, that
a group of distinct and unique weak learners when ensembled together form a strong learner. To prove
this hypothesis we have suggested a framework where various texture extraction techniques have been
ensembled after the convolution layers.

## Description:
 
- Texture analysis from images plays a vital role in many industries. Image texture has multiple definitions
according to multiple authors and each one of them defined different ways to calculate different qualities
of texture such as smoothness, roughness, uniformity, homogeneity, etc

- Convolution Neural Network (CNN) has been very successful
in capturing the local and global spatial features, which play a key role in texture classification. CNN
preserves the relative spatial information with the help of convolution layers and aggregates the spatial
information using the pooling layers
 
-  CNN tends to lose the locally rich
features and that’s the major reason why CNN under-perform while identifying textures. To overcome this problem various techniques have been introduced where texture extraction layers are used before the
fully connected layers, which allows the model to focus on local features along with the global features.

- We proposed a new solution to encounter the issue of using various texture feature extraction
techniques togeather. We developed a framework which incorporates combination of various texture related techniques and demonstrated the effectiveness on various texture datasets. The framework have not only resulted in a state-of-the-art result but also made us understand the impact of different techniques in
a combination. Ensemble methods have always produced a improved result when compared to individual model performance. This technique can easily be incorporated in standard CNNs as well as sophisticated models such as Deep Ten, Fenet etc.

- We combined [DeepTEN](https://arxiv.org/abs/1612.02844), where feature extraction, dictionary learning and encoding representation all
happen at the same time. DeepTEN is a flexible framework which allows arbitrary input image size
making it easier to combine with any model. Next is [FENet](https://proceedings.neurips.cc/paper/2021/file/c04c19c2c2474dbf5f7ac4372c5b9af1-Paper.pdf) which focuses on discriminating textures based on the fractal dimension. The third member is a [histogram layer](https://arxiv.org/pdf/2001.00215.pdf) that captures the texture information directly from the feature maps and is based on the fundamental of local histograms which can be used to
distinguish textures.

- While we only considered two remarkable texture datasets such as  [KTH](https://www.csc.kth.se/cvap/databases/kth-tips/index.html) and [FMD](https://people.csail.mit.edu/celiu/CVPR2010/FMD/) and achieved SOTA result, this technique can also be applied to more datasets such as [DTD](https://www.robots.ox.ac.uk/~vgg/data/dtd/download/dtd-r1.0.1.tar.gz), [GTOS](https://www.ece.rutgers.edu/~kdana/gts/index.html), [GTOS_MOBILE](https://drive.google.com/file/d/1Hd1G7aKhsPPMbNrk4zHNJAzoXvUzWJ9M/view) and result in State of the Art. 


## Required Python Dependencies

To run the code, Install the dependencies first using the following commands

```bash
  pip3 install pillow==8.2.0
  pip3 install torch==1.7.1
  pip3 install torchvision==0.8.2
  pip3 install torch-encoding==1.2.2b20200808
  pip3 install mlflow
  pip3 install barbar
  pip3 install Ninja
```


## Code Run

Run the code using following command on colab or terminal


For FMD dataset
```bash
  python main.py --dataset=FMD  --n_classes=10 --train_need --test_need  --test_BS=16 --train_BS=16  --model=FENet --use_pretrained --num_epochs=60 --scheduler="cosine" --lr=0.001
```
 For KTH dataset
```bash
  python main.py --dataset=KTH --n_classes=11 --train_need --test_need  --test_BS=32 --train_BS=32  --model=FENet --use_pretrained --num_epochs=40 --scheduler="cosine" --lr=0.01
```   
## Usage

Clone the repository:

```bash
```   
## Architecture

General Architecture

![Arch](https://github.com/faisu07/Texture_Analysis/blob/main/Architecture.jpeg) 

Our Architecture

![Arch](https://github.com/faisu07/Texture_Analysis/blob/main/Our%20ARCHITECTURE_latest.jpeg)  

<!-- ## Results

![Results of triple combination](https://github.com/faisu07/Texture_Analysis/blob/main/results3.png)    

![Results of single combination](https://github.com/faisu07/Texture_Analysis/blob/main/resukts2.png)    

![Results of double combination](https://github.com/faisu07/Texture_Analysis/blob/main/results.png) -->



## Research Paper

[Paper](https://arxiv.org/abs/2206.04158)


## Authors

- [@Vijay Pandey](https://www.linkedin.com/in/vijay-pandey-29a0a35a)
- [@Trapti Kalra](https://www.linkedin.com/in/traptikalra/)
- [@Mohammed Faisal](www.linkedin.com/in/mohammed-faisal-771b8818b)
- [@Mayank Gubba](https://www.linkedin.com/in/mayank-gubba/)


## Appendix

Any additional information goes here

