# PICNN
## About
This repository is the official PyTorch implementation of "PICNN: A Pathway towards Interpretable Convolutional Neural Networks", Wengang Guo, Jiayi Yang, Huilin Yin, Qijun Chen, Wei Ye, AAAI 2024.
## Experiment
* Apply the python environment:
  
  ```bash
  conda create -n PICNN python=3.8
  conda activate PICNN
  ./requirements.sh
  ```
  
* Run the experiment:

  Type
  ```bash
  python main.py -configFileName='./configs/cifar10.yml' -backbone='resnet18' -criterion='StandardCE'
  ```
  for standard CNNs (STD) using ResNet as the backbone.

  Type
  ```bash
  python main.py -configFileName='./configs/cifar10.yml' -backbone='resnet18' -criterion='ClassSpecificCE'
  ```
  for PICNN.


  
