#!/bin/bash

#----RADAR----
radar_path=/data #change this to your data folder
### Experiments on SPDNet
[ $? -eq 0 ] && python SPDNet-MLR -m dataset=RADAR dataset.path=radar_path nnet.model.architecture=[20,16,14,12,10,8],[20,16,8] nnet.model.classifier=LogEigMLR
### Experiments on SPDNet-LEM
[ $? -eq 0 ] && python SPDNet-MLR.py -m dataset=RADAR dataset.path=radar_path nnet.model.architecture=[20,16,14,12,10,8],[20,16,8] nnet.model.classifier=SPDMLR\
 nnet.model.metric=SPDLogEuclideanMetric nnet.model.beta=1.,0.

### Experiments on SPDNet-LCM
[ $? -eq 0 ] && python SPDNet-MLR.py -m dataset=RADAR dataset.path=radar_path nnet.model.architecture=[20,16,14,12,10,8],[20,16,8] nnet.model.classifier=SPDMLR\
 nnet.model.metric=SPDLogCholeskyMetric nnet.model.power=1.,0.5

#----HDM05----
hdm05_path=/data #change this to your data folder
### Experiments on SPDNet
[ $? -eq 0 ] && python SPDNet-MLR -m dataset=HDM05 dataset.path=hdm05_path nnet.model.architecture=[93,30],[93,70,30],[93,70,50,30] nnet.model.classifier=LogEigMLR
### Experiments on SPDNet-LEM
[ $? -eq 0 ] && python SPDNet-MLR -m dataset=HDM05 dataset.path=hdm05_path nnet.model.architecture=[93,30],[93,70,30],[93,70,50,30] nnet.model.classifier=LogEigMLR\
nnet.model.metric=SPDLogEuclideanMetric

### Experiments on SPDNet-LCM
[ $? -eq 0 ] && python SPDNet-MLR -m dataset=HDM05 dataset.path=hdm05_path nnet.model.architecture=[93,30],[93,70,30],[93,70,50,30] nnet.model.classifier=LogEigMLR\
nnet.model.metric=SPDLogCholeskyMetric