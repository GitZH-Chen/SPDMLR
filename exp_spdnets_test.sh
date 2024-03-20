#!/bin/bash

#----HDM05----
hdm05_path=/data #change this to your data folder

### Experiments on SPDNet-LCM
[ $? -eq 0 ] && python SPDNet-MLR -m dataset=HDM05 dataset.path=hdm05_path nnet.model.architecture=[93,30],[93,70,30],[93,70,50,30] nnet.model.classifier=SPDMLR\
nnet.model.metric=SPDLogCholeskyMetric nnet.model.power=1.,0.5