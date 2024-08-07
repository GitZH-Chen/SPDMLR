#!/bin/bash

data_dir=/data #change this to your data folder
### Experiments on SPDDSMBN
[ $? -eq 0 ] && python TSMNet-MLR.py -m data_dir=$data_dir evaluation=inter-subject+uda,inter-session+uda

### Experiments on SPDDSMBN+SPDMLR-(1,0)-LEM
[ $? -eq 0 ] && python TSMNet-MLR.py -m data_dir=$data_dir evaluation=inter-subject+uda,inter-session+uda nnet.model.metric=SPDLogEuclideanMetric

### Experiments on SPDDSMBN+SPDMLR-(\theta)-LCM
# inter-session inter-subject
[ $? -eq 0 ] && python TSMNet-MLR.py -m data_dir=$data_dir evaluation=inter-session+uda,inter-subject+uda nnet.model.metric=SPDLogCholeskyMetric nnet.model.power=1.,1.5