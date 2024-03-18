import math
from typing import Optional, Union
import torch
import torch.nn as nn

import spdnets.cplx.nn as nn_cplx
import spdnets.modules as modules
from spdnets.SPDMLR import SPDRMLR


class SPDNet(nn.Module):
    def __init__(self,args):
        super(__class__, self).__init__()
        dims = [int(dim) for dim in args.architecture]
        self.feature = []
        if args.dataset == 'RADAR':
            self.feature.append(nn_cplx.SplitSignal_cplx(2, 20, 10))
            self.feature.append(nn_cplx.CovPool_cplx())
            self.feature.append(modules.ReEig())

        for i in range(len(dims) - 2):
            shape=[dims[i], dims[i + 1]]
            self.feature.append(modules.BiMap(shape,init_mode=args.init_mode))
            self.feature.append(modules.ReEig())

        self.feature.append(modules.BiMap([dims[-2], dims[-1]],init_mode=args.init_mode))
        self.feature = nn.Sequential(*self.feature)

        self.construct_classifier(args.classifier,dims[-1],args.class_num,args.metric,args.power,args.alpha,args.beta)

    def forward(self, x):
        x_spd = self.feature(x)
        y = self.classifier(x_spd)
        return y

    def construct_classifier(self,classifier,subspacedims,nclasses_,metric,power,alpha,beta):
        if classifier=='SPDMLR':
            self.classifier = torch.nn.Sequential(
                SPDRMLR(n=subspacedims,c=nclasses_,metric=metric,power=power,alpha=alpha,beta=beta)
                )
        elif classifier=='LogEigMLR':
            """Following SPDNet and SPDNetBN, we use the full matrices"""
            tsdim = int( subspacedims ** 2 )
            self.classifier = torch.nn.Sequential(
                modules.LogEig(subspacedims),
                torch.nn.Linear(tsdim, nclasses_),
            )
        else:
            raise Exception(f'wrong clssifier {classifier}')
