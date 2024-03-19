[//]: # ([<img src="https://img.shields.io/badge/arXiv-2206.01323-b31b1b"></img>]&#40;https://arxiv.org/abs/2403.11261&#41;)

[//]: # ([<img src="https://img.shields.io/badge/OpenReview|forum-pp7onaiM4VB-8c1b13"></img>]&#40;https://openreview.net/forum?id=okYdj8Ysru&#41;)

[//]: # ([<img src="https://img.shields.io/badge/OpenReview|pdf-pp7onaiM4VB-8c1b13"></img>]&#40;https://openreview.net/pdf?id=okYdj8Ysru&#41;)


# Riemannian Multinomial Logistics Regression for SPD Neural Networks

This is the official code for our CVPR 2024 publication: *Riemannian Multinomial Logistics Regression for SPD Neural Networks.*. 

[//]: # ([[OpenReview]&#40;https://openreview.net/forum?id=okYdj8Ysru&#41;].)

If you find this project helpful, please consider citing us as follows:


```bib

@inproceedings{
anonymous2024riemannian,
title={Riemannian Multinomial Logistics Regression for {SPD} Neural Networks},
author={Anonymous},
booktitle={Conference on Computer Vision and Pattern Recognition 2024},
year={2024}
}
```

```bib
@inproceedings{chen2024a,
title={A Lie Group Approach to Riemannian Batch Normalization},
author={Ziheng Chen and Yue Song and Yunmei Liu and Nicu Sebe},
booktitle={The Twelfth International Conference on Learning Representations},
year={2024},
url={https://openreview.net/forum?id=okYdj8Ysru}
}
```

In case you have any problem, do not hesitate to contact me ziheng_ch@163.com.

## Requirements

Install necessary dependencies by `conda`:

```setup
conda env create --file environment.yaml
```

**Note** that the [hydra](https://hydra.cc/) package is used to manage configuration files.

## Experiments

The implementation is based on the official code of 
    
- *Riemannian batch normalization for SPD neural networks* [[Neurips 2019](https://papers.nips.cc/paper_files/paper/2019/hash/6e69ebbfad976d4637bb4b39de261bf7-Abstract.html)] [[code](https://papers.nips.cc/paper_files/paper/2019/file/6e69ebbfad976d4637bb4b39de261bf7-Supplemental.zip)].
- *SPD domain-specific batch normalization to crack interpretable unsupervised domain adaptation in EEG* [[Neurips 2022](https://openreview.net/forum?id=pp7onaiM4VB)] [[code](https://github.com/rkobler/TSMNet.git)].

### Dataset

The synthetic [Radar](https://www.dropbox.com/s/dfnlx2bnyh3kjwy/data.zip?e=1&dl=0) dataset is released by SPDNetBN. We further release our preprocessed [HDM05](https://www.dropbox.com/scl/fi/x2ouxjwqj3zrb1idgkg2g/HDM05.zip?rlkey=4f90ktgzfz28x3i2i4ylu6dvu&dl=0) dataset.

The [Hinss2021](https://doi.org/10.5281/zenodo.5055046) dataset is publicly available. 
The [moabb](https://neurotechx.github.io/moabb/) and [mne](https://mne.tools) packages are used to download and preprocess these datasets. 
There is no need to manually download and preprocess the datasets.
This is done automatically.

Please download the datasets and put them in your personal folder.
If necessary, change the `path` accordingly in
`conf/SPDNet/dataset/HDM05.yaml`, `conf/SPDNet/dataset/RADAR.yaml`, and `data_dir` in `conf/TSMNet/TSMNetMLR.yaml`.

### Running experiments

To run experiments on the SPDNet, run this command:

```train
bash exp_spdnets.sh
```
To run experiments on the TSMNet, run this command:
```train
bash exp_eeg.sh
```

These scripts contain the experiments shown in Tab. 3-6

**Note:** You also can change the `data_dir` in `exp_eeg.sh` or `xx_path` in `exp_spdnets.sh`, which will override the hydra config.



