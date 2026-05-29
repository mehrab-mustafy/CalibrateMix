# CalibrateMix

Official implementation for **CalibrateMix**, a semi-supervised learning method built on top of the [USB / semilearn framework](USB_REPO_LINK_HERE). CalibrateMix extends SoftMatch with calibration-aware sample difficulty estimates and mixup between easy and hard labeled/unlabeled examples.

Links:

- Paper: PAPER_LINK_HERE
- arXiv: ARXIV_LINK_HERE
- Base framework: USB_REPO_LINK_HERE

## Overview

This repository contains the CalibrateMix implementation used for classic computer vision semi-supervised learning experiments. The main algorithm is registered as:

```text
softmatch_calibratemix
```

The implementation is located at:

```text
semilearn/algorithms/softmatch/softmatch_calibratemix.py
```

CalibrateMix keeps the SoftMatch training objective and adds an AUM/APM-based post-warmup mixup stage. During warmup, the model trains with the standard SoftMatch loss. After warmup, the method estimates labeled sample difficulty and unlabeled pseudo-label margin in memory, splits batches into easy and hard groups, and applies mixup between complementary groups.

## Repository Structure

```text
config/classic_cv/softmatch/        CalibrateMix experiment YAML files
semilearn/algorithms/softmatch/     SoftMatch and CalibrateMix implementations
semilearn/                          USB / semilearn training framework
eval.py                            Standalone checkpoint evaluation script
train.py                           Training entrypoint
requirements.txt                   Python dependencies
```

## Installation

Create and activate a Python environment, then install the requirements:

```bash
conda create -n calibratemix python=3.9
conda activate calibratemix
pip install -r requirements.txt
```

Install the PyTorch build appropriate for your CUDA version if the default package resolver does not select the version you need.

## Datasets

Datasets are handled through the USB dataset pipeline. By default, configs use:

```text
./data
```

as the data root. The supported classic CV configs in this repository cover CIFAR-10, CIFAR-100, SVHN, and STL-10 label settings.

## Training

Run training with a YAML config:

```bash
python train.py --c config/classic_cv/softmatch/softmatch_cifar10_40_0.yaml
```

Other provided configs include:

```text
config/classic_cv/softmatch/softmatch_cifar10_250_0.yaml
config/classic_cv/softmatch/softmatch_cifar10_4000_0.yaml
config/classic_cv/softmatch/softmatch_cifar100_400_0.yaml
config/classic_cv/softmatch/softmatch_cifar100_2500_0.yaml
config/classic_cv/softmatch/softmatch_cifar100_10000_0.yaml
config/classic_cv/softmatch/softmatch_svhn_40_0.yaml
config/classic_cv/softmatch/softmatch_svhn_250_0.yaml
config/classic_cv/softmatch/softmatch_svhn_1000_0.yaml
config/classic_cv/softmatch/softmatch_stl10_40_0.yaml
config/classic_cv/softmatch/softmatch_stl10_250_0.yaml
config/classic_cv/softmatch/softmatch_stl10_1000_0.yaml
```

The configs save checkpoints under:

```text
saved_models/classic_cv/
```

## Important Config Options

The YAML files use:

```yaml
algorithm: softmatch_calibratemix
amp: True
```

For CIFAR-100 experiments, the backbone is set to:

```yaml
net: wrn_28_8
```

CalibrateMix-specific arguments have defaults in the algorithm file:

```yaml
apm_delta: 0.997
apm_warmup_iter: 150000
```

These can be added to a YAML file if you want the warmup and APM smoothing settings to be explicit.

## Evaluation

Evaluate a saved checkpoint with:

```bash
python eval.py \
  --dataset cifar10 \
  --num_classes 10 \
  --net wrn_28_2 \
  --load_path saved_models/classic_cv/softmatch_calibratemix_cifar10_40_0/latest_model.pth
```

For CIFAR-100 checkpoints, use `--num_classes 100` and `--net wrn_28_8`.

The evaluation script reports:

```text
Test Accuracy
Test Error Rate
Test ECE
```

ECE uses 10 confidence bins by default. You can override this with:

```bash
--ece_bins 10
```

## Method Notes

CalibrateMix stores labeled AUM and unlabeled pseudo-margin statistics in memory during training and saves them in checkpoints. This avoids repeatedly writing and reading CSV files during the post-warmup stage and allows resumed runs to keep the difficulty estimates.

The post-warmup mixup stage begins when:

```text
iteration > apm_warmup_iter
```

## Acknowledgments

This codebase is based on the USB / semilearn framework. Please cite and acknowledge USB when using this repository.

- USB repository: USB_REPO_LINK_HERE
- USB paper: USB_PAPER_LINK_HERE

## Citation

If you find this repository useful, please cite our paper:

```bibtex
@article{calibratemix,
  title   = {CALIBRATEMIX_TITLE_HERE},
  author  = {AUTHOR_LIST_HERE},
  journal = {ARXIV_OR_VENUE_HERE},
  year    = {YEAR_HERE}
}
```

Please also cite the USB framework:

```bibtex
@inproceedings{usb,
  title     = {USB: A Unified Semi-supervised Learning Benchmark for Classification},
  author    = {Wang, Yidong and Chen, Hao and Fan, Yue and Sun, Wang and Tao, Ran and Hou, Wenxin and Wang, Ran and Yang, Linyi and Zhou, Zhi and Guo, Lan-Zhe and Qi, Heli and Wu, Zhen and Li, Yu-Feng and Nakamura, Satoshi and Ye, Wei and Savvides, Marios and Raj, Bhiksha and Shinozaki, Takahiro and Schiele, Bernt and Xie, Xing and Zhang, Yue and Sugiyama, Masashi and Liu, Weiyang and Smith, Noah A. and Wang, Wen-tau and Jia, Xu and Li, Bo},
  booktitle = {NeurIPS Datasets and Benchmarks},
  year      = {2022}
}
```
