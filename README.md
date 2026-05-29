# CalibrateMix

Official implementation for **CalibrateMix**, a semi-supervised learning method built on top of the [USB / semilearn framework](USB_REPO_LINK_HERE). CalibrateMix extends SoftMatch with calibration-aware sample difficulty estimates and mixup between easy and hard labeled/unlabeled examples.

Links:

- Paper: [PAPER_LINK_HERE](https://ojs.aaai.org/index.php/AAAI/article/view/39696)
- arXiv: [ARXIV_LINK_HERE](https://arxiv.org/abs/2511.12964)
- Base framework: [USB_Paper](https://arxiv.org/abs/2208.07204) , [USB_Repo](https://github.com/microsoft/Semi-supervised-learning)

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

## Evaluation

Evaluate a saved checkpoint with:

```bash
python eval.py \
  --dataset cifar10 \
  --num_classes 10 \
  --net wrn_28_2 \
  --load_path saved_models/classic_cv/softmatch_calibratemix_cifar10_40_0/model_best.pth
```

The evaluation script reports:

```text
Test Accuracy
Test Error Rate
Test ECE
```

## Acknowledgments

This codebase is based on the USB / semilearn framework. Please cite and acknowledge USB when using this repository. We also thank the following projects for their inspiring work:
[FixMatch](https://arxiv.org/abs/2001.07685)
[FlexMatch](https://arxiv.org/abs/2110.08263)
[SoftMatch](https://arxiv.org/abs/2301.10921)



## Citation

If you find this repository useful, please cite our paper:

```bibtex
@inproceedings{rahman2026calibration,
  title={On the Calibration of Image Semi-Supervised Learning Models},
  author={Rahman, Mehrab Mustafy and Mohan, Jayanth and Sosea, Tiberiu and Caragea, Cornelia},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={40},
  number={30},
  pages={25073--25081},
  year={2026}
}
```
