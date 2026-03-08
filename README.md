# Leveraging Label Proportion Prior for Class-Imbalanced Semi-Supervised Learning

Official implementation of [**LEVERAGING LABEL PROPORTION PRIOR FOR CLASS-IMBALANCED SEMI-SUPERVISED LEARNING**](https://arxiv.org/abs/2603.02957) (Kohki Akiba, Shinnosuke Matsuo, Shota Harada, Ryoma Bise @ Kyushu University)

## Overview

In semi-supervised learning (SSL), pseudo-labeling tends to favor majority classes under class imbalance, degrading minority-class performance. We introduce **Proportion Loss** from **Learning from Label Proportions (LLP)** into SSL as a regularization term for the first time, aligning model predictions with the global class distribution to mitigate this bias. We further stabilize training under severe imbalance with a **stochastic variant** of Proportion Loss that models mini-batch class composition via a multivariate hypergeometric distribution.

- **Proportion Loss**: Regularizes agreement between batch-wise predicted proportions and the global class proportion estimated from labeled data, using cross-entropy.
- **Stochastic variant**: At each iteration, target proportions are sampled from a multivariate hypergeometric distribution to reduce overfitting to a fixed noisy proportion.

The method is implemented on top of FixMatch and ReMixMatch. On Long-tailed CIFAR-10 it consistently improves over baselines and is competitive or better than existing CISSL methods, especially when the labeled ratio is small (β = 2%, 4%).

## Setup

**RandAugment** is used for data augmentation. Place a RandAugment implementation (e.g. [pytorch-randaugment](https://github.com/ildoonet/pytorch-randaugment)) under `dataset/RandAugment/`.

```bash
# Example: place RandAugment under dataset/
# Use git submodule or clone to provide dataset/RandAugment
```

## Dataset

- **CIFAR-10-LT**: Long-tailed CIFAR-10. The number of samples per class is controlled by exponential decay; you set the imbalance ratio γ and the labeled ratio β.
- Configuration is in the `dataset` section of `config/*.json`:
  - `imb_ratio`: Imbalance ratio γ (e.g. 10, 20, 50, 100)
  - `label_ratio`: Labeled ratio β (e.g. 2, 4, 10, 20)
  - `path`: Directory for CIFAR-10 (e.g. `./data`)
  - `imbalancetype`: `"long"` (long-tailed) or `"step"`

## Training

Run `main.py` with a config file:

```bash
# FixMatch + Ours (Proportion Loss)
python main.py --config config/train_config1.json

# ReMixMatch + Ours
python main.py --config config/train_config2.json

# FixMatch baseline (no Proportion Loss)
python main.py --config config/train_config3.json

# ReMixMatch baseline
python main.py --config config/train_config4.json
```

To match Table 1 in the paper, set `dataset.label_ratio` and `dataset.imb_ratio` to combinations such as (γ=10, β=2%), (γ=20, β=4%), (γ=50, β=10%), (γ=100, β=20%), and set `training.method` to `"ours"` (proposed) or `"vanilla"` (baseline). For multiple seeds, change `training.manualSeed` and run again.

## Config options

| Option | Description |
|--------|-------------|
| `training.base` | `"fixmatch"` or `"remixmatch"` |
| `training.method` | `"ours"` (with Proportion Loss) / `"vanilla"` (without) |
| `training.lambda_p` | Weight λ for Proportion Loss |
| `training.T_prop` | Temperature for proportion softmax (scaling of logits before softmax) |
| `training.tau` | Pseudo-label confidence threshold |
| `training.mu` | Unlabeled batch size = `batch_size * mu` |
| `dataset.imb_ratio` | Imbalance ratio γ |
| `dataset.label_ratio` | Labeled ratio β (%) |

Logs and checkpoints are written under `output_dir` (default `./results`), in paths that include dataset name, base method, label ratio, imbalance ratio, seed, and method name. Use TensorBoard on the `results/...` directory to view training curves.

## Project structure

```
.
├── main.py                 # Entry point (data, model, trainer setup and training)
├── config/
│   ├── train_config1.json  # FixMatch + Ours
│   ├── train_config2.json  # ReMixMatch + Ours
│   ├── train_config3.json  # FixMatch vanilla
│   └── train_config4.json  # ReMixMatch vanilla
├── dataset/
│   ├── fix_cifar10.py      # CIFAR-10-LT for FixMatch
│   ├── remix_cifar10.py    # CIFAR-10-LT for ReMixMatch
│   └── RandAugment/        # RandAugment (place separately)
├── models/
│   ├── wideresnet.py       # Wide ResNet-28-2 for FixMatch
│   ├── wideresnetwithrot.py# Wide ResNet with rotation head for ReMixMatch
│   ├── ema.py              # EMA model
│   └── resnext.py
├── trainer/
│   ├── trainer_fixmatch.py # FixMatch + Proportion Loss training loop
│   ├── trainer_remixmatch.py # ReMixMatch + Proportion Loss training loop
│   └── validator.py       # Validation and test
└── utils/
    ├── train_tools.py      # ProportionLoss, cosine scheduler, interleave, etc.
    ├── misc.py             # Imbalance data generation (make_imb_data), accuracy, set_seed
    ├── fixmatch_.py
    └── remixmatch_.py
```

- **Proportion Loss**: Implemented in `utils/train_tools.py` as `ProportionLoss`, matching predicted and target proportions via cross-entropy.
- **Stochastic target proportions**: In `trainer_fixmatch.py` and `trainer_remixmatch.py`, class counts are sampled with `rng.multivariate_hypergeometric(U_SAMPLES_PER_CLASS, nsample=batch_size*mu)`, normalized, and used as the target proportion for Proportion Loss.

## Results (paper)

- **CIFAR-10-LT**: Adding the proposed method to FixMatch and ReMixMatch improves over baselines across (γ, β) settings and is competitive or better than DARP and CReST+PDA, especially at β = 2% and 4%.
- See Table 1 in the paper [arXiv:2603.02957](https://arxiv.org/abs/2603.02957) for exact numbers.

## Citation

```bibtex
@article{akiba2026leveraging,
  title={Leveraging Label Proportion Prior for Class-Imbalanced Semi-Supervised Learning},
  author={Akiba, Kohki and Matsuo, Shinnosuke and Harada, Shota and Bise, Ryoma},
  journal={arXiv preprint arXiv:2603.02957},
  year={2026}
}
```

## Acknowledgments

As stated in the paper, this work was supported by JSPS KAKENHI (JP25K22846), ASPIRE (JPMJAP2403), and JST ACT-X (JPMJAX23CR).
