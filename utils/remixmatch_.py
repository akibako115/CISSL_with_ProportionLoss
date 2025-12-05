import torch
import torch.nn.functional as F
import numpy as np
import argparse


def get_args():
    parser = argparse.ArgumentParser(description="PyTorch fixMatch Training")

    # Optimization options
    parser.add_argument("--epochs",  default=500, type=int, metavar="N", help="number of total epochs to run",)
    parser.add_argument("--start-epoch", default=0, type=int, metavar="N", help="manual epoch number (useful on restarts)",)
    parser.add_argument("--val-iteration", type=int, default=500, help="Frequency for the evaluation",)
    parser.add_argument("--batch-size", default=64, type=int, metavar="N", help="train batchsize",)
    parser.add_argument("--mu", default=7, type=int, help="coefficient of unlabeled batch size",)

    # Checkpoints
    parser.add_argument("--resume", default="", type=str, metavar="PATH", help="path to latest checkpoint (default: none)",)
    parser.add_argument("--out", default="result", help="Directory to output the result",)

    # Miscs
    parser.add_argument("--manualSeed", type=int, default=0, help="manual seed")
    
    # Device options
    parser.add_argument("--gpu", default="0", type=str, help="id(s) for CUDA_VISIBLE_DEVICES")

    # Dataset options
    parser.add_argument("--dataset", type=str, default="cifar10", help="Dataset")
    parser.add_argument("--num_max", type=int, default=1500, help="Number of samples in the maximal class",)
    parser.add_argument("--label_ratio", type=float, default=2, help="percentage of labeled data",)
    parser.add_argument("--imb_ratio", type=int, default=5, help="Imbalance ratio",)
    parser.add_argument("--imbalancetype", type=str, default="long", help="Long tailed or step imbalanced",)
    parser.add_argument("--num_val", type=int, default=10, help="Number of validation data")

    # Model options
    parser.add_argument("--arch", default="wideresnet", type=str, choices=["wideresnet", "resnext"], help="architecture name",)

    # Hyperparameters for RemixMatch
    parser.add_argument("--lambda-u-mixed", default=1.5, type=float, help="coefficient of proportion loss",)
    parser.add_argument("--lambda-u", default=0.5, type=float, help="coefficient of unlabeled loss",)
    parser.add_argument("--lambda-r", default=0.5, type=float, help="coefficient of rotation loss",)
    parser.add_argument("--T", default=0.5, type=float, help="pseudo label temperature",)

    # Learning strategy options
    parser.add_argument("--lr", "--learning-rate", default=0.03, type=float, metavar="LR", help="initial learning rate",)
    parser.add_argument("--nesterov", action="store_true", default=True, help="use nesterov momentum",)
    parser.add_argument("--wdecay", default=5e-4, type=float, help="weight decay",)
    parser.add_argument("--ema-decay", default=0.999, type=float, help="EMA decay",)
    parser.add_argument("--warmup", default=0, type=float, help="warmup epochs (unlabeled data based)",)

    # Optional for our method
    parser.add_argument("--ours", action="store_true", default=False, help="use our method")
    parser.add_argument("--lambda-p", default=0.01, type=float, help="coefficient of proportion loss",)
    parser.add_argument("--T-prop", default=1, type=float, help="LLP temperature",)

    # Optional for DARP
    parser.add_argument('--warm', type=int, default=200, help='Number of warm up epoch for DARP')
    parser.add_argument('--alpha', default=2.0, type=float, help='hyperparameter for removing noisy entries')
    parser.add_argument('--darp', action='store_true', help='Applying DARP')
    parser.add_argument('--est', action='store_true', help='Using estimated distribution for unlabeled dataset')
    parser.add_argument('--iter_T', type=int, default=10, help='Number of iteration (T) for DARP')
    parser.add_argument('--num_iter', type=int, default=10, help='Scheduling for updating pseudo-labels')

    # Optional for CReST
    parser.add_argument("--crest", action="store_true", default=False, help="use CReST")
    parser.add_argument("--crest-period", type=int, default=20, help="CReST period")
    parser.add_argument("--crest-alpha", type=float, default=1/3, help="CReST alpha")
    parser.add_argument("--crest-remove", action="store_true", default=True, help="remove selected samples from unlabeled dataset")
    parser.add_argument("--crest-tau", type=float, default=0.95, help="CReST tau")
    parser.add_argument("--crest-t-min", type=float, default=0.5, help="CReST t-min")
    
    args = parser.parse_args()
    state = {k: v for k, v in args._get_kwargs()}

    return args, state


def interleave_offsets(batch, nu):
    groups = [batch // (nu + 1)] * (nu + 1)
    for x in range(batch - sum(groups)):
        groups[-x - 1] += 1
    offsets = [0]
    for g in groups:
        offsets.append(offsets[-1] + g)
    assert offsets[-1] == batch
    return offsets


def interleave(xy, batch):
    nu = len(xy) - 1
    offsets = interleave_offsets(batch, nu)
    xy = [[v[offsets[p] : offsets[p + 1]] for p in range(nu + 1)] for v in xy]
    for i in range(1, nu + 1):
        xy[0][i], xy[i][i] = xy[i][i], xy[0][i]
    return [torch.cat(v, dim=0) for v in xy]


def linear_rampup(current, rampup_length=0):
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current / rampup_length, 0.0, 1.0)
        return float(current)


class SemiLoss(object):
    def __call__(
        self, args, outputs_x, targets_x, outputs_u, targets_u, epoch, mask=None
    ):
        Lx = -torch.mean(torch.sum(F.log_softmax(outputs_x, dim=1) * targets_x, dim=1))
        Lu = -torch.mean(torch.sum(F.log_softmax(outputs_u, dim=1) * targets_u, dim=1))

        return Lx, Lu, args.lambda_u * linear_rampup(epoch, args.epochs)
