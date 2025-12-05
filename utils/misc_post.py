"""Some helper functions for PyTorch, including:
- get_mean_and_std: calculate the mean and std value of dataset.
"""

import logging
import numpy as np
from sklearn.metrics import precision_score, recall_score, accuracy_score
import torch

logger = logging.getLogger(__name__)

__all__ = ["get_mean_and_std", "accuracy", "AverageMeter", "pseudo_statics"]


def get_mean_and_std(dataset):
    """Compute the mean and std value of dataset."""
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=4
    )

    mean = torch.zeros(3)
    std = torch.zeros(3)
    logger.info("==> Computing mean and std..")
    for inputs, targets in dataloader:
        for i in range(3):
            mean[i] += inputs[:, i, :, :].mean()
            std[i] += inputs[:, i, :, :].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


class AverageMeter(object):
    """Computes and stores the average and current value
    Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def pseudo_statics(targets_u, groundtruth_u, mask, num_classes=10):
    """Calculate statistics for pseudo-labels."""
    targets_u = targets_u.cpu().numpy()
    groundtruth_u = groundtruth_u.cpu().numpy()
    groundtruth_counts = np.bincount(groundtruth_u, minlength=num_classes)
    mask = mask.cpu().numpy()

    if np.sum(mask) == 0:
        logger.warning("No valid pseudo-labels found. Returning zeros.")
        return np.zeros(num_classes), np.zeros(num_classes), 0.0, np.zeros(num_classes)

    targets_u = targets_u[mask == 1]
    targets_counts = np.bincount(targets_u, minlength=num_classes)
    groundtruth_u = groundtruth_u[mask == 1]

    # flatten arrays in case they are not 1D
    targets_u = targets_u.flatten()
    groundtruth_u = groundtruth_u.flatten()

    # precision and recall (macro average over classes)
    precision = precision_score(
        groundtruth_u,
        targets_u,
        average=None,
        zero_division=0,
        labels=np.arange(num_classes),
    )
    accuracy = accuracy_score(
        groundtruth_u,
        targets_u,
        normalize=True,
        sample_weight=None,
    )

    recall = np.zeros(num_classes)
    for i, (t_count, g_count) in enumerate(zip(targets_counts, groundtruth_counts)):
        recall[i] = t_count * precision[i] / g_count if g_count > 0 else 0.0

    return (precision, recall, accuracy, targets_counts)
