
from torch.optim.lr_scheduler import LambdaLR
import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

def get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps,
    num_training_steps,
    num_cycles=7.0 / 16.0,
    last_epoch=-1,
):
    def _lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        no_progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(0, math.cos(math.pi * num_cycles * no_progress))

    return LambdaLR(optimizer, _lr_lambda, last_epoch)

def interleave(xy, batch):
    nu = len(xy) - 1
    offsets = interleave_offsets(batch, nu)
    xy = [[v[offsets[p] : offsets[p + 1]] for p in range(nu + 1)] for v in xy]
    for i in range(1, nu + 1):
        xy[0][i], xy[i][i] = xy[i][i], xy[0][i]
    return [torch.cat(v, dim=0) for v in xy]

def interleave_offsets(batch, nu):
    groups = [batch // (nu + 1)] * (nu + 1)
    for x in range(batch - sum(groups)):
        groups[-x - 1] += 1
    offsets = [0]
    for g in groups:
        offsets.append(offsets[-1] + g)
    assert offsets[-1] == batch
    return offsets

def linear_rampup(current, rampup_length=0):
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current / rampup_length, 0.0, 1.0)
        return float(current)

class ProportionLoss(nn.Module):
    def __init__(self, metric="ce", eps=1e-8):
        super().__init__()
        self.metric = metric
        self.eps = eps
    
    def cross_entropy_loss(self, input, target, eps=1e-8):
        input = torch.clamp(input, eps, 1 - eps)
        loss = -target * torch.log(input + eps)
        return loss

    def forward(self, input, target):
        if self.metric == "ce":
            loss = self.cross_entropy_loss(input, target, eps=self.eps)
        elif self.metric == "l1":
            loss = F.l1_loss(input, target, reduction="none")
        elif self.metric == "mse":
            loss = F.mse_loss(input, target, reduction="none")
        else:
            raise NameError("metric {} is not supported".format(self.metric))

        loss = torch.sum(loss, dim=-1).mean()
        return loss
