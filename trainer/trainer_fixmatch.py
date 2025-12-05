from __future__ import annotations

import os
import sys
import shutil

import numpy as np
from numpy.random import Generator
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from .validator import Validator
from utils.misc import AverageMeter
from utils.train_tools import ProportionLoss
from tqdm import tqdm

class Trainer:
    def __init__(
        self,
        model: nn.Module,
        ema_model: nn.Module,
        labeled_loader: DataLoader,
        unlabeled_loader: DataLoader,
        optimizer: Optimizer,
        scheduler: _LRScheduler,
        rng: Generator,
        U_SAMPLES_PER_CLASS: list[int],
        checkpoint_path: str,
        writer: SummaryWriter,
        train_config: dict,
        validator: Validator,
        tester: Validator,
    ):
        self.model = model
        self.ema_model = ema_model
        self.labeled_loader = labeled_loader
        self.unlabeled_loader = unlabeled_loader
        self.optimizer = optimizer
        self.scheduler = scheduler

        # training config
        self.method = train_config["method"]
        self.epochs = train_config["epochs"]
        self.val_iteration = train_config["val_iteration"]
        self.device = train_config["device"]
        self.tau = train_config["tau"]
        self.T = train_config["T"]
        self.mu = train_config["mu"]
        self.lambda_p = train_config["lambda_p"]
        self.lambda_u = train_config["lambda_u"]
        self.num_class = train_config["num_class"]
        self.batch_size = train_config["batch_size"]
        self.T_prop = train_config["T_prop"]
        # others
        self.rng = rng
        self.U_SAMPLES_PER_CLASS = U_SAMPLES_PER_CLASS
        self.checkpoint_path = checkpoint_path
        self.writer = writer
        self.loggers = {
            "train_losses": AverageMeter(),
            "train_losses_x": AverageMeter(),
            "train_losses_u": AverageMeter(),
            "train_losses_p": AverageMeter(),
            "train_mask_rate": AverageMeter(),
        }

        # validator class : track validation metrics
        self.validator = validator
        self.tester = tester

        # loss function
        self.proportion_loss = ProportionLoss()

    def train(self):
        print("Start Training...")
        for epoch in range(self.epochs):
            print(f"Epoch: {epoch + 1}/{self.epochs}")

            self.train_epoch(epoch)
            is_best = self.validate(epoch, self.validator) # eval on validation set
            _ = self.validate(epoch, self.tester) # eval on test set
            self.save_checkpoint(epoch, is_best)

            if is_best:
                print(f"Best epoch: {epoch + 1}")
            
        print("Training Finished!")

    def train_epoch(self, epoch):
        # reset metrics
        self._reset_metrics()

        self.model.train()

        labeled_iter = iter(self.labeled_loader)
        unlabeled_iter = iter(self.unlabeled_loader)

        progress = tqdm(range(self.val_iteration))
        for batch_idx in progress:
            try:
                inputs_x, targets_x, _ = next(labeled_iter)
            except:
                labeled_iter = iter(self.labeled_loader)
                inputs_x, targets_x, _ = next(labeled_iter)

            try:
                (inputs_u1, inputs_u2), groundtruth_u, idx_u = next(unlabeled_iter)
            except:
                unlabeled_iter = iter(self.unlabeled_loader)
                (inputs_u1, inputs_u2), groundtruth_u, idx_u = next(unlabeled_iter)

            inputs_x, targets_x = inputs_x.to(self.device), targets_x.to(self.device, non_blocking=True)
            inputs_u1, inputs_u2 = inputs_u1.to(self.device), inputs_u2.to(self.device)
            logits_x = self.model(inputs_x)
            logits_u1, logits_u2 = self.model(inputs_u1), self.model(inputs_u2)

            # create pseudo label
            prob_u1 = torch.softmax(logits_u1.detach() / self.T, dim=-1)
            max_probs, targets_u = torch.max(prob_u1, dim=-1)
            mask = max_probs.ge(self.tau).float()

            # predict proportion for batch
            confidence_u1 = F.softmax(logits_u1 / self.T_prop, dim=-1)
            prop_u1 = confidence_u1.mean(dim=0)

            # calculate loss for labeled data
            Lx = F.cross_entropy(logits_x, targets_x, reduction="mean")
            # calculate loss for unlabeled
            Lu = F.cross_entropy(logits_u2, targets_u, reduction="none")
            Lu = (Lu * mask).mean()

            # calculate proportion loss for unlabled data (ours)
            if "ours" in self.method:
                est_counts = torch.tensor(
                    self.rng.multivariate_hypergeometric(self.U_SAMPLES_PER_CLASS, nsample=self.batch_size*self.mu),
                    device=self.device,
                    dtype=torch.float32,
                )
                est_props = est_counts / torch.sum(est_counts)
                # calculate proportion loss for unlabeled
                Lp = self.proportion_loss(prop_u1, est_props)
            else:
                Lp = torch.tensor(0.0, device=self.device)

            loss = Lx + self.lambda_u * Lu + self.lambda_p * Lp

            self.loggers["train_losses"].update(loss.item())
            self.loggers["train_losses_x"].update(Lx.item())
            self.loggers["train_losses_u"].update(Lu.item())
            self.loggers["train_losses_p"].update(Lp.item())
            self.loggers["train_mask_rate"].update(mask.mean().item())

            # compute gradient and do SGD step
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            self.ema_model.update(self.model)
            self.optimizer.zero_grad(set_to_none=True)

            if epoch % 10 == 0:
                progress.set_postfix(
                    loss=f"{self.loggers['train_losses'].avg:.3f}",
                    Lx=f"{self.loggers['train_losses_x'].avg:.3f}",
                    Lu=f"{self.loggers['train_losses_u'].avg:.3f}",
                    Lp=f"{self.loggers['train_losses_p'].avg:.3f}",
                    Mask=f"{self.loggers['train_mask_rate'].avg:.3f}")

        self._record_metrics(epoch)
        progress.close()

    def validate(self, epoch, user: Validator):
        is_best = user.validate(self.ema_model.ema, epoch)
        return is_best

    def _record_metrics(self, epoch):
        self.writer.add_scalar("train1/loss", self.loggers["train_losses"].avg, epoch)
        self.writer.add_scalar("train2/loss_x", self.loggers["train_losses_x"].avg, epoch)
        self.writer.add_scalar("train3/loss_u", self.loggers["train_losses_u"].avg, epoch)
        self.writer.add_scalar("train4/loss_p", self.loggers["train_losses_p"].avg, epoch)
        self.writer.add_scalar("train5/mask_rate", self.loggers["train_mask_rate"].avg, epoch)
        self._reset_metrics()

    def _reset_metrics(self):
        for key in self.loggers.keys():
            self.loggers[key].reset()

    def save_checkpoint(self, epoch, is_best, filename="checkpoint.pth.tar"):
        state = {
            "epoch": epoch + 1,
            "state_dict": self.model.state_dict(),
            "ema_state_dict": self.ema_model.ema.state_dict(),
            "val_acc": self.validator.loggers["val_acc"].avg,
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
        }
        filepath = os.path.join(self.checkpoint_path, filename)
        torch.save(state, filepath)
        if is_best:
            shutil.copyfile(filepath, os.path.join(self.checkpoint_path, "model_best.pth.tar"))

def trainer_factory(
    model: nn.Module,
    ema_model: nn.Module,
    labeled_loader: DataLoader,
    unlabeled_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    optimizer: Optimizer,
    scheduler: _LRScheduler,
    rng: Generator,
    U_SAMPLES_PER_CLASS: list[int],
    checkpoint_path: str,
    writer: SummaryWriter,
    train_config: dict,
):
    validator = Validator(val_loader, train_config["num_class"], train_config["device"], writer, save_name="val")
    tester = Validator(test_loader, train_config["num_class"], train_config["device"], writer, save_name="test")
    trainer = Trainer(
        model,
        ema_model,
        labeled_loader,
        unlabeled_loader,
        optimizer,
        scheduler,
        rng,
        U_SAMPLES_PER_CLASS,
        checkpoint_path,
        writer,
        train_config,
        validator,
        tester,
    )
    return trainer