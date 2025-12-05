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
from utils.train_tools import interleave, linear_rampup, ProportionLoss
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
        self.lambda_u_mixed = train_config["lambda_u_mixed"]
        self.lambda_r = train_config["lambda_r"]
        self.num_class = train_config["num_class"]
        self.batch_size = train_config["batch_size"]
        self.T_prop = train_config["T_prop"]
        # others
        self.rng = rng
        self.U_SAMPLES_PER_CLASS = U_SAMPLES_PER_CLASS
        self.U_SAMPLES_PROPORTION = torch.tensor(U_SAMPLES_PER_CLASS / np.sum(U_SAMPLES_PER_CLASS)).to(self.device)
        self.checkpoint_path = checkpoint_path
        self.writer = writer
        self.loggers = {
            "train_losses": AverageMeter(),
            "train_losses_x": AverageMeter(),
            "train_losses_u": AverageMeter(),
            "train_losses_p": AverageMeter(),
        }

        # validator class : track validation metrics
        self.validator = validator
        self.tester = tester

        # loss function
        self.proportion_loss = ProportionLoss()

    def train(self):
        print("Start Training...")
        for epoch in range(self.epochs):
            self.train_epoch(epoch)
            is_best = self.validate(epoch, self.validator) # eval on validation set
            _ = self.validate(epoch, self.tester) # eval on test set
            self.save_checkpoint(epoch, is_best)
            
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
                inputs_x, targets_x, _ = next(labeled_train_iter)
            except:
                labeled_train_iter = iter(self.labeled_loader)
                inputs_x, targets_x, _ = next(labeled_train_iter)

            try:
                (inputs_u1, inputs_u2, inputs_u3), groundtruth_u, idx_u = next(
                    unlabeled_train_iter
                )
            except:
                unlabeled_train_iter = iter(self.unlabeled_loader)
                (inputs_u1, inputs_u2, inputs_u3), groundtruth_u, idx_u = next(
                    unlabeled_train_iter
                )

            # Measure data loading time
            batch_size = inputs_x.size(0)

            # Transform label to one-hot
            targets_x = torch.zeros(batch_size, self.num_class).scatter_(1, targets_x.view(-1,1), 1)
            inputs_x, targets_x = inputs_x.to(self.device), targets_x.to(self.device, non_blocking=True)
            inputs_u1, inputs_u2, inputs_u3  = inputs_u1.to(self.device), inputs_u2.to(self.device), inputs_u3.to(self.device)

            # Rotate images
            temp = []
            targets_r = torch.randint(0, 4, (inputs_u2.size(0),)).long()
            for i in range(inputs_u2.size(0)):
                inputs_rot = torch.rot90(inputs_u2[i], targets_r[i], [1, 2]).reshape(1, 3, 32, 32)
                temp.append(inputs_rot)
            inputs_r = torch.cat(temp, 0)
            targets_r = torch.zeros(batch_size * self.mu, 4).scatter_(1, targets_r.view(-1, 1), 1)
            inputs_r, targets_r = inputs_r.to(self.device), targets_r.to(self.device, non_blocking=True)

            # Generate the pseudo labels
            outputs_u1, _ = self.model(inputs_u1)
            with torch.no_grad():
                p = torch.softmax(outputs_u1, dim=1)

                # Tracking the empirical distribution on the unlabeled samples (ReMixMatch)
                real_batch_idx = batch_idx + epoch * self.val_iteration
                if real_batch_idx == 0:
                    emp_distb_u = p.mean(0, keepdim=True)
                elif real_batch_idx // 128 == 0:
                    emp_distb_u = torch.cat([emp_distb_u, p.mean(0, keepdim=True)], 0)
                else:
                    emp_distb_u = emp_distb_u[-127:]
                    emp_distb_u = torch.cat([emp_distb_u, p.mean(0, keepdim=True)], 0)

                pa = p * (self.U_SAMPLES_PROPORTION + 1e-6) / (emp_distb_u.mean(0).to(self.device) + 1e-6)
                p = pa / pa.sum(dim=1, keepdim=True)

                # Temperature scaling
                pt = p ** (1 / self.T)
                targets_u = (pt / pt.sum(dim=1, keepdim=True)).detach()

                # Update the saved predictions with current one
                p = targets_u

            # Mixup
            all_inputs = torch.cat([inputs_x, inputs_u1, inputs_u2, inputs_u3], dim=0)
            all_targets = torch.cat([targets_x, targets_u, targets_u, targets_u], dim=0)

            l = np.random.beta(0.75, 0.75)
            l = max(l, 1-l)
            idx = torch.randperm(all_inputs.size(0))

            input_a, input_b = all_inputs, all_inputs[idx]
            target_a, target_b = all_targets, all_targets[idx]

            mixed_input = l * input_a + (1 - l) * input_b
            mixed_target = l * target_a + (1 - l) * target_b

            # interleave labeled and unlabed samples between batches to get correct batchnorm calculation
            mixed_input = list(torch.split(mixed_input, batch_size))
            mixed_input = interleave(mixed_input, batch_size)

            logits = [self.model(mixed_input[0])[0]]
            for input in mixed_input[1:]:
                logits.append(self.model(input)[0])

            # put interleaved samples back
            logits = interleave(logits, batch_size)
            logits_x = logits[0]
            logits_u = torch.cat(logits[1:], dim=0)

            Lx = -torch.mean(torch.sum(F.log_softmax(logits_x, dim=1) * mixed_target[:batch_size], dim=1))
            Lu_mixed = -torch.mean(torch.sum(F.log_softmax(logits_u, dim=1) * mixed_target[batch_size:], dim=1))
            w = self.lambda_u_mixed * linear_rampup(epoch, self.epochs)

            _, logits_r = self.model(inputs_r)
            Lu_mixed *= w
            Lr = -1 * torch.mean(torch.sum(F.log_softmax(logits_r, dim=1) * targets_r, dim=1))
            Lr *= self.lambda_r

            # Entropy minimization for unlabeled samples (strong augmented)
            outputs_u2, _ = self.model(inputs_u2)
            Lu = -1 * torch.mean(torch.sum(F.log_softmax(outputs_u2, dim=1) * targets_u, dim=1))
            Lu *= self.lambda_u * linear_rampup(epoch+batch_idx/self.val_iteration, self.epochs)

            # calculate proportion loss for unlabled data (ours)
            if "ours" in self.method:
                confidence_u1 = F.softmax(outputs_u1 / self.T_prop, dim=1)
                pred_u1_prop = confidence_u1.mean(dim=0)

                sample_props = torch.tensor(
                    self.rng.multivariate_hypergeometric(self.U_SAMPLES_PER_CLASS, nsample=batch_size * self.mu),
                    device=self.device,
                    dtype=torch.float32,
                )
                sample_props = sample_props / torch.sum(sample_props)

                # calculate proportion loss for unlabeled
                Lp = self.proportion_loss(pred_u1_prop, sample_props)
            else:
                Lp = torch.tensor(0.0, device=self.device)

            loss = Lx + Lu_mixed + Lr + Lu + self.lambda_p * Lp

            self.loggers["train_losses"].update(loss.item())
            self.loggers["train_losses_x"].update(Lx.item())
            self.loggers["train_losses_u"].update(Lu.item())
            self.loggers["train_losses_p"].update(Lp.item())

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
                    Lp=f"{self.loggers['train_losses_p'].avg:.3f}")

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