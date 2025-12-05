import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from utils.misc import accuracy, AverageMeter
import numpy as np
from torch.utils.tensorboard import SummaryWriter

class Validator:
    def __init__(
        self, 
        val_loader: DataLoader,
        num_class: int,
        device: str,
        writer: SummaryWriter,
        save_name: str,
    ):
        self.val_loader = val_loader
        self.num_class = num_class
        self.device = device
        self.writer = writer
        self.save_name = save_name
        self.loggers = {
            "val_losses": AverageMeter(),
            "val_acc": AverageMeter(),
            "accperclass": np.zeros((num_class)),
        }
        self.best_epoch = 0
        self.best_val_acc = 0

        self.val_size = len(self.val_loader.dataset) // self.num_class

    def validate(self, ema_model, epoch):
        self._reset_metrics()

        ema_model.eval()
        accperclass = torch.zeros((self.num_class), device=self.device)

        with torch.no_grad():
            for batch_idx, (inputs_x, targets_x, _) in enumerate(self.val_loader):
                inputs_x, targets_x = inputs_x.to(self.device, non_blocking=True), targets_x.to(self.device, non_blocking=True)
                logits_x = ema_model(inputs_x)
                loss = F.cross_entropy(logits_x, targets_x, reduction="mean")
                prec1, prec5 = accuracy(logits_x, targets_x, topk=(1, 5))

                self.loggers["val_losses"].update(loss.item(), inputs_x.size(0))
                self.loggers["val_acc"].update(prec1.item(), inputs_x.size(0))

                preds = torch.argmax(F.softmax(logits_x, dim=-1), dim=-1)
                targetsonehot = torch.zeros(inputs_x.size(0), self.num_class, device=self.device).scatter_(
                    1, targets_x.view(-1, 1).long(), 1
                )
                outputs2onehot = torch.zeros(inputs_x.size(0), self.num_class, device=self.device).scatter_(
                    1, preds.view(-1, 1).long(), 1
                )
                accperclass += torch.sum(
                    targetsonehot * outputs2onehot, dim=0
                )
        
        self.loggers["accperclass"] = accperclass.cpu().numpy().astype(np.int64) / self.val_size

        is_best = self.loggers["val_acc"].avg > self.best_val_acc
        if is_best:
            self.best_val_acc = self.loggers["val_acc"].avg
            self.best_epoch = epoch
        
        self._record_metrics(epoch)

        print(f"{self.save_name} acc: {self.loggers['val_acc'].avg:.3f}, loss: {self.loggers['val_losses'].avg:.3f}")

        return is_best

    def _reset_metrics(self):
        for key in self.loggers.keys():
            if key != "accperclass":
                self.loggers[key].reset()
            else:
                self.loggers[key] = np.zeros((self.num_class))

    def _record_metrics(self, epoch: int):
        self.writer.add_scalar(f"{self.save_name}1/loss", self.loggers["val_losses"].avg, epoch)
        self.writer.add_scalar(f"{self.save_name}2/acc", self.loggers["val_acc"].avg, epoch)
        self.writer.add_scalars(f"{self.save_name}3/accperclass", {str(i): self.loggers["accperclass"][i] for i in range(self.num_class)}, epoch)