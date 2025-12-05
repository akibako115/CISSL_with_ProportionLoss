# This code is constructed based on Pytorch Implementation of DARP(https://github.com/bbuing9/DARP)
from __future__ import print_function

import argparse
import logging
import os
import shutil
import time
import random
import math
from tqdm import tqdm

import numpy as np
from sklearn.metrics import precision_score, recall_score

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
import torch.utils.data as data
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from utils import AverageMeter, accuracy, pseudo_statics
from utils.darp_ import estimate_pseudo, opt_solver
from utils.fixmatch_ import get_args
from models.ema import ModelEMA

logger = logging.getLogger(__name__)

# Global variables
best_val_acc = 0
best_test_acc = 0

# 最後に現れる nn.Linear を返す
def main():
    global best_test_acc
    global best_val_acc

    # Parse the arguments
    args, state = get_args()

    # Set the random seed manually for reproducibility
    set_seed(args.manualSeed)

    # Create output dir and set SummaryWriter
    if os.path.exists(args.out):
        shutil.rmtree(args.out)
    os.makedirs(args.out, exist_ok=True)
    args.writer = SummaryWriter(args.out)

    # Set the dataset and model types
    if args.dataset == "cifar10":
        import dataset.fix_cifar10 as dataset

        print(f"==> Preparing imbalanced CIFAR10")
        args.num_class = 10

        if args.arch == "wideresnet":
            args.model_depth = 28
            args.model_width = 2
            print("==> creating WRN-28-2 with abc")
        elif args.arch == "resnext":
            args.model_cardinality = 4
            args.model_depth = 28
            args.model_width = 4
            print("==> creating ResNeXt-28-4 with abc")

    elif args.dataset == "svhn":
        # todo: my_svhn
        import dataset.fix_svhn as dataset

        print(f"==> Preparing imbalanced SVHN")
        args.num_class = 10

        if args.arch == "wideresnet":
            args.model_depth = 28
            args.model_width = 2
            print("==> creating WRN-28-2 with abc")
        elif args.arch == "resnext":
            args.model_cardinality = 4
            args.model_depth = 28
            args.model_width = 4
            print("==> creating ResNeXt-28-4 with abc")

    elif args.dataset == "cifar100":
        # todo: my_cifar100
        import dataset.fix_cifar100 as dataset

        print(f"==> Preparing imbalanced CIFAR100")
        args.num_class = 100

        if args.arch == "wideresnet":
            args.model_depth = 28
            args.model_width = 8
            print("==> creating WRN-28-8 with abc")
        elif args.arch == "resnext":
            args.model_cardinality = 8
            args.model_depth = 29
            args.model_width = 64
            print("==> creating ResNeXt-29-8 with abc")

    # Use CUDA
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    args.device = torch.device("cuda", 0)

    # Set up logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(dict(args._get_kwargs()))

    # Prepare datasets
    args.num_max = args.num_max - int(500*args.label_ratio/100)  # 500 is for validation set per class
    
    N_SAMPLES_PER_CLASS = make_imb_data(
        args.num_max, 
        args.num_class, 
        args.imb_ratio, 
        args.imbalancetype
    )
    U_SAMPLES_PER_CLASS = (100 - args.label_ratio) / args.label_ratio * np.array(N_SAMPLES_PER_CLASS)
    U_SAMPLES_PER_CLASS = np.round(U_SAMPLES_PER_CLASS).astype(int)
    U_SAMPLES_PROPORTION = U_SAMPLES_PER_CLASS / np.sum(U_SAMPLES_PER_CLASS)
    
    if args.dataset == "cifar10":
        train_labeled_set, train_unlabeled_set, val_set, test_set = dataset.get_cifar10(
            "../data", N_SAMPLES_PER_CLASS, U_SAMPLES_PER_CLASS
        )
    elif args.dataset == "svhn":
        train_labeled_set, train_unlabeled_set, val_set, test_set = dataset.get_svhn(
            "../data", N_SAMPLES_PER_CLASS, U_SAMPLES_PER_CLASS
        )
    elif args.dataset == "cifar100":
        train_labeled_set, train_unlabeled_set, val_set, test_set = dataset.get_cifar100(
            "../data", N_SAMPLES_PER_CLASS, U_SAMPLES_PER_CLASS
        )

    # Prepare data loaders
    labeled_trainloader = data.DataLoader(
        train_labeled_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        drop_last=True,
    )
    unlabeled_trainloader = data.DataLoader(
        train_unlabeled_set,
        batch_size=args.batch_size * args.mu,
        shuffle=True,
        num_workers=4,
        drop_last=True,
    )
    val_loader = data.DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False, num_workers=4
    )
    test_loader = data.DataLoader(
        test_set, batch_size=args.batch_size, shuffle=False, num_workers=4
    )

    # Create models
    def create_model(args):
        if args.arch == "wideresnet":
            import models.wideresnet_cRT as models

            model = models.build_wideresnet(
                depth=args.model_depth,
                widen_factor=args.model_width,
                dropout=0,
                num_classes=args.num_class,
            )
        elif args.arch == "resnext":
            import models.resnext as models

            model = models.build_resnext(
                cardinality=args.model_cardinality,
                depth=args.model_depth,
                width=args.model_width,
                num_classes=args.num_class,
            )

        logger.info(
            "Total params: {:.2f}M".format(
                sum(p.numel() for p in model.parameters()) / 1e6
            )
        )
        return model
    
    model = create_model(args)
    model.to(args.device)
    ema_model = ModelEMA(args, model, args.ema_decay)

    checkpoint = torch.load(f'../results_fixmatch/my_fixmatch/cifar10@{args.imb_ratio}.{args.label_ratio}.lp{args.lambda_p}.T{args.temp}.r{args.seed}.mu{args.mu}/model_best.pth.tar', map_location=args.device)
    ema_model.ema.load_state_dict(checkpoint["ema_state_dict"])
    model = ema_model.ema
    model.cuda()

    model.freeze_backbone()
    model.reset_classifier(args.num_class)

    # 特徴抽出
    feats, labels = [], []
    with torch.no_grad():
        for x, y, _ in data.DataLoader(train_labeled_set, batch_size=256, shuffle=False, num_workers=4):
            feats.append(model.forward_feats(x.cuda()).cpu())
            labels.append(y)
    feats = torch.cat(feats); labels = torch.cat(labels)

    # バランスサンプラ
    counts = torch.bincount(labels, minlength=args.num_class).float().clamp_min(1)
    weights = (1.0 / counts)[labels]
    sampler = data.WeightedRandomSampler(weights=weights, num_samples=len(labels), replacement=True)
    crt_loader = data.DataLoader(data.TensorDataset(feats, labels), batch_size=256, sampler=sampler)

    # 線形層のみ学習
    opt = torch.optim.AdamW(model.fc.parameters(), lr=1e-3, weight_decay=1e-4)
    crit = nn.CrossEntropyLoss()

    for epoch in range(20):
        model.eval(); model.fc.train()
        for inputs, targets in crt_loader:
            inputs, targets = inputs.cuda(non_blocking=True), targets.cuda(non_blocking=True)
            logits = model(inputs)
            loss = crit(logits, targets)
            opt.zero_grad(); loss.backward(); opt.step()

            args.writer.add_scalar("train/1.train_loss", loss.item(), epoch)

            _, _, _ = validate(args, test_loader, model, epoch, "test")
    

def validate(args, valloader, model, epoch, mode):

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    model.eval(); model.fc.eval()

    accperclass = np.zeros((args.num_class))

    end = time.time()

    p_bar = tqdm(valloader)

    with torch.no_grad():
        for batch_idx, (inputs, targets, _) in enumerate(valloader):
            # measure data loading time
            data_time.update(time.time() - end)

            # compute loss
            inputs, targets = inputs.to(args.device), targets.to(args.device, non_blocking=True)
            outputs = model(inputs)
            preds = torch.argmax(F.softmax(outputs, dim=-1), dim=-1)
            loss = F.cross_entropy(outputs, targets)

            # compute acc per class
            targetsonehot = torch.zeros(inputs.size()[0], args.num_class).scatter_(
                1, targets.cpu().view(-1, 1).long(), 1
            )
            outputs2onehot = torch.zeros(inputs.size()[0], args.num_class).scatter_(
                1, preds.cpu().view(-1, 1).long(), 1
            )
            accperclass = accperclass + torch.sum(
                targetsonehot * outputs2onehot, dim=0
            ).cpu().detach().numpy().astype(np.int64)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(outputs, targets, topk=(1, 5))
            losses.update(loss.item(), inputs.size(0))
            top1.update(prec1.item(), inputs.size(0))
            top5.update(prec5.item(), inputs.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            p_bar.set_description(
                "Test Iter: {batch:4}/{iter:4}. Data: {data:.3f}s. Batch: {bt:.3f}s. Loss: {loss:.4f}. top1: {top1:.2f}. top5: {top5:.2f}. ".format(
                    batch=batch_idx + 1,
                    iter=len(valloader),
                    data=data_time.avg,
                    bt=batch_time.avg,
                    loss=losses.avg,
                    top1=top1.avg,
                    top5=top5.avg,
                )
            )
        p_bar.close()

    if args.dataset == "cifar10":
        accperclass = accperclass / 1000
    elif args.dataset == "svhn":
        accperclass = accperclass / 1500
    elif args.dataset == "cifar100":
        accperclass = accperclass / 100

    args.writer.add_scalar(f"{mode}/1.{mode}_acc", top1.avg, epoch)
    args.writer.add_scalar(f"{mode}/2.{mode}_loss", losses.avg, epoch)

    logger.info(f" {mode} top-1 acc: {top1.avg:.2f}")
    logger.info(f" {mode} top-5 acc: {top5.avg:.2f}")

    if mode == "test":
        args.writer.add_scalars(
            "test/3.test_accperclass",
            {str(i): accperclass[i] for i in range(args.num_class)},
            epoch,
        )

    return (losses.avg, top1.avg, accperclass)



# --- Utility functions ---
def save_checkpoint(state, is_best, checkpoint, filename="checkpoint.pth.tar"):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, "model_best.pth.tar"))


def set_seed(seed):
    # np.random.seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_imb_data(max_num, class_num, gamma, imb):
    if imb == "long":
        mu = np.power(1 / gamma, 1 / (class_num - 1))
        class_num_list = []
        for i in range(class_num):
            if i == (class_num - 1):
                class_num_list.append(int(max_num / gamma))
            else:
                class_num_list.append(int(max_num * np.power(mu, i)))
        print(class_num_list)
    if imb == "step":
        class_num_list = []
        for i in range(class_num):
            if i < int((class_num) / 2):
                class_num_list.append(int(max_num))
            else:
                class_num_list.append(int(max_num / gamma))
        print(class_num_list)
    return list(class_num_list)