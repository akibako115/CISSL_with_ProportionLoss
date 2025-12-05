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

from utils import AverageMeter, accuracy
from utils.remixmatch_ import get_args, SemiLoss, linear_rampup, interleave
from utils.crest_ import crest_select, PseudoLabeledFromU
from models.ema import ModelEMA

logger = logging.getLogger(__name__)

# Global variables
best_val_acc = 0
best_test_acc = 0
rng_seed = 122807528840384100672342137672332424406

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

    # Set rng
    rng = np.random.default_rng(rng_seed)

    # Set the dataset and model types
    if args.dataset == "cifar10":
        import dataset.remix_cifar10 as dataset

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
        import dataset.remix_svhn as dataset

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
        import dataset.remix_cifar100 as dataset

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

    # Set initial values for empirical distribution of unlabeled data
    emp_distb_u = torch.ones(args.num_class) / args.num_class

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
    U_SAMPLES_PROPORTION = torch.tensor(U_SAMPLES_PER_CLASS / np.sum(U_SAMPLES_PER_CLASS)).to(args.device, dtype=torch.float32)
    
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
    current_l_dataset = train_labeled_set  # 最初は元のラベル付き
    labeled_trainloader = data.DataLoader(
        current_l_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=min(16, os.cpu_count() or 8),
        drop_last=True,
        pin_memory=True,
        persistent_workers=True,
    )

    remaining_u_indices = list(range(len(train_unlabeled_set)))
    current_u_dataset = data.Subset(train_unlabeled_set, remaining_u_indices)

    # 以降の unlabeled_trainloader は current_u_dataset を使う
    unlabeled_trainloader = data.DataLoader(
        current_u_dataset,
        batch_size=args.batch_size * args.mu,
        shuffle=True,
        num_workers=min(16, os.cpu_count() or 8),
        drop_last=True,
        pin_memory=True,
        persistent_workers=True,
    )
    val_loader = data.DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False, num_workers=min(16, os.cpu_count() or 8),
        pin_memory=True,
        persistent_workers=True,
    )
    test_loader = data.DataLoader(
        test_set, batch_size=args.batch_size, shuffle=False, num_workers=min(16, os.cpu_count() or 8),
        pin_memory=True,
        persistent_workers=True,
    )

    # Create models
    def create_model(args):
        if args.arch == "wideresnet":
            import models.wideresnetwithrot as models

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
    
    model = create_model(args).to(args.device)
    model.to(memory_format=torch.channels_last)
    ema_model = ModelEMA(args, model, args.ema_decay)

    # Create optimizer and scheduler
    no_decay = ["bias", "bn"]
    grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": args.wdecay,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]

    optimizer = optim.SGD(
        grouped_parameters, lr=args.lr, momentum=0.9, nesterov=args.nesterov
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, args.warmup, args.epochs * args.val_iteration
    )

    # Resume from checkpoint if specified
    args.start_epoch = 0
    if args.resume:
        logger.info("=> Resuming from checkpoint...")
        assert os.path.isfile(args.resume), "Error: no checkpoint directory found!"
        args.out = os.path.dirname(args.resume)
        checkpoint = torch.load(args.resume)
        best_test_acc = checkpoint["best_test_acc"]
        best_val_acc = checkpoint["best_val_acc"]
        args.start_epoch = checkpoint["epoch"]
        model.load_state_dict(checkpoint["state_dict"])
        ema_model.ema.load_state_dict(checkpoint["ema_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])

    logger.info("***** Running training *****")
    logger.info(
        f"  Task = {args.dataset}. Num Max = {args.num_max}. Imb Ratio = {args.imb_ratio}. Label Ratio = {args.label_ratio}. Seed = {args.manualSeed}."
    )
    logger.info(f"  Num Epochs = {args.epochs}")
    logger.info(f"  Num Evaluation step = {args.val_iteration}")
    logger.info(f"  Total train batch size = {args.batch_size}")

    G = max(1, min(6, sum(1 for e in range(args.start_epoch, args.epochs)
                if ((e + 1) % args.crest_period == 0) and (e + 1 < args.epochs))))
    gen = 0

    for epoch in range(args.start_epoch, args.epochs):
        print("\nEpoch: [%d | %d] LR: %f" % (epoch + 1, args.epochs, state["lr"]))

        # Training part
        (
            train_loss,
            train_loss_x_mixed,
            train_loss_u_mixed,
            train_loss_u,
            train_loss_rot,
            emp_distb_u
        ) = train(
            args,
            labeled_trainloader,
            unlabeled_trainloader,
            model,
            ema_model,
            optimizer,
            scheduler,
            epoch,
            U_SAMPLES_PROPORTION,
            U_SAMPLES_PER_CLASS,
            emp_distb_u,
        )

        test_loss, test_acc, testclassacc = validate(
            args, test_loader, ema_model.ema, epoch, mode="test"
        )
        val_loss, val_acc, _ = validate(
            args, val_loader, ema_model.ema, epoch, mode="val"
        )

        is_best = val_acc > best_val_acc
        best_val_acc = max(val_acc, best_val_acc)

        if is_best:
            best_test_acc = test_acc

        if ((epoch + 1) % 10 == 0 and (epoch + 1) <= 100) or ((epoch + 1) % 100 == 0):
            # Save a checkpoint every 10 epochs
            save_checkpoint(
                {
                "epoch": epoch + 1,
                "state_dict": model.state_dict(),
                "ema_state_dict": ema_model.ema.state_dict(),
                "acc": test_acc,
                "best_test_acc": best_test_acc,
                "best_val_acc": best_val_acc,
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                },
                False,
                args.out,
                filename=f"checkpoint_epoch_{epoch+1}.pth.tar"
            )

        # Save models
        save_checkpoint(
            {
                "epoch": epoch + 1,
                "state_dict": model.state_dict(),
                "ema_state_dict": ema_model.ema.state_dict(),
                "acc": test_acc,
                "best_test_acc": best_test_acc,
                "best_val_acc": best_val_acc,
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
            },
            is_best,
            args.out,
        )

        if args.crest and ((epoch + 1) % args.crest_period == 0) and (gen < 6):
            gen += 1
            t_cur = (1 - gen / float(G)) * 1.0 + (gen / float(G)) * args.crest_t_min
            logger.info("=> Running CReST selection ...")
            chosen_idx, chosen_lbl, rates, cand_counts = crest_select(
                args, ema_model.ema, current_u_dataset, current_l_dataset, args.num_class, t_cur
            )
            logger.info(f"CReST candidates per class: {cand_counts}")
            logger.info(f"CReST rates: {[round(r,3) for r in rates]}")
            logger.info(f"CReST selected: {len(chosen_idx)}")

            if len(chosen_idx) > 0:
                crest_pl_ds = PseudoLabeledFromU(train_unlabeled_set, chosen_idx, chosen_lbl)
                current_l_dataset = data.ConcatDataset([current_l_dataset, crest_pl_ds])
                labeled_trainloader = data.DataLoader(
                    current_l_dataset,
                    batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True
                )
                if args.crest_remove:
                    chosen_set = set(chosen_idx)
                    remaining_u_indices = [i for i in remaining_u_indices if i not in chosen_set]
                    current_u_dataset = data.Subset(train_unlabeled_set, remaining_u_indices)
                    unlabeled_trainloader = data.DataLoader(
                        current_u_dataset,
                        batch_size=args.batch_size * args.mu,
                        shuffle=True, num_workers=4, drop_last=True
                    )

    args.writer.close()


def train(
    args,
    labeled_trainloader,
    unlabeled_trainloader,
    model,
    ema_model,
    optimizer,
    scheduler,
    epoch,
    U_SAMPLES_PROPORTION,
    U_SAMPLES_PER_CLASS,
    emp_distb_u,
):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    losses_x = AverageMeter()
    losses_u_mixed = AverageMeter()
    losses_u = AverageMeter()
    losses_r = AverageMeter()
    end = time.time()

    model.train()
    labeled_train_iter = iter(labeled_trainloader)
    unlabeled_train_iter = iter(unlabeled_trainloader)

    p_bar = tqdm(range(args.val_iteration))
    for batch_idx in range(args.val_iteration):
        try:
            inputs_x, targets_x, _ = next(labeled_train_iter)
        except:
            labeled_train_iter = iter(labeled_trainloader)
            inputs_x, targets_x, _ = next(labeled_train_iter)

        try:
            (inputs_u1, inputs_u2, inputs_u3), groundtruth_u, idx_u = next(
                unlabeled_train_iter
            )
        except:
            unlabeled_train_iter = iter(unlabeled_trainloader)
            (inputs_u1, inputs_u2, inputs_u3), groundtruth_u, idx_u = next(
                unlabeled_train_iter
            )

        # Measure data loading time
        data_time.update(time.time() - end)
        batch_size = inputs_x.size(0)

        inputs_x = inputs_x.to(args.device, non_blocking=True)
        targets_x = torch.zeros(batch_size, args.num_class).scatter_(1, targets_x.view(-1,1), 1).to(args.device, non_blocking=True)
        inputs_u1, inputs_u2, inputs_u3  = inputs_u1.to(args.device, non_blocking=True), inputs_u2.to(args.device, non_blocking=True), inputs_u3.to(args.device, non_blocking=True)

        # Rotate images
        temp = []
        targets_r = torch.randint(0, 4, (inputs_u2.size(0),)).long()
        for i in range(inputs_u2.size(0)):
            inputs_rot = torch.rot90(inputs_u2[i], targets_r[i], [1, 2]).reshape(1, 3, 32, 32)
            temp.append(inputs_rot)
        inputs_r = torch.cat(temp, 0)
        targets_r = torch.zeros(batch_size * args.mu, 4).scatter_(1, targets_r.view(-1, 1), 1)
        inputs_r, targets_r = inputs_r.to(args.device), targets_r.to(args.device, non_blocking=True)

        # Generate the pseudo labels
        with torch.no_grad():
            outputs_u1, _ = model(inputs_u1)
            p = torch.softmax(outputs_u1, dim=1)

            # Tracking the empirical distribution on the unlabeled samples (ReMixMatch)
            real_batch_idx = batch_idx + epoch * args.val_iteration
            if real_batch_idx == 0:
                emp_distb_u = p.mean(0, keepdim=True)
            elif real_batch_idx // 128 == 0:
                emp_distb_u = torch.cat([emp_distb_u, p.mean(0, keepdim=True)], 0)
            else:
                emp_distb_u = emp_distb_u[-127:]
                emp_distb_u = torch.cat([emp_distb_u, p.mean(0, keepdim=True)], 0)

            pa = p * (U_SAMPLES_PROPORTION + 1e-6) / (emp_distb_u.mean(0).to(args.device) + 1e-6)
            p = pa / pa.sum(dim=1, keepdim=True)

            # Temperature scaling
            pt = p ** (1 / args.T)
            targets_u = (pt / pt.sum(dim=1, keepdim=True))

            # Update the saved predictions with current one
            p = targets_u

        # Mixup
        all_inputs = torch.cat([inputs_x, inputs_u1, inputs_u2, inputs_u3], dim=0)
        all_targets = torch.cat([targets_x, targets_u, targets_u, targets_u], dim=0)

        l = np.random.beta(0.75, 0.75)
        l = max(l, 1-l)
        idx = torch.randperm(all_inputs.size(0), device=args.device)

        input_a, input_b = all_inputs, all_inputs[idx]
        target_a, target_b = all_targets, all_targets[idx]

        mixed_input = l * input_a + (1 - l) * input_b
        mixed_target = l * target_a + (1 - l) * target_b

        # interleave labeled and unlabed samples between batches to get correct batchnorm calculation
        mixed_input = list(torch.split(mixed_input, batch_size))
        mixed_input = interleave(mixed_input, batch_size)

        logits = [model(mixed_input[0])[0]]
        for input in mixed_input[1:]:
            logits.append(model(input)[0])

        outputs_u2, _ = model(inputs_u2)
        _, logits_r = model(inputs_r)

        # put interleaved samples back
        logits = interleave(logits, batch_size)
        logits_x = logits[0]
        logits_u = torch.cat(logits[1:], dim=0)

        criterion = SemiLoss()
        Lx, Lu_mixed, w = criterion(args, logits_x, mixed_target[:batch_size], logits_u, mixed_target[batch_size:], epoch+batch_idx/args.val_iteration)
        Lu_mixed *= w
        Lr = -1 * torch.mean(torch.sum(F.log_softmax(logits_r, dim=1) * targets_r, dim=1))
        Lr *= args.lambda_r

        # Entropy minimization for unlabeled samples (strong augmented)
        Lu = -1 * torch.mean(torch.sum(F.log_softmax(outputs_u2, dim=1) * targets_u, dim=1))
        Lu *= args.lambda_u * linear_rampup(epoch+batch_idx/args.val_iteration, args.epochs)

        loss = Lx + Lu_mixed + Lr + Lu

        losses.update(loss.item())
        losses_x.update(Lx.item())
        losses_u_mixed.update(Lu_mixed.item())
        losses_u.update(Lu.item())
        losses_r.update(Lr.item())

        # Backward pass
        loss.backward()
        optimizer.step()
        scheduler.step()
        ema_model.update(model)
        model.zero_grad()

        batch_time.update(time.time() - end)
        end = time.time()

        p_bar.set_description(
            "Train Epoch: {epoch}/{epochs:4}. Iter: {batch:4}/{iter:4}. LR: {lr:.4f}. Data: {data:.3f}s. Batch: {bt:.3f}s. Loss: {loss:.4f}. Loss_x: {loss_x_mixed:.4f}. Loss_u_mixed: {loss_u_mixed:.4f}. Loss_u: {loss_u:.4f}. Loss_rot: {loss_rot:.4f}.".format(
                epoch=epoch + 1,
                epochs=args.epochs,
                batch=batch_idx + 1,
                iter=args.val_iteration,
                lr=args.lr,
                data=data_time.avg,
                bt=batch_time.avg,
                loss=losses.avg,
                loss_x_mixed=losses_x.avg,
                loss_u_mixed=losses_u_mixed.avg,
                loss_u=losses_u.avg,
                loss_rot=losses_r.avg,
            )
        )
        p_bar.update()

    p_bar.close()

    args.writer.add_scalar("train/1.train_loss", losses.avg, epoch)
    args.writer.add_scalar("train/2.train_loss_x_mixed", losses_x.avg, epoch)
    args.writer.add_scalar("train/3.train_loss_u_mixed", losses_u_mixed.avg, epoch)
    args.writer.add_scalar("train/4.train_loss_u", losses_u.avg, epoch)
    args.writer.add_scalar("train/5.train_loss_rot", losses_r.avg, epoch)

    return (
        losses.avg,
        losses_x.avg,
        losses_u_mixed.avg,
        losses_u.avg,
        losses_r.avg,
        emp_distb_u
    )


def validate(args, valloader, model, epoch, mode):

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    model.eval()

    accperclass = np.zeros((args.num_class))
    target_list = []
    prediction_list = []
    prediction_perclass = np.zeros(args.num_class)

    end = time.time()

    p_bar = tqdm(valloader)

    with torch.no_grad():
        for batch_idx, (inputs, targets, _) in enumerate(valloader):
            # measure data loading time
            data_time.update(time.time() - end)

            # compute loss
            inputs, targets = inputs.to(args.device), targets.to(args.device, non_blocking=True)
            outputs, _ = model(inputs)
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

            target_list.extend(targets.cpu().numpy().tolist())
            prediction_list.extend(preds.cpu().numpy().tolist())

            prediction_perclass += np.bincount(preds.cpu().numpy(), minlength=args.num_class)

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

    precision_perclass = precision_score(
        target_list, prediction_list, average=None, zero_division=0
    )
    recall_perclass = recall_score(
        target_list, prediction_list, average=None, zero_division=0
    )

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
        args.writer.add_scalars(
            "test/4.test_precision_perclass",
            {str(i): precision_perclass[i] for i in range(args.num_class)},
            epoch
        )
        args.writer.add_scalars(
            "test/5.test_recall_perclass",
            {str(i): recall_perclass[i] for i in range(args.num_class)},
            epoch
        )
        args.writer.add_scalars(
            "test/6.test_prediction_perclass",
            {str(i): prediction_perclass[i] for i in range(args.num_class)},
            epoch
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


class WeightEMA(object):
    def __init__(self, model, ema_model, alpha=0.999, args=None):
        self.model = model
        self.ema_model = ema_model
        self.alpha = alpha
        self.params = list(model.state_dict().values())
        self.ema_params = list(ema_model.state_dict().values())
        self.wd = 0.02 * args.lr

        for param, ema_param in zip(self.params, self.ema_params):
            param.data.copy_(ema_param.data)

    def step(self):
        one_minus_alpha = 1.0 - self.alpha
        for param, ema_param in zip(self.params, self.ema_params):
            ema_param = ema_param.float()
            param = param.float()
            ema_param.mul_(self.alpha)
            ema_param.add_(param * one_minus_alpha)
            # customized weight decay
            param.mul_(1 - self.wd)


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

if __name__ == "__main__":
    main()
