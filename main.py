import json
import os
from utils.misc import set_seed, make_imb_data
from utils.train_tools import get_cosine_schedule_with_warmup
import shutil
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from torch.utils.data import DataLoader
import torch.optim as optim
from models.ema import ModelEMA
import argparse

rng_seed = 122807528840384100672342137672332424406

def get_datasets(dataset_config, train_config):
    N_SAMPLES_PER_CLASS = make_imb_data(
        dataset_config["num_max"],
        dataset_config["num_class"], 
        dataset_config["imb_ratio"], 
        dataset_config["imbalancetype"]
    )
    U_SAMPLES_PER_CLASS = (100 - dataset_config["label_ratio"]) / dataset_config["label_ratio"] * np.array(N_SAMPLES_PER_CLASS)
    U_SAMPLES_PER_CLASS = np.round(U_SAMPLES_PER_CLASS).astype(int)

    if dataset_config["name"] == "cifar10":
        if train_config["base"].lower() == "fixmatch":
            from dataset.fix_cifar10 import get_cifar10
            labeled_set, unlabeled_set, val_set, test_set = get_cifar10(dataset_config["path"], N_SAMPLES_PER_CLASS, U_SAMPLES_PER_CLASS)
        elif train_config["base"].lower() == "remixmatch":
            from dataset.remix_cifar10 import get_cifar10
            labeled_set, unlabeled_set, val_set, test_set = get_cifar10(dataset_config["path"], N_SAMPLES_PER_CLASS, U_SAMPLES_PER_CLASS)

    return labeled_set, unlabeled_set, val_set, test_set, U_SAMPLES_PER_CLASS

def get_model(model_config, train_config):
    if model_config["name"] == "wideresnet":
        if train_config["base"].lower() == "fixmatch":
            import models.wideresnet as models
        elif train_config["base"].lower() == "remixmatch":
            import models.wideresnetwithrot as models

        model = models.build_wideresnet(
            depth=model_config["depth"],
            widen_factor=model_config["width"],
            dropout=0,
            num_classes=model_config["num_class"],
        )

    no_decay = ["bias", "bn"]
    grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": model_config["wdecay"],
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

    print(
        "Total params: {:.2f}M".format(
            sum(p.numel() for p in model.parameters()) / 1e6
        )
    )

    return model, grouped_parameters

def get_optimizer(optimizer_config, grouped_parameters):
    if optimizer_config["name"] == "SGD":
        optimizer = optim.SGD(grouped_parameters, lr=optimizer_config["lr"], momentum=optimizer_config["momentum"], nesterov=optimizer_config["nesterov"])
    
    return optimizer

def get_scheduler(scheduler_config, optimizer):
    if scheduler_config["name"].lower() == "cosine":
        scheduler = get_cosine_schedule_with_warmup(optimizer, scheduler_config["num_warmup_steps"], scheduler_config["num_training_steps"], scheduler_config["num_cycles"])
    return scheduler

def get_configparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    return parser.parse_args()


def main():
    args = get_configparser()
    with open(args.config, "r", encoding="utf-8") as f:
        config_dict = json.load(f)

    # split config into different parts
    train_config = config_dict["training"]
    model_config = config_dict["model"]
    dataset_config = config_dict["dataset"]
    optimizer_config = config_dict["optimizer"]
    scheduler_config = config_dict["scheduler"]
    output_dir = config_dict["output_dir"]

    # set some default values
    num_max = 50 * dataset_config["label_ratio"]
    dataset_config["num_max"] = num_max - int(500*dataset_config["label_ratio"]/100)  # 500 is for validation set per class
    model_config["num_class"] = dataset_config["num_class"]
    train_config["num_class"] = dataset_config["num_class"]
    scheduler_config["num_training_steps"] = train_config["epochs"] * train_config["val_iteration"]

    # set seed and output path
    set_seed(train_config["manualSeed"])
    output_path = os.path.join(output_dir, f"{dataset_config['name']}", f"{train_config['base']}", f"l{dataset_config['label_ratio']}_i{dataset_config['imb_ratio']}_r{train_config['manualSeed']}", f"{train_config['method']}")
    
    if os.path.exists(output_path):
        shutil.rmtree(output_path)

    os.makedirs(output_path, exist_ok=True)
    writer = SummaryWriter(output_path)
    rng = np.random.default_rng(rng_seed)

    # save config
    with open(os.path.join(output_path, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config_dict, f, indent=4, ensure_ascii=False)

    # get datasets
    labeled_set, unlabeled_set, val_set, test_set, U_SAMPLES_PER_CLASS = get_datasets(dataset_config, train_config)

    # get loaders
    labeled_loader = DataLoader(labeled_set, batch_size=train_config["batch_size"], shuffle=True, num_workers=4, drop_last=True)
    unlabeled_loader = DataLoader(unlabeled_set, batch_size=train_config["batch_size"] * train_config["mu"], shuffle=True, num_workers=4, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=train_config["batch_size"], shuffle=False, num_workers=4)
    test_loader = DataLoader(test_set, batch_size=train_config["batch_size"], shuffle=False, num_workers=4)

    # get model and ema model
    model, grouped_parameters = get_model(model_config, train_config)
    model.to(train_config["device"])
    ema_model = ModelEMA(train_config["device"], model, train_config["ema_decay"])
    optimizer = get_optimizer(optimizer_config, grouped_parameters)
    scheduler = get_scheduler(scheduler_config, optimizer)

    # get trainer
    if "fixmatch" in train_config["base"].lower():
        from trainer.trainer_fixmatch import trainer_factory
        trainer = trainer_factory(model, ema_model, labeled_loader, unlabeled_loader, val_loader, test_loader, optimizer, scheduler, rng, U_SAMPLES_PER_CLASS, output_path, writer, train_config)
    elif "remixmatch" in train_config["base"].lower():
        from trainer.trainer_remixmatch import trainer_factory
        trainer = trainer_factory(model, ema_model, labeled_loader, unlabeled_loader, val_loader, test_loader, optimizer, scheduler, rng, U_SAMPLES_PER_CLASS, output_path, writer, train_config)

    # train
    trainer.train()

if __name__ == "__main__":
    main()