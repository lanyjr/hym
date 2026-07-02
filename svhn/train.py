import argparse
import importlib

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.utils.data import ConcatDataset

from tqdm.auto import tqdm
import wandb

from resnet_3_96 import ResNet3_96
from opt import *


def import_obj(path: str):
    if ":" in path:
        mod, name = path.split(":")
    else:
        mod, name = path.rsplit(".", 1)
    m = importlib.import_module(mod)
    return getattr(m, name)


def build_optimizer(opt_name, params, lr, momentum, weight_decay=0.0, beta=None):
    opt_name = opt_name.lower()

    if "muon" in opt_name:
        params = list(params)
        muon_params = []
        adam_params = []
        for p in params:
            if p.ndim >= 2:
                muon_params.append(p)
            else:
                adam_params.append(p)

        param_groups = []
        if adam_params:
            param_groups.append(dict(
                params=adam_params,
                # lr=1e-3,
                lr=lr,
                betas=(0.9, 0.999),
                eps=1e-10,
                use_muon=False,
            ))
        if muon_params:
            param_groups.append(dict(
                params=muon_params,
                lr=lr,
                momentum=momentum,
                use_muon=True,
            ))

    if opt_name == "sgd":
        return SGD(params, lr=lr, momentum=momentum, nesterov=False, weight_decay=weight_decay)
    elif opt_name == "sf_sgd1":
        return SF_SGD1(params, lr=lr, momentum=momentum, nesterov=False, weight_decay=weight_decay, beta=beta)
    elif opt_name == "sf_sgd2":
        return SF_SGD2(params, lr=lr, momentum=momentum, nesterov=False, weight_decay=weight_decay, beta=beta)
    elif opt_name == "dm_sgd":
        return DM_SGD(params, lr=lr, momentum=momentum, nesterov=False, weight_decay=weight_decay, beta=beta)
    elif opt_name == "dm_sgd2":
        return DM_SGD2(params, lr=lr, momentum=momentum, nesterov=False, weight_decay=weight_decay, beta=beta)
    elif opt_name == "dm_sgd3": 
        return DM_SGD3(params, lr=lr, momentum=momentum, nesterov=False, weight_decay=weight_decay, beta=beta)
    elif opt_name == 'muon':
        return SingleDeviceMuonWithAuxAdam(param_groups)
    elif opt_name == 'dm_muon':
        return DM_Muon(param_groups, beta=beta)
    raise ValueError(f"Unknown optimizer: {opt_name}")


@torch.no_grad()
def evaluate(model, loader, device, eval_on_x=False, opt=None):
    if eval_on_x and opt is not None:
        opt.exchange("x")

    model.eval()
    loss_fn = nn.CrossEntropyLoss()
    total_loss, total_correct, total_n = 0.0, 0, 0

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        loss = loss_fn(logits, y)

        bs = x.size(0)
        total_loss += loss.item() * bs
        total_correct += (logits.argmax(dim=1) == y).sum().item()
        total_n += bs

    if eval_on_x and opt is not None:
        opt.exchange("y")

    return total_loss / total_n, total_correct / total_n


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--lr", type=float, required=True)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--beta", type=float, default=0.9)
    p.add_argument("--wd", type=float, default=0.0)
    p.add_argument("--epochs", type=int, required=True)
    p.add_argument("--batch_size", type=int, required=True)
    p.add_argument("--opt", type=str, default="sgd")
    p.add_argument("--device", type=str, default="cuda:0", help='e.g. "cuda:0", "cuda:1", "cpu"')
    p.add_argument("--sched", type=str, default="none", choices=["multistep", "cosine", "none"])
    p.add_argument("--eval", type=str, default="False", choices=["False", "True"])
    p.add_argument("--wandb_project", type=str, default="double momentum")
    args = p.parse_args()

    eval_on_x = (args.eval == "True")
    device = torch.device(args.device)

    wandb.init(
        project=args.wandb_project,
        group="svhn-resnet3-96",
        name=args.opt,
        config=vars(args),
    )

    # SVHN 常用 normalize（使用训练集统计量的常见近似）
    mean = (0.4377, 0.4438, 0.4728)
    std = (0.1980, 0.2010, 0.1970)

    train_tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    test_tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    # train_ds = datasets.SVHN(root="./data", split="train", download=True, transform=train_tfm)
    train_ds = ConcatDataset([
    datasets.SVHN(root="./data", split="train", download=True, transform=train_tfm),
    datasets.SVHN(root="./data", split="extra", download=True, transform=train_tfm),
    ])
    test_ds = datasets.SVHN(root="./data", split="test", download=True, transform=test_tfm)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=4, pin_memory=True)

    model = ResNet3_96(num_classes=10).to(device)
    optimizer = build_optimizer(args.opt, model.parameters(), args.lr, args.momentum, args.wd, args.beta)

    if args.sched == "multistep":
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20, 40], gamma=0.1)
    elif args.sched == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    else:
        scheduler = None

    loss_fn = nn.CrossEntropyLoss()

    global_step = 0
    for epoch in range(args.epochs):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}", leave=False)

        running_loss = 0.0
        running_correct = 0
        running_n = 0

        for x, y in pbar:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            optimizer.step()

            bs = x.size(0)
            running_loss += loss.item() * bs
            running_correct += (logits.argmax(dim=1) == y).sum().item()
            running_n += bs

            pbar.set_postfix({
                "loss": f"{(running_loss/running_n):.4f}",
                "acc": f"{(running_correct/running_n):.4f}",
                "lr": f"{optimizer.param_groups[0]['lr']:g}",
            })

            wandb.log(
                {"train/loss": loss.item(), "train/lr": optimizer.param_groups[0]["lr"], "global_step": global_step},
                step=global_step,
            )
            global_step += 1

        train_loss_avg = running_loss / running_n
        train_acc_avg = running_correct / running_n
        eval_loss, eval_acc = evaluate(model, test_loader, device, eval_on_x, optimizer)

        wandb.log(
            {
                "epoch": epoch,
                "train/loss_avg": train_loss_avg,
                "train/acc_avg": train_acc_avg,
                "eval/loss": eval_loss,
                "eval/acc": eval_acc,
            },
            step=global_step,
        )
        print(f"Epoch {epoch+1}/{args.epochs} | eval_loss={eval_loss:.4f} eval_acc={eval_acc:.4f}")

        if scheduler is not None:
            scheduler.step()

    wandb.finish()


if __name__ == "__main__":
    main()