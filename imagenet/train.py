import argparse
import importlib
import os
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from tqdm.auto import tqdm
import wandb

from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, MultiStepLR, SequentialLR

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
    elif opt_name == "adamw":
        return AdamW(params, lr=lr, betas=(momentum, 0.999), weight_decay=weight_decay)
    elif opt_name == "dm_adamw":
        return DM_AdamW(params, lr=lr, betas=(momentum, 0.999), weight_decay=weight_decay, beta=beta)
    raise ValueError(f"Unknown optimizer: {opt_name}")


@torch.no_grad()
def evaluate(model, loader, device, amp_dtype):
    model.eval()
    loss_fn = nn.CrossEntropyLoss()

    total_loss = 0.0
    total_correct1 = 0
    total_correct5 = 0
    total_n = 0

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=(device.type == "cuda")):
            logits = model(x)
            loss = loss_fn(logits, y)

        bs = x.size(0)
        total_loss += loss.item() * bs
        total_n += bs

        _, pred = logits.topk(5, dim=1, largest=True, sorted=True)
        correct = pred.eq(y.view(-1, 1))
        total_correct1 += correct[:, :1].sum().item()
        total_correct5 += correct[:, :5].sum().item()

    return (
        total_loss / total_n,
        total_correct1 / total_n,
        total_correct5 / total_n,
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--lr", type=float, required=True)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--beta", type=float, default=0.9)
    p.add_argument("--wd", type=float, default=1e-4)
    p.add_argument("--epochs", type=int, default = 100)
    p.add_argument("--batch_size", type=int, default= 256)
    p.add_argument("--opt", type=str, default="sgd")
    p.add_argument("--device", type=str, default="cuda:0", help='e.g. "cuda:0", "cuda:1", "cpu"')
    p.add_argument("--sched", type=str, default="none", choices=["multistep", "cosine", "none"])
    p.add_argument("--wandb_project", type=str, default="double momentum")
    p.add_argument("--wandb_mode", type=str,default ='online')

    # ImageNet paths
    p.add_argument("--train_dir", type=str, default='./imagenet/train')
    p.add_argument("--val_dir", type=str, default='./imagenet/val')

    # precision
    p.add_argument("--precision", type=str, default="bf16", choices=["fp32", "fp16", "bf16"])

    args = p.parse_args()
    device = torch.device(args.device)

    # A100 推荐打开
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    if args.precision == "bf16":
        amp_dtype = torch.bfloat16
        use_amp = True
        use_scaler = False
    elif args.precision == "fp16":
        amp_dtype = torch.float16
        use_amp = True
        use_scaler = (device.type == "cuda")
    else:
        amp_dtype = None
        use_amp = False
        use_scaler = False

    scaler = torch.cuda.amp.GradScaler(enabled=use_scaler)

    wandb.init(
        project=args.wandb_project,
        group="imagenet-resnet50",
        name=f"{args.opt}-{args.precision}",
        config=vars(args),
        mode = args.wandb_mode,
    )

    # Standard ImageNet preprocessing
    train_tfm = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406),
                             std=(0.229, 0.224, 0.225)),
    ])
    val_tfm = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406),
                             std=(0.229, 0.224, 0.225)),
    ])

    train_ds = datasets.ImageFolder(args.train_dir, transform=train_tfm)
    val_ds = datasets.ImageFolder(args.val_dir, transform=val_tfm)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=16,
        pin_memory=True,
        persistent_workers=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=max(args.batch_size, 256),
        shuffle=False,
        num_workers=16,
        pin_memory=True,
        persistent_workers=True,
    )

    model = models.resnet50(num_classes=1000).to(device)
    model = torch.compile(model)
    optimizer = build_optimizer(args.opt, model.parameters(), args.lr, args.momentum, args.wd, args.beta)

    # if args.sched == "multistep":
    #     scheduler = torch.optim.lr_scheduler.MultiStepLR(
    #         optimizer,
    #         milestones=[int(args.epochs * 0.3), int(args.epochs * 0.6), int(args.epochs * 0.8)],
    #         gamma=0.1,
    #     )
    # elif args.sched == "cosine":
    #     scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    # else:
    #     scheduler = None

    total_epochs = args.epochs
    warmup_epochs = int(0.05 * total_epochs)

    if args.sched == "cosine":
        warmup_scheduler = LinearLR(
            optimizer, 
            start_factor=0.01, 
            total_iters=warmup_epochs
        )
        
        main_scheduler = CosineAnnealingLR(
            optimizer, 
            T_max=total_epochs - warmup_epochs,   
            eta_min=0.0                           
        )
    elif args.sched == "multistep":
        warmup_scheduler = LinearLR(optimizer, start_factor=0.01, total_iters=warmup_epochs)
        
        main_scheduler = MultiStepLR(
            optimizer, 
            milestones=[int(total_epochs*0.5), int(total_epochs*0.75)], 
            gamma=0.1
        )

    if args.sched != 'none':
        scheduler = SequentialLR(
            optimizer, 
            schedulers=[warmup_scheduler, main_scheduler], 
            milestones=[warmup_epochs]   # warmup 结束后切换到 main_scheduler
        )
    else:
        scheduler = None

    loss_fn = nn.CrossEntropyLoss()
    global_step = 0

    for epoch in range(args.epochs):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}", leave=False)

        running_loss = 0.0
        running_correct1 = 0
        running_n = 0
        t0 = time.time()

        for x, y in pbar:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(
                device_type="cuda",
                dtype=amp_dtype,
                enabled=(use_amp and device.type == "cuda")
            ):
                logits = model(x)
                loss = loss_fn(logits, y)

            if use_scaler:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            bs = x.size(0)
            running_loss += loss.item() * bs
            running_n += bs

            pred1 = logits.argmax(dim=1)
            running_correct1 += (pred1 == y).sum().item()

            cur_lr = optimizer.param_groups[0]["lr"]
            pbar.set_postfix({
                "loss": f"{(running_loss/running_n):.4f}",
                "acc1": f"{(running_correct1/running_n):.4f}",
                "lr": f"{cur_lr:g}",
            })

            if global_step % 10 == 0:
                wandb.log(
                    {
                        "train/loss": loss.item(),
                        "train/lr": cur_lr,
                        "global_step": global_step,
                    },
                    step=global_step,
                )
            global_step += 1

        train_loss_avg = running_loss / running_n
        train_acc1_avg = running_correct1 / running_n

        val_loss, val_acc1, val_acc5 = evaluate(model, val_loader, device, amp_dtype if use_amp else torch.float32)
        epoch_time = time.time() - t0

        wandb.log(
            {
                "epoch": epoch,
                "train/loss_avg": train_loss_avg,
                "train/acc1_avg": train_acc1_avg,
                "val/loss": val_loss,
                "val/acc1": val_acc1,
                "val/acc5": val_acc5,
                "time/epoch_sec": epoch_time,
            },
            step=global_step,
        )
        print(
            f"Epoch {epoch+1}/{args.epochs} | "
            f"val_loss={val_loss:.4f} val_acc1={val_acc1:.4f} val_acc5={val_acc5:.4f},epoch_time={epoch_time:.4f}"
        )

        if scheduler is not None:
            scheduler.step()

    wandb.finish()


if __name__ == "__main__":
    main()