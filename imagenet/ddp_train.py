import argparse
import importlib
import os
import time

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
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
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--opt", type=str, default="sgd")
    p.add_argument("--device", type=str, default="cuda:0", help='e.g. "cuda:0", "cuda:1", "cpu"')  # 单卡时仍可使用
    p.add_argument("--sched", type=str, default="none", choices=["multistep", "cosine", "none"])
    p.add_argument("--wandb_project", type=str, default="double momentum")
    p.add_argument("--wandb_mode", type=str, default='online')

    # ImageNet paths
    p.add_argument("--train_dir", type=str, default='./imagenet/train')
    p.add_argument("--val_dir", type=str, default='./imagenet/val')

    # precision
    p.add_argument("--precision", type=str, default="bf16", choices=["fp32", "fp16", "bf16"])

    # ==================== DDP 参数 ====================
    p.add_argument("--local_rank", type=int, default=-1, help="torchrun 自动传入")
    p.add_argument("--local-rank", type=int, default=-1)
    p.add_argument("--dist_backend", type=str, default="nccl")
    p.add_argument("--init_method", type=str, default="env://")

    args = p.parse_args()

    # ===================== DDP 初始化 =====================
    if "LOCAL_RANK" in os.environ:
        args.local_rank = int(os.environ["LOCAL_RANK"])

    if args.local_rank == -1:   # 单卡模式
        distributed = False
        args.rank = 0
        args.world_size = 1
    else:
        distributed = True
        # 初始化进程组
        dist.init_process_group(
            backend=args.dist_backend,
            init_method=args.init_method,
            world_size=int(os.environ.get("WORLD_SIZE", 1)),
            rank=int(os.environ.get("RANK", 0)),
        )
        args.rank = dist.get_rank()
        args.world_size = dist.get_world_size()

    torch.cuda.set_device(args.local_rank)
    device = torch.device(f"cuda:{args.local_rank}")

    print(f"[Rank {args.rank}/{args.world_size}] Using device: {device}", flush=True)

    # ===================== AMP 配置 =====================
    if args.precision == "bf16":
        amp_dtype = torch.bfloat16
        use_amp = True
        use_scaler = False
    elif args.precision == "fp16":
        amp_dtype = torch.float16
        use_amp = True
        use_scaler = True
    else:
        amp_dtype = None
        use_amp = False
        use_scaler = False

    scaler = torch.cuda.amp.GradScaler(enabled=use_scaler)

    # ===================== Wandb 只在 rank 0 初始化 =====================
    if args.rank == 0:
        wandb.init(
            project=args.wandb_project,
            group="imagenet-resnet50",
            name=f"{args.opt}-{args.precision}-ddp{args.world_size}",
            config=vars(args),
            mode=args.wandb_mode,
        )

    # ===================== 数据预处理 =====================
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

    # ===================== DistributedSampler =====================
    if distributed:
        train_sampler = DistributedSampler(train_ds, shuffle=True)
        val_sampler = DistributedSampler(val_ds, shuffle=False)
    else:
        train_sampler = None
        val_sampler = None

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=16,
        pin_memory=True,
        persistent_workers=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=max(args.batch_size, 256),
        sampler=val_sampler,
        num_workers=16,
        pin_memory=True,
        persistent_workers=True,
    )

    # ===================== 模型 =====================
    model = models.resnet50(num_classes=1000).to(device)
    # model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    if distributed:
        model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank)

    model = torch.compile(model)

    optimizer = build_optimizer(args.opt, model.parameters(), args.lr, args.momentum, args.wd, args.beta)

    # ===================== Scheduler =====================
    total_epochs = args.epochs
    warmup_epochs = int(0.05 * total_epochs)

    if args.sched == "cosine":
        warmup_scheduler = LinearLR(optimizer, start_factor=0.01, total_iters=warmup_epochs)
        main_scheduler = CosineAnnealingLR(optimizer, T_max=total_epochs - warmup_epochs, eta_min=0.0)
    elif args.sched == "multistep":
        warmup_scheduler = LinearLR(optimizer, start_factor=0.01, total_iters=warmup_epochs)
        main_scheduler = MultiStepLR(optimizer, milestones=[int(total_epochs*0.5), int(total_epochs*0.75)], gamma=0.1)
    else:
        warmup_scheduler = main_scheduler = None

    if args.sched != 'none':
        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, main_scheduler],
            milestones=[warmup_epochs]
        )
    else:
        scheduler = None

    loss_fn = nn.CrossEntropyLoss()
    global_step = 0

    for epoch in range(args.epochs):
        if distributed:
            train_sampler.set_epoch(epoch)  # 重要：保证每个 epoch shuffle 不同

        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}", 
                   leave=False, disable=(args.rank != 0))

        running_loss = 0.0
        running_correct1 = 0
        running_n = 0
        t0 = time.time()

        for x, y in pbar:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
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
            if args.rank == 0:
                pbar.set_postfix({
                    "loss": f"{(running_loss/running_n):.4f}",
                    "acc1": f"{(running_correct1/running_n):.4f}",
                    "lr": f"{cur_lr:g}",
                })

            if global_step % 10 == 0 and args.rank == 0:
                wandb.log({
                    "train/loss": loss.item(),
                    "train/lr": cur_lr,
                    "global_step": global_step,
                }, step=global_step)

            global_step += 1

        train_loss_avg = running_loss / running_n
        train_acc1_avg = running_correct1 / running_n

        # ===================== Validation (仅 rank 0) =====================
        if args.rank == 0:
            val_model = model.module if distributed else model
            val_loss, val_acc1, val_acc5 = evaluate(
                val_model, val_loader, device, amp_dtype if use_amp else torch.float32
            )
            epoch_time = time.time() - t0

            wandb.log({
                "epoch": epoch,
                "train/loss_avg": train_loss_avg,
                "train/acc1_avg": train_acc1_avg,
                "val/loss": val_loss,
                "val/acc1": val_acc1,
                "val/acc5": val_acc5,
                "time/epoch_sec": epoch_time,
            }, step=global_step)

            print(
                f"Epoch {epoch+1}/{args.epochs} | "
                f"val_loss={val_loss:.4f} val_acc1={val_acc1:.4f} val_acc5={val_acc5:.4f} "
                f"epoch_time={epoch_time:.2f}s"
            )

        if distributed:
            dist.barrier()  # 同步所有卡

        if scheduler is not None:
            scheduler.step()

    if args.rank == 0:
        wandb.finish()

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()