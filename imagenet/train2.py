import argparse
import os
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm.auto import tqdm
import wandb
import timm
from timm.data import create_transform, Mixup

from opt import *



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


def interpolate_pos_embed_if_needed(model, checkpoint_model):
    if "pos_embed" not in checkpoint_model:
        return checkpoint_model

    pos_embed_checkpoint = checkpoint_model["pos_embed"]
    embedding_size = pos_embed_checkpoint.shape[-1]
    num_patches = model.patch_embed.num_patches
    num_extra_tokens = model.pos_embed.shape[-2] - num_patches

    orig_num_tokens = pos_embed_checkpoint.shape[-2]
    orig_size = int((orig_num_tokens - num_extra_tokens) ** 0.5)
    new_size = int(num_patches ** 0.5)

    if orig_size != new_size:
        print(f"Interpolating position embedding from {orig_size}x{orig_size} to {new_size}x{new_size}")
        extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
        pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
        pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
        pos_tokens = torch.nn.functional.interpolate(
            pos_tokens, size=(new_size, new_size), mode="bicubic", align_corners=False
        )
        pos_tokens = pos_tokens.permute(0, 2, 3, 1).reshape(-1, new_size * new_size, embedding_size)
        new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
        checkpoint_model["pos_embed"] = new_pos_embed

    return checkpoint_model


def load_mae_pretrained(model, ckpt_path):
    print(f"Loading MAE pretrained checkpoint from: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location="cpu")

    if isinstance(checkpoint, dict):
        if "model" in checkpoint:
            checkpoint_model = checkpoint["model"]
        elif "state_dict" in checkpoint:
            checkpoint_model = checkpoint["state_dict"]
        else:
            checkpoint_model = checkpoint
    else:
        checkpoint_model = checkpoint

    # remove common prefixes
    new_ckpt = {}
    for k, v in checkpoint_model.items():
        nk = k
        if nk.startswith("module."):
            nk = nk[len("module."):]
        if nk.startswith("encoder."):
            nk = nk[len("encoder."):]
        if nk.startswith("backbone."):
            nk = nk[len("backbone."):]
        new_ckpt[nk] = v
    checkpoint_model = new_ckpt

    # MAE pretrain often contains decoder weights, drop them
    keys_to_remove = []
    for k in checkpoint_model.keys():
        if k.startswith("decoder_"):
            keys_to_remove.append(k)
        if k.startswith("mask_token"):
            keys_to_remove.append(k)
    for k in keys_to_remove:
        del checkpoint_model[k]

    checkpoint_model = interpolate_pos_embed_if_needed(model, checkpoint_model)

    # classifier head is task-specific, remove if shape mismatch
    for k in ["head.weight", "head.bias", "fc_norm.weight", "fc_norm.bias"]:
        if k in checkpoint_model and k in model.state_dict():
            if checkpoint_model[k].shape != model.state_dict()[k].shape:
                print(f"Removing incompatible key from checkpoint: {k}")
                del checkpoint_model[k]

    msg = model.load_state_dict(checkpoint_model, strict=False)
    print("Load result:")
    print("  missing keys:", msg.missing_keys)
    print("  unexpected keys:", msg.unexpected_keys)


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

        k = min(5, logits.shape[1])
        _, pred = logits.topk(k, dim=1, largest=True, sorted=True)
        correct = pred.eq(y.view(-1, 1))
        total_correct1 += correct[:, :1].sum().item()
        total_correct5 += correct[:, :k].sum().item()

    return total_loss / total_n, total_correct1 / total_n, total_correct5 / total_n


def main():
    p = argparse.ArgumentParser()

    # optimization
    p.add_argument("--lr", type=float, required=True)
    p.add_argument("--min_lr", type=float, default=1e-6)
    p.add_argument("--wd", type=float, default=0.05)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--epochs", type=int, default = 100)
    p.add_argument("--batch_size", type=int, default = 1024)
    p.add_argument("--opt", type=str, default="sgd")
    p.add_argument("--sched", type=str, default="cosine", choices=["cosine", "multistep", "none"])
    p.add_argument("--warmup_epochs", type=int, default=0)
    p.add_argument("--beta", type=float, default=0.9)

    # model
    p.add_argument("--model", type=str, default="vit_small_patch16_224")
    p.add_argument("--num_classes", type=int, default=1000)
    p.add_argument("--drop_path", type=float, default=0.1)
    p.add_argument("--pretrained_ckpt", type=str, default="./checkpoints/mae_pretrained.pth")

    # finetune behavior
    p.add_argument("--layer_decay", type=float, default=0.65, help="Set <1.0 if you want layer-wise lr decay")
    p.add_argument("--freeze_patch_embed", action="store_true")
    p.add_argument("--freeze_blocks", type=int, default=0, help="Freeze first N transformer blocks")

    # data
    p.add_argument("--train_dir", type=str, default="./imagenet/train")
    p.add_argument("--val_dir", type=str, default="./imagenet/val")
    p.add_argument("--num_workers", type=int, default=8)

    # runtime
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--precision", type=str, default="bf16", choices=["fp32", "fp16", "bf16"])
    p.add_argument("--wandb_project", type=str, default="mae-imagenet-finetune")
    p.add_argument("--run_name", type=str, default=None)
    p.add_argument("--output_dir", type=str, default="./outputs_mae_finetune")
    p.add_argument("--save_freq", type=int, default=50)

    args = p.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device(args.device)

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
        amp_dtype = torch.float32
        use_amp = False
        use_scaler = False

    scaler = torch.cuda.amp.GradScaler(enabled=use_scaler)

    if args.run_name is None:
        args.run_name = f"{args.model}-ft-{args.opt}-{args.precision}"

    wandb.init(
        project=args.wandb_project,
        group="imagenet-mae-finetune",
        name=args.run_name,
        config=vars(args),
    )

    # ImageNet transforms for ViT finetuning
    train_tfm = create_transform(
        input_size=224,
        is_training=True,
        color_jitter=0.4,
        auto_augment='rand-m9-mstd0.5-inc1',   # RandAugment
        re_prob=0.25,                            # Random Erasing
        re_mode='pixel',
        re_count=1,
        interpolation='bicubic',
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    )

    val_tfm = transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406),
                            std=(0.229, 0.224, 0.225)),
    ])

    # Mixup / Cutmix
    mixup_fn = Mixup(
        mixup_alpha=0.8,
        cutmix_alpha=1.0,
        label_smoothing=0.1,
        num_classes=args.num_classes,
    )

    train_ds = datasets.ImageFolder(args.train_dir, transform=train_tfm)
    val_ds = datasets.ImageFolder(args.val_dir, transform=val_tfm)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=(args.num_workers > 0),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=max(args.batch_size, 256),
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=(args.num_workers > 0),
    )

    checkpoint_path = "./checkpoints/mae_vit_base_224_2.pth"

    print(f"Loading MAE pretrained weights from local file: {checkpoint_path}")

    model = timm.create_model(
        'vit_base_patch16_224',           # 注意：这里用基础模型名，不要加 .mae
        pretrained=False,                 # 不从网络下载
        num_classes=args.num_classes,
        drop_path_rate=args.drop_path,
        img_size=224,
    ).to(device)

    # 从本地文件加载权重
    if os.path.exists(checkpoint_path):
        # state_dict = torch.load(checkpoint_path, map_location="cpu")
        
        # # timm 的 MAE 权重通常在 "model" 键下，或者直接是 state_dict
        # if isinstance(state_dict, dict) and "model" in state_dict:
        #     state_dict = state_dict["model"]
        
        # # 加载权重（允许部分 key 不匹配）
        # msg = model.load_state_dict(state_dict, strict=False)
        # print("Load result:")
        # print("  missing keys:", len(msg.missing_keys))
        # print("  unexpected keys:", len(msg.unexpected_keys))
        # print(" Successfully loaded local MAE pretrained weights!")
        load_mae_pretrained(model, checkpoint_path)
    else:
        print(f" Checkpoint not found at {checkpoint_path}")
        print("Please download it first.")

    model = torch.compile(model, mode="default", fullgraph=False)

    if args.freeze_patch_embed:
        for p_ in model.patch_embed.parameters():
            p_.requires_grad = False

    if args.freeze_blocks > 0:
        for i in range(min(args.freeze_blocks, len(model.blocks))):
            for p_ in model.blocks[i].parameters(): 
                p_.requires_grad = False

    # trainable_params = [p_ for p_ in model.parameters() if p_.requires_grad]
    trainable_params = []
    if args.layer_decay < 1.0:
        # 按层衰减
        layer_decay = args.layer_decay
        num_layers = len(model.blocks)
        for i, block in enumerate(model.blocks):
            decay = layer_decay ** (num_layers - i)
            for n, p in block.named_parameters():
                if 'norm' in n or 'bias' in n:
                    trainable_params.append({'params': p, 'weight_decay': 0., 'lr': args.lr * decay})
                else:
                    trainable_params.append({'params': p, 'weight_decay': args.wd, 'lr': args.lr * decay})
        
        # patch embed 和 head
        trainable_params.append({'params': model.patch_embed.parameters(), 
                            'lr': args.lr * layer_decay** (num_layers+1), 'weight_decay': args.wd})
        trainable_params.append({'params': model.head.parameters(), 
                            'lr': args.lr, 'weight_decay': args.wd})
    else:
        trainable_params = [p for p in model.parameters() if p.requires_grad]


    optimizer = build_optimizer(
    args.opt,
    trainable_params,
    args.lr,
    args.momentum,
    args.wd,
    args.beta,
)

    if args.sched == "multistep":
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=[int(args.epochs * 0.6), int(args.epochs * 0.8)],
            gamma=0.1,
        )
    elif args.sched == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, args.epochs - args.warmup_epochs),
            eta_min=args.min_lr,
        )
    else:
        scheduler = None

    # loss_fn = nn.CrossEntropyLoss()
    if mixup_fn is not None:
        loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
    else:
        loss_fn = nn.CrossEntropyLoss()
    global_step = 0
    best_acc1 = 0.0

    for epoch in range(args.epochs):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}", leave=False)

        running_loss = 0.0
        running_correct1 = 0
        running_n = 0
        t0 = time.time()

        # simple warmup
        if epoch < args.warmup_epochs:
            warmup_lr = args.lr * float(epoch + 1) / float(args.warmup_epochs)
            for pg in optimizer.param_groups:
                pg["lr"] = warmup_lr

        for x, y in pbar:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            if mixup_fn is not None:
                x, y = mixup_fn(x, y)

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

            # pred1 = logits.argmax(dim=1)
            # running_correct1 += (pred1 == y).sum().item()
            
            pred1 = logits.argmax(dim=1)
            
            if y.dim() == 1:                    # 普通 hard label
                correct1 = (pred1 == y).sum().item()
            else:                               # Mixup 后的 soft label
                # 取 soft label 的 argmax 作为“伪真实标签”计算准确率
                target = y.argmax(dim=1)
                correct1 = (pred1 == target).sum().item()
            
            running_correct1 += correct1

            cur_lr = optimizer.param_groups[0]["lr"]
            pbar.set_postfix({
                "loss": f"{(running_loss / running_n):.4f}",
                "acc1": f"{(running_correct1 / running_n):.4f}",
                "lr": f"{cur_lr:g}",
            })

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

        val_loss, val_acc1, val_acc5 = evaluate(model, val_loader, device, amp_dtype)
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
            f"train_loss={train_loss_avg:.4f} train_acc1={train_acc1_avg:.4f} | "
            f"val_loss={val_loss:.4f} val_acc1={val_acc1:.4f} val_acc5={val_acc5:.4f}"
        )

        if epoch >= args.warmup_epochs and scheduler is not None:
            scheduler.step()

        ckpt = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "args": vars(args),
            "best_acc1": best_acc1,
        }

        if (epoch + 1) % args.save_freq == 0:
            save_path = os.path.join(args.output_dir, f"checkpoint_epoch_{epoch+1}.pth")
            torch.save(ckpt, save_path)

        if val_acc1 > best_acc1:
            best_acc1 = val_acc1
            ckpt["best_acc1"] = best_acc1
            best_path = os.path.join(args.output_dir, "best.pth")
            torch.save(ckpt, best_path)
            print(f"Saved best checkpoint to {best_path}, acc1={best_acc1:.4f}")

    wandb.finish()


if __name__ == "__main__":
    main()