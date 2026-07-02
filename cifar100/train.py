import argparse
import importlib

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torchvision.models.densenet import DenseNet
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR, MultiStepLR

import math

from tqdm.auto import tqdm
import wandb

from densenet import DenseNetBC

from opt import * 


def import_obj(path: str):
    if ":" in path:
        mod, name = path.split(":")
    else:
        mod, name = path.rsplit(".", 1)
    m = importlib.import_module(mod)
    return getattr(m, name)


def build_optimizer(opt_name, params, lr, momentum, wd = 0.0,beta = None):
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
        return SGD(params, lr=lr, momentum=momentum, nesterov=False, weight_decay=wd)
    elif opt_name == 'sf_sgd1':
        return SF_SGD1(params, lr=lr, momentum=momentum, nesterov=False, weight_decay=wd, beta=beta)
    elif opt_name == 'sf_sgd2':
        return SF_SGD2(params, lr=lr, momentum=momentum, nesterov=False, weight_decay=wd, beta=beta)
    elif opt_name == 'dm_sgd':
        return DM_SGD(params, lr=lr, momentum=momentum, nesterov=False, weight_decay=wd, beta=beta)
    elif opt_name == 'dm_sgd2':
        return DM_SGD2(params, lr=lr, momentum=momentum, nesterov=False, weight_decay=wd, beta=beta)
    elif opt_name == 'dm_sgd3':
        return DM_SGD3(params, lr=lr, momentum=momentum, nesterov=False, weight_decay=wd, beta=beta)
    elif opt_name == 'muon':
        return SingleDeviceMuonWithAuxAdam(param_groups)
    elif opt_name == 'dm_muon':
        return DM_Muon(param_groups, beta=beta)

    raise ValueError(f"Unknown optimizer: {opt_name}")


@torch.no_grad()
def evaluate(model, loader, device, eval_on_x = False, opt = None):
    if eval_on_x and opt is not None:
        opt.exchange('x')

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
        opt.exchange('y')

    return total_loss / total_n, total_correct / total_n


class CosineBetaScheduler:
    """
    Cosine decay `beta` in optimizer.param_groups to 0.0 over T_max steps.
    Usage:
        sched = CosineBetaScheduler(optimizer, T_max=10000)
        for step in range(...):
            ...
            sched.step()
    """
    def __init__(self, optimizer, T_max: int):
        if T_max <= 0:
            raise ValueError("T_max must be > 0")
        self.optimizer = optimizer
        self.T_max = int(T_max)
        self.last_step = -1

        # record initial beta for each group (None if not present)
        self.base_betas = []
        for g in self.optimizer.param_groups:
            self.base_betas.append(g.get("beta", None))

    def get_beta_scale(self, step: int) -> float:
        t = min(max(step, 0), self.T_max)
        return 0.5 * (1.0 + math.cos(math.pi * t / self.T_max))  # 1 -> 0

    def step(self):
        self.last_step += 1
        scale = self.get_beta_scale(self.last_step)

        for g, base in zip(self.optimizer.param_groups, self.base_betas):
            if base is None or "beta" not in g:
                continue
            g["beta"] = float(base) * scale

        return scale  # optional: for logging

    def state_dict(self):
        return {
            "T_max": self.T_max,
            "last_step": self.last_step,
            "base_betas": self.base_betas,
        }

    def load_state_dict(self, state):
        self.T_max = int(state["T_max"])
        self.last_step = int(state["last_step"])
        self.base_betas = list(state["base_betas"])


class StageBetaScheduler:
    """
    分阶段设置 beta 值。
    可用于：前期保持0.9，中期降到0.5，后期降到0.1。
    """
    def __init__(
        self,
        optimizer,
        milestones: list[int],
        betas: list[float],
        last_step: int = -1
    ):
        if len(milestones) + 1 != len(betas):
            raise ValueError("milestones 数量必须比 betas 少 1")
        if not all(0 < b <= 0.999 for b in betas):
            raise ValueError("beta 值必须在 (0, 0.999] 范围内")

        self.optimizer = optimizer
        self.milestones = sorted(milestones)      # 切换点
        self.betas = list(betas)                  # 每个阶段的目标 beta
        self.last_step = last_step

        # 记录每个 param_group 的初始 beta（用于 state_dict 恢复）
        self.base_betas = [g.get("beta", None) for g in optimizer.param_groups]

        if last_step >= 0:
            self.step()  # 恢复到正确阶段

    def get_beta(self, step: int) -> float:
        """根据当前 step 返回应该使用的 beta 值"""
        for milestone, beta in zip(self.milestones, self.betas):
            if step < milestone:
                return beta
        return self.betas[-1]                     # 最后一个阶段

    def step(self):
        self.last_step += 1
        current_beta = self.get_beta(self.last_step)

        for group, base in zip(self.optimizer.param_groups, self.base_betas):
            if base is None or "beta" not in group:
                continue
            group["beta"] = float(current_beta)   # 直接设置目标值

        return current_beta

    def state_dict(self):
        return {
            "milestones": self.milestones,
            "betas": self.betas,
            "last_step": self.last_step,
            "base_betas": self.base_betas,
        }

    def load_state_dict(self, state_dict):
        self.milestones = list(state_dict["milestones"])
        self.betas = list(state_dict["betas"])
        self.last_step = int(state_dict["last_step"])
        self.base_betas = list(state_dict["base_betas"])
        self.step()   # 恢复当前 beta


class DenseNetCIFAR100(DenseNet):
    def __init__(self):
        # 严格按照图片中的参数
        super().__init__(
            block_config=(6, 12, 24, 16),   #
            growth_rate=12,                
            num_init_features=24,       
            bn_size=4,
            drop_rate=0.0,
            num_classes=100                
        )
        
        self.features.conv0 = nn.Conv2d(
            in_channels=3,
            out_channels=24,         
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )
        
        self.features.pool0 = nn.Identity()

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--lr", type=float, required=True)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--beta", type=float, default=0.9)
    p.add_argument("--wd", type=float, default=2e-4)
    p.add_argument("--epochs", type=int, default = 300)
    p.add_argument("--batch_size", type=int, default = 64)
    p.add_argument("--opt", type=str, default="sgd")
    p.add_argument("--device", type=str, default="cuda:0", help='e.g. "cuda:0", "cuda:1", "cpu"')
    p.add_argument("--sched", type=str, default="none", choices=["multistep", "cosine", "none"])
    p.add_argument("--eval", type=str, default="False", choices=["False", "True"])
    p.add_argument("--beta_sched", type=str, default="False", choices=["False", "True"])

    p.add_argument("--wandb_project", type=str, default="double momentum")
    args = p.parse_args()

    eval_on_x = (args.eval == 'True')
    beta_sched = (args.beta_sched == 'True')
    device = torch.device(args.device)

    wandb.init(
        project=args.wandb_project,
        group="cifar100-densenet",
        name=args.opt,
        config=vars(args),
    )

    # CIFAR100 mean/std (常用一组)
    mean = (0.5071, 0.4867, 0.4408)
    std = (0.2675, 0.2565, 0.2761)

    train_tfm = transforms.Compose([
        transforms.Pad(4, padding_mode='reflect'),  # 先把图片填充到40x40
        transforms.RandomCrop(32),                  # 再随机裁剪回32x32
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    test_tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    train_ds = datasets.CIFAR100(root="./data", train=True, download=True, transform=train_tfm)
    test_ds = datasets.CIFAR100(root="./data", train=False, download=True, transform=test_tfm)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=16, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=16, pin_memory=True)

    # DenseNet (Huang et al., 2017) - torchvision
    # model = DenseNetBC(depth=100, growth_rate=12, compression=0.5, num_classes=100, drop_rate=0.0).to(device)
    model = DenseNetCIFAR100()
    model = model.to(device)

    optimizer = build_optimizer(args.opt, model.parameters(), args.lr, args.momentum, args.wd, args.beta)

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

    if beta_sched is True:
        # beta_scheduler = CosineBetaScheduler(optimizer,T_max=args.epochs)
        beta_scheduler = StageBetaScheduler(optimizer=optimizer, milestones=[ int(1 / 3 * args.epochs), int(2 / 3 * args.epochs)],betas=[0.9, 0.5, 0.1])
    else:
        beta_scheduler = None


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

            beta = optimizer.param_groups[0].get("beta", None)
            wandb.log(
                {"train/loss": loss.item(), "train/lr": optimizer.param_groups[0]["lr"], "global_step": global_step, 'beta': beta},
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
        if beta_scheduler is not None:
            beta_scheduler.step()

    wandb.finish()


if __name__ == "__main__":
    main()