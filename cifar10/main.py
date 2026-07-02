'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import wandb


import torchvision
import torchvision.transforms as transforms

import os
import argparse

from models import *
from utils import progress_bar

from opt import *

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
parser.add_argument("--device", type=str, default="cuda:0", help='e.g. "cuda:0", "cuda:1", "cpu"')
parser.add_argument("--sched", type=str, default="cos")
parser.add_argument("--opt", type=str, default="sgd")
parser.add_argument('--beta', default=0.9, type=float, help='beta')
parser.add_argument('--wd', default=5e-4, type=float, help='weight_decay')
parser.add_argument("--beta_sched", type=str, default="None", choices=["None", "cosine", "linear", 'stepwise'])
parser.add_argument('--epochs', default=200, type=int, help='epochs')

args = parser.parse_args()

device = args.device
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
step = 0


def build_optimizer(opt_name, params, lr, momentum,weight_decay = 0.0,beta = None):
    opt_name = opt_name.lower()

    if opt_name == "sgd":
        return SGD(params, lr=lr, momentum=momentum, nesterov=False, weight_decay=weight_decay)
        # return optim.SGD(params, lr=lr, momentum=0.9, weight_decay=5e-4)
    elif opt_name == 'dm_sgd3':
        return DM_SGD3(params, lr=lr, momentum=momentum, nesterov=False, weight_decay=weight_decay, beta=beta)

    raise ValueError(f"Unknown optimizer: {opt_name}")

class CosineBetaScheduler:
    """
    Cosine decay `beta` in optimizer.param_groups to 0.0 over T_max steps.
    """
    def __init__(self, optimizer, T_max: int):
        if T_max <= 0:
            raise ValueError("T_max must be > 0")
        self.optimizer = optimizer
        self.T_max = int(T_max)
        self.last_step = -1

        self.base_betas = []
        for g in self.optimizer.param_groups:
            self.base_betas.append(g.get("beta", None))

    def get_beta_scale(self, step: int) -> float:
        t = min(max(step, 0), self.T_max)
        return 0.5 * (1.0 + math.cos(math.pi * t / self.T_max))

    def step(self):
        self.last_step += 1
        scale = self.get_beta_scale(self.last_step)

        for g, base in zip(self.optimizer.param_groups, self.base_betas):
            if base is None or "beta" not in g:
                continue
            g["beta"] = float(base) * scale
        return scale

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

class LinearBetaScheduler:
    """
    线性衰减 `beta` 从初始值逐渐降到 0.0。
    
    Usage:
        sched = LinearBetaScheduler(optimizer, T_max=20000)
        for step in range(...):
            ...
            sched.step()
    """
    def __init__(self, optimizer, T_max: int, last_step: int = -1):
        if T_max <= 0:
            raise ValueError("T_max must be > 0")
        
        self.optimizer = optimizer
        self.T_max = int(T_max)
        self.last_step = last_step

        # 记录每个 param_group 的初始 beta 值
        self.base_betas = []
        for g in self.optimizer.param_groups:
            self.base_betas.append(g.get("beta", None))

        # 如果是从 checkpoint 恢复，立即应用当前 beta
        if last_step >= 0:
            self._update_beta()

    def get_beta(self, step: int) -> float:
        """返回当前 step 应该使用的 beta 值"""
        t = min(max(step, 0), self.T_max)
        scale = max(0.0, 1.0 - t / self.T_max)   # 线性从 1.0 衰减到 0.0
        return scale

    def _update_beta(self):
        scale = self.get_beta(self.last_step)
        for group, base_beta in zip(self.optimizer.param_groups, self.base_betas):
            if base_beta is None or "beta" not in group:
                continue
            group["beta"] = float(base_beta) * scale

    def step(self):
        self.last_step += 1
        self._update_beta()
        return self.get_beta(self.last_step)

    def state_dict(self):
        return {
            "T_max": self.T_max,
            "last_step": self.last_step,
            "base_betas": self.base_betas,
        }

    def load_state_dict(self, state_dict):
        self.T_max = int(state_dict["T_max"])
        self.last_step = int(state_dict["last_step"])
        self.base_betas = list(state_dict["base_betas"])
        self._update_beta()   # 恢复对应 beta 值

wandb.init(
    project = "double momentum",
    group = 'cifar10-github',
    name = args.opt,
    config={k: getattr(args, k) for k in ["lr", "opt",'beta','device','sched']},
)

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=128, shuffle=True, num_workers=16)

testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=16)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

# Model
print('==> Building model..')
# net = VGG('VGG19')
# net = ResNet18()
# net = PreActResNet18()
# net = GoogLeNet()
# net = DenseNet121()
# net = ResNeXt29_2x64d()
# net = MobileNet()
# net = MobileNetV2()
# net = DPN92()
# net = ShuffleNetG2()
# net = SENet18()
# net = ShuffleNetV2(1)
# net = EfficientNetB0()
# net = RegNetX_200MF()
# net = SimpleDLA()
net = DLA()
net = net.to(device)
if 'cuda' in device:
    # net = torch.nn.DataParallel(net)
    cudnn.benchmark = True


if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.pth')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

criterion = nn.CrossEntropyLoss()

optimizer = build_optimizer(args.opt, net.parameters(),lr=args.lr,momentum=args.momentum,weight_decay=args.wd,beta=args.beta)

if args.sched == 'cos':
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
else:
    scheduler = None

if args.beta_sched == 'linear':
    # beta_scheduler = CosineBetaScheduler(optimizer, T_max=args.epochs)
    # beta_scheduler = StageBetaScheduler(optimizer=optimizer, milestones=[ int(1 / 3 * args.epochs), int(2 / 3 * args.epochs)],betas=[0.9, 0.5, 0.1])
    beta_scheduler = LinearBetaScheduler(optimizer, T_max = args.epochs)
elif args.beta_sched == 'cosine':
    beta_scheduler = CosineBetaScheduler(optimizer, T_max=args.epochs)
elif args.beta_sched == 'None':
    beta_scheduler = None


# Training
def train(epoch):
    global step
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        step += 1
        road_beta = optimizer.param_groups[0].get("beta", None)
        wandb.log({
            "train/loss": loss.item(),
            "train/acc": 100. * correct / total,
            "train/lr": optimizer.param_groups[0]['lr'],
            'beta' : road_beta,
            "global_step" : step,
        }, step=step)

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))



def test(epoch):
    global best_acc
    global step
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    avg_test_loss = test_loss / len(testloader)
    avg_test_acc = 100. * correct / total

    # === 每个 epoch 记录 valid loss 和 valid acc ===
    print(step)
    wandb.log({
        "val/loss": avg_test_loss,
        "val/acc": avg_test_acc,
        "epoch": epoch,
    }, step=step)

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.pth')
        best_acc = acc


for epoch in range(start_epoch, start_epoch+args.epochs):
    train(epoch)
    test(epoch)
    if scheduler is not None:
        scheduler.step()
    if beta_scheduler is not None:
        beta_scheduler.step()

    wandb.log({"lr": optimizer.param_groups[0]['lr']}, step=step)
