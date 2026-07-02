"""
This training script can be run both on a single gpu in debug mode,
and also in a larger training run with distributed data parallel (ddp).

To run on a single GPU, example:
$ python train.py --batch_size=32 --compile=False

To run with DDP on 4 gpus on 1 node, example:
$ torchrun --standalone --nproc_per_node=4 train.py

To run with DDP on 4 gpus across 2 nodes, example:
- Run on the first (master) node with example IP 123.456.123.456:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=123.456.123.456 --master_port=1234 train.py
- Run on the worker node:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=123.456.123.456 --master_port=1234 train.py
(If your cluster does not have Infiniband interconnect prepend NCCL_IB_DISABLE=1)
"""



"""
This training script can be run both on a single gpu in debug mode,
and also in a larger training run with distributed data parallel (ddp).

To run on a single GPU, example:
$ python train.py --batch_size=32 --compile=False

To run with DDP on 4 gpus on 1 node, example:
$ torchrun --standalone --nproc_per_node=4 train.py

To run with DDP on 4 gpus across 2 nodes, example:
- Run on the first (master) node with example IP 123.456.123.456:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=123.456.123.456 --master_port=1234 train.py
- Run on the worker node:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=123.456.123.456 --master_port=1234 train.py
(If your cluster does not have Infiniband interconnect prepend NCCL_IB_DISABLE=1)
"""




#### 单卡运行   python train.py config/train_gpt2.py
#### 分布式运行   torchrun --standalone --nproc_per_node=8 train.py config/train_gpt2.py


import os
import time
import math
import pickle
from contextlib import nullcontext

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

# 导入模型和优化器
from model import GPTConfig, GPT
from muon import Muon, SingleDeviceMuon, MuonWithAuxAdam, SingleDeviceMuonWithAuxAdam
from muon import SGD ,SF_SGD, DM_SGD, DM_Muon, DM_Muon2, DM_Muon3

import scipy
from scipy import linalg
import numpy as np
import torch.nn.functional as F


# -----------------------------------------------------------------------------
# default config values designed to train a gpt2 (124M) on OpenWebText
# I/O
out_dir = 'out_openwebtext'
eval_interval = 100
log_interval = 1
eval_iters = 200
eval_only = False
always_save_checkpoint = True
init_from = 'scratch'  # 'scratch' or 'resume' or 'gpt2*'
    
    # wandb logging
wandb_log = False  
wandb_project = 'owt'
wandb_run_name = 'gpt2'
    
    # data
dataset = 'openwebtext'
gradient_accumulation_steps = 5 * 8
batch_size = 12
block_size = 1024
    
    # model
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0
bias = False
    
    # optimizer
optimizer_type = 'adamw'  # 'adamw' or 'muon'
learning_rate = 6e-4
max_iters = 30000
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0
    
    # muon specific
muon_momentum = 0.95
muon_lr = 0.02
use_muon_for_hidden_only = True
road_beta = 0.9
    
    # learning rate decay
decay_lr = True
warmup_iters = 2000
lr_decay_iters = 600000
min_lr = 6e-5
    
backend = 'nccl' # 'nccl', 'gloo', etc.
    # system
device = 'cuda' if torch.cuda.is_available() else 'cpu'
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
# dtype = 'float8_e4m3fn'
compile = True

# -----------------------------------------------------------------------------
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open('configurator.py').read()) # overrides from command line or config file
config = {k: globals()[k] for k in config_keys} # will be useful for logging
# -----------------------------------------------------------------------------

wandb_run_name = f'gpt2-124M-{optimizer_type}'


ddp = int(os.environ.get('RANK', -1)) != -1
if ddp:
    init_process_group(backend=backend)
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 
    seed_offset = ddp_rank 
    assert gradient_accumulation_steps % ddp_world_size == 0
    gradient_accumulation_steps //= ddp_world_size
else:
    master_process = True
    seed_offset = 0
    ddp_world_size = 1
tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * block_size
print(f"tokens per iteration will be: {tokens_per_iter:,}")

if master_process:
    os.makedirs(out_dir, exist_ok=True)
torch.manual_seed(1337 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True 
torch.backends.cudnn.allow_tf32 = True 
device_type = 'cuda' if 'cuda' in device else 'cpu' 
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
# if dtype == 'float8_e4m3fn':
#     ptdtype = torch.float8_e4m3fn
# elif dtype == 'float8_e5m2':
#     ptdtype = torch.float8_e5m2
# else:
#     ptdtype = {'float32': torch.float32, 
#                'bfloat16': torch.bfloat16, 
#                'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)



data_dir = os.path.join('data', dataset)
def get_batch(split):
    if split == 'train':
        data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    else:
        data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    if device_type == 'cuda':
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y

iter_num = 0
best_val_loss = 1e9

meta_path = os.path.join(data_dir, 'meta.pkl')
meta_vocab_size = None
if os.path.exists(meta_path):
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    meta_vocab_size = meta['vocab_size']
    print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")

model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                  bias=bias, vocab_size=None, dropout=dropout) 
if init_from == 'scratch':
    print("Initializing a new model from scratch")
    if meta_vocab_size is None:
        print("defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)")
    model_args['vocab_size'] = meta_vocab_size if meta_vocab_size is not None else 50304
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
elif init_from == 'resume':
    print(f"Resuming training from {out_dir}")
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = checkpoint['model_args']
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = checkpoint_model_args[k]
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    iter_num = checkpoint['iter_num']
    best_val_loss = checkpoint['best_val_loss']
elif init_from.startswith('gpt2'):
    print(f"Initializing from OpenAI GPT-2 weights: {init_from}")
    override_args = dict(dropout=dropout)
    model = GPT.from_pretrained(init_from, override_args)
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = getattr(model.config, k)
if block_size < model.config.block_size:
    model.crop_block_size(block_size)
    model_args['block_size'] = block_size
model.to(device)

scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

if use_muon_for_hidden_only:
    hidden_weights = []
    other_params = []
    
    for name, param in model.named_parameters():
        if param.ndim >= 2 and "embed" not in name and "lm_head" not in name:
            hidden_weights.append(param)
        else:
            other_params.append(param)

    param_groups = [
        {
            'params': other_params, 
            'lr': learning_rate,
            'betas': (beta1, beta2),
            'eps': 1e-8,
            'weight_decay': weight_decay,
            'use_muon': False
        },
        {
            'params': hidden_weights,
            'lr': muon_lr,
            'momentum': muon_momentum,
            'weight_decay': weight_decay,
            'use_muon': True
        }
    ]
    
    if ddp:
        if optimizer_type == 'Muon':
            optimizer = SingleDeviceMuonWithAuxAdam(param_groups)
            print("using single-device MuonWithAuxAdam optimizer")
        elif optimizer_type == 'Muon_Road':
            optimizer = MuonWithAuxAdamRoad(param_groups,road_beta=road_beta)
            print("using distributed MuonWithAuxAdamRoad optimizer")
        elif optimizer_type.lower() == 'dm_muon3':
            print('====================================dm muon3=====================================')
            optimizer = DM_Muon3(param_groups,road_beta=road_beta)
        elif optimizer_type.lower() == 'adam':
            print('==================================== adam =====================================')
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=learning_rate,
                betas=(beta1, beta2),
                weight_decay = weight_decay
            )
        else:
            raise ValueError(f'{optimizer_type} is not supported')
    else:
        if optimizer_type == 'Muon':
            optimizer = SingleDeviceMuonWithAuxAdam(param_groups)
            print("using single-device MuonWithAuxAdam optimizer")
        elif optimizer_type == 'Muon_Road':
            optimizer = SingleDeviceMuonWithAuxAdamRoad(param_groups,road_beta=road_beta)
            print("using single-device MuonWithAuxAdamRoad optimizer")
        elif optimizer_type.lower() == 'sgd':
            print('====================================sgd=====================================')
            optimizer = SGD(model.parameters(),
                                lr=lr,
                                momentum = muon_momentum, 
                                weight_decay = weight_decay,
                                road_beta = road_beta)
        elif optimizer_type.lower() == 'sf_sgd':
            print('====================================sf sgd=====================================')
            optimizer = SF_SGD(model.parameters(),
                                lr=lr,
                                momentum = muon_momentum, 
                                weight_decay = weight_decay,
                                road_beta = road_beta)
        elif optimizer_type.lower() == 'dm_sgd':
            print('====================================dm sgd=====================================')
            optimizer = DM_SGD(model.parameters(),
                                lr=lr,
                                momentum = muon_momentum, 
                                weight_decay = weight_decay,
                                road_beta = road_beta)
        elif optimizer_type.lower() == 'dm_muon':
            print('====================================dm muon=====================================')
            optimizer = DM_Muon(param_groups,road_beta=road_beta)
        elif optimizer_type.lower() == 'dm_muon2':
            print('====================================dm muon2=====================================')
            optimizer = DM_Muon2(param_groups,road_beta=road_beta)
        elif optimizer_type.lower() == 'dm_muon3':
            print('====================================dm muon3=====================================')
            optimizer = DM_Muon3(param_groups,road_beta=road_beta)
        else:
            raise ValueError(f'{optimizer_type} is not supported')
else:
    params = list(model.parameters())
    if ddp:
        optimizer = Muon(params, lr=muon_lr, weight_decay=weight_decay, 
                        momentum=muon_momentum)
        print("using distributed Muon optimizer")
    else:
        optimizer = SingleDeviceMuon(params, lr=muon_lr, weight_decay=weight_decay, 
                                    momentum=muon_momentum)
        print("using single-device Muon optimizer")


if init_from == 'resume':
    optimizer.load_state_dict(checkpoint['optimizer'])
checkpoint = None 


if compile:
    print("compiling the model... (takes a ~minute)")
    unoptimized_model = model
    model = torch.compile(model) # requires PyTorch 2.0


if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            with ctx:
                logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


def get_lr(it, lr=learning_rate):
    if it < warmup_iters:
        return lr * (it + 1) / (warmup_iters + 1)
    if it > lr_decay_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (lr - min_lr)

# logging
if wandb_log and master_process:
    import wandb
    wandb.init(
        project=wandb_project,
        name=wandb_run_name,
        config=config,
        mode=wandb_mode if 'wandb_mode' in globals() else "online",   # 支持 offline
        save_code=True,
        reinit=True,          # 防止 DDP 冲突
    )
    print(f"WandB initialized on rank {ddp_rank if ddp else 0}")
else:
    wandb = None  # 其他 rank 不使用 wandb

# training loop
X, Y = get_batch('train') # fetch the very first batch
t0 = time.time()
local_iter_num = 0 # number of iterations in the lifetime of this process
raw_model = model.module if ddp else model # unwrap DDP container if needed
running_mfu = -1.0



while True:
    lr = get_lr(iter_num, learning_rate) if decay_lr else learning_rate
    m_lr = get_lr(iter_num, muon_lr) if decay_lr else muon_lr
    for param_group in optimizer.param_groups:
        if param_group.get("use_muon", True):
            param_group['lr'] = m_lr
        else:
            param_group['lr'] = lr

    log = {
            "iter": iter_num,
            "lr": lr,
            'muon_lr':m_lr,
            "mfu": running_mfu*100, 
            }

    if iter_num % eval_interval == 0 and master_process:
        losses = estimate_loss()
        print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        
        log["train/loss"]= losses['train']
        log["val/loss"]= losses['val']
        if losses['val'] < best_val_loss or always_save_checkpoint:
            best_val_loss = losses['val']
            if iter_num > 0:
                checkpoint = {
                    'model': raw_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'model_args': model_args,
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                    'config': config,
                }
                print(f"saving checkpoint to {out_dir}")
                torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))
    if iter_num == 0 and eval_only:
        break
 
    for micro_step in range(gradient_accumulation_steps):
        if ddp:
            model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
        with ctx:
            logits, loss = model(X, Y)
            loss = loss / gradient_accumulation_steps 
        X, Y = get_batch('train')
        scaler.scale(loss).backward()
    
    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)
    

    

    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % log_interval == 0 and master_process:
        lossf = loss.item() * gradient_accumulation_steps
        if local_iter_num >= 5: # let the training loop settle a bit
            mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
            running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
        print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%")
        log['loss'] = lossf
    iter_num += 1
    local_iter_num += 1

    if wandb_log and master_process and wandb is not None:
            wandb.log(log)

    if iter_num > max_iters:
        break

if ddp:
    destroy_process_group()