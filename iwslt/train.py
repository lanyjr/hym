import argparse
import importlib
import math
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from datasets import load_dataset
import sentencepiece as spm
import sacrebleu
from tqdm.auto import tqdm
import wandb

from lstm_seq2seq import Seq2Seq
from opt import *

class WarmupLinearDecayLR:
    """
    Linear warmup for `warmup_steps`, then linear decay to 0 by `total_steps`.
    Call .step() every training iteration.
    """
    def __init__(self, optimizer, total_steps, warmup_steps, base_lr):
        self.optimizer = optimizer
        self.total_steps = int(total_steps)
        self.warmup_steps = int(warmup_steps)
        self.base_lr = float(base_lr)
        self.step_num = 0  # counts how many times step() has been called

        # set initial lr (optional)
        self._set_lr(0.0)

    def _set_lr(self, lr):
        for pg in self.optimizer.param_groups:
            pg["lr"] = lr

    def get_lr(self):
        s = self.step_num
        if self.warmup_steps > 0 and s < self.warmup_steps:
            return self.base_lr * (s + 1) / self.warmup_steps
        # decay
        denom = max(1, self.total_steps - self.warmup_steps)
        rem = max(0, self.total_steps - (s + 1))
        return self.base_lr * (rem / denom)

    def step(self):
        lr = self.get_lr()
        self._set_lr(lr)
        self.step_num += 1
        return lr


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


def ensure_spm(text_iter, out_dir, vocab_size):
    os.makedirs(out_dir, exist_ok=True)
    prefix = os.path.join(out_dir, f"spm_{vocab_size}")
    model_path = prefix + ".model"
    if os.path.exists(model_path):
        return model_path

    corpus_path = os.path.join(out_dir, "corpus.txt")
    with open(corpus_path, "w", encoding="utf-8") as f:
        for t in text_iter:
            f.write(t.replace("\n", " ") + "\n")

    spm.SentencePieceTrainer.Train(
        input=corpus_path,
        model_prefix=prefix,
        vocab_size=vocab_size,
        model_type="bpe",
        character_coverage=1.0,
        pad_id=0, bos_id=1, eos_id=2, unk_id=3,
    )
    return model_path


def collate_fn(examples, sp_src, sp_tgt, max_len, device):
    pad_id, bos_id, eos_id = 0, 1, 2

    src_seqs, tgt_in_seqs, tgt_out_seqs, src_lens = [], [], [], []
    for ex in examples:
        src = sp_src.EncodeAsIds(ex["de"])[: max_len - 1] + [eos_id]
        tgt = sp_tgt.EncodeAsIds(ex["en"])[: max_len - 1] + [eos_id]

        tgt_in = [bos_id] + tgt[:-1]
        tgt_out = tgt

        src_seqs.append(src)
        tgt_in_seqs.append(tgt_in)
        tgt_out_seqs.append(tgt_out)
        src_lens.append(len(src))

    def pad(seqs):
        m = max(len(s) for s in seqs)
        out = torch.full((len(seqs), m), pad_id, dtype=torch.long)
        for i, s in enumerate(seqs):
            out[i, : len(s)] = torch.tensor(s, dtype=torch.long)
        return out

    batch = {
        "src": pad(src_seqs).to(device),
        "src_len": torch.tensor(src_lens, dtype=torch.long, device=device),
        "tgt_in": pad(tgt_in_seqs).to(device),
        "tgt_out": pad(tgt_out_seqs).to(device),
    }
    return batch


@torch.no_grad()
def evaluate(model, loader, device, loss_fn):
    model.eval()
    total_loss = 0.0
    total_tok = 0

    for batch in loader:
        logits = model(batch["src"], batch["src_len"], batch["tgt_in"])  # [B,T,V]
        loss = loss_fn(logits.view(-1, logits.size(-1)), batch["tgt_out"].view(-1))
        ntok = (batch["tgt_out"] != 0).sum().item()
        total_loss += loss.item() * ntok
        total_tok += ntok

    avg_loss = total_loss / max(total_tok, 1)
    ppl = math.exp(min(avg_loss, 20.0))
    return avg_loss, ppl


@torch.no_grad()
def eval_bleu_subset(model, loader, sp_tgt, max_len, num_batches=50):
    # 为了不拖慢训练，默认只取前 num_batches 个 batch 做 greedy BLEU
    model.eval()
    hyps, refs = [], []
    eos_id = 2

    for i, batch in enumerate(loader):
        if i >= num_batches:
            break
        out_ids = model.greedy_decode(batch["src"], batch["src_len"], max_len=max_len)
        for b in range(out_ids.size(0)):
            hyp_ids = [t for t in out_ids[b].tolist() if t not in (0, eos_id)]
            ref_ids = [t for t in batch["tgt_out"][b].tolist() if t not in (0, eos_id)]
            hyps.append(sp_tgt.DecodeIds(hyp_ids))
            refs.append(sp_tgt.DecodeIds(ref_ids))

    if not hyps:
        return 0.0
    return sacrebleu.corpus_bleu(hyps, [refs]).score


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--lr", type=float, required=True)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--beta", type=float, default=0.9)
    p.add_argument("--wd", type=float, default=0.0)
    p.add_argument("--epochs", type=int, required=True)
    p.add_argument("--batch_size", type=int, required=True)
    p.add_argument("--opt", type=str, default="sgd")
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--sched", type=str, default="none", choices=["multistep", "cosine", 'line',"none"])
    p.add_argument("--wandb_project", type=str, default="double momentum")

    # 翻译任务最少需要控制这两个；不给也能跑，但不好用
    p.add_argument("--sp_vocab", type=int, default=16000)
    p.add_argument("--max_len", type=int, default=100)

    args = p.parse_args()
    device = torch.device(args.device)

    wandb.init(
        project=args.wandb_project,
        group="iwslt14-de-en-lstm",
        name=args.opt,
        config=vars(args),
    )

    # dataset
    ds = load_dataset("iwslt2017", "iwslt2017-de-en", trust_remote_code=True)
    train_raw, valid_raw, test_raw = ds["train"], ds["validation"], ds["test"]

    def map_ex(ex):
        return {"de": ex["translation"]["de"], "en": ex["translation"]["en"]}

    train = train_raw.map(map_ex, remove_columns=train_raw.column_names)
    valid = valid_raw.map(map_ex, remove_columns=valid_raw.column_names)
    test = test_raw.map(map_ex, remove_columns=test_raw.column_names)

    # sentencepiece (cache under ./data/iwslt_spm/)
    sp_dir = "./data/iwslt_spm"
    src_spm_path = ensure_spm((ex["de"] for ex in train), os.path.join(sp_dir, "src"), args.sp_vocab)
    tgt_spm_path = ensure_spm((ex["en"] for ex in train), os.path.join(sp_dir, "tgt"), args.sp_vocab)
    sp_src = spm.SentencePieceProcessor(model_file=src_spm_path)
    sp_tgt = spm.SentencePieceProcessor(model_file=tgt_spm_path)

    collate = lambda xs: collate_fn(xs, sp_src, sp_tgt, args.max_len, device)

    train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=True, num_workers=0, collate_fn=collate)
    valid_loader = DataLoader(valid, batch_size=256, shuffle=False, num_workers=0, collate_fn=collate)
    test_loader  = DataLoader(test,  batch_size=256, shuffle=False, num_workers=0, collate_fn=collate)

    # model
    model = Seq2Seq(
        src_vocab=sp_src.get_piece_size(),
        tgt_vocab=sp_tgt.get_piece_size(),
        emb_dim=512,
        hid_dim=512,
        num_layers=1,
        dropout=0.3,
    ).to(device)

    optimizer = build_optimizer(args.opt, model.parameters(), args.lr, args.momentum, args.wd, args.beta)

    if args.sched == "multistep":
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 20, 30], gamma=0.5)
    elif args.sched == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    elif args.sched == 'line':
        total_steps = args.epochs * len(train_loader)
        scheduler = WarmupLinearDecayLR(optimizer, total_steps=total_steps, warmup_steps=4000, base_lr=args.lr)
    else:
        scheduler = None

    # loss_fn = nn.CrossEntropyLoss(ignore_index=0)
    loss_fn = nn.CrossEntropyLoss(ignore_index=0, label_smoothing=0.1)

    scaler = torch.cuda.amp.GradScaler()
    global_step = 0
    for epoch in range(args.epochs):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}", leave=False)

        running_loss = 0.0
        running_tok = 0

        for batch in pbar:
            optimizer.zero_grad(set_to_none=True)
            # logits = model(batch["src"], batch["src_len"], batch["tgt_in"])
            # loss = loss_fn(logits.view(-1, logits.size(-1)), batch["tgt_out"].view(-1))
            # loss.backward()
            # optimizer.step()
            with torch.cuda.amp.autocast(dtype=torch.float16):
                logits = model(batch["src"], batch["src_len"], batch["tgt_in"])
                loss = loss_fn(logits.view(-1, logits.size(-1)), batch["tgt_out"].view(-1))
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            if args.sched == 'line':
                scheduler.step()


            ntok = (batch["tgt_out"] != 0).sum().item()
            running_loss += loss.item() * ntok
            running_tok += ntok

            cur_lr = optimizer.param_groups[0]["lr"]
            ppl = math.exp(min(loss.item(), 20.0))
            pbar.set_postfix({"loss": f"{loss.item():.3f}", "ppl": f"{ppl:.2f}", "lr": f"{cur_lr:g}"})

            wandb.log(
                {"train/loss": loss.item(), "train/ppl": ppl, "train/lr": cur_lr, "global_step": global_step},
                step=global_step,
            )
            global_step += 1

        train_loss_avg = running_loss / max(running_tok, 1)
        train_ppl_avg = math.exp(min(train_loss_avg, 20.0))

        valid_loss, valid_ppl = evaluate(model, valid_loader, device, loss_fn)
        bleu_subset = eval_bleu_subset(model, test_loader, sp_tgt, args.max_len, num_batches=50)

        wandb.log(
            {
                "epoch": epoch,
                "train/loss_avg": train_loss_avg,
                "train/ppl_avg": train_ppl_avg,
                "valid/loss": valid_loss,
                "valid/ppl": valid_ppl,
                "test/bleu_greedy@subset": bleu_subset,
            },
            step=global_step,
        )
        print(f"Epoch {epoch+1}/{args.epochs} | valid_ppl={valid_ppl:.2f} bleu(subset)={bleu_subset:.2f}")

        if scheduler is not None and args.sched != 'line':
            scheduler.step()

    wandb.finish()


if __name__ == "__main__":
    main()