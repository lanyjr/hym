import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, vocab_size, emb_dim=512, hid_dim=512, num_layers=1, dropout=0.2):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.rnn = nn.LSTM(emb_dim, hid_dim, num_layers=num_layers, batch_first=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_len):
        x = self.dropout(self.emb(src))  # [B,S,E]
        packed = nn.utils.rnn.pack_padded_sequence(x, src_len.cpu(), batch_first=True, enforce_sorted=False)
        out_p, (h, c) = self.rnn(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out_p, batch_first=True)  # [B,S,H]
        return out, (h, c)


class LuongAttention(nn.Module):
    def __init__(self, hid_dim):
        super().__init__()
        self.scale = 1.0 / math.sqrt(hid_dim)

    def forward(self, dec_h, enc_out, src_mask):
        # dec_h: [B,H], enc_out: [B,S,H], src_mask: [B,S] bool
        scores = torch.bmm(enc_out, dec_h.unsqueeze(2)).squeeze(2) * self.scale  # [B,S]
        # scores = scores.masked_fill(~src_mask, -1e9)
        scores = scores.masked_fill(~src_mask, torch.finfo(scores.dtype).min)
        attn = F.softmax(scores, dim=1)
        ctx = torch.bmm(attn.unsqueeze(1), enc_out).squeeze(1)  # [B,H]
        return ctx


class Decoder(nn.Module):
    def __init__(self, vocab_size, emb_dim=512, hid_dim=512, num_layers=1, dropout=0.2, tie_weights=True):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.rnn = nn.LSTM(emb_dim, hid_dim, num_layers=num_layers, batch_first=True)
        self.attn = LuongAttention(hid_dim)

        # NEW: map [h;ctx] (2*hid_dim) -> emb_dim
        self.pre_out = nn.Linear(hid_dim + hid_dim, emb_dim, bias=False)

        # NEW: proj now takes emb_dim, so it can tie with embedding
        self.proj = nn.Linear(emb_dim, vocab_size, bias=False)

        if tie_weights:
            if self.proj.weight.shape != self.emb.weight.shape:
                raise ValueError("Weight tying requires proj.weight and emb.weight to have same shape")
            self.proj.weight = self.emb.weight

        self.dropout = nn.Dropout(dropout)

    def forward(self, tgt_in, enc_out, enc_state, src_mask):
        x = self.dropout(self.emb(tgt_in))
        dec_out, _ = self.rnn(x, enc_state)

        logits = []
        for t in range(dec_out.size(1)):
            h_t = dec_out[:, t, :]
            ctx = self.attn(h_t, enc_out, src_mask)
            z = self.pre_out(torch.cat([h_t, ctx], dim=1))   # NEW
            logits.append(self.proj(z))                      # NEW
        return torch.stack(logits, dim=1)


class Seq2Seq(nn.Module):
    def __init__(self, src_vocab, tgt_vocab, emb_dim=512, hid_dim=512, num_layers=1, dropout=0.2):
        super().__init__()
        self.enc = Encoder(src_vocab, emb_dim, hid_dim, num_layers, dropout)
        self.dec = Decoder(tgt_vocab, emb_dim, hid_dim, num_layers, dropout)

    def forward(self, src, src_len, tgt_in):
        enc_out, enc_state = self.enc(src, src_len)
        src_mask = (src != 0)
        return self.dec(tgt_in, enc_out, enc_state, src_mask)

    @torch.no_grad()
    def greedy_decode(self, src, src_len, max_len, bos_id=1, eos_id=2):
        self.eval()
        enc_out, state = self.enc(src, src_len)
        src_mask = (src != 0)

        B = src.size(0)
        ys = torch.full((B, 1), bos_id, dtype=torch.long, device=src.device)

        for _ in range(max_len):
            x = self.dec.dropout(self.dec.emb(ys[:, -1:]))  # [B,1,E]
            dec_out, state = self.dec.rnn(x, state)         # [B,1,H]
            h_t = dec_out[:, 0, :]
            ctx = self.dec.attn(h_t, enc_out, src_mask)
            # logit = self.dec.proj(torch.cat([h_t, ctx], dim=1))
            z = self.dec.pre_out(torch.cat([h_t, ctx], dim=1))
            logit = self.dec.proj(z)
            next_tok = logit.argmax(dim=1, keepdim=True)
            ys = torch.cat([ys, next_tok], dim=1)
            if (next_tok.squeeze(1) == eos_id).all():
                break

        return ys[:, 1:]  # drop BOS