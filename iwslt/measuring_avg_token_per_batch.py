import math
import sentencepiece as spm
from torch.utils.data import DataLoader
from datasets import load_dataset


# ====== hard-coded config ======
SRC_SPM_MODEL = "./data/iwslt_spm/src/spm_16000.model"
TGT_SPM_MODEL = "./data/iwslt_spm/tgt/spm_16000.model"

MAX_LEN = 100
# BATCH_SIZE = 128
BATCH_SIZE = 177
NUM_BATCHES = 300
# ===============================

EOS_ID = 2


def collate_count_tokens(examples, sp_src, sp_tgt, max_len):
    src_tokens = 0
    tgt_tokens = 0
    for ex in examples:
        src = sp_src.EncodeAsIds(ex["de"])[: max_len - 1] + [EOS_ID]
        tgt = sp_tgt.EncodeAsIds(ex["en"])[: max_len - 1] + [EOS_ID]
        src_tokens += len(src)
        tgt_tokens += len(tgt)
    return {"src_tokens": src_tokens, "tgt_tokens": tgt_tokens, "nsent": len(examples)}


class RunningStats:
    # Welford
    def __init__(self):
        self.n = 0
        self.mean = 0.0
        self.M2 = 0.0

    def update(self, x: float):
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        delta2 = x - self.mean
        self.M2 += delta * delta2

    def get_mean_std(self):
        if self.n < 2:
            return self.mean, 0.0
        var = self.M2 / (self.n - 1)
        return self.mean, math.sqrt(max(var, 0.0))


def main():
    sp_src = spm.SentencePieceProcessor(model_file=SRC_SPM_MODEL)
    sp_tgt = spm.SentencePieceProcessor(model_file=TGT_SPM_MODEL)

    ds = load_dataset("iwslt2017", "iwslt2017-de-en", trust_remote_code=True)
    train_raw = ds["train"]

    def map_ex(ex):
        return {"de": ex["translation"]["de"], "en": ex["translation"]["en"]}

    train = train_raw.map(map_ex, remove_columns=train_raw.column_names)

    loader = DataLoader(
        train,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        collate_fn=lambda xs: collate_count_tokens(xs, sp_src, sp_tgt, MAX_LEN),
    )

    src_stats = RunningStats()
    tgt_stats = RunningStats()
    sent_stats = RunningStats()

    tot_src = tot_tgt = tot_sent = 0
    nb = 0
    for batch in loader:
        src_tokens = float(batch["src_tokens"])
        tgt_tokens = float(batch["tgt_tokens"])
        nsent = float(batch["nsent"])

        tot_src += src_tokens
        tot_tgt += tgt_tokens
        tot_sent += nsent
        nb += 1

        src_stats.update(src_tokens)
        tgt_stats.update(tgt_tokens)
        sent_stats.update(nsent)

        if nb >= NUM_BATCHES:
            break

    avg_tgt_per_sent = tot_tgt / tot_sent

    src_mean, src_std = src_stats.get_mean_std()
    tgt_mean, tgt_std = tgt_stats.get_mean_std()
    sent_mean, sent_std = sent_stats.get_mean_std()

    print(f"Measured over {nb} batches (batch_size={BATCH_SIZE}, max_len={MAX_LEN})")
    print(f"sentences/batch: mean={sent_mean:.2f} std={sent_std:.2f}")
    print(f"src tokens/batch: mean={src_mean:.2f} std={src_std:.2f}")
    print(f"tgt tokens/batch: mean={tgt_mean:.2f} std={tgt_std:.2f}")
    print(f"avg tgt tokens/sentence: {avg_tgt_per_sent:.2f}")

    target = 4096
    rec_bs = int(round(BATCH_SIZE * (target / max(tgt_mean, 1e-9))))
    # print(f"Recommended batch_size for ~{target} tgt tokens/batch: {rec_bs}")
    print(f"tgt tokens/batch std at batch_size={BATCH_SIZE}: {tgt_std:.2f}")


if __name__ == "__main__":
    main()