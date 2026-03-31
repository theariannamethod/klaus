#!/usr/bin/env python3
"""
Klaus — Pure LM training. One transformer per language, ~10.5M params.
Pre-tokenizes into .bin, trains with random sampling (nanoGPT-style).

Usage:
  python train_klaus.py --lang en [--tokens 20000000] [--steps 50000]
"""
import os, sys, math, lzma, shutil, time, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import sentencepiece as spm

# ─── Config ───────────────────────────────────────────────────────────
VOCAB     = 4096
DIM       = 384
N_HEADS   = 6
N_LAYERS  = 6
HDIM      = 768
MAX_SEQ   = 256
DROPOUT   = 0.1
LR        = 3e-4
MIN_LR    = 3e-5
BATCH     = 64
ACCUM     = 2       # effective batch = 128
WARMUP    = 2000
WD        = 0.1

DATA_DIR    = '/home/ubuntu/klaus/data'
WEIGHTS_DIR = '/home/ubuntu/klaus/weights'

LANGS = {
    'en': {'file': 'en_fineweb.txt.xz', 'xz': True, 'min_len': 20,
            'prompts': ['The meaning of life is', 'I feel so angry because',
                        'She walked into the room and', 'Once upon a time there was a']},
    'he': {'file': 'he_fineweb2.txt', 'xz': False, 'min_len': 10,
            'prompts': ['היום אני מרגיש', 'השמש זורחת על', 'אני אוהב את', 'הבוקר היה קשה כי']},
    'ru': {'file': 'ru_fineweb2.txt', 'xz': False, 'min_len': 10,
            'prompts': ['Сегодня я чувствую себя', 'В этом городе есть',
                        'Она посмотрела на него и', 'Жизнь прекрасна когда']},
    'fr': {'file': 'fr_fineweb2.txt', 'xz': False, 'min_len': 10,
            'prompts': ['La vie est belle quand', 'Je me sens tellement',
                        'Elle marchait dans la rue', 'Le soleil brillait sur']},
}

# ─── Model ────────────────────────────────────────────────────────────
class RMSNorm(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.w = nn.Parameter(torch.ones(d))
    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * self.w

class Attn(nn.Module):
    def __init__(self):
        super().__init__()
        self.nh = N_HEADS
        self.hd = DIM // N_HEADS
        self.wq = nn.Linear(DIM, DIM, bias=False)
        self.wk = nn.Linear(DIM, DIM, bias=False)
        self.wv = nn.Linear(DIM, DIM, bias=False)
        self.wo = nn.Linear(DIM, DIM, bias=False)

    def forward(self, x):
        B, S, _ = x.shape
        q = self.wq(x).view(B, S, self.nh, self.hd).transpose(1, 2)
        k = self.wk(x).view(B, S, self.nh, self.hd).transpose(1, 2)
        v = self.wv(x).view(B, S, self.nh, self.hd).transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True,
                                             dropout_p=DROPOUT if self.training else 0.0)
        return self.wo(out.transpose(1, 2).contiguous().view(B, S, -1))

class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1 = RMSNorm(DIM)
        self.attn = Attn()
        self.ln2 = RMSNorm(DIM)
        self.w1 = nn.Linear(DIM, HDIM, bias=False)
        self.w2 = nn.Linear(DIM, HDIM, bias=False)
        self.w3 = nn.Linear(HDIM, DIM, bias=False)
        self.drop = nn.Dropout(DROPOUT)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        h = self.ln2(x)
        x = x + self.drop(self.w3(F.silu(self.w1(h)) * self.w2(h)))
        return x

class Klaus(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok = nn.Embedding(VOCAB, DIM)
        self.pos = nn.Embedding(MAX_SEQ, DIM)
        self.blocks = nn.ModuleList([Block() for _ in range(N_LAYERS)])
        self.norm = RMSNorm(DIM)
        self.head = nn.Linear(DIM, VOCAB, bias=False)
        self.head.weight = self.tok.weight
        self.drop = nn.Dropout(DROPOUT)

    def forward(self, x):
        B, S = x.shape
        h = self.drop(self.tok(x) + self.pos(torch.arange(S, device=x.device)))
        for b in self.blocks:
            h = b(h)
        return self.head(self.norm(h))

# ─── Data ─────────────────────────────────────────────────────────────
def open_src(path, xz):
    if xz:
        return lzma.open(path, 'rt', encoding='utf-8', errors='ignore')
    return open(path, encoding='utf-8', errors='ignore')

def ensure_bpe(lang):
    prefix = f'{WEIGHTS_DIR}/{lang}_bpe_{VOCAB}'
    path = f'{prefix}.model'
    if os.path.exists(path):
        sp = spm.SentencePieceProcessor()
        sp.load(path)
        print(f'BPE {lang.upper()}: {sp.get_piece_size()} pieces (cached)')
        return sp

    lc = LANGS[lang]
    src = f'{DATA_DIR}/{lc["file"]}'
    bpe_txt = f'{DATA_DIR}/{lang}_bpe_train.txt'

    if not os.path.exists(bpe_txt):
        print(f'Preparing BPE text for {lang.upper()}...')
        n = 0
        with open_src(src, lc['xz']) as fin, open(bpe_txt, 'w') as fout:
            for line in fin:
                line = line.strip()
                if line and len(line) >= lc['min_len']:
                    fout.write(line + '\n')
                    n += 1
                    if n >= 2_000_000:
                        break
        print(f'  {n:,} lines')

    print(f'Training BPE {lang.upper()} ({VOCAB})...')
    spm.SentencePieceTrainer.train(
        input=bpe_txt, model_prefix=prefix, vocab_size=VOCAB,
        model_type='bpe', character_coverage=0.9999, num_threads=8,
        max_sentence_length=4096, shuffle_input_sentence=True, byte_fallback=True,
    )
    sp = spm.SentencePieceProcessor()
    sp.load(path)
    print(f'  {sp.get_piece_size()} pieces')
    return sp

def tokenize(lang, sp, max_tokens):
    bin_path = f'{DATA_DIR}/{lang}_train.bin'
    if os.path.exists(bin_path):
        existing = os.path.getsize(bin_path) // 2
        if existing >= max_tokens:
            print(f'{lang.upper()} tokens: {existing:,} (cached)')
            return bin_path

    lc = LANGS[lang]
    src = f'{DATA_DIR}/{lc["file"]}'
    print(f'Tokenizing {lang.upper()} → {max_tokens:,} tokens...')

    tokens = []
    t0 = time.time()
    with open_src(src, lc['xz']) as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line or len(line) < lc['min_len']:
                continue
            tokens.extend(sp.encode(line))
            if len(tokens) >= max_tokens:
                tokens = tokens[:max_tokens]
                break
            if (i + 1) % 200_000 == 0:
                print(f'  {len(tokens):,} tokens...')

    arr = np.array(tokens, dtype=np.uint16)
    arr.tofile(bin_path)
    print(f'  {len(tokens):,} tokens in {time.time()-t0:.0f}s → {os.path.getsize(bin_path)/1e6:.1f}MB')
    return bin_path

def get_batch(data, lo, hi, device):
    ix = np.random.randint(lo, hi - MAX_SEQ - 1, size=BATCH)
    x = torch.stack([torch.from_numpy(data[i:i+MAX_SEQ].astype(np.int64)) for i in ix]).to(device)
    y = torch.stack([torch.from_numpy(data[i+1:i+1+MAX_SEQ].astype(np.int64)) for i in ix]).to(device)
    return x, y

# ─── Generate ─────────────────────────────────────────────────────────
@torch.no_grad()
def generate(model, sp, prompt, n=60, temp=0.8, top_k=40):
    was_training = model.training
    model.eval()
    ids = sp.encode(prompt)
    for _ in range(n):
        ctx = ids[-MAX_SEQ:]
        logits = model(torch.tensor([ctx], device='cuda'))[0, -1] / temp
        topk_v, topk_i = logits.topk(top_k)
        nid = topk_i[torch.multinomial(F.softmax(topk_v, dim=-1), 1)].item()
        if nid <= 2:
            break
        ids.append(nid)
    if was_training:
        model.train()
    return sp.decode(ids)

# ─── Train ────────────────────────────────────────────────────────────
def train(lang, max_tokens, total_steps):
    dev = torch.device('cuda')
    print(f'\n{"="*70}')
    print(f' KLAUS {lang.upper()} — Pure LM Training')
    print(f' {max_tokens//1_000_000}M tokens, {total_steps//1000}K steps')
    print(f'{"="*70}\n')

    sp = ensure_bpe(lang)
    bin_path = tokenize(lang, sp, max_tokens)

    # Load memmap
    data = np.memmap(bin_path, dtype=np.uint16, mode='r')
    n = len(data)
    val_n = min(n // 50, 500_000)  # 2% or 500K max
    train_n = n - val_n
    print(f'Data split: train={train_n:,} | val={val_n:,}')

    # Model
    raw_model = Klaus().to(dev)
    n_params = sum(p.numel() for p in raw_model.parameters())
    print(f'Model: {n_params:,} params ({n_params/1e6:.1f}M)')

    model = torch.compile(raw_model)
    print('torch.compile: ON')

    # Optimizer
    opt = torch.optim.AdamW(raw_model.parameters(), lr=LR, weight_decay=WD, betas=(0.9, 0.95))

    def lr_fn(step):
        if step < WARMUP:
            return step / WARMUP
        progress = (step - WARMUP) / max(1, total_steps - WARMUP)
        coeff = 0.5 * (1 + math.cos(math.pi * progress))
        return MIN_LR / LR + (1 - MIN_LR / LR) * coeff

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_fn)

    eff = BATCH * ACCUM * MAX_SEQ
    passes = total_steps * eff / train_n
    print(f'Effective batch: {BATCH*ACCUM} seqs = {eff:,} tokens/step')
    print(f'Data passes: ~{passes:.1f} | Warmup: {WARMUP} | LR: {LR}→{MIN_LR}')
    print('─' * 70)

    best_val = float('inf')
    losses = []
    t0 = time.time()

    for step in range(1, total_steps + 1):
        model.train()
        opt.zero_grad(set_to_none=True)

        step_loss = 0.0
        for _ in range(ACCUM):
            x, y = get_batch(data, 0, train_n, dev)
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                logits = model(x)
                loss = F.cross_entropy(logits.view(-1, VOCAB), y.view(-1)) / ACCUM
            loss.backward()
            step_loss += loss.item()

        torch.nn.utils.clip_grad_norm_(raw_model.parameters(), 1.0)
        opt.step()
        sched.step()
        losses.append(step_loss)

        # Log every 100 steps
        if step % 100 == 0:
            avg = sum(losses[-100:]) / len(losses[-100:])
            dt = time.time() - t0
            tps = step * eff / dt
            print(f'step {step:6d}/{total_steps} | train={avg:.4f} | lr={sched.get_last_lr()[0]:.2e} | {tps/1e6:.1f}M t/s')
            sys.stdout.flush()

        # Eval every 2000 steps
        if step % 2000 == 0 or step == total_steps:
            model.eval()
            vl = []
            with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.bfloat16):
                for _ in range(50):
                    x, y = get_batch(data, train_n, n, dev)
                    logits = model(x)
                    vl.append(F.cross_entropy(logits.view(-1, VOCAB), y.view(-1)).item())
            val_avg = sum(vl) / len(vl)
            tr_avg = sum(losses[-500:]) / min(len(losses), 500)

            print(f'\n  >>> EVAL step {step}')
            print(f'  >>> TRAIN = {tr_avg:.4f}  |  VAL = {val_avg:.4f}')

            for p in LANGS[lang]['prompts']:
                out = generate(raw_model, sp, p)
                print(f'  >> {out[:200]}')
            print()
            sys.stdout.flush()

            if val_avg < best_val:
                best_val = val_avg
                path = f'{WEIGHTS_DIR}/klaus_{lang}.pt'
                torch.save({
                    'model': raw_model.state_dict(),
                    'step': step, 'train_loss': tr_avg, 'val_loss': val_avg,
                    'cfg': {'vocab': VOCAB, 'dim': DIM, 'n_heads': N_HEADS,
                            'n_layers': N_LAYERS, 'hdim': HDIM, 'max_seq': MAX_SEQ}
                }, path)
                shutil.copy2(path, f'{WEIGHTS_DIR}/klaus_{lang}_backup.pt')
                bpe_src = f'{WEIGHTS_DIR}/{lang}_bpe_{VOCAB}.model'
                shutil.copy2(bpe_src, f'{WEIGHTS_DIR}/klaus_{lang}_bpe.model')
                print(f'  SAVED best val={val_avg:.4f}\n')

    dt = time.time() - t0
    final_train = sum(losses[-500:]) / min(len(losses), 500)
    print(f'{"="*70}')
    print(f'DONE {lang.upper()}: {total_steps} steps in {dt/60:.1f} min')
    print(f'Final: train={final_train:.4f} | best_val={best_val:.4f}')
    print(f'Weights: {WEIGHTS_DIR}/klaus_{lang}.pt')
    print(f'BPE: {WEIGHTS_DIR}/klaus_{lang}_bpe.model')
    print(f'{"="*70}')

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--lang', required=True, choices=['en', 'he', 'ru', 'fr'])
    p.add_argument('--tokens', type=int, default=10_000_000,
                   help='Tokens to pre-tokenize (default: 10M)')
    p.add_argument('--steps', type=int, default=100_000,
                   help='Training steps (default: 100K)')
    a = p.parse_args()
    train(a.lang, a.tokens, a.steps)
