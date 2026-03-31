#!/usr/bin/env python3
"""Pre-tokenize HE/RU/FR with 200M tokens each (CPU only, no GPU needed)."""
import sys, os, time
sys.path.insert(0, '/home/ubuntu/klaus')
from train_klaus import ensure_bpe, tokenize

TARGET = 200_000_000

for lang in ['he', 'ru', 'fr']:
    print(f'\n{"="*60}')
    print(f'  Preparing {lang.upper()} — {TARGET//1_000_000}M tokens')
    print(f'{"="*60}')
    t0 = time.time()
    sp = ensure_bpe(lang)
    tokenize(lang, sp, TARGET)
    dt = time.time() - t0
    bin_path = f'/home/ubuntu/klaus/data/{lang}_train.bin'
    actual = os.path.getsize(bin_path) // 2
    print(f'{lang.upper()} DONE: {actual:,} tokens in {dt:.0f}s ({os.path.getsize(bin_path)/1e6:.0f}MB)')

print(f'\n{"="*60}')
print('ALL PREP DONE')
print(f'{"="*60}')
