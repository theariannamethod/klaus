#!/usr/bin/env python3
"""Pre-train BPE and tokenize data for HE/RU/FR while EN trains on GPU."""
import os, sys, time, lzma
import numpy as np
import sentencepiece as spm

VOCAB = 4096
DATA_DIR = '/home/ubuntu/klaus/data'
WEIGHTS_DIR = '/home/ubuntu/klaus/weights'
MAX_TOKENS = 10_000_000  # 10M tokens per lang

LANGS = {
    'he': {'file': 'he_fineweb2.txt', 'xz': False, 'min_len': 10},
    'ru': {'file': 'ru_fineweb2.txt', 'xz': False, 'min_len': 10},
    'fr': {'file': 'fr_fineweb2.txt', 'xz': False, 'min_len': 10},
}

def prep(lang):
    lc = LANGS[lang]
    src = f'{DATA_DIR}/{lc["file"]}'
    bpe_prefix = f'{WEIGHTS_DIR}/{lang}_bpe_{VOCAB}'
    bpe_path = f'{bpe_prefix}.model'
    bin_path = f'{DATA_DIR}/{lang}_train.bin'

    # Step 1: BPE training text
    bpe_txt = f'{DATA_DIR}/{lang}_bpe_train.txt'
    if not os.path.exists(bpe_txt):
        print(f'[{lang.upper()}] Extracting BPE training text...')
        n = 0
        with open(src, encoding='utf-8', errors='ignore') as fin, open(bpe_txt, 'w') as fout:
            for line in fin:
                line = line.strip()
                if line and len(line) >= lc['min_len']:
                    fout.write(line + '\n')
                    n += 1
                    if n >= 2_000_000:
                        break
        print(f'  {n:,} lines → {os.path.getsize(bpe_txt)/1e6:.0f}MB')
    else:
        print(f'[{lang.upper()}] BPE text exists: {os.path.getsize(bpe_txt)/1e6:.0f}MB')

    # Step 2: Train BPE
    if not os.path.exists(bpe_path):
        print(f'[{lang.upper()}] Training BPE {VOCAB}...')
        t0 = time.time()
        spm.SentencePieceTrainer.train(
            input=bpe_txt, model_prefix=bpe_prefix, vocab_size=VOCAB,
            model_type='bpe', character_coverage=0.9999, num_threads=8,
            max_sentence_length=4096, shuffle_input_sentence=True, byte_fallback=True,
        )
        print(f'  Done in {time.time()-t0:.0f}s')
    else:
        print(f'[{lang.upper()}] BPE exists: {bpe_path}')

    sp = spm.SentencePieceProcessor()
    sp.load(bpe_path)
    print(f'[{lang.upper()}] BPE: {sp.get_piece_size()} pieces')

    # Step 3: Tokenize
    if os.path.exists(bin_path) and os.path.getsize(bin_path) >= MAX_TOKENS * 2:
        print(f'[{lang.upper()}] Tokens exist: {os.path.getsize(bin_path)//2:,}')
        return

    print(f'[{lang.upper()}] Tokenizing {MAX_TOKENS:,} tokens...')
    tokens = []
    t0 = time.time()
    with open(src, encoding='utf-8', errors='ignore') as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line or len(line) < lc['min_len']:
                continue
            tokens.extend(sp.encode(line))
            if len(tokens) >= MAX_TOKENS:
                tokens = tokens[:MAX_TOKENS]
                break
            if (i + 1) % 200_000 == 0:
                print(f'  {len(tokens):,} tokens...')

    arr = np.array(tokens, dtype=np.uint16)
    arr.tofile(bin_path)
    print(f'  {len(tokens):,} tokens in {time.time()-t0:.0f}s → {os.path.getsize(bin_path)/1e6:.1f}MB')

if __name__ == '__main__':
    langs = sys.argv[1:] if len(sys.argv) > 1 else ['he', 'ru', 'fr']
    for lang in langs:
        print(f'\n{"="*50}')
        prep(lang)
    print('\nAll done.')
