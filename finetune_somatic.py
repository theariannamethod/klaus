#!/usr/bin/env python3
"""Fine-tune Klaus LMs on somatic corpus. Low LR, few epochs, preserve general capability."""
import os, sys, math, time, argparse, random
import torch
import torch.nn as nn
import torch.nn.functional as F
import sentencepiece as spm
sys.path.insert(0, '/home/ubuntu/klaus')
from train_chambers import KlausLM, WEIGHTS_DIR, DIM, N_HEADS, N_LAYERS, HDIM, VOCAB, MAX_SEQ

BATCH = 16
LR = 1e-4  # very low — preserve general capability
MAX_EPOCHS = 15
LANGS = ['en', 'ru', 'fr', 'he']
DATA_DIR = '/home/ubuntu/klaus/data'


def load_corpus(lang, sp):
    """Load somatic corpus + original training data (mixed)."""
    path = DATA_DIR + '/%s_somatic_corpus.txt' % lang
    if not os.path.exists(path):
        print('  %s: no somatic corpus found' % lang.upper())
        return []

    lines = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if len(line) > 5:
                lines.append(line)

    # Tokenize
    all_ids = []
    for line in lines:
        ids = sp.encode(line)
        if len(ids) > 3:
            all_ids.append(ids)

    print('  %s: %d somatic sentences, %d tokens avg' % (
        lang.upper(), len(all_ids),
        sum(len(x) for x in all_ids) // max(1, len(all_ids))))
    return all_ids


def make_batches(all_ids, batch_size, seq_len=128):
    """Create training batches from tokenized sentences."""
    # Concatenate all sentences with separator
    flat = []
    for ids in all_ids:
        flat.extend(ids[:seq_len])

    # Split into chunks
    batches = []
    for i in range(0, len(flat) - seq_len, seq_len // 2):
        chunk = flat[i:i + seq_len]
        if len(chunk) == seq_len:
            batches.append(chunk)

    # Shuffle and group into batches
    random.shuffle(batches)
    result = []
    for i in range(0, len(batches) - batch_size, batch_size):
        batch = batches[i:i + batch_size]
        result.append(torch.tensor(batch, dtype=torch.long))

    return result


def train_lang(lang, epochs=MAX_EPOCHS):
    dev = torch.device('cuda')

    # Load tokenizer
    sp = spm.SentencePieceProcessor()
    sp.load(WEIGHTS_DIR + '/klaus_%s_bpe.model' % lang)

    # Load somatic corpus
    all_ids = load_corpus(lang, sp)
    if not all_ids:
        return

    # Also load some general text to prevent catastrophic forgetting
    gen_path = DATA_DIR + '/%s_bpe_train.txt' % lang
    if os.path.exists(gen_path):
        gen_lines = []
        with open(gen_path) as f:
            for i, line in enumerate(f):
                if i >= 2000:
                    break  # 5K general lines
                ids = sp.encode(line.strip())
                if len(ids) > 3:
                    gen_lines.append(ids)
        print('  %s: +%d general lines (anti-forgetting)' % (lang.upper(), len(gen_lines)))
        all_ids.extend(gen_lines)
        random.shuffle(all_ids)

    batches = make_batches(all_ids, BATCH, seq_len=64)
    print('  %s: %d batches' % (lang.upper(), len(batches)))

    if not batches:
        print('  %s: not enough data' % lang.upper())
        return

    # Load model
    lm = KlausLM().to(dev)
    ck_path = WEIGHTS_DIR + '/klaus_somatic_lms.pt'
    ck = torch.load(ck_path, map_location=dev, weights_only=False)
    if 'lms' in ck and lang in ck['lms']:
        lm.load_state_dict(ck['lms'][lang])
        print('  %s: LM loaded from chambers checkpoint' % lang.upper())
    else:
        sd = torch.load(WEIGHTS_DIR + '/klaus_%s.pt' % lang, map_location=dev, weights_only=False)
        lm.load_state_dict(sd['model'])
        print('  %s: LM loaded from individual checkpoint' % lang.upper())

    opt = torch.optim.AdamW(lm.parameters(), lr=LR, weight_decay=0.01)
    t0 = time.time()
    best_loss = 999

    for epoch in range(epochs):
        random.shuffle(batches)
        total_loss = 0
        n = 0
        for bi, batch in enumerate(batches):
            batch = batch.to(dev)
            logits, _ = lm(batch)
            # Shift: predict next token
            loss = F.cross_entropy(
                logits[:, :-1].contiguous().view(-1, VOCAB),
                batch[:, 1:].contiguous().view(-1)
            )
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(lm.parameters(), 1.0)
            opt.step()
            total_loss += loss.item()
            n += 1

            if (bi + 1) % 50 == 0:
                avg = total_loss / n
                print('  %s epoch %d step %d/%d | loss=%.4f | %ds' % (
                    lang.upper(), epoch + 1, bi + 1, len(batches), avg, time.time() - t0))
                sys.stdout.flush()

        avg_loss = total_loss / max(n, 1)
        print('  %s epoch %d DONE | loss=%.4f | %ds' % (
            lang.upper(), epoch + 1, avg_loss, time.time() - t0))

        if avg_loss < best_loss:
            best_loss = avg_loss
            # Save back into chambers checkpoint
            ck['lms'][lang] = lm.state_dict()
            torch.save(ck, ck_path)
            print('  SAVED (loss=%.4f)' % avg_loss)

    # Test generation
    lm.eval()
    test_prompts = {
        'en': ['my chest feels', 'I am shaking', 'a warmth in my'],
        'ru': ['у меня в груди', 'я дрожу', 'тепло внутри'],
        'fr': ['ma poitrine', 'je tremble', 'une chaleur dans'],
        'he': ['החזה שלי', 'אני רועד', 'חום בתוך'],
    }
    prompts = test_prompts.get(lang, test_prompts['en'])
    print('\n  Generation test:')
    for prompt in prompts:
        ids = sp.encode(prompt)
        tokens = list(ids)
        for _ in range(20):
            input_t = torch.tensor([tokens], device=dev)
            with torch.no_grad():
                logits, _ = lm(input_t)
                probs = F.softmax(logits[0, -1] / 0.7, dim=0)
                next_tok = torch.multinomial(probs, 1).item()
            if next_tok == 0:
                break
            tokens.append(next_tok)
        gen = sp.decode(tokens[len(ids):])
        print('    "%s" → %s' % (prompt, gen.strip()[:60]))

    print()


def main():
    print('=' * 60)
    print(' Klaus Somatic Fine-Tuning')
    print(' LR=%s, epochs=%d, batch=%d' % (LR, MAX_EPOCHS, BATCH))
    print('=' * 60 + '\n')

    for lang in LANGS:
        print('\n--- %s ---' % lang.upper())
        train_lang(lang)

    print('\n' + '=' * 60)
    print(' DONE. All LMs fine-tuned on somatic corpus.')
    print('=' * 60)


if __name__ == '__main__':
    main()
