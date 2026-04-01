#!/usr/bin/env python3
"""Klaus generative inference — somatic text generation via Dario equation."""
import torch, sys, os, math
import torch.nn.functional as F
import sentencepiece as spm
sys.path.insert(0, '/home/ubuntu/klaus')
from train_chambers import KlausFull, CNAMES, LANGS, WEIGHTS_DIR

# Somatic seed words with chamber affinities [FEAR,LOVE,RAGE,VOID,FLOW,COMPLEX]
SEED_EN = {
    'pulse':(0.4,0.0,0.8,0.0,0.3,0.2), 'tremor':(0.8,0.0,0.2,0.2,0.0,0.3),
    'burning':(0.3,0.1,0.9,0.0,0.1,0.2), 'clenching':(0.4,0.0,0.8,0.1,0.0,0.3),
    'aching':(0.2,0.1,0.2,0.7,0.0,0.3), 'tightness':(0.6,0.0,0.5,0.3,0.0,0.3),
    'sinking':(0.5,0.0,0.0,0.9,0.0,0.2), 'heaviness':(0.2,0.0,0.1,0.8,0.0,0.2),
    'shaking':(0.8,0.0,0.4,0.1,0.0,0.3), 'freezing':(0.7,0.0,0.0,0.5,0.0,0.2),
    'warmth':(0.0,0.9,0.0,0.0,0.6,0.1), 'softness':(0.0,0.8,0.0,0.0,0.5,0.2),
    'floating':(0.0,0.3,0.0,0.2,0.8,0.3), 'pressure':(0.4,0.0,0.5,0.5,0.0,0.4),
    'chest':(0.4,0.5,0.4,0.3,0.2,0.3), 'throat':(0.6,0.2,0.3,0.5,0.0,0.3),
    'stomach':(0.5,0.1,0.3,0.6,0.1,0.3), 'jaw':(0.2,0.0,0.9,0.1,0.0,0.2),
    'fists':(0.1,0.0,0.9,0.0,0.0,0.2), 'spine':(0.7,0.0,0.2,0.2,0.1,0.4),
    'breath':(0.5,0.3,0.1,0.3,0.3,0.3), 'heart':(0.3,0.8,0.2,0.2,0.3,0.3),
    'tears':(0.3,0.3,0.1,0.4,0.0,0.8), 'blood':(0.4,0.0,0.8,0.1,0.1,0.3),
    'cold':(0.7,0.0,0.0,0.6,0.0,0.2), 'heat':(0.2,0.2,0.7,0.0,0.3,0.2),
    'empty':(0.1,0.0,0.0,0.9,0.0,0.3), 'numb':(0.1,0.0,0.0,0.8,0.0,0.3),
    'tight':(0.5,0.0,0.6,0.2,0.0,0.3), 'soft':(0.0,0.7,0.0,0.1,0.4,0.2),
    'sharp':(0.4,0.0,0.6,0.0,0.2,0.4), 'deep':(0.3,0.4,0.1,0.5,0.2,0.5),
}

SEED_RU = {
    'пульс':(0.4,0.0,0.8,0.0,0.3,0.2), 'дрожь':(0.8,0.0,0.2,0.2,0.0,0.3),
    'жар':(0.3,0.1,0.8,0.0,0.2,0.2), 'холод':(0.7,0.0,0.0,0.5,0.0,0.2),
    'тошнота':(0.5,0.0,0.2,0.7,0.0,0.3), 'боль':(0.3,0.0,0.4,0.5,0.0,0.4),
    'тяжесть':(0.2,0.0,0.1,0.8,0.0,0.2), 'горит':(0.3,0.1,0.9,0.0,0.1,0.2),
    'сжимает':(0.5,0.0,0.6,0.3,0.0,0.3), 'тепло':(0.0,0.9,0.0,0.0,0.6,0.1),
    'грудь':(0.4,0.5,0.4,0.3,0.2,0.3), 'горло':(0.6,0.2,0.3,0.5,0.0,0.3),
    'живот':(0.5,0.1,0.3,0.6,0.1,0.3), 'челюсть':(0.2,0.0,0.9,0.1,0.0,0.2),
    'кулаки':(0.1,0.0,0.9,0.0,0.0,0.2), 'сердце':(0.3,0.8,0.2,0.2,0.3,0.3),
    'слёзы':(0.3,0.3,0.1,0.4,0.0,0.8), 'пусто':(0.1,0.0,0.0,0.9,0.0,0.3),
    'давит':(0.4,0.0,0.5,0.5,0.0,0.4), 'ноет':(0.2,0.0,0.1,0.8,0.0,0.3),
}


def build_somatic_affinity(sp, seeds):
    """Map seed words to BPE token affinities."""
    aff = torch.zeros(sp.get_piece_size(), 6)
    for word, scores in seeds.items():
        toks = sp.encode(word)
        for t in toks:
            if t < aff.shape[0]:
                for c in range(6):
                    aff[t, c] = max(aff[t, c], scores[c])
    n = (aff.sum(1) > 0.01).sum().item()
    return aff, n


def generate(model, sp, lang, prompt, max_tokens=48, temp=0.7, top_k=32, boost_strength=2.0, device='cuda'):
    """Generate somatic text using LM + Dario injection."""
    seeds = SEED_RU if lang == 'ru' else SEED_EN
    somatic_aff, n_soma = build_somatic_affinity(sp, seeds)
    somatic_aff = somatic_aff.to(device)

    ids = sp.encode(prompt)
    tokens = list(ids)

    # Get chambers from input
    input_t = torch.tensor([tokens], device=device)
    with torch.no_grad():
        logits, act, raw = model(input_t, lang)
        chambers = act[0]  # (6,)

    generated = []
    for step in range(max_tokens):
        input_t = torch.tensor([tokens], device=device)
        with torch.no_grad():
            logits_out, _, _ = model(input_t, lang)
            logit = logits_out[0, -1]  # (VOCAB,)

        # Dario injection: somatic boost based on chambers
        boost = (somatic_aff * chambers.unsqueeze(0)).sum(1) * boost_strength
        logit = logit + boost

        # Temperature + top-k sampling
        logit = logit / temp
        if top_k > 0:
            vals, idx = logit.topk(top_k)
            logit = torch.full_like(logit, float('-inf'))
            logit.scatter_(0, idx, vals)

        probs = F.softmax(logit, dim=0)
        next_tok = torch.multinomial(probs, 1).item()

        if next_tok == 0:  # EOS/UNK
            break

        tokens.append(next_tok)
        generated.append(next_tok)

    return sp.decode(generated)


def main():
    dev = torch.device('cuda')
    model = KlausFull().to(dev)
    ck = torch.load(WEIGHTS_DIR + '/klaus_chambers.pt', map_location=dev, weights_only=False)
    for l in LANGS:
        if l in ck.get('lms', {}):
            model.lms[l].load_state_dict(ck['lms'][l])
    model.chambers.load_state_dict(ck['chambers'])
    for l in LANGS:
        if l in ck['res_projs']:
            model.res_projs[l].load_state_dict(ck['res_projs'][l])
    model.eval()

    sps = {}
    for l in LANGS:
        sp = spm.SentencePieceProcessor()
        sp.load(WEIGHTS_DIR + '/klaus_%s_bpe.model' % l)
        sps[l] = sp

    print('=' * 60)
    print(' KLAUS — Live Somatic Generation')
    print('=' * 60)

    tests = {
        'en': [
            ('I am terrified', 'FEAR'),
            ('I love you so much', 'LOVE'),
            ('I want to destroy everything', 'RAGE'),
            ('Nothing matters', 'VOID'),
        ],
        'ru': [
            ('мне страшно', 'FEAR'),
            ('я тебя люблю', 'LOVE'),
            ('ненавижу всё', 'RAGE'),
            ('мне всё равно', 'VOID'),
        ],
    }

    for lang in ['en', 'ru']:
        print('\n--- %s ---' % lang.upper())
        for prompt, expected in tests[lang]:
            # Get chambers
            ids = sps[lang].encode(prompt)
            input_t = torch.tensor([ids], device=dev)
            with torch.no_grad():
                _, act, _ = model(input_t, lang)
                ch = act[0].cpu().numpy()
            dom = CNAMES[ch.argmax()]
            ch_str = ' '.join('%.2f' % ch[i] for i in range(6))

            # Generate
            out = generate(model, sps[lang], lang, prompt, max_tokens=32, temp=0.7, boost_strength=3.0, device=dev)

            print('\n  "%s"' % prompt)
            print('  [%s] %s' % (dom, ch_str))
            print('  → %s' % out.strip())

    # Interactive mode
    print('\n' + '=' * 60)
    print(' Interactive: type to feel. /quit to exit.')
    print('=' * 60)
    while True:
        try:
            prompt = input('\nyou: ').strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not prompt or prompt == '/quit':
            break
        # detect lang
        lang = 'en'
        for ch in prompt:
            if '\u0400' <= ch <= '\u04ff':
                lang = 'ru'; break
        out = generate(model, sps[lang], lang, prompt, max_tokens=40, temp=0.7, boost_strength=3.0, device=dev)
        # chambers
        ids = sps[lang].encode(prompt)
        input_t = torch.tensor([ids], device=dev)
        with torch.no_grad():
            _, act, _ = model(input_t, lang)
            ch = act[0].cpu().numpy()
        dom = CNAMES[ch.argmax()]
        print('  [%s] → %s' % (dom, out.strip()))


if __name__ == '__main__':
    main()
