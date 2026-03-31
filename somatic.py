#!/usr/bin/env python3
"""
Klaus Somatic Module — single source of truth for:
- Somatic phrase vocabularies (all languages)
- Lexical priors (all languages)
- Dario field injection
- Pipeline function
"""
import torch
import torch.nn.functional as F
import numpy as np

CNAMES = ['FEAR', 'LOVE', 'RAGE', 'VOID', 'FLOW', 'COMPLEX']
DARIO_BOOST = 5.0  # unified boost factor
SOFTMAX_TEMP = 0.8

# ─── Somatic Phrases ───
SOMATIC = {
    'ru': [
        ('дрожь по всему телу', 0), ('холод внутри', 0), ('хочу спрятаться', 0),
        ('сжимается в животе', 0), ('не могу дышать', 0), ('ноги подкашиваются', 0),
        ('мурашки по коже', 0), ('сердце замирает', 0),
        ('тепло в груди', 1), ('хочу обнять', 1), ('сердце тает', 1),
        ('лёгкость внутри', 1), ('улыбка сама появляется', 1), ('хочу быть рядом', 1),
        ('мягко внутри', 1), ('дыхание ровное', 1),
        ('жжёт в груди', 2), ('кулаки сжимаются', 2), ('хочется ударить', 2),
        ('кровь кипит', 2), ('челюсть сжата', 2), ('бешеный пульс', 2),
        ('давление в висках', 2), ('скрежет зубов', 2),
        ('пусто внутри', 3), ('ничего не чувствую', 3), ('тяжесть во всём теле', 3),
        ('хочу исчезнуть', 3), ('всё серое', 3), ('нет сил', 3),
        ('онемение', 3), ('как будто не здесь', 3),
        ('любопытство внутри', 4), ('энергия поднимается', 4), ('хочу узнать больше', 4),
        ('лёгкое возбуждение', 4), ('глаза широко открыты', 4), ('тянет вперёд', 4),
        ('ком в горле', 5), ('стыд жжёт лицо', 5), ('слёзы подступают', 5),
        ('не знаю что чувствую', 5), ('всё смешалось внутри', 5), ('тревога и радость одновременно', 5),
    ],
    'en': [
        ('trembling all over', 0), ('cold inside', 0), ('want to hide', 0),
        ('stomach clenching', 0), ('cannot breathe', 0), ('legs giving way', 0),
        ('warmth in my chest', 1), ('want to embrace', 1), ('heart melting', 1),
        ('lightness inside', 1), ('smiling without reason', 1),
        ('burning in my chest', 2), ('fists clenching', 2), ('want to hit', 2),
        ('blood boiling', 2), ('jaw tight', 2), ('pounding pulse', 2),
        ('empty inside', 3), ('feeling nothing', 3), ('heaviness everywhere', 3),
        ('want to disappear', 3), ('everything grey', 3), ('no strength left', 3),
        ('curiosity rising', 4), ('energy building', 4), ('eyes wide open', 4),
        ('excitement bubbling', 4), ('mind racing', 4), ('pulled forward', 4),  # FIX #14
        ('lump in throat', 5), ('shame burning face', 5), ('tears coming', 5),
        ('do not know what I feel', 5), ('everything mixed up inside', 5),
    ],
    'fr': [
        ('tremblements partout', 0), ('froid interieur', 0), ('envie de me cacher', 0),
        ('estomac noue', 0), ('ne peux pas respirer', 0), ('jambes qui flanchent', 0),
        ('chaleur dans la poitrine', 1), ('envie de serrer', 1), ('coeur qui fond', 1),
        ('legerete interieure', 1), ('sourire involontaire', 1), ('envie de rester', 1),
        ('brulure dans la poitrine', 2), ('poings serres', 2), ('envie de frapper', 2),
        ('sang qui bout', 2), ('machoire serree', 2), ('pouls qui bat', 2),
        ('vide interieur', 3), ('ne ressens rien', 3), ('lourdeur partout', 3),
        ('envie de disparaitre', 3), ('tout est gris', 3), ('plus de force', 3),
        ('curiosite qui monte', 4), ('energie qui monte', 4), ('yeux grands ouverts', 4),
        ('boule dans la gorge', 5), ('honte brulante', 5), ('larmes qui montent', 5),
        ('ne sais pas ce que je ressens', 5), ('tout melange', 5),
    ],
    'he': [
        ('\u05e8\u05d5\u05e2\u05d3 \u05d1\u05db\u05dc \u05d4\u05d2\u05d5\u05e3', 0),
        ('\u05e7\u05d5\u05e8 \u05e4\u05e0\u05d9\u05de\u05d9', 0),
        ('\u05e8\u05d5\u05e6\u05d4 \u05dc\u05d4\u05e1\u05ea\u05ea\u05e8', 0),
        ('\u05d1\u05d8\u05df \u05e0\u05e7\u05e9\u05e8\u05ea', 0),
        ('\u05dc\u05d0 \u05d9\u05db\u05d5\u05dc \u05dc\u05e0\u05e9\u05d5\u05dd', 0),
        ('\u05d7\u05d5\u05dd \u05d1\u05d7\u05d6\u05d4', 1),
        ('\u05e8\u05d5\u05e6\u05d4 \u05dc\u05d7\u05d1\u05e7', 1),
        ('\u05d4\u05dc\u05d1 \u05e0\u05de\u05e1', 1),
        ('\u05e7\u05dc\u05d5\u05ea \u05e4\u05e0\u05d9\u05de\u05d9\u05ea', 1),
        ('\u05d7\u05d9\u05d5\u05da \u05de\u05e2\u05e6\u05de\u05d5', 1),
        ('\u05e9\u05e8\u05d9\u05e4\u05d4 \u05d1\u05d7\u05d6\u05d4', 2),
        ('\u05d0\u05d2\u05e8\u05d5\u05e4\u05d9\u05dd \u05e7\u05e4\u05d5\u05e6\u05d9\u05dd', 2),
        ('\u05e8\u05d5\u05e6\u05d4 \u05dc\u05d4\u05db\u05d5\u05ea', 2),
        ('\u05d4\u05d3\u05dd \u05e8\u05d5\u05ea\u05d7', 2),
        ('\u05e8\u05d9\u05e7 \u05d1\u05e4\u05e0\u05d9\u05dd', 3),
        ('\u05dc\u05d0 \u05de\u05e8\u05d2\u05d9\u05e9 \u05db\u05dc\u05d5\u05dd', 3),
        ('\u05db\u05d1\u05d3\u05d5\u05ea \u05d1\u05db\u05dc \u05d4\u05d2\u05d5\u05e3', 3),
        ('\u05e8\u05d5\u05e6\u05d4 \u05dc\u05d4\u05d9\u05e2\u05dc\u05dd', 3),
        ('\u05d4\u05db\u05dc \u05d0\u05e4\u05d5\u05e8', 3),
        ('\u05d0\u05d9\u05df \u05db\u05d5\u05d7', 3),
        # FIX #13: HE FLOW phrases (were missing)
        ('\u05e1\u05e7\u05e8\u05e0\u05d5\u05ea \u05e2\u05d5\u05dc\u05d4', 4),
        ('\u05d0\u05e0\u05e8\u05d2\u05d9\u05d4 \u05e2\u05d5\u05dc\u05d4', 4),
        ('\u05e2\u05d9\u05e0\u05d9\u05d9\u05dd \u05e4\u05e7\u05d5\u05d7\u05d5\u05ea \u05dc\u05e8\u05d5\u05d5\u05d7\u05d4', 4),
        ('\u05de\u05e9\u05d5\u05da \u05e4\u05e0\u05d9\u05de\u05d4', 4),
        ('\u05d2\u05d5\u05e9 \u05d1\u05d2\u05e8\u05d5\u05df', 5),
        ('\u05d1\u05d5\u05e9\u05d4 \u05e9\u05d5\u05e8\u05e4\u05ea', 5),
        ('\u05d3\u05de\u05e2\u05d5\u05ea \u05e2\u05d5\u05dc\u05d5\u05ea', 5),
        ('\u05dc\u05d0 \u05d9\u05d5\u05d3\u05e2 \u05de\u05d4 \u05d0\u05e0\u05d9 \u05de\u05e8\u05d2\u05d9\u05e9', 5),
    ],
}

# ─── Lexical Priors (all languages) ───
# FIX #7: EN priors added
EN_LEXICAL = {
    0: ['scared', 'afraid', 'terrif', 'frighten', 'panic', 'anxious', 'dread', 'nervous'],
    1: ['love', 'adore', 'cherish', 'happy', 'joy', 'grateful', 'tender'],
    2: ['hate', 'angry', 'furious', 'rage', 'disgust', 'annoy', 'infuriat', 'punch'],
    3: ['nothing', 'empty', 'numb', 'void', 'hollow', 'dead inside', 'apathy', 'bored', 'meaningless'],
    4: ['curious', 'interest', 'fascin', 'excit', 'wonder', 'amaz'],
    5: ['confus', 'shame', 'embarrass', 'weird', 'strange', 'overwhelm'],
}

RU_LEXICAL = {
    0: ['страшн', 'боюсь', 'ужас', 'паник', 'жуть', 'кошмар', 'испуг'],
    1: ['люблю', 'обожаю', 'нежн', 'дорог', 'родн', 'счастл'],
    2: ['бесишь', 'бесит', 'ненавиж', 'злюсь', 'злит', 'ярост', 'бешен', 'задолбал', 'достал', 'убью', 'бесят', 'раздраж'],
    3: ['пусто', 'равно', 'безразлич', 'тоскл', 'одинок', 'бессмысл', 'устал', 'апати', 'скучн', 'тошно',
        'грустн', 'грусть', 'печаль', 'печальн'],  # FIX #17
    4: ['интересн', 'любопытн', 'увлека', 'круто', 'захватыва'],
    5: ['стыдн', 'сложно', 'непонятн', 'странн'],
}

FR_LEXICAL = {
    0: ['peur', 'terrifi', 'effraye', 'angoiss', 'panique'],
    1: ['aime', 'amour', 'adore', 'tendress', 'bonheur', 'heureu'],
    2: ['deteste', 'furieu', 'rage', 'colere', 'haine', 'enerve', 'agace', 'insupport'],
    3: ['vide', 'rien', 'ennui', 'lasse', 'fatigue', 'indiffere', 'triste', 'seul', 'ressens rien'],
    4: ['excite', 'curie', 'fascin', 'passion', 'intere'],
    5: ['honte', 'confus', 'bizarre', 'etrange'],
}

HE_LEXICAL = {
    0: ['\u05e4\u05d7\u05d3', '\u05de\u05e4\u05d7\u05d3', '\u05d0\u05d9\u05de\u05d4', '\u05e4\u05d0\u05e0\u05d9\u05e7', '\u05de\u05e4\u05d7\u05d9\u05d3'],
    1: ['\u05d0\u05d5\u05d4\u05d1', '\u05d0\u05d4\u05d1\u05d4', '\u05d0\u05d5\u05d4\u05d1\u05ea', '\u05d7\u05d9\u05d1\u05d4', '\u05e9\u05de\u05d7', '\u05de\u05d0\u05d5\u05e9\u05e8'],
    2: ['\u05e9\u05d5\u05e0\u05d0', '\u05e9\u05e0\u05d0\u05d4', '\u05db\u05d5\u05e2\u05e1', '\u05e2\u05e6\u05d1\u05e0\u05d9', '\u05de\u05e8\u05d2\u05d9\u05d6', '\u05de\u05ea\u05e2\u05e6\u05d1\u05df'],
    3: ['\u05e8\u05d9\u05e7', '\u05dc\u05d0 \u05de\u05e8\u05d2\u05d9\u05e9', '\u05d0\u05d3\u05d9\u05e9', '\u05dc\u05d0 \u05d0\u05db\u05e4\u05ea', '\u05e2\u05e6\u05d5\u05d1', '\u05de\u05d3\u05d5\u05db\u05d0'],
    4: ['\u05de\u05e2\u05e0\u05d9\u05d9\u05df', '\u05e1\u05e7\u05e8\u05df', '\u05de\u05d3\u05d4\u05d9\u05dd', '\u05de\u05e8\u05ea\u05e7'],
    5: ['\u05d1\u05d5\u05e9\u05d4', '\u05de\u05d1\u05d5\u05dc\u05d1\u05dc', '\u05de\u05d5\u05d6\u05e8'],
}

LEXICAL_BOOST = {'en': 0.25, 'ru': 0.25, 'fr': 0.35, 'he': 0.9}
LEXICAL_MAP = {'en': EN_LEXICAL, 'ru': RU_LEXICAL, 'fr': FR_LEXICAL, 'he': HE_LEXICAL}


def apply_lexical_priors(prompt, chambers, lang):
    """Apply lexical priors to chamber activations."""
    if lang not in LEXICAL_MAP:
        return chambers
    modified = chambers.copy()
    prompt_lower = prompt.lower()
    boost = LEXICAL_BOOST.get(lang, 0.25)
    for ch_idx, words in LEXICAL_MAP[lang].items():
        for w in words:
            if w in prompt_lower:
                modified[ch_idx] += boost
                break
    return modified


def fix_zero_chambers(chambers):
    """FIX #8: When all chambers near zero, normalize to prevent BPE noise domination."""
    if chambers.max() < 0.05:
        # All chambers dead — normalize to uniform so Dario field has some effect
        # This happens when input is emotionally flat
        chambers = chambers + 0.01  # tiny floor
    return chambers


def somatic_response(model, sp, lang, prompt, device='cuda'):
    """Full Klaus pipeline: LM → chambers → lexical priors → Dario → phrases."""
    from model import enc_batch

    if lang not in SOMATIC:
        lang = 'en'

    phrases = SOMATIC[lang]
    phrases_text = [p[0] for p in phrases]
    phrases_ch = [p[1] for p in phrases]
    phrases_bpe = [sp.encode(p) for p in phrases_text]

    padded, mask = enc_batch([prompt], sp)
    tokens = padded.to(device)
    mask_dev = mask.to(device)

    with torch.no_grad():
        logits, act, raw = model(tokens, lang, mask=mask_dev)
        last_logits = logits[0, -1]
        chambers = act[0].cpu().numpy()

    # Apply lexical priors
    chambers = apply_lexical_priors(prompt, chambers, lang)

    # Fix zero chambers
    chambers = fix_zero_chambers(chambers)

    # Phase 1: BPE logits → phrase scores (Penelope trick)
    base_scores = []
    for pt in phrases_bpe:
        if len(pt) > 0:
            base_scores.append(sum(last_logits[t].item() for t in pt) / len(pt))
        else:
            base_scores.append(-999)
    base_scores = torch.tensor(base_scores)

    # Phase 2: Dario field
    chamber_boost = torch.zeros(len(phrases))
    for i, (text, ch_idx) in enumerate(phrases):
        chamber_boost[i] = chambers[ch_idx] * DARIO_BOOST

    final_scores = base_scores + chamber_boost
    probs = F.softmax(final_scores / SOFTMAX_TEMP, dim=0)
    top5 = probs.topk(5)

    dominant_idx = int(np.argmax(chambers))
    dominant = CNAMES[dominant_idx]
    ch_str = ' '.join('%s=%.2f' % (CNAMES[i], chambers[i]) for i in range(6))

    return {
        'chambers': chambers,
        'dominant': dominant,
        'dominant_idx': dominant_idx,
        'ch_str': ch_str,
        'top5_indices': top5.indices.tolist(),
        'top5_probs': top5.values.tolist(),
        'phrases_text': phrases_text,
        'phrases_ch': phrases_ch,
    }
