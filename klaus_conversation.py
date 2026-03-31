#!/usr/bin/env python3
"""
Klaus Conversation Mode — interactive somatic dialogue.
Full pipeline: LM + Chambers + Phrases + Dario + State Memory + Trajectory.
"""
import sys, os, time, readline
sys.path.insert(0, '/home/ubuntu/klaus')
import torch
import torch.nn.functional as F
import numpy as np
import sentencepiece as spm
from train_chambers import KlausFull, CNAMES, LANGS, VOCAB, DIM, MAX_SEQ, WEIGHTS_DIR
from state_memory import StateMemory, TrajectoryAlert

N_CHAMBERS = 6

# Somatic phrases
SOMATIC = {
    'ru': [
        ('дрожь по всему телу',0),('холод внутри',0),('хочу спрятаться',0),
        ('сжимается в животе',0),('не могу дышать',0),('ноги подкашиваются',0),
        ('мурашки по коже',0),('сердце замирает',0),
        ('тепло в груди',1),('хочу обнять',1),('сердце тает',1),
        ('лёгкость внутри',1),('улыбка сама появляется',1),('хочу быть рядом',1),
        ('мягко внутри',1),('дыхание ровное',1),
        ('жжёт в груди',2),('кулаки сжимаются',2),('хочется ударить',2),
        ('кровь кипит',2),('челюсть сжата',2),('бешеный пульс',2),
        ('давление в висках',2),('скрежет зубов',2),
        ('пусто внутри',3),('ничего не чувствую',3),('тяжесть во всём теле',3),
        ('хочу исчезнуть',3),('всё серое',3),('нет сил',3),
        ('онемение',3),('как будто не здесь',3),
        ('любопытство внутри',4),('энергия поднимается',4),('хочу узнать больше',4),
        ('лёгкое возбуждение',4),('глаза широко открыты',4),('тянет вперёд',4),
        ('ком в горле',5),('стыд жжёт лицо',5),('слёзы подступают',5),
        ('не знаю что чувствую',5),('всё смешалось внутри',5),('тревога и радость одновременно',5),
    ],
    'en': [
        ('trembling all over',0),('cold inside',0),('want to hide',0),
        ('stomach clenching',0),('cannot breathe',0),('legs giving way',0),
        ('warmth in my chest',1),('want to embrace',1),('heart melting',1),
        ('lightness inside',1),('smiling without reason',1),
        ('burning in my chest',2),('fists clenching',2),('want to hit',2),
        ('blood boiling',2),('jaw tight',2),('pounding pulse',2),
        ('empty inside',3),('feeling nothing',3),('heaviness everywhere',3),
        ('want to disappear',3),('everything grey',3),('no strength left',3),
        ('curiosity rising',4),('energy building',4),('eyes wide open',4),
        ('lump in throat',5),('shame burning face',5),('tears coming',5),
        ('do not know what I feel',5),('everything mixed up inside',5),
    ],
    'fr': [
        ('tremblements partout',0),('froid interieur',0),('envie de me cacher',0),
        ('estomac noue',0),('ne peux pas respirer',0),('jambes qui flanchent',0),
        ('chaleur dans la poitrine',1),('envie de serrer',1),('coeur qui fond',1),
        ('legerete interieure',1),('sourire involontaire',1),('envie de rester',1),
        ('brulure dans la poitrine',2),('poings serres',2),('envie de frapper',2),
        ('sang qui bout',2),('machoire serree',2),('pouls qui bat',2),
        ('vide interieur',3),('ne ressens rien',3),('lourdeur partout',3),
        ('envie de disparaitre',3),('tout est gris',3),('plus de force',3),
        ('curiosite qui monte',4),('energie qui monte',4),('yeux grands ouverts',4),
        ('boule dans la gorge',5),('honte brulante',5),('larmes qui montent',5),
        ('ne sais pas ce que je ressens',5),('tout melange',5),
    ],
}

# Lexical priors
RU_LEXICAL = {
    0: ['страшн','боюсь','ужас','паник','жуть','кошмар','испуг'],
    1: ['люблю','обожаю','нежн','дорог','родн','счастл'],
    2: ['бесишь','бесит','ненавиж','злюсь','злит','ярост','бешен','задолбал','достал','убью','бесят','раздраж'],
    3: ['пусто','равно','безразлич','тоскл','одинок','бессмысл','устал','апати','скучн','тошно'],
    4: ['интересн','любопытн','увлека','круто','захватыва'],
    5: ['стыдн','сложно','непонятн','странн'],
}

FR_LEXICAL = {
    0: ['peur','terrifi','effraye','angoiss','panique'],
    1: ['aime','amour','adore','tendress','bonheur','heureu'],
    2: ['deteste','furieu','rage','colere','haine','enerve','agace','insupport'],
    3: ['vide','rien','ennui','lasse','fatigue','indiffere','triste','seul'],
    4: ['excite','curie','fascin','passion','intere'],
    5: ['honte','confus','bizarre','etrange'],
}

def apply_lexical_priors(prompt, chambers, lang):
    if lang not in ('ru', 'fr'):
        return chambers
    modified = chambers.copy()
    prompt_lower = prompt.lower()
    BOOST = {'ru': 0.25, 'fr': 0.35}.get(lang, 0.25)
    lexicon = RU_LEXICAL if lang == 'ru' else FR_LEXICAL
    for ch_idx, words in lexicon.items():
        for w in words:
            if w in prompt_lower:
                modified[ch_idx] += BOOST
                break
    return modified


def detect_lang(text):
    """Simple language detection."""
    for ch in text:
        if '\u0400' <= ch <= '\u04ff': return 'ru'
        if '\u0590' <= ch <= '\u05ff': return 'he'
    # Check for French markers
    fr_markers = ['je ', 'tu ', 'il ', 'nous', 'vous', 'les ', 'des ', 'est ', 'pas ', 'une ']
    text_lower = text.lower()
    if any(m in text_lower for m in fr_markers):
        return 'fr'
    return 'en'


def respond(model, sps, mem, lang, prompt):
    """Full Klaus response: LM + Chambers + Phrases + Dario + Memory."""
    if lang not in SOMATIC:
        lang = 'en'  # fallback

    sp = sps[lang]
    phrases = SOMATIC[lang]
    phrases_text = [p[0] for p in phrases]
    phrases_ch = [p[1] for p in phrases]
    phrases_bpe = [sp.encode(p) for p in phrases_text]

    ids = sp.encode(prompt)
    tokens = torch.tensor([ids], device='cuda')

    with torch.no_grad():
        logits, act, raw = model(tokens, lang)
        last_logits = logits[0, -1]
        chambers = act[0].cpu().numpy()

    # Apply lexical priors
    chambers = apply_lexical_priors(prompt, chambers, lang)

    # Apply state memory modulation (history shapes perception)
    chambers = mem.modulate_chambers(chambers)

    # Record state and check trajectory
    alert = mem.record(chambers, lang, prompt)

    # Phase 1: BPE logits -> phrase scores
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
        chamber_boost[i] = chambers[ch_idx] * 5.0  # stronger chamber signal

    final_scores = base_scores + chamber_boost
    probs = F.softmax(final_scores / 0.8, dim=0)
    top5 = probs.topk(5)

    dominant = CNAMES[int(np.argmax(chambers))]
    ch_str = ' '.join('%s=%.2f' % (CNAMES[i], chambers[i]) for i in range(6))

    # Output
    print()
    print('  [%s] %s' % (dominant, ch_str))

    # Top 3 somatic phrases
    for i in range(min(3, len(top5.indices))):
        idx = top5.indices[i].item()
        p = top5.values[i].item()
        print('    %.0f%% %s' % (p * 100, phrases_text[idx]))

    # Alert
    if alert:
        print('  *** %s (%.0f%%) — %s' % (alert.alert_type.upper(), alert.severity * 100, alert.description))

    # Fingerprint after 3+ turns
    if mem.n_events() >= 3:
        _, fp_s = mem.get_fingerprint()
        fp_str = ' '.join('%.1f' % fp_s[i] for i in range(6))
        print('  fingerprint: %s' % fp_str)


def main():
    print('Loading Klaus...')
    model = KlausFull().cuda()
    ck = torch.load(WEIGHTS_DIR + '/klaus_chambers.pt', map_location='cpu', weights_only=False)
    if 'lms' in ck:
        for l in LANGS:
            if l in ck['lms']:
                model.lms[l].load_state_dict(ck['lms'][l])
    else:
        model.load_weights()
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

    mem = StateMemory()

    print('Klaus ready. Type to talk. /quit to exit. /reset to clear memory.')
    print('Language auto-detected (EN/RU/FR). HE not in conversation mode yet.')
    print('=' * 50)

    while True:
        try:
            prompt = input('\nyou: ').strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not prompt:
            continue
        if prompt == '/quit':
            break
        if prompt == '/reset':
            mem.reset()
            print('  [memory reset]')
            continue
        if prompt == '/state':
            print(mem.get_trajectory_summary())
            continue

        lang = detect_lang(prompt)
        respond(model, sps, mem, lang, prompt)


if __name__ == '__main__':
    main()
