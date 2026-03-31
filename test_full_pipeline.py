#!/usr/bin/env python3
"""Klaus full pipeline test: LM + chambers + phrase aggregation + Dario field."""
import torch,sys,json; sys.path.insert(0,'/home/ubuntu/klaus')
import torch.nn.functional as F
import sentencepiece as spm
from train_chambers import KlausFull, CNAMES, LANGS, VOCAB, DIM, MAX_SEQ, WEIGHTS_DIR

# Somatic phrases (same as test_phrase_output.py)
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
    'he': [
        ('\u05e8\u05d5\u05e2\u05d3 \u05d1\u05db\u05dc \u05d4\u05d2\u05d5\u05e3',0),
        ('\u05e7\u05d5\u05e8 \u05e4\u05e0\u05d9\u05de\u05d9',0),
        ('\u05e8\u05d5\u05e6\u05d4 \u05dc\u05d4\u05e1\u05ea\u05ea\u05e8',0),
        ('\u05d1\u05d8\u05df \u05e0\u05e7\u05e9\u05e8\u05ea',0),
        ('\u05dc\u05d0 \u05d9\u05db\u05d5\u05dc \u05dc\u05e0\u05e9\u05d5\u05dd',0),
        ('\u05d7\u05d5\u05dd \u05d1\u05d7\u05d6\u05d4',1),
        ('\u05e8\u05d5\u05e6\u05d4 \u05dc\u05d7\u05d1\u05e7',1),
        ('\u05d4\u05dc\u05d1 \u05e0\u05de\u05e1',1),
        ('\u05e7\u05dc\u05d5\u05ea \u05e4\u05e0\u05d9\u05de\u05d9\u05ea',1),
        ('\u05d7\u05d9\u05d5\u05da \u05de\u05e2\u05e6\u05de\u05d5',1),
        ('\u05e9\u05e8\u05d9\u05e4\u05d4 \u05d1\u05d7\u05d6\u05d4',2),
        ('\u05d0\u05d2\u05e8\u05d5\u05e4\u05d9\u05dd \u05e7\u05e4\u05d5\u05e6\u05d9\u05dd',2),
        ('\u05e8\u05d5\u05e6\u05d4 \u05dc\u05d4\u05db\u05d5\u05ea',2),
        ('\u05d4\u05d3\u05dd \u05e8\u05d5\u05ea\u05d7',2),
        ('\u05e8\u05d9\u05e7 \u05d1\u05e4\u05e0\u05d9\u05dd',3),
        ('\u05dc\u05d0 \u05de\u05e8\u05d2\u05d9\u05e9 \u05db\u05dc\u05d5\u05dd',3),
        ('\u05db\u05d1\u05d3\u05d5\u05ea \u05d1\u05db\u05dc \u05d4\u05d2\u05d5\u05e3',3),
        ('\u05e8\u05d5\u05e6\u05d4 \u05dc\u05d4\u05d9\u05e2\u05dc\u05dd',3),
        ('\u05d4\u05db\u05dc \u05d0\u05e4\u05d5\u05e8',3),
        ('\u05d0\u05d9\u05df \u05db\u05d5\u05d7',3),
        ('\u05d2\u05d5\u05e9 \u05d1\u05d2\u05e8\u05d5\u05df',5),
        ('\u05d1\u05d5\u05e9\u05d4 \u05e9\u05d5\u05e8\u05e4\u05ea',5),
        ('\u05d3\u05de\u05e2\u05d5\u05ea \u05e2\u05d5\u05dc\u05d5\u05ea',5),
        ('\u05dc\u05d0 \u05d9\u05d5\u05d3\u05e2 \u05de\u05d4 \u05d0\u05e0\u05d9 \u05de\u05e8\u05d2\u05d9\u05e9',5),
    ],
}


# Lexical priors for RU (inference-time injection, same principle as dario.c)
RU_LEXICAL = {
    0: ['страшн','боюсь','ужас','паник','жуть','кошмар','испуг'],
    1: ['люблю','обожаю','нежн','дорог','родн','счастл'],
    2: ['бесишь','бесит','ненавиж','злюсь','злит','ярост','бешен','задолбал','достал','убью','бесят','раздраж'],
    3: ['пусто','равно','безразлич','тоскл','одинок','бессмысл','устал','апати','скучн','тошно'],
    4: ['интересн','любопытн','увлека','круто','захватыва'],
    5: ['стыдн','сложно','непонятн','странн'],
}

# HE lexical priors
HE_LEXICAL = {
    0: ['פחד','מפחד','אימה','פאניק','מפחיד'],
    1: ['אוהב','אהבה','אוהבת','חיבה','שמח','מאושר'],
    2: ['שונא','שנאה','כועס','עצבני','מרגיז','מתעצבן'],
    3: ['ריק','לא מרגיש','אדיש','לא אכפת','עצוב','מדוכא'],
    4: ['מעניין','סקרן','מדהים'],
    5: ['בושה','מבולבל','מוזר'],
}

def apply_lexical_priors(prompt, chambers, lang):
    import numpy as np
    if lang not in ('ru', 'he'):
        return chambers
    modified = chambers.copy()
    prompt_lower = prompt.lower()
    BOOST = 0.25 if lang == "ru" else 0.9  # HE needs stronger boost (VOID=84% in data)
    lexicon = RU_LEXICAL if lang == 'ru' else HE_LEXICAL
    for ch_idx, words in lexicon.items():
        for w in words:
            if w in prompt_lower:
                modified[ch_idx] += BOOST
                break
    return modified

def run_test(model, sps, lang, prompt):
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
    
    # Phase 1: BPE logits → phrase scores (Penelope trick)
    base_scores = []
    for pt in phrases_bpe:
        if len(pt) > 0:
            base_scores.append(sum(last_logits[t].item() for t in pt) / len(pt))
        else:
            base_scores.append(-999)
    base_scores = torch.tensor(base_scores)
    
    # Phase 2: Dario field — chambers modulate phrase scores
    chamber_boost = torch.zeros(len(phrases))
    for i, (text, ch_idx) in enumerate(phrases):
        chamber_boost[i] = chambers[ch_idx] * 3.0  # boost factor
    
    # Combined: base + chamber modulation
    final_scores = base_scores + chamber_boost
    probs = F.softmax(final_scores / 0.8, dim=0)
    
    # Top 5
    top5 = probs.topk(5)
    
    # Chamber distribution
    ch_str = ' '.join(f'{CNAMES[i]}={chambers[i]:.2f}' for i in range(6))
    dominant = CNAMES[chambers.argmax()]
    
    print(f'\n  Input: {prompt}')
    print(f'  Chambers: {ch_str}')
    print(f'  Dominant: {dominant}')
    print(f'  Somatic response:')
    for i in range(5):
        idx = top5.indices[i].item()
        p = top5.values[i].item()
        ch = CNAMES[phrases_ch[idx]]
        marker = ' ←' if phrases_ch[idx] == chambers.argmax() else ''
        print(f'    {p:.1%} [{ch:7s}] {phrases_text[idx]}{marker}')

eq="="*60; dash="─"*60
# Load model
print('Loading full Klaus model...')
model = KlausFull().cuda()
# Load from unified checkpoint (LMs + chambers + res_projs)
ck = torch.load(f'{WEIGHTS_DIR}/klaus_chambers.pt', map_location='cpu', weights_only=False)
if 'lms' in ck:
    for l in LANGS:
        if l in ck['lms']:
            model.lms[l].load_state_dict(ck['lms'][l])
            print(f'  {l.upper()} loaded from chambers checkpoint')
else:
    model.load_weights()
model.chambers.load_state_dict(ck['chambers'])
for l in LANGS:
    if l in ck['res_projs']:
        model.res_projs[l].load_state_dict(ck['res_projs'][l])
print(f'Chambers loaded (acc={ck["acc"]:.1%})')
model.eval()

sps = {}
for l in LANGS:
    sp = spm.SentencePieceProcessor(); sp.load(f'{WEIGHTS_DIR}/klaus_{l}_bpe.model'); sps[l] = sp

print('=' * 60)
print(f' KLAUS Full Pipeline — LM + Chambers + Phrases + Dario')
print('=' * 60)

# RU tests
print('-' * 60)
for p in ['я ненавижу тебя','мне так страшно','я тебя люблю','мне всё равно','ты меня бесишь']:
    run_test(model, sps, 'ru', p)

# EN tests
print('-' * 60)
for p in ['I hate you so much','I am terrified','I love you deeply','I feel nothing']:
    run_test(model, sps, 'en', p)

# FR tests
print('-' * 60)
for p in ['je te deteste','j\'ai tellement peur','je t\'aime profondement','je ne ressens rien']:
    run_test(model, sps, 'fr', p)

# HE tests
print('-' * 60)
for p in ['\u05d0\u05e0\u05d9 \u05e9\u05d5\u05e0\u05d0 \u05d0\u05d5\u05ea\u05da','\u05d0\u05e0\u05d9 \u05de\u05e4\u05d7\u05d3','\u05d0\u05e0\u05d9 \u05d0\u05d5\u05d4\u05d1 \u05d0\u05d5\u05ea\u05da','\u05d0\u05e0\u05d9 \u05dc\u05d0 \u05de\u05e8\u05d2\u05d9\u05e9 \u05db\u05dc\u05d5\u05dd']:
    run_test(model, sps, 'he', p)

print('=' * 60)
