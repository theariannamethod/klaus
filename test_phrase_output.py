#!/usr/bin/env python3
"""
Klaus Phase 2a+2b — Somatic phrase vocabulary + Penelope trick test.
Tests whether trained LM weights can score somatic phrases through BPE aggregation.
NO training needed — uses existing weights.
"""
import torch, sys, json
sys.path.insert(0, '/home/ubuntu/klaus')
from train_klaus import Klaus, VOCAB, DIM, MAX_SEQ
import torch.nn.functional as F
import sentencepiece as spm

# ─── Somatic Phrase Vocabulary (per language, with chamber affinity) ───
# Chamber indices: 0=FEAR, 1=LOVE, 2=RAGE, 3=VOID, 4=FLOW, 5=COMPLEX
SOMATIC_VOCAB = {
    'ru': [
        # FEAR (0)
        ('дрожь по всему телу', 0), ('холод внутри', 0), ('хочу спрятаться', 0),
        ('сжимается в животе', 0), ('не могу дышать', 0), ('ноги подкашиваются', 0),
        ('мурашки по коже', 0), ('сердце замирает', 0),
        # LOVE (1)
        ('тепло в груди', 1), ('хочу обнять', 1), ('сердце тает', 1),
        ('лёгкость внутри', 1), ('улыбка сама появляется', 1), ('хочу быть рядом', 1),
        ('мягко внутри', 1), ('дыхание ровное', 1),
        # RAGE (2)
        ('жжёт в груди', 2), ('кулаки сжимаются', 2), ('хочется ударить', 2),
        ('кровь кипит', 2), ('челюсть сжата', 2), ('бешеный пульс', 2),
        ('давление в висках', 2), ('скрежет зубов', 2),
        # VOID (3)
        ('пусто внутри', 3), ('ничего не чувствую', 3), ('тяжесть во всём теле', 3),
        ('хочу исчезнуть', 3), ('всё серое', 3), ('нет сил', 3),
        ('онемение', 3), ('как будто не здесь', 3),
        # FLOW (4)
        ('любопытство внутри', 4), ('энергия поднимается', 4), ('хочу узнать больше', 4),
        ('лёгкое возбуждение', 4), ('глаза широко открыты', 4), ('тянет вперёд', 4),
        # COMPLEX (5)
        ('ком в горле', 5), ('стыд жжёт лицо', 5), ('слёзы подступают', 5),
        ('не знаю что чувствую', 5), ('всё смешалось внутри', 5), ('тревога и радость одновременно', 5),
    ],
    'en': [
        ('trembling all over', 0), ('cold inside', 0), ('want to hide', 0),
        ('stomach clenching', 0), ('can not breathe', 0), ('legs giving way', 0),
        ('warmth in my chest', 1), ('want to embrace', 1), ('heart melting', 1),
        ('lightness inside', 1), ('smiling without reason', 1),
        ('burning in my chest', 2), ('fists clenching', 2), ('want to hit', 2),
        ('blood boiling', 2), ('jaw tight', 2), ('pounding pulse', 2),
        ('empty inside', 3), ('feeling nothing', 3), ('heaviness everywhere', 3),
        ('want to disappear', 3), ('everything grey', 3), ('no strength left', 3),
        ('curiosity rising', 4), ('energy building', 4), ('eyes wide open', 4),
        ('lump in throat', 5), ('shame burning face', 5), ('tears coming', 5),
        ('don not know what I feel', 5), ('everything mixed up inside', 5),
    ],
    'he': [
        ('רעד בכל הגוף', 0), ('קור בפנים', 0), ('רוצה להתחבא', 0),
        ('הבטן מתכווצת', 0), ('לא יכול לנשום', 0),
        ('חום בחזה', 1), ('רוצה לחבק', 1), ('הלב נמס', 1),
        ('קלילות בפנים', 1),
        ('שריפה בחזה', 2), ('אגרופים נסגרים', 2), ('רוצה להכות', 2),
        ('הדם רותח', 2), ('דופק מטורף', 2),
        ('ריק בפנים', 3), ('לא מרגיש כלום', 3), ('כבדות בכל הגוף', 3),
        ('רוצה להיעלם', 3),
        ('סקרנות עולה', 4), ('אנרגיה עולה', 4),
        ('גוש בגרון', 5), ('בושה שורפת את הפנים', 5), ('דמעות מגיעות', 5),
    ],
    'fr': [
        ('tremblement partout', 0), ('froid intérieur', 0), ('envie de me cacher', 0),
        ('estomac qui se serre', 0), ('je ne peux pas respirer', 0),
        ('chaleur dans la poitrine', 1), ('envie de serrer dans mes bras', 1),
        ('coeur qui fond', 1), ('légèreté intérieure', 1),
        ('brûlure dans la poitrine', 2), ('poings serrés', 2), ('envie de frapper', 2),
        ('sang qui bout', 2), ('pouls qui bat fort', 2),
        ('vide intérieur', 3), ('je ne ressens rien', 3), ('lourdeur partout', 3),
        ('envie de disparaître', 3),
        ('curiosité qui monte', 4), ('énergie qui monte', 4),
        ('boule dans la gorge', 5), ('honte qui brûle le visage', 5), ('larmes qui montent', 5),
    ],
}

def bpe_logits_to_phrase_scores(bpe_logits, phrases_bpe, temperature=1.0):
    """Penelope trick: aggregate BPE logits per phrase."""
    scores = []
    for phrase_tokens in phrases_bpe:
        if len(phrase_tokens) > 0:
            score = sum(bpe_logits[t].item() for t in phrase_tokens) / len(phrase_tokens)
        else:
            score = -float('inf')
        scores.append(score)
    return torch.tensor(scores)

def test_language(lang, prompts):
    """Test phrase scoring for one language."""
    wpath = f'/home/ubuntu/klaus/weights/klaus_{lang}.pt'
    bpath = f'/home/ubuntu/klaus/weights/klaus_{lang}_bpe.model'

    ckpt = torch.load(wpath, map_location='cuda', weights_only=False)
    sp = spm.SentencePieceProcessor()
    sp.load(bpath)

    model = Klaus().cuda()
    model.load_state_dict(ckpt['model'])
    model.eval()

    # Pre-encode all somatic phrases to BPE tokens
    phrases = SOMATIC_VOCAB[lang]
    phrases_text = [p[0] for p in phrases]
    phrases_chamber = [p[1] for p in phrases]
    phrases_bpe = [sp.encode(p) for p in phrases_text]

    CNAMES = ['FEAR', 'LOVE', 'RAGE', 'VOID', 'FLOW', 'COMPLEX']

    print(f'\n{"="*60}')
    print(f' {lang.upper()} | val={ckpt["val_loss"]:.4f} | {len(phrases)} somatic phrases')
    print(f'{"="*60}')

    for prompt in prompts:
        ids = sp.encode(prompt)
        tokens = torch.tensor([ids], device='cuda')

        with torch.no_grad():
            logits = model(tokens)
            last_logits = logits[0, -1]  # last position logits

        # Penelope trick: BPE logits → phrase scores
        phrase_scores = bpe_logits_to_phrase_scores(last_logits, phrases_bpe)

        # Softmax to get probabilities
        probs = F.softmax(phrase_scores / 0.8, dim=0)

        # Top 5 phrases
        top5 = probs.topk(5)

        print(f'\n  Prompt: "{prompt}"')
        print(f'  Top somatic responses:')
        for i in range(5):
            idx = top5.indices[i].item()
            prob = top5.values[i].item()
            chamber = CNAMES[phrases_chamber[idx]]
            print(f'    {prob:.1%} [{chamber:7s}] {phrases_text[idx]}')

        # Chamber distribution (sum probs per chamber)
        ch_probs = [0.0] * 6
        for i, p in enumerate(probs):
            ch_probs[phrases_chamber[i]] += p.item()
        print(f'  Chamber dist: ' + ' '.join(f'{CNAMES[c]}={ch_probs[c]:.2f}' for c in range(6)))

    del model
    torch.cuda.empty_cache()

# ─── Test all 4 languages ───
print('KLAUS Phase 2a+2b — Somatic Phrase Output Test')
print('Penelope trick: BPE logits → phrase aggregation')
print('NO training — existing LM weights only')

test_language('ru', [
    'я ненавижу тебя',
    'мне так страшно',
    'я тебя люблю',
    'мне всё равно',
])

test_language('en', [
    'I hate you so much',
    'I am terrified',
    'I love you deeply',
    'I feel nothing',
])

test_language('he', [
    'אני שונא אותך',
    'אני מפחד',
    'אני אוהב אותך',
])

test_language('fr', [
    'je te déteste',
    'j ai tellement peur',
    'je t aime profondément',
])

print(f'\n{"="*60}')
print('DONE — Phrase output test complete.')
print(f'{"="*60}')
