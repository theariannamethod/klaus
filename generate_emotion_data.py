#!/usr/bin/env python3
"""Generate labeled emotion data via GPT-4.1 API for Klaus chambers training."""
import json, os, time, sys
from openai import OpenAI

API_KEY = os.environ.get("OPENAI_API_KEY", "")
client = OpenAI(api_key=API_KEY)

CHAMBERS = {
    0: "FEAR — anxiety, terror, panic, dread, nervousness, worry",
    1: "LOVE — warmth, affection, gratitude, joy, tenderness, caring",
    2: "RAGE — anger, fury, irritation, disgust, hatred, frustration",
    3: "VOID — emptiness, apathy, numbness, depression, exhaustion, indifference",
    4: "FLOW — curiosity, excitement, amusement, interest, wonder, engagement",
    5: "COMPLEX — confusion, shame, embarrassment, mixed feelings, ambivalence",
}

def generate_batch(lang, chamber_id, n=50):
    """Generate n labeled sentences for a specific chamber in a language."""
    lang_names = {'ru': 'Russian', 'he': 'Hebrew', 'fr': 'French', 'en': 'English'}
    lang_name = lang_names.get(lang, lang)
    chamber_desc = CHAMBERS[chamber_id]

    prompt = f"""Generate exactly {n} short sentences (1-2 sentences each) in {lang_name} that clearly express the emotion: {chamber_desc}

Requirements:
- Each sentence should be a natural, authentic expression of this specific emotion
- Varied contexts: relationships, work, daily life, existential, physical sensations
- Mix of intensities: mild to extreme
- No labels or numbers, just one sentence per line
- Must be clearly THIS emotion, not ambiguous
- Native-sounding, not translated
- No quotation marks"""

    try:
        response = client.chat.completions.create(
            model="gpt-4.1",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.9,
            max_tokens=4000,
        )
        text = response.choices[0].message.content
        lines = [l.strip() for l in text.strip().split('\n') if l.strip() and len(l.strip()) > 5]
        # Remove numbering if present
        cleaned = []
        for l in lines:
            # Strip "1. " or "- " prefixes
            if len(l) > 3 and l[0].isdigit() and (l[1] == '.' or l[1] == ')'):
                l = l[2:].strip()
            elif len(l) > 4 and l[:2].isdigit() and (l[2] == '.' or l[2] == ')'):
                l = l[3:].strip()
            if l.startswith('- '):
                l = l[2:]
            if len(l) > 5:
                cleaned.append(l)
        return cleaned
    except Exception as e:
        print('  ERROR: %s' % e)
        return []

def main():
    output_dir = '/home/ubuntu/klaus/data'

    # Languages and chambers that need more data
    tasks = [
        # HE: missing RAGE, FLOW, COMPLEX entirely
        ('he', 0, 100),  # FEAR (only 296)
        ('he', 2, 150),  # RAGE (0!)
        ('he', 4, 150),  # FLOW (0!)
        ('he', 5, 150),  # COMPLEX (0!)
        ('he', 1, 50),   # LOVE (3123, enough but add variety)
        # RU: weak FEAR (589), no VOID in GoEmotions
        ('ru', 0, 150),  # FEAR
        ('ru', 3, 100),  # VOID (only from CEDR)
        ('ru', 5, 100),  # COMPLEX (1202)
        ('ru', 2, 100),  # RAGE (more variety)
        # EN: weak FEAR (562)
        ('en', 0, 100),  # FEAR
        ('en', 3, 100),  # VOID (2207)
        # FR: weak FEAR (584)
        ('fr', 0, 100),  # FEAR
        ('fr', 3, 100),  # VOID (2150)
    ]

    all_data = {}
    for lang, ch, n in tasks:
        key = lang
        if key not in all_data:
            all_data[key] = []

        print('%s %s: generating %d...' % (lang.upper(), CHAMBERS[ch].split(' —')[0], n))
        # Generate in batches of 50
        generated = []
        for batch_start in range(0, n, 50):
            batch_n = min(50, n - batch_start)
            lines = generate_batch(lang, ch, batch_n)
            generated.extend(lines)
            if batch_start + 50 < n:
                time.sleep(1)

        for text in generated:
            all_data[key].append({'text': text, 'chamber': ch})
        print('  got %d sentences' % len(generated))

    # Save per language
    for lang, items in all_data.items():
        path = os.path.join(output_dir, '%s_gpt_emotion.jsonl' % lang)
        with open(path, 'w') as f:
            for d in items:
                f.write(json.dumps(d, ensure_ascii=False) + '\n')

        # Count per chamber
        ch_count = {}
        for d in items:
            ch_count[d['chamber']] = ch_count.get(d['chamber'], 0) + 1
        ch_str = ' '.join('%s=%d' % (CHAMBERS[k].split(' —')[0], v) for k, v in sorted(ch_count.items()))
        print('\nSaved %s: %d samples | %s' % (path, len(items), ch_str))

    print('\nDONE. Total: %d samples across %d languages.' % (
        sum(len(v) for v in all_data.values()), len(all_data)))

if __name__ == '__main__':
    main()
