#!/usr/bin/env python3
"""Generate somatic text corpus for Klaus LM fine-tuning via GPT-4.1."""
import json, os, time, sys
from openai import OpenAI

API_KEY = os.environ.get("OPENAI_API_KEY", "")
if not API_KEY:
    # Read from file
    try:
        with open('/home/ubuntu/klaus/.openai_key') as f:
            API_KEY = f.read().strip()
    except:
        pass

client = OpenAI(api_key=API_KEY)

CHAMBERS = ['FEAR','LOVE','RAGE','VOID','FLOW','COMPLEX']
CHAMBER_DESC = {
    'FEAR': 'fear, anxiety, panic, terror — body sensations of threat',
    'LOVE': 'love, warmth, tenderness, joy — body sensations of connection',
    'RAGE': 'anger, fury, hatred, frustration — body sensations of aggression',
    'VOID': 'emptiness, numbness, apathy, depression — body sensations of absence',
    'FLOW': 'curiosity, excitement, wonder, engagement — body sensations of exploration',
    'COMPLEX': 'confusion, shame, guilt, mixed emotions — body sensations of ambivalence',
}

def generate_batch(lang, chamber, n=100):
    lang_map = {'en':'English','ru':'Russian','fr':'French','he':'Hebrew'}
    lang_name = lang_map.get(lang, lang)
    ch_desc = CHAMBER_DESC[chamber]

    prompt = """Generate %d short somatic/body-sensation descriptions in %s.
Theme: %s (%s)

Rules:
- Each line is ONE body sensation description, 5-15 words
- First person: "I feel...", "my chest...", "hands are..."
- ONLY physical/body sensations, not thoughts or abstract emotions
- Natural, diverse, authentic language
- Mix intensity: subtle to overwhelming
- Include: body parts, temperature, movement, pressure, pain, texture
- NO numbering, NO quotes, just raw text, one per line

Examples (English, for reference):
my hands won't stop shaking
there is ice in my veins
my chest burns like fire
everything feels heavy and slow
a tingling runs down my spine
my throat closes up""" % (n, lang_name, chamber, ch_desc)

    try:
        response = client.chat.completions.create(
            model="gpt-4.1",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.95,
            max_tokens=8000,
        )
        text = response.choices[0].message.content
        lines = [l.strip() for l in text.strip().split('\n') if l.strip() and len(l.strip()) > 8]
        # Clean numbering
        cleaned = []
        for l in lines:
            if len(l) > 3 and l[0].isdigit() and l[1] in '.-)':
                l = l[2:].strip()
            elif len(l) > 4 and l[:2].isdigit() and l[2] in '.-)':
                l = l[3:].strip()
            if l.startswith('- '): l = l[2:]
            if len(l) > 8:
                cleaned.append(l)
        return cleaned
    except Exception as e:
        print('  ERROR: %s' % str(e)[:100])
        return []


def main():
    out_dir = '/home/ubuntu/klaus/data'
    n_per_chamber = 200  # 200 per chamber * 6 = 1200 per language

    for lang in ['en', 'ru', 'fr', 'he']:
        all_lines = []
        for ch in CHAMBERS:
            print('%s %s: generating %d...' % (lang.upper(), ch, n_per_chamber), end=' ', flush=True)
            lines = []
            # Generate in batches of 100
            for batch in range(0, n_per_chamber, 100):
                batch_n = min(100, n_per_chamber - batch)
                got = generate_batch(lang, ch, batch_n)
                lines.extend(got)
                if batch + 100 < n_per_chamber:
                    time.sleep(0.5)
            print('got %d' % len(lines))
            all_lines.extend(lines)

        path = os.path.join(out_dir, '%s_somatic_corpus.txt' % lang)
        with open(path, 'w') as f:
            for line in all_lines:
                f.write(line + '\n')
        print('%s: %d sentences -> %s\n' % (lang.upper(), len(all_lines), path))

    print('DONE.')


if __name__ == '__main__':
    main()
