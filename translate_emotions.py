#!/usr/bin/env python3
"""Translate EN emotion dataset to RU/FR/HE using GPT-4.1 batch API."""

import os, json, time, sys
from openai import OpenAI
from datasets import load_from_disk, Dataset

API_KEY = os.environ.get('OPENAI_API_KEY', '')
client = OpenAI(api_key=API_KEY)

LABEL_NAMES = {0:'sadness', 1:'joy', 2:'love', 3:'anger', 4:'fear', 5:'surprise'}
LANG_PROMPTS = {
    'ru': 'Translate the following English text to Russian. Return ONLY the translation, nothing else.',
    'fr': 'Translate the following English text to French. Return ONLY the translation, nothing else.',
    'he': 'Translate the following English text to Hebrew. Return ONLY the translation, nothing else.',
}

def translate_batch(texts, lang, batch_size=50):
    """Translate a batch of texts using GPT-4.1-mini for speed/cost."""
    prompt = LANG_PROMPTS[lang]
    results = []
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        # Pack multiple texts into one request for efficiency
        numbered = '\n'.join(f'{j+1}. {t}' for j, t in enumerate(batch))
        
        try:
            resp = client.chat.completions.create(
                model='gpt-4.1-mini',
                messages=[
                    {'role': 'system', 'content': f'{prompt} The input contains numbered lines. Translate each line and keep the numbering. Do not add or remove lines.'},
                    {'role': 'user', 'content': numbered}
                ],
                temperature=0.3,
                max_tokens=4096,
            )
            
            output = resp.choices[0].message.content.strip()
            lines = []
            for line in output.split('\n'):
                line = line.strip()
                if line and line[0].isdigit():
                    # Remove numbering
                    parts = line.split('.', 1)
                    if len(parts) > 1:
                        lines.append(parts[1].strip())
                    else:
                        lines.append(line)
                elif line:
                    lines.append(line)
            
            # Pad or trim to match input
            while len(lines) < len(batch):
                lines.append(batch[len(lines)])  # fallback to original
            lines = lines[:len(batch)]
            results.extend(lines)
            
        except Exception as e:
            print(f'  ERROR at batch {i}: {e}')
            results.extend(batch)  # fallback to original
            time.sleep(5)
            continue
        
        if (i // batch_size) % 20 == 0:
            print(f'  {lang}: {len(results)}/{len(texts)} translated')
    
    return results

def main():
    lang = sys.argv[1] if len(sys.argv) > 1 else 'ru'
    max_samples = int(sys.argv[2]) if len(sys.argv) > 2 else 1000  # start small to test
    
    print(f'Loading EN dataset...')
    ds = load_from_disk('/home/ubuntu/klaus/data/en')
    
    # Get texts and labels
    if 'train' in ds:
        texts = [x['text'] for x in ds['train']][:max_samples]
        labels = [x['label'] for x in ds['train']][:max_samples]
    else:
        texts = [x['text'] for x in ds][:max_samples]
        labels = [x['label'] for x in ds][:max_samples]
    
    print(f'Translating {len(texts)} texts to {lang}...')
    translated = translate_batch(texts, lang)
    
    # Save
    out_ds = Dataset.from_dict({'text': translated, 'label': labels})
    out_path = f'/home/ubuntu/klaus/data/{lang}_translated'
    out_ds.save_to_disk(out_path)
    print(f'Saved {len(translated)} to {out_path}')
    
    # Show samples
    print(f'\nSamples:')
    for i in range(min(5, len(translated))):
        print(f'  [{LABEL_NAMES[labels[i]]}] EN: {texts[i][:60]}')
        print(f'  [{LABEL_NAMES[labels[i]]}] {lang.upper()}: {translated[i][:60]}')
        print()

if __name__ == '__main__':
    main()
