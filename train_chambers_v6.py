#!/usr/bin/env python3
"""
Klaus Chambers v6 — bigger chambers + clean data + GPT data.
Train from scratch, joint LM (tiny lr) + chambers (high lr).
"""
import os, sys, math, json, random, time, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import sentencepiece as spm

# Config
DIM=384; N_HEADS=6; N_LAYERS=6; HDIM=768; VOCAB=4096; MAX_SEQ=256; DROPOUT=0.1
RES_DIM=150  # was 100, now 150 for richer representation
N_CHAMBERS=6; CF_ITERS=5; CF_K=0.02
BATCH=32; LR_CH=1e-3; LR_LM=0; WARMUP=500; WD=0.1
WEIGHTS_DIR='/home/ubuntu/klaus/weights'
DATA_DIR='/home/ubuntu/klaus/data'
CNAMES=['FEAR','LOVE','RAGE','VOID','FLOW','COMPLEX']
DECAY=[0.90,0.93,0.85,0.97,0.88,0.94]
LANGS=['en','he','ru','fr']

GO_LABELS=['admiration','amusement','anger','annoyance','approval','caring',
           'confusion','curiosity','desire','disappointment','disapproval',
           'disgust','embarrassment','excitement','fear','gratitude','grief',
           'joy','love','nervousness','optimism','pride','realization',
           'relief','remorse','sadness','surprise','neutral']
GO_TO_CH={'admiration':1,'amusement':4,'approval':1,'excitement':4,
          'gratitude':1,'joy':1,'love':1,'optimism':4,'relief':4,
          'pride':1,'desire':1,'caring':1,'sadness':3,'grief':3,
          'remorse':3,'disappointment':3,'embarrassment':5,'anger':2,
          'annoyance':2,'disapproval':2,'disgust':2,'fear':0,
          'nervousness':0,'confusion':5,'surprise':5,'realization':5,
          'curiosity':4,'neutral':-1}
HE_MAP={0:3, 1:1, 2:0}
RU_CEDR_MAP={0:1, 1:3, 2:5, 3:0, 4:2}

# Import working model from train_chambers.py
sys.path.insert(0,'/home/ubuntu/klaus')
from train_chambers import KlausFull as _KlausFull, KlausLM, Attn, Block, RMSNorm

class Chambers(nn.Module):
    def __init__(s):
        super().__init__()
        # BIGGER: 150->256->128->64->1
        s.ch=nn.ModuleList([nn.Sequential(
            nn.Linear(RES_DIM,256),nn.SiLU(),nn.Dropout(0.1),
            nn.Linear(256,128),nn.SiLU(),nn.Dropout(0.1),
            nn.Linear(128,64),nn.SiLU(),
            nn.Linear(64,1)
        ) for _ in range(N_CHAMBERS)])
        s.coupling=nn.Parameter(torch.tensor([
            [ 0.0, -0.3,  0.6,  0.4, -0.2,  0.3],
            [-0.3,  0.0, -0.5, -0.7,  0.6,  0.4],
            [ 0.6, -0.5,  0.0,  0.3, -0.3,  0.2],
            [ 0.4, -0.7,  0.3,  0.0, -0.4,  0.5],
            [-0.2,  0.6, -0.3, -0.4,  0.0,  0.3],
            [ 0.3,  0.4,  0.2,  0.5,  0.3,  0.0]],dtype=torch.float32))
        s.decay=torch.tensor(DECAY)
    def forward(s,res):
        raw=torch.stack([c(res).squeeze(-1) for c in s.ch],dim=1)
        act=torch.sigmoid(raw); d=s.decay.to(res.device)
        for _ in range(CF_ITERS):
            act=act*d
            delta=torch.matmul(act,s.coupling.to(act.device))*CF_K
            act=torch.sigmoid(raw+delta)
        return act, raw

class KlausFull(nn.Module):
    def __init__(s):
        super().__init__()
        s.lms=nn.ModuleDict({l:KlausLM() for l in LANGS})
        s.res_projs=nn.ModuleDict({l:nn.Linear(DIM,RES_DIM) for l in LANGS})
        s.chambers=Chambers()
    def load_weights(s):
        for l in LANGS:
            p=WEIGHTS_DIR+'/klaus_'+l+'.pt'
            if os.path.exists(p):
                ck=torch.load(p,map_location='cpu',weights_only=False)
                s.lms[l].load_state_dict(ck['model'])
                print('  %s loaded (val=%.4f)'%(l.upper(),ck.get('val_loss',0)))
    def forward(s,tokens,lang):
        logits,mid=s.lms[lang](tokens)
        pooled=mid.mean(dim=1)
        res=s.res_projs[lang](pooled)
        act,raw=s.chambers(res)
        return logits, act, raw

# Data loading — clean + GPT
def load_emo():
    data={l:[] for l in LANGS}

    # EN: clean GoEmotions + GPT
    for path in [DATA_DIR+'/en_goemotions_final.jsonl', DATA_DIR+'/en_gpt_emotion.jsonl']:
        if os.path.exists(path):
            with open(path) as f:
                for line in f:
                    d=json.loads(line)
                    if 'chamber' in d:
                        data['en'].append((d['text'],d['chamber']))
                    else:
                        for li in d.get('labels',[]):
                            ln=GO_LABELS[li] if li<len(GO_LABELS) else 'neutral'
                            c=GO_TO_CH.get(ln,-1)
                            if c>=0: data['en'].append((d['text'],c))

    # FR: clean GoEmotions + GPT
    for path in [DATA_DIR+'/fr_goemotions_final.jsonl', DATA_DIR+'/fr_gpt_emotion.jsonl']:
        if os.path.exists(path):
            with open(path) as f:
                for line in f:
                    d=json.loads(line)
                    if 'chamber' in d:
                        data['fr'].append((d['text'],d['chamber']))
                    else:
                        labs=d.get('labels',[])
                        if isinstance(labs,str): labs=json.loads(labs)
                        for li in labs:
                            li=int(li); ln=GO_LABELS[li] if li<len(GO_LABELS) else 'neutral'
                            c=GO_TO_CH.get(ln,-1)
                            if c>=0: data['fr'].append((d['text'],c))

    # HE: sentiment + GPT
    p=DATA_DIR+'/he_sentiment_real.jsonl'
    if os.path.exists(p):
        with open(p) as f:
            for line in f:
                d=json.loads(line); c=HE_MAP.get(d.get('label',-1),-1)
                if c>=0: data['he'].append((d['text'],c))
    p=DATA_DIR+'/he_gpt_emotion.jsonl'
    if os.path.exists(p):
        with open(p) as f:
            for line in f:
                d=json.loads(line)
                if 'chamber' in d:
                    data['he'].append((d['text'],d['chamber']))

    # RU: CEDR + clean GoEmotions + sentiment + GPT
    p=DATA_DIR+'/ru_cedr.jsonl'
    if os.path.exists(p):
        with open(p) as f:
            for line in f:
                d=json.loads(line)
                for li in d.get('labels',[]):
                    c=RU_CEDR_MAP.get(li,-1)
                    if c>=0: data['ru'].append((d['text'],c))
    p=DATA_DIR+'/ru_goemotions_final.jsonl'
    if os.path.exists(p):
        with open(p) as f:
            for line in f:
                d=json.loads(line)
                if 'chamber' in d:
                    data['ru'].append((d['text'],d['chamber']))
                else:
                    labs=d.get('labels',[])
                    if isinstance(labs,str): labs=json.loads(labs)
                    for li in labs:
                        li=int(li); ln=GO_LABELS[li] if li<len(GO_LABELS) else 'neutral'
                        c=GO_TO_CH.get(ln,-1)
                        if c>=0: data['ru'].append((d['text'],c))
    # RU sentiment negative
    p=DATA_DIR+'/ru_sentiment_trimmed.jsonl'
    if os.path.exists(p):
        rage_w=['ужас','кошмар','отврат','хам','грубо','нагл','безобраз','обман','мрази','идиот','тупо','бесит','злит','ненавижу','дерьм','говно','скот','урод','хуж','отстой','позор']
        void_w=['грустн','печальн','тоскл','пуст','безразлич','одинок','уныл','апати','депресс','скучн','бессмысл','устал','разочаров','потеря','тоска','горе','больно']
        with open(p) as f:
            for line in f:
                d=json.loads(line)
                if d.get('sentiment')!=2: continue
                t=d['text'].lower()
                if any(w in t for w in rage_w): data['ru'].append((d['text'],2))
                elif any(w in t for w in void_w): data['ru'].append((d['text'],3))
    # RU GPT
    p=DATA_DIR+'/ru_gpt_emotion.jsonl'
    if os.path.exists(p):
        with open(p) as f:
            for line in f:
                d=json.loads(line)
                if 'chamber' in d:
                    data['ru'].append((d['text'],d['chamber']))

    # Cap LOVE for RU
    by_ch_ru={}
    for t,c in data['ru']: by_ch_ru.setdefault(c,[]).append((t,c))
    if len(by_ch_ru.get(1,[]))>5000:
        random.shuffle(by_ch_ru[1]); by_ch_ru[1]=by_ch_ru[1][:5000]
        data['ru']=[];
        for c in by_ch_ru: data['ru'].extend(by_ch_ru[c])

    for l in LANGS:
        ch_dist={}
        for t,c in data[l]: ch_dist[c]=ch_dist.get(c,0)+1
        dist_str=' '.join('%s=%d'%(CNAMES[i],ch_dist.get(i,0)) for i in range(6))
        print('  %s: %d samples | %s'%(l.upper(),len(data[l]),dist_str))
    return data

def enc_batch(texts,sp):
    ids=[sp.encode(str(t))[:MAX_SEQ] for t in texts]
    mx=max(len(x) for x in ids)
    return torch.tensor([x+[0]*(mx-len(x)) for x in ids],dtype=torch.long)

def train(steps):
    dev=torch.device('cuda')
    print('\n'+'='*60)
    print(' KLAUS Chambers v6 — bigger + clean + GPT data')
    print(' %dK steps' % (steps//1000))
    print('='*60+'\n')
    sps={}
    for l in LANGS:
        sp=spm.SentencePieceProcessor(); sp.load(WEIGHTS_DIR+'/klaus_%s_bpe.model'%l); sps[l]=sp
    print('Loading emotion data...'); emo=load_emo()
    active=[l for l in LANGS if len(emo[l])>0]
    print('Active langs: %s\n' % active)
    print('Building model...')
    model=KlausFull().to(dev); model.load_weights()
    # Freeze LM
    for l in LANGS:
        for p in model.lms[l].parameters(): p.requires_grad=False
    n_train=sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('  Trainable: %d (chambers=%d + res_projs=%d)' % (
        n_train,
        sum(p.numel() for p in model.chambers.parameters()),
        sum(p.numel() for n,p in model.named_parameters() if 'res_proj' in n)))

    opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                          lr=LR_CH,weight_decay=WD,betas=(0.9,0.95))
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=steps,eta_min=1e-5)
    crit=nn.CrossEntropyLoss()
    print('-'*60)

    best_acc=0; t0=time.time(); losses=[]; accs=[]
    for step in range(1,steps+1):
        model.train()
        lang=random.choice(active)
        lang_data=emo[lang]
        by_ch=[[] for _ in range(N_CHAMBERS)]
        for t,c in lang_data: by_ch[c].append((t,c))
        batch=[]
        per_ch=max(1,BATCH//N_CHAMBERS)
        for c in range(N_CHAMBERS):
            if len(by_ch[c])>0:
                batch.extend(random.choices(by_ch[c],k=per_ch))
        random.shuffle(batch); batch=batch[:BATCH]
        texts=[t for t,c in batch]
        tgt=torch.tensor([c for t,c in batch],dtype=torch.long,device=dev)
        tokens=enc_batch(texts,sps[lang]).to(dev)
        _,act,raw=model(tokens,lang)
        loss=crit(raw,tgt)
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.0)
        opt.step(); sched.step()
        losses.append(loss.item())
        pred=raw.argmax(dim=1); accs.append((pred==tgt).float().mean().item())
        if step%100==0:
            al=sum(losses[-100:])/100; aa=sum(accs[-100:])/100
            print('step %6d/%d | loss=%.4f | acc=%.1f%% | %ds'%(step,steps,al,aa*100,time.time()-t0))
            sys.stdout.flush()
        if step%2000==0 or step==steps:
            model.eval(); print('\n  >>> EVAL step %d'%step)
            for el in active:
                samp=random.sample(emo[el],min(200,len(emo[el])))
                tx=[t for t,c in samp]; tg=torch.tensor([c for t,c in samp],dtype=torch.long,device=dev)
                with torch.no_grad():
                    tk=enc_batch(tx,sps[el]).to(dev); _,a,r=model(tk,el)
                    l=crit(r,tg).item(); ac=(r.argmax(1)==tg).float().mean().item()
                    cd=a.mean(0).cpu().numpy()
                cstr=' '.join('%s=%.2f'%(CNAMES[i],cd[i]) for i in range(6))
                print('  %s: loss=%.4f acc=%.1f%% | %s'%(el.upper(),l,ac*100,cstr))
            ta=sum(accs[-500:])/max(len(accs[-500:]),1)
            if ta>best_acc:
                best_acc=ta
                sd_ch=model.chambers.state_dict()
                sd_rp={l:model.res_projs[l].state_dict() for l in LANGS}
                sd_lms={l:model.lms[l].state_dict() for l in LANGS}
                torch.save({'chambers':sd_ch,'res_projs':sd_rp,'lms':sd_lms,'step':step,'acc':ta},
                           WEIGHTS_DIR+'/klaus_chambers_v6.pt')
                print('  SAVED (acc=%.1f%%)'%(ta*100))
            print(); sys.stdout.flush()
    dt=time.time()-t0
    print('='*60)
    print('DONE: %d steps in %.1f min | best acc=%.1f%%'%(steps,dt/60,best_acc*100))
    print('='*60)

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--steps',type=int,default=100000)
    train(p.parse_args().steps)
