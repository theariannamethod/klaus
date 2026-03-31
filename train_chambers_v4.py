#!/usr/bin/env python3
"""
Klaus Phase 2c — Chambers training.
Dual loss: LM (low lr) + chamber classification (high lr) on emotion data.
Loads 4 pre-trained LM weights. Tunneling: hidden states from layer 3 (mid-network).
"""
import os, sys, math, json, random, time, argparse, shutil
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import sentencepiece as spm

# ─── Config ───
DIM=384; N_HEADS=6; N_LAYERS=6; HDIM=768; VOCAB=4096; MAX_SEQ=256; DROPOUT=0.1
RES_DIM=100; N_CHAMBERS=6; CF_ITERS=5; CF_K=0.02
BATCH=32; LR_CH=5e-4; LR_LM=0; WARMUP=300; WD=0.1  # LR_LM=0: freeze LM weights
WEIGHTS_DIR='/home/ubuntu/klaus/weights'
DATA_DIR='/home/ubuntu/klaus/data'
CNAMES=['FEAR','LOVE','RAGE','VOID','FLOW','COMPLEX']
DECAY=[0.90,0.93,0.85,0.97,0.88,0.94]
LANGS=['en','he','ru','fr']

# GoEmotions → 6 chambers
GO_LABELS=['admiration','amusement','anger','annoyance','approval','caring',
           'confusion','curiosity','desire','disappointment','disapproval',
           'disgust','embarrassment','excitement','fear','gratitude','grief',
           'joy','love','nervousness','optimism','pride','realization',
           'relief','remorse','sadness','surprise','neutral']
GO_TO_CH={'admiration':4,'amusement':4,'approval':4,'excitement':4,
          'gratitude':1,'joy':1,'love':1,'optimism':4,'relief':5,
          'pride':5,'desire':1,'caring':1,'sadness':3,'grief':3,
          'remorse':3,'disappointment':3,'embarrassment':5,'anger':2,
          'annoyance':2,'disapproval':2,'disgust':2,'fear':0,
          'nervousness':0,'confusion':5,'surprise':5,'realization':5,
          'curiosity':4,'neutral':-1}
HE_MAP={0:3, 1:1, 2:0}
RU_CEDR_MAP={0:1, 1:3, 2:5, 3:0, 4:2}

# ─── Model (same arch as train_klaus.py) ───
class RMSNorm(nn.Module):
    def __init__(s,d): super().__init__(); s.w=nn.Parameter(torch.ones(d))
    def forward(s,x): return x*torch.rsqrt(x.pow(2).mean(-1,keepdim=True)+1e-6)*s.w

class Attn(nn.Module):
    def __init__(s):
        super().__init__(); s.nh=N_HEADS; s.hd=DIM//N_HEADS
        s.wq=nn.Linear(DIM,DIM,bias=False); s.wk=nn.Linear(DIM,DIM,bias=False)
        s.wv=nn.Linear(DIM,DIM,bias=False); s.wo=nn.Linear(DIM,DIM,bias=False)
    def forward(s,x):
        B,S,_=x.shape
        q=s.wq(x).view(B,S,s.nh,s.hd).transpose(1,2)
        k=s.wk(x).view(B,S,s.nh,s.hd).transpose(1,2)
        v=s.wv(x).view(B,S,s.nh,s.hd).transpose(1,2)
        o=F.scaled_dot_product_attention(q,k,v,is_causal=True,dropout_p=DROPOUT if s.training else 0.0)
        return s.wo(o.transpose(1,2).contiguous().view(B,S,-1))

class Block(nn.Module):
    def __init__(s):
        super().__init__()
        s.ln1=RMSNorm(DIM); s.attn=Attn(); s.ln2=RMSNorm(DIM)
        s.w1=nn.Linear(DIM,HDIM,bias=False); s.w2=nn.Linear(DIM,HDIM,bias=False)
        s.w3=nn.Linear(HDIM,DIM,bias=False); s.drop=nn.Dropout(DROPOUT)
    def forward(s,x):
        x=x+s.attn(s.ln1(x)); h=s.ln2(x)
        x=x+s.drop(s.w3(F.silu(s.w1(h))*s.w2(h))); return x

class KlausLM(nn.Module):
    """Same as Klaus in train_klaus.py but exposes mid-layer hidden states."""
    def __init__(s):
        super().__init__()
        s.tok=nn.Embedding(VOCAB,DIM); s.pos=nn.Embedding(MAX_SEQ,DIM)
        s.blocks=nn.ModuleList([Block() for _ in range(N_LAYERS)])
        s.norm=RMSNorm(DIM); s.head=nn.Linear(DIM,VOCAB,bias=False)
        s.head.weight=s.tok.weight; s.drop=nn.Dropout(DROPOUT)
    def forward(s,x):
        B,S=x.shape
        h=s.drop(s.tok(x)+s.pos(torch.arange(S,device=x.device)))
        mid=None
        for i,b in enumerate(s.blocks):
            h=b(h)
            if i==2: mid=h  # layer 3 = mid-network tunneling
        h=s.norm(h)
        return s.head(h), mid  # logits + mid-layer hidden states

# ─── Chambers ───
class Chambers(nn.Module):
    def __init__(s):
        super().__init__()
        s.ch=nn.ModuleList([nn.Sequential(
            nn.Linear(RES_DIM,128),nn.SiLU(),nn.Linear(128,64),nn.SiLU(),
            nn.Linear(64,32),nn.SiLU(),nn.Linear(32,1)
        ) for _ in range(N_CHAMBERS)])
        s.coupling=nn.Parameter(torch.tensor([
            [ 0.0, -0.3,  0.6,  0.4, -0.2,  0.3],  # FEAR
            [-0.3,  0.0, -0.5, -0.7,  0.6,  0.4],  # LOVE
            [ 0.6, -0.5,  0.0,  0.3, -0.3,  0.2],  # RAGE
            [ 0.4, -0.7,  0.3,  0.0, -0.4,  0.5],  # VOID
            [-0.2,  0.6, -0.3, -0.4,  0.0,  0.3],  # FLOW
            [ 0.3,  0.4,  0.2,  0.5,  0.3,  0.0],  # COMPLEX
        ],dtype=torch.float32))
        s.decay=torch.tensor(DECAY)
    def forward(s,res):
        raw=torch.stack([c(res).squeeze(-1) for c in s.ch],dim=1) # (B,6)
        act=torch.sigmoid(raw); d=s.decay.to(res.device)
        for _ in range(CF_ITERS):
            act=act*d; old=act.clone()
            infl=CF_K*torch.sin(old.unsqueeze(1)-old.unsqueeze(2)) # (B,6,6)
            act=act+(infl*s.coupling.unsqueeze(0)).sum(2)
            act=act.clamp(0,1)
        return act, raw

# ─── Full model ───
class KlausFull(nn.Module):
    def __init__(s):
        super().__init__()
        s.lms=nn.ModuleDict({l:KlausLM() for l in LANGS})
        s.res_projs=nn.ModuleDict({l:nn.Linear(DIM,RES_DIM) for l in LANGS})
        s.chambers=Chambers()
    def load_weights(s):
        for l in LANGS:
            p=f'{WEIGHTS_DIR}/klaus_{l}.pt'
            if os.path.exists(p):
                ck=torch.load(p,map_location='cpu',weights_only=False)
                # Map old keys (no mid output) to new model
                s.lms[l].load_state_dict(ck['model'])
                print(f'  {l.upper()} loaded (val={ck.get("val_loss","?"):.4f})')
    def forward(s,tokens,lang):
        logits,mid=s.lms[lang](tokens) # mid = layer 3 hidden states
        pooled=mid.mean(dim=1) # (B,DIM)
        res=s.res_projs[lang](pooled) # (B,RES_DIM)
        act,raw=s.chambers(res)
        return logits, act, raw

# ─── Data loading ───
def load_emo():
    data={l:[] for l in LANGS}
    # EN GoEmotions
    p=f'{DATA_DIR}/en_goemotions.jsonl'
    if os.path.exists(p):
        with open(p) as f:
            for line in f:
                d=json.loads(line)
                for li in d['labels']:
                    ln=GO_LABELS[li] if li<len(GO_LABELS) else 'neutral'
                    c=GO_TO_CH.get(ln,-1)
                    if c>=0: data['en'].append((d['text'],c))
    # FR
    p=f'{DATA_DIR}/fr_go_emotions.jsonl'
    if os.path.exists(p):
        with open(p) as f:
            for line in f:
                d=json.loads(line); labs=d['labels']
                if isinstance(labs,str): labs=json.loads(labs)
                for li in labs:
                    li=int(li); ln=GO_LABELS[li] if li<len(GO_LABELS) else 'neutral'
                    c=GO_TO_CH.get(ln,-1)
                    if c>=0: data['fr'].append((d['text'],c))
    # HE
    p=f'{DATA_DIR}/he_sentiment_real.jsonl'
    if os.path.exists(p):
        with open(p) as f:
            for line in f:
                d=json.loads(line); c=HE_MAP.get(d.get('label',-1),-1)
                if c>=0: data['he'].append((d['text'],c))
    # RU CEDR
    p=f'{DATA_DIR}/ru_cedr.jsonl'
    if os.path.exists(p):
        with open(p) as f:
            for line in f:
                d=json.loads(line)
                for li in d.get('labels',[]):
                    c=RU_CEDR_MAP.get(li,-1)
                    if c>=0: data['ru'].append((d['text'],c))
    # RU GoEmotions
    p=f'{DATA_DIR}/ru_goemotions.jsonl'
    if os.path.exists(p):
        with open(p) as f:
            for line in f:
                d=json.loads(line); labs=d.get('labels',[])
                if isinstance(labs,str): labs=json.loads(labs)
                for li in labs:
                    li=int(li); ln=GO_LABELS[li] if li<len(GO_LABELS) else 'neutral'
                    c=GO_TO_CH.get(ln,-1)
                    if c>=0: data['ru'].append((d['text'],c))
    # RU sentiment (negative → RAGE/VOID by keyword)
    p=f'{DATA_DIR}/ru_sentiment_trimmed.jsonl'
    if os.path.exists(p):
        rage_w=['ужас','кошмар','отврат','хам','грубо','нагл','безобраз','обман','мрази','идиот','тупо','бесит','злит','ненавижу','дерьм','говно','скот','урод','хуж','отстой','позор']
        void_w=['грустн','печальн','тоскл','пуст','безразлич','одинок','уныл','апати','депресс','скучн','бессмысл','устал','разочаров','потеря','тоска','горе','больно']
        n_rage=n_void=0
        with open(p) as f:
            for line in f:
                d=json.loads(line)
                if d.get('sentiment')!=2: continue
                t=d['text'].lower()
                if any(w in t for w in rage_w):
                    data['ru'].append((d['text'],2)); n_rage+=1  # RAGE
                elif any(w in t for w in void_w):
                    data['ru'].append((d['text'],3)); n_void+=1  # VOID
        print(f'  RU sentiment: +{n_rage} RAGE, +{n_void} VOID')
    # Cap LOVE for RU (prevent domination)
    by_ch_ru = {}
    for t,c in data['ru']:
        by_ch_ru.setdefault(c,[]).append((t,c))
    if len(by_ch_ru.get(1,[])) > 5000:
        import random as _r; _r.shuffle(by_ch_ru[1])
        by_ch_ru[1] = by_ch_ru[1][:5000]
        data['ru'] = []
        for c in by_ch_ru: data['ru'].extend(by_ch_ru[c])
        print(f'  RU LOVE capped at 5000')
    for l in LANGS:
        # Print per-chamber distribution
        ch_dist = {}
        for t,c in data[l]: ch_dist[c] = ch_dist.get(c,0)+1
        dist_str = ' '.join(f'{CNAMES[i]}={ch_dist.get(i,0)}' for i in range(6))
        print(f'  {l.upper()}: {len(data[l]):,} samples | {dist_str}')
    return data

def enc_batch(texts,sp):
    ids=[sp.encode(str(t))[:MAX_SEQ] for t in texts]
    mx=max(len(x) for x in ids)
    return torch.tensor([x+[0]*(mx-len(x)) for x in ids],dtype=torch.long)

# ─── Train ───
def train(steps):
    dev=torch.device('cuda')
    print(f'\n{"="*60}\n KLAUS Phase 2c — Chambers Training\n {steps//1000}K steps\n{"="*60}\n')
    sps={}
    for l in LANGS:
        sp=spm.SentencePieceProcessor(); sp.load(f'{WEIGHTS_DIR}/klaus_{l}_bpe.model'); sps[l]=sp
    print('Loading emotion data...'); emo=load_emo()
    active=[l for l in LANGS if len(emo[l])>0]
    print(f'Active langs: {active}')
    print('\nBuilding model...')
    model=KlausFull().to(dev); model.load_weights()
    # Load existing chambers checkpoint
    ckpt_path=WEIGHTS_DIR+'/klaus_chambers.pt'
    if os.path.exists(ckpt_path):
        ckpt=torch.load(ckpt_path,map_location=dev,weights_only=False)
        model.chambers.load_state_dict(ckpt['chambers'])
        if 'res_projs' in ckpt:
            for l in LANGS:
                if l in ckpt['res_projs']:
                    model.res_projs[l].load_state_dict(ckpt['res_projs'][l])
        if 'lms' in ckpt:
            for l in LANGS:
                if l in ckpt['lms']:
                    model.lms[l].load_state_dict(ckpt['lms'][l])
        print('  Chambers loaded from checkpoint (continuing)')
    # Freeze LM weights
    for l in LANGS:
        for p in model.lms[l].parameters(): p.requires_grad=False
    n_train=sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'  Trainable: {n_train:,} (chambers + res_projs)')
    # Optimizer — only chambers + res_projs
    opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                          lr=LR_CH,weight_decay=WD,betas=(0.9,0.95))
    # Focal loss: focus on hard examples
    class FocalLoss(nn.Module):
        def __init__(self, gamma=2.0):
            super().__init__()
            self.gamma = gamma
        def forward(self, logits, targets):
            ce = F.cross_entropy(logits, targets, reduction='none')
            pt = torch.exp(-ce)
            return ((1 - pt) ** self.gamma * ce).mean()
    crit=FocalLoss(gamma=2.0)
    print(f'{"─"*60}')
    best_acc=0; t0=time.time(); losses=[]; accs=[]
    for step in range(1,steps+1):
        model.train()
        lang=random.choice(active)
        # Balanced sampling: equal samples per chamber
        lang_data=emo[lang]
        by_ch=[[] for _ in range(N_CHAMBERS)]
        for t,c in lang_data: by_ch[c].append((t,c))
        batch=[]
        per_ch=max(1,BATCH//N_CHAMBERS)
        for c in range(N_CHAMBERS):
            if len(by_ch[c])>0:
                batch.extend(random.choices(by_ch[c],k=per_ch))
        random.shuffle(batch)
        batch=batch[:BATCH]
        texts=[t for t,c in batch]
        tgt=torch.tensor([c for t,c in batch],dtype=torch.long,device=dev)
        tokens=enc_batch(texts,sps[lang]).to(dev)
        _,act,raw=model(tokens,lang)
        loss=crit(raw,tgt)
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.0)
        opt.step()
        losses.append(loss.item())
        pred=raw.argmax(dim=1); accs.append((pred==tgt).float().mean().item())
        if step%100==0:
            al=sum(losses[-100:])/100; aa=sum(accs[-100:])/100
            print(f'step {step:6d}/{steps} | loss={al:.4f} | acc={aa:.1%} | {time.time()-t0:.0f}s')
            sys.stdout.flush()
        if step%2000==0 or step==steps:
            model.eval(); print(f'\n  >>> EVAL step {step}')
            for el in active:
                samp=random.sample(emo[el],min(200,len(emo[el])))
                tx=[t for t,c in samp]; tg=torch.tensor([c for t,c in samp],dtype=torch.long,device=dev)
                with torch.no_grad():
                    tk=enc_batch(tx,sps[el]).to(dev); _,a,r=model(tk,el)
                    l=crit(r,tg).item(); ac=(r.argmax(1)==tg).float().mean().item()
                    cd=a.mean(0).cpu().numpy()
                print(f'  {el.upper()}: loss={l:.4f} acc={ac:.1%} | '+' '.join(f'{CNAMES[i]}={cd[i]:.2f}' for i in range(6)))
            ta=sum(accs[-500:])/max(len(accs[-500:]),1)
            if ta>best_acc:
                best_acc=ta
                sd_ch=model.chambers.state_dict()
                sd_rp={l:model.res_projs[l].state_dict() for l in LANGS}
                torch.save({'chambers':sd_ch,'res_projs':sd_rp,'step':step,'acc':ta},
                           f'{WEIGHTS_DIR}/klaus_chambers.pt')
                print(f'  SAVED (acc={ta:.1%})')
            print(); sys.stdout.flush()
    dt=time.time()-t0
    print(f'{"="*60}\nDONE: {steps} steps in {dt/60:.1f} min | best acc={best_acc:.1%}')
    print(f'Chambers: {WEIGHTS_DIR}/klaus_chambers.pt\n{"="*60}')

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--steps',type=int,default=20000)
    train(p.parse_args().steps)
