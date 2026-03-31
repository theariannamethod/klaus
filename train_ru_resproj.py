#!/usr/bin/env python3
"""Fine-tune ONLY RU res_proj to fix RU chamber accuracy.
Chambers frozen (EN works). Only learn RU->chamber mapping."""
import os, sys, json, random, time, math
import torch, torch.nn as nn, torch.nn.functional as F
import sentencepiece as spm

DIM=384; N_HEADS=6; N_LAYERS=6; HDIM=768; VOCAB=4096; MAX_SEQ=256; DROPOUT=0.1
RES_DIM=100; N_CHAMBERS=6; CF_ITERS=5; CF_K=0.02; BATCH=32
WEIGHTS_DIR='/home/ubuntu/klaus/weights'
DATA_DIR='/home/ubuntu/klaus/data'
CNAMES=['FEAR','LOVE','RAGE','VOID','FLOW','COMPLEX']
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
RU_CEDR_MAP={0:1, 1:3, 2:5, 3:0, 4:2}

class RMSNorm(nn.Module):
    def __init__(s,d): super().__init__(); s.w=nn.Parameter(torch.ones(d))
    def forward(s,x): return x*torch.rsqrt(x.pow(2).mean(-1,keepdim=True)+1e-6)*s.w

class Attn(nn.Module):
    def __init__(s):
        super().__init__(); s.nh=N_HEADS; s.hd=DIM//N_HEADS
        s.wq=nn.Linear(DIM,DIM,bias=False); s.wk=nn.Linear(DIM,DIM,bias=False)
        s.wv=nn.Linear(DIM,DIM,bias=False); s.wo=nn.Linear(DIM,DIM,bias=False)
    def forward(s,x):
        B,T,_=x.shape; q=s.wq(x).view(B,T,s.nh,s.hd).transpose(1,2)
        k=s.wk(x).view(B,T,s.nh,s.hd).transpose(1,2)
        v=s.wv(x).view(B,T,s.nh,s.hd).transpose(1,2)
        a=torch.matmul(q,k.transpose(-2,-1))/math.sqrt(s.hd)
        m=torch.triu(torch.ones(T,T,device=x.device),1).bool()
        a.masked_fill_(m,float('-inf')); a=F.softmax(a,dim=-1)
        return s.wo(torch.matmul(a,v).transpose(1,2).contiguous().view(B,T,-1))

class Block(nn.Module):
    def __init__(s):
        super().__init__(); s.ln1=RMSNorm(DIM); s.attn=Attn()
        s.ln2=RMSNorm(DIM); s.ff=nn.Sequential(nn.Linear(DIM,HDIM),nn.GELU(),nn.Linear(HDIM,DIM))
    def forward(s,x): x=x+s.attn(s.ln1(x)); return x+s.ff(s.ln2(x))

class Chambers(nn.Module):
    def __init__(s):
        super().__init__()
        s.ch=nn.ModuleList([nn.Sequential(
            nn.Linear(RES_DIM,128),nn.SiLU(),
            nn.Linear(128,64),nn.SiLU(),
            nn.Linear(64,32),nn.SiLU(),
            nn.Linear(32,1)) for _ in range(N_CHAMBERS)])
        s.coupling=nn.Parameter(torch.tensor([
            [ 0.0, -0.3,  0.6,  0.4, -0.2,  0.3],
            [-0.3,  0.0, -0.5, -0.7,  0.6,  0.4],
            [ 0.6, -0.5,  0.0,  0.3, -0.3,  0.2],
            [ 0.4, -0.7,  0.3,  0.0, -0.4,  0.5],
            [-0.2,  0.6, -0.3, -0.4,  0.0,  0.3],
            [ 0.3,  0.4,  0.2,  0.5,  0.3,  0.0]]))
        s.decay=torch.tensor([0.90,0.93,0.85,0.97,0.88,0.94])
    def forward(s,res):
        raw=torch.stack([c(res).squeeze(-1) for c in s.ch],dim=1)
        act=torch.sigmoid(raw); d=s.decay.to(res.device)
        for _ in range(CF_ITERS):
            act=act*d
            delta=torch.matmul(act,s.coupling.to(act.device))*CF_K
            act=torch.sigmoid(raw+delta)
        return act, raw

class KlausLM(nn.Module):
    def __init__(s):
        super().__init__(); s.emb=nn.Embedding(VOCAB,DIM)
        s.blocks=nn.ModuleList([Block() for _ in range(N_LAYERS)])
        s.ln=RMSNorm(DIM); s.head=nn.Linear(DIM,VOCAB,bias=False)
    def forward(s,x):
        h=s.emb(x)
        mid=None
        for i,b in enumerate(s.blocks):
            h=b(h)
            if i==2: mid=h
        return s.head(s.ln(h)), mid

def load_ru_emo():
    data=[]
    p=DATA_DIR+'/ru_cedr.jsonl'
    if os.path.exists(p):
        with open(p) as f:
            for line in f:
                d=json.loads(line)
                for li in d.get('labels',[]):
                    c=RU_CEDR_MAP.get(li,-1)
                    if c>=0: data.append((d['text'],c))
    p=DATA_DIR+'/ru_goemotions.jsonl'
    if os.path.exists(p):
        with open(p) as f:
            for line in f:
                d=json.loads(line); labs=d.get('labels',[])
                if isinstance(labs,str): labs=json.loads(labs)
                for li in labs:
                    li=int(li)
                    ln=GO_LABELS[li] if li<len(GO_LABELS) else 'neutral'
                    c=GO_TO_CH.get(ln,-1)
                    if c>=0: data.append((d['text'],c))
    # Sentiment negative -> RAGE/VOID
    p=DATA_DIR+'/ru_sentiment_trimmed.jsonl'
    if os.path.exists(p):
        rage_w=['ужас','кошмар','отврат','хам','грубо','нагл','безобраз','обман',
                'мрази','идиот','тупо','бесит','злит','ненавижу','дерьм','говно',
                'скот','урод','хуж','отстой','позор']
        void_w=['грустн','печальн','тоскл','пуст','безразлич','одинок','уныл',
                'апати','депресс','скучн','бессмысл','устал','разочаров','потеря',
                'тоска','горе','больно']
        with open(p) as f:
            for line in f:
                d=json.loads(line)
                if d.get('sentiment')!=2: continue
                t=d['text'].lower()
                if any(w in t for w in rage_w): data.append((d['text'],2))
                elif any(w in t for w in void_w): data.append((d['text'],3))
    # Cap LOVE at 5000
    by_ch={}
    for t,c in data: by_ch.setdefault(c,[]).append((t,c))
    if len(by_ch.get(1,[]))>5000:
        random.shuffle(by_ch[1]); by_ch[1]=by_ch[1][:5000]
    data=[]
    for c in by_ch: data.extend(by_ch[c])
    for i in range(N_CHAMBERS):
        n=len([1 for _,c in data if c==i])
        print('  %s: %d' % (CNAMES[i], n))
    print('  Total: %d' % len(data))
    return data

def enc_batch(texts,sp):
    ids=[sp.encode(str(t))[:MAX_SEQ] for t in texts]
    mx=max(len(x) for x in ids)
    return torch.tensor([x+[0]*(mx-len(x)) for x in ids],dtype=torch.long)

def train(steps=15000, lr=3e-4):
    dev=torch.device('cuda')
    print('\n' + '='*60)
    print(' RU res_proj fine-tune (%d steps, lr=%s)' % (steps, lr))
    print('='*60 + '\n')

    # Load RU LM
    lm=KlausLM().to(dev)
    sd=torch.load(WEIGHTS_DIR+'/klaus_ru.pt',map_location=dev)
    lm.load_state_dict(sd['model'] if 'model' in sd else sd, strict=False)
    for p in lm.parameters(): p.requires_grad=False
    lm.eval()
    print('  RU LM loaded')

    # Load chambers (FROZEN)
    chambers=Chambers().to(dev)
    ckpt=torch.load(WEIGHTS_DIR+'/klaus_chambers.pt',map_location=dev)
    chambers.load_state_dict(ckpt['chambers'])
    for p in chambers.parameters(): p.requires_grad=False
    chambers.eval()
    print('  Chambers loaded (FROZEN)')

    # RU res_proj (TRAINABLE)
    res_proj=nn.Linear(DIM,RES_DIM).to(dev)
    if 'res_projs' in ckpt and 'ru' in ckpt['res_projs']:
        res_proj.load_state_dict(ckpt['res_projs']['ru'])
    n_params=sum(p.numel() for p in res_proj.parameters())
    print('  RU res_proj: %d params (TRAINABLE)' % n_params)

    # Load tokenizer + data
    sp=spm.SentencePieceProcessor(); sp.load(WEIGHTS_DIR+'/klaus_ru_bpe.model')
    print('\nLoading RU emotion data...')
    data=load_ru_emo()
    by_ch=[[] for _ in range(N_CHAMBERS)]
    for t,c in data: by_ch[c].append((t,c))

    # Class weights for loss
    counts=torch.tensor([max(1,len(by_ch[i])) for i in range(N_CHAMBERS)],
                        dtype=torch.float,device=dev)
    weights=counts.max()/counts
    weights=weights/weights.sum()*N_CHAMBERS
    wstr=' '.join('%s=%.2f' % (CNAMES[i],weights[i].item()) for i in range(N_CHAMBERS))
    print('  Class weights: %s' % wstr)

    crit=nn.CrossEntropyLoss(weight=weights)
    opt=torch.optim.AdamW(res_proj.parameters(),lr=lr,weight_decay=0.01)

    best_acc=0; t0=time.time(); losses=[]; accs=[]
    for step in range(1,steps+1):
        res_proj.train()
        batch=[]
        per_ch=max(1,BATCH//N_CHAMBERS)
        for c in range(N_CHAMBERS):
            if len(by_ch[c])>0:
                batch.extend(random.choices(by_ch[c],k=per_ch))
        random.shuffle(batch); batch=batch[:BATCH]
        texts=[t for t,c in batch]
        tgt=torch.tensor([c for t,c in batch],dtype=torch.long,device=dev)
        tokens=enc_batch(texts,sp).to(dev)

        with torch.no_grad():
            _,mid=lm(tokens)
            pooled=mid.mean(dim=1)

        res=res_proj(pooled)
        act,raw=chambers(res)
        loss=crit(raw,tgt)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(res_proj.parameters(),1.0)
        opt.step()

        losses.append(loss.item())
        pred=raw.argmax(dim=1); accs.append((pred==tgt).float().mean().item())

        if step%200==0:
            al=sum(losses[-200:])/200; aa=sum(accs[-200:])/200
            print('step %5d/%d | loss=%.4f | acc=%.1f%% | %ds' % (step,steps,al,aa*100,time.time()-t0))
            sys.stdout.flush()

        if step%3000==0 or step==steps:
            res_proj.eval()
            samp=random.sample(data,min(500,len(data)))
            tx=[t for t,c in samp]
            tg=torch.tensor([c for t,c in samp],dtype=torch.long,device=dev)
            with torch.no_grad():
                tk=enc_batch(tx,sp).to(dev); _,mid=lm(tk); pooled=mid.mean(dim=1)
                res=res_proj(pooled); a,r=chambers(res)
                l=crit(r,tg).item(); ac=(r.argmax(1)==tg).float().mean().item()
                cd=a.mean(0).cpu().numpy()
            cstr=' '.join('%s=%.2f' % (CNAMES[i],cd[i]) for i in range(N_CHAMBERS))
            print('  EVAL: loss=%.4f acc=%.1f%% | %s' % (l,ac*100,cstr))
            ta=sum(accs[-500:])/max(len(accs[-500:]),1)
            if ta>best_acc:
                best_acc=ta
                ckpt['res_projs']['ru']=res_proj.state_dict()
                ckpt['acc_ru']=ta
                torch.save(ckpt,WEIGHTS_DIR+'/klaus_chambers.pt')
                print('  SAVED RU res_proj (acc=%.1f%%)' % (ta*100))
            print()

    dt=time.time()-t0
    print('='*60)
    print('DONE: %d steps in %.1f min | best RU acc=%.1f%%' % (steps,dt/60,best_acc*100))
    print('='*60)

if __name__=='__main__':
    train()
