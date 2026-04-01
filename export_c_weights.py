#!/usr/bin/env python3
"""Export Klaus PyTorch weights to C binary format for klaus.c inference."""
import torch
import struct
import sys
import os
sys.path.insert(0, '/home/ubuntu/klaus')

WEIGHTS_DIR = '/home/ubuntu/klaus/weights'
LANGS = ['en', 'ru', 'fr', 'he']  # must match LANG_EN=0,LANG_RU=1,LANG_FR=2,LANG_HE=3 in C

def write_tensor(f, t, name=""):
    """Write tensor as raw float32."""
    data = t.detach().cpu().float().contiguous().numpy()
    f.write(data.tobytes())
    if name:
        print("  %-40s %s  %.2f KB" % (name, list(t.shape), data.nbytes / 1024))

def export(ckpt_path, out_path):
    print("Loading checkpoint: %s" % ckpt_path)
    ck = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    print("acc=%.1f%% step=%d" % (ck.get('acc', 0) * 100, ck.get('step', 0)))

    # Verify keys
    assert 'lms' in ck, "Checkpoint must contain 'lms' key"
    assert 'chambers' in ck, "Checkpoint must contain 'chambers' key"
    assert 'res_projs' in ck, "Checkpoint must contain 'res_projs' key"

    total_bytes = 0
    with open(out_path, 'wb') as f:
        # Magic + version
        f.write(b'KLAU')
        f.write(struct.pack('i', 1))  # version

        # Config
        DIM = 384; N_HEADS = 6; N_LAYERS = 6; HDIM = 768; VOCAB = 4096; RES_DIM = 100
        f.write(struct.pack('6i', DIM, N_HEADS, N_LAYERS, HDIM, VOCAB, RES_DIM))

        # ─── 4 Transformers ───
        for lang in LANGS:
            sd = ck['lms'][lang]
            print("\n[%s LM]" % lang.upper())

            # tok_emb (VOCAB, DIM)
            write_tensor(f, sd['tok.weight'], 'tok.weight')

            # pos_emb (MAX_SEQ, DIM) — get actual size
            pos = sd['pos.weight']
            write_tensor(f, pos, 'pos.weight')

            # N_LAYERS blocks
            for i in range(N_LAYERS):
                pfx = 'blocks.%d.' % i
                # attn_norm (RMSNorm weight)
                write_tensor(f, sd[pfx + 'ln1.w'], pfx + 'ln1.w (attn_norm)')
                # wq, wk, wv, wo
                write_tensor(f, sd[pfx + 'attn.wq.weight'], pfx + 'wq')
                write_tensor(f, sd[pfx + 'attn.wk.weight'], pfx + 'wk')
                write_tensor(f, sd[pfx + 'attn.wv.weight'], pfx + 'wv')
                write_tensor(f, sd[pfx + 'attn.wo.weight'], pfx + 'wo')
                # ffn_norm
                write_tensor(f, sd[pfx + 'ln2.w'], pfx + 'ln2.w (ffn_norm)')
                # w1, w2, w3 (SwiGLU)
                write_tensor(f, sd[pfx + 'w1.weight'], pfx + 'w1')
                write_tensor(f, sd[pfx + 'w2.weight'], pfx + 'w2')
                write_tensor(f, sd[pfx + 'w3.weight'], pfx + 'w3')

            # final_norm
            write_tensor(f, sd['norm.w'], 'norm.w (final_norm)')

            # res_proj (DIM -> RES_DIM)
            rp = ck['res_projs'][lang]
            write_tensor(f, rp['weight'], 'res_proj.weight')
            write_tensor(f, rp['bias'], 'res_proj.bias')

        # ─── Chambers (shared) ───
        print("\n[CHAMBERS]")
        ch = ck['chambers']
        for c in range(6):
            pfx = 'ch.%d.' % c
            write_tensor(f, ch[pfx + '0.weight'], pfx + 'w1 (100->128)')
            write_tensor(f, ch[pfx + '0.bias'], pfx + 'b1')
            write_tensor(f, ch[pfx + '2.weight'], pfx + 'w2 (128->64)')
            write_tensor(f, ch[pfx + '2.bias'], pfx + 'b2')
            write_tensor(f, ch[pfx + '4.weight'], pfx + 'w3 (64->32)')
            write_tensor(f, ch[pfx + '4.bias'], pfx + 'b3')
            write_tensor(f, ch[pfx + '6.weight'], pfx + 'w4 (32->1)')
            write_tensor(f, ch[pfx + '6.bias'], pfx + 'b4')

        # Coupling matrix (6x6)
        write_tensor(f, ch['coupling'], 'coupling')

        # Decay (hardcoded in C but write anyway)
        decay = torch.tensor([0.90, 0.93, 0.85, 0.97, 0.88, 0.94])
        write_tensor(f, decay, 'decay')

        total_bytes = f.tell()

    # BPE vocabularies — write separately per language
    import sentencepiece as spm
    for lang in LANGS:
        bpe_path = out_path.replace('.bin', '_%s.bpe' % lang)
        sp = spm.SentencePieceProcessor()
        sp.load(WEIGHTS_DIR + '/klaus_%s_bpe.model' % lang)
        with open(bpe_path, 'w') as f:
            for i in range(sp.get_piece_size()):
                piece = sp.id_to_piece(i)
                f.write("%d\t%s\n" % (i, piece))
        print("BPE vocab %s: %d pieces -> %s" % (lang.upper(), sp.get_piece_size(), bpe_path))

    print("\nExported: %s (%.1f MB)" % (out_path, total_bytes / 1024 / 1024))


if __name__ == '__main__':
    export(
        WEIGHTS_DIR + '/klaus_chambers.pt',
        WEIGHTS_DIR + '/klaus.bin'
    )
