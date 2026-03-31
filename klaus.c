/*
 * klaus.c -- Somatic Engine
 *
 * 4 per-language decoder-only transformers, shared chambers,
 * Dario equation logit injection, Hebbian co-occurrence.
 *
 * Input: text -> Output: bodily sensation descriptions.
 * Single file, zero dependencies.
 *
 * HOW KLAUS SPEAKS:
 *   Transformer learns LANGUAGE (CC-100 LM loss).
 *   Chambers learn EMOTION (emotion data, chamber loss).
 *   Somatic output = logit injection via Dario equation.
 *   Hardcoded somatic seed vocabulary with chamber affinities
 *   BOOSTS somatic tokens in the logit space.
 *   Chambers = body.  Dario equation = nervous system.  lm_head = mouth.
 *   Body shifts what the mouth already knows how to say.
 *
 * Build: cc -O2 -o klaus klaus.c -lm
 * Usage: ./klaus weights.bin [--interactive]
 *
 * RRPRAM (from postgpt):
 *   "The tokenizer IS the training."
 *   BPE merges encode which tokens naturally sit together.
 *   At init: build merge affinity table from BPE vocabulary.
 *   At generation: tokens that merge with recent context get rhythmic boost.
 *   No training needed. Rhythm from structure.
 *
 * Ancestors: haze/cloud (chambers), dario.c (equation), postgpt (RRPRAM)
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <time.h>

/* ================================================================
 *  CONFIG
 * ================================================================ */

#define DIM         384
#define N_HEADS     6
#define HEAD_DIM    (DIM / N_HEADS)
#define N_LAYERS    6
#define HDIM        768
#define MAX_SEQ     256
#define BPE_VOCAB   4096
#define RES_DIM     100
#define N_CHAMBERS  6
#define N_LANGS     4
#define XFIRE_ITERS 5
#define XFIRE_K     0.05f
#define MEM_SLOTS   16
#define MEM_DECAY   0.85f
#define MEM_BLEND   0.2f
#define MAX_GEN     64
#define GEN_TEMP    0.7f
#define TOP_K_SAMP  32

/* Dario equation coefficients (base, before chamber modulation) */
#define ALPHA_BASE  2.0f    /* somatic boost strength */
#define BETA_BASE   0.5f    /* bigram strength */
#define GAMMA_BASE  0.3f    /* Hebbian strength */
#define DELTA_BASE  0.4f    /* RRPRAM rhythmic strength */

/* RRPRAM merge table */
#define MAX_MERGES  16384
#define RRPRAM_WIN  6       /* look-back for rhythm */

/* co-occurrence */
#define MAX_COOC    4096
#define COOC_WINDOW 6
#define COOC_DECAY  0.98f

/* chamber indices */
enum { CH_FEAR=0, CH_LOVE, CH_RAGE, CH_VOID, CH_FLOW, CH_COMPLEX };

/* ================================================================
 *  TYPES
 * ================================================================ */

typedef struct {
    float tok_emb[BPE_VOCAB * DIM];
    float pos_emb[MAX_SEQ * DIM];
    struct {
        float attn_norm[DIM];
        float wq[DIM * DIM], wk[DIM * DIM], wv[DIM * DIM], wo[DIM * DIM];
        float ffn_norm[DIM];
        float w1[DIM * HDIM], w2[DIM * HDIM], w3[HDIM * DIM];
    } layers[N_LAYERS];
    float final_norm[DIM];
    float res_proj[DIM * RES_DIM];
} Transformer;

typedef struct {
    struct {
        float w1[RES_DIM * 128]; float b1[128];
        float w2[128 * 64];     float b2[64];
        float w3[64 * 32];      float b3[32];
        float w4[32];           float b4[1];
    } ch[N_CHAMBERS];
    float coupling[N_CHAMBERS * N_CHAMBERS];
} Chambers;

typedef struct { char text[64]; int len; } BPEPiece;

/* Hebbian co-occurrence field -- grows through conversation */
typedef struct {
    int src[MAX_COOC], dst[MAX_COOC];
    float count[MAX_COOC];
    int n;
} CoocField;

/* RRPRAM merge table -- built from BPE vocabulary, no training */
typedef struct {
    uint16_t a[MAX_MERGES], b[MAX_MERGES];
    int n;
} MergeTable;

typedef struct {
    Transformer xf[N_LANGS];
    Chambers    chambers;
    float       decay[N_CHAMBERS];
    BPEPiece    bpe[BPE_VOCAB];
    /* somatic affinity: how much each chamber boosts each BPE token */
    float       somatic_aff[BPE_VOCAB * N_CHAMBERS];
    /* Hebbian co-occurrence */
    CoocField   cooc;
    /* RRPRAM -- rhythmic resonance from BPE structure */
    MergeTable  merges;
    /* membrane */
    float mem_res[MEM_SLOTS][RES_DIM];
    float mem_ch[MEM_SLOTS][N_CHAMBERS];
    float mem_w[MEM_SLOTS];
    int   mem_ptr, mem_n;
} Klaus;

/* ================================================================
 *  MATH
 * ================================================================ */

static void rmsnorm(float *o, const float *x, const float *w, int n) {
    float ss = 0.0f;
    for (int i = 0; i < n; i++) ss += x[i] * x[i];
    ss = 1.0f / sqrtf(ss / n + 1e-6f);
    for (int i = 0; i < n; i++) o[i] = x[i] * ss * w[i];
}

static void matmul(float *o, const float *x, const float *w, int n, int d) {
    for (int i = 0; i < d; i++) {
        float v = 0.0f;
        for (int j = 0; j < n; j++) v += x[j] * w[j * d + i];
        o[i] = v;
    }
}

static void matmul_bias(float *o, const float *x, const float *w,
                        const float *b, int n, int d) {
    matmul(o, x, w, n, d);
    for (int i = 0; i < d; i++) o[i] += b[i];
}

static float swish(float x)   { return x / (1.0f + expf(-x)); }
static float sigmoid(float x) { return 1.0f / (1.0f + expf(-x)); }
static float clampf(float x, float lo, float hi) {
    return x < lo ? lo : x > hi ? hi : x;
}

static void softmax(float *x, int n) {
    float mx = x[0];
    for (int i = 1; i < n; i++) if (x[i] > mx) mx = x[i];
    float s = 0.0f;
    for (int i = 0; i < n; i++) { x[i] = expf(x[i] - mx); s += x[i]; }
    for (int i = 0; i < n; i++) x[i] /= s;
}

/* ================================================================
 *  SOMATIC SEED VOCABULARY
 *  Hardcoded body/sensation words with per-chamber affinities.
 *  At init: BPE-encoded, mapped to somatic_aff[vocab * 6].
 *  Like SEED_WORDS in dario.c but organized by body, not physics.
 * ================================================================ */

typedef struct {
    const char *word;
    float aff[N_CHAMBERS]; /* FEAR LOVE RAGE VOID FLOW COMPLEX */
} SomWord;

/*                                         FEAR LOVE RAGE VOID FLOW CMPLX */
static const SomWord SEED_EN[] = {
    /* sensations */
    {"pulse",       {0.4, 0.0, 0.8, 0.0, 0.3, 0.2}},
    {"tremor",      {0.8, 0.0, 0.2, 0.2, 0.0, 0.3}},
    {"burning",     {0.3, 0.1, 0.9, 0.0, 0.1, 0.2}},
    {"clenching",   {0.4, 0.0, 0.8, 0.1, 0.0, 0.3}},
    {"tingling",    {0.5, 0.2, 0.1, 0.0, 0.4, 0.5}},
    {"throbbing",   {0.3, 0.0, 0.7, 0.1, 0.3, 0.2}},
    {"aching",      {0.2, 0.1, 0.2, 0.7, 0.0, 0.3}},
    {"tightness",   {0.6, 0.0, 0.5, 0.3, 0.0, 0.3}},
    {"sinking",     {0.5, 0.0, 0.0, 0.9, 0.0, 0.2}},
    {"nausea",      {0.5, 0.0, 0.2, 0.7, 0.0, 0.3}},
    {"heaviness",   {0.2, 0.0, 0.1, 0.8, 0.0, 0.2}},
    {"weakness",    {0.5, 0.0, 0.0, 0.7, 0.0, 0.2}},
    {"shaking",     {0.8, 0.0, 0.4, 0.1, 0.0, 0.3}},
    {"freezing",    {0.7, 0.0, 0.0, 0.5, 0.0, 0.2}},
    {"sweating",    {0.6, 0.0, 0.3, 0.1, 0.2, 0.2}},
    {"warmth",      {0.0, 0.9, 0.0, 0.0, 0.6, 0.1}},
    {"softness",    {0.0, 0.8, 0.0, 0.0, 0.5, 0.2}},
    {"floating",    {0.0, 0.3, 0.0, 0.2, 0.8, 0.3}},
    {"pressure",    {0.4, 0.0, 0.5, 0.5, 0.0, 0.4}},
    {"vibrating",   {0.2, 0.1, 0.3, 0.0, 0.4, 0.8}},
    /* body parts */
    {"chest",       {0.4, 0.5, 0.4, 0.3, 0.2, 0.3}},
    {"throat",      {0.6, 0.2, 0.3, 0.5, 0.0, 0.3}},
    {"stomach",     {0.5, 0.1, 0.3, 0.6, 0.1, 0.3}},
    {"jaw",         {0.2, 0.0, 0.9, 0.1, 0.0, 0.2}},
    {"fists",       {0.1, 0.0, 0.9, 0.0, 0.0, 0.2}},
    {"spine",       {0.7, 0.0, 0.2, 0.2, 0.1, 0.4}},
    {"temples",     {0.4, 0.0, 0.3, 0.3, 0.0, 0.6}},
    {"shoulders",   {0.3, 0.0, 0.4, 0.4, 0.0, 0.3}},
    {NULL, {0}}
};

static const SomWord SEED_RU[] = {
    /* sensations */
    {"\xd0\xbf\xd1\x83\xd0\xbb\xd1\x8c\xd1\x81",               /* пульс */
                    {0.4, 0.0, 0.8, 0.0, 0.3, 0.2}},
    {"\xd0\xb4\xd1\x80\xd0\xbe\xd0\xb6\xd1\x8c",               /* дрожь */
                    {0.8, 0.0, 0.2, 0.2, 0.0, 0.3}},
    {"\xd0\xb6\xd0\xb0\xd1\x80",                                 /* жар */
                    {0.3, 0.1, 0.8, 0.0, 0.2, 0.2}},
    {"\xd1\x85\xd0\xbe\xd0\xbb\xd0\xbe\xd0\xb4",               /* холод */
                    {0.7, 0.0, 0.0, 0.5, 0.0, 0.2}},
    {"\xd1\x82\xd0\xbe\xd1\x88\xd0\xbd\xd0\xbe\xd1\x82\xd0\xb0", /* тошнота */
                    {0.5, 0.0, 0.2, 0.7, 0.0, 0.3}},
    {"\xd0\xb1\xd0\xbe\xd0\xbb\xd1\x8c",                       /* боль */
                    {0.3, 0.0, 0.4, 0.5, 0.0, 0.4}},
    {"\xd1\x82\xd1\x8f\xd0\xb6\xd0\xb5\xd1\x81\xd1\x82\xd1\x8c", /* тяжесть */
                    {0.2, 0.0, 0.1, 0.8, 0.0, 0.2}},
    {"\xd0\xb3\xd0\xbe\xd1\x80\xd0\xb8\xd1\x82",               /* горит */
                    {0.3, 0.1, 0.9, 0.0, 0.1, 0.2}},
    {"\xd1\x81\xd0\xb6\xd0\xb8\xd0\xbc\xd0\xb0\xd0\xb5\xd1\x82\xd1\x81\xd1\x8f", /* сжимается */
                    {0.5, 0.0, 0.6, 0.3, 0.0, 0.3}},
    {"\xd1\x81\xd0\xba\xd1\x80\xd0\xb8\xd0\xbf\xd0\xb8\xd1\x82", /* скрипит */
                    {0.2, 0.0, 0.8, 0.1, 0.0, 0.3}},
    {"\xd0\xba\xd0\xbe\xd0\xbb\xd0\xb5\xd1\x82",               /* колет */
                    {0.4, 0.0, 0.5, 0.1, 0.0, 0.5}},
    {"\xd0\xbd\xd0\xbe\xd0\xb5\xd1\x82",                       /* ноет */
                    {0.2, 0.0, 0.1, 0.8, 0.0, 0.3}},
    {"\xd1\x82\xd0\xb5\xd0\xbf\xd0\xbb\xd0\xbe",               /* тепло */
                    {0.0, 0.9, 0.0, 0.0, 0.6, 0.1}},
    {"\xd0\xb4\xd1\x80\xd0\xbe\xd0\xb6\xd1\x83",               /* дрожу */
                    {0.8, 0.0, 0.3, 0.1, 0.0, 0.3}},
    {"\xd0\xb4\xd0\xb0\xd0\xb2\xd0\xbb\xd0\xb5\xd0\xbd\xd0\xb8\xd0\xb5", /* давление */
                    {0.4, 0.0, 0.5, 0.5, 0.0, 0.4}},
    /* body parts */
    {"\xd0\xb3\xd1\x80\xd1\x83\xd0\xb4\xd1\x8c",               /* грудь */
                    {0.4, 0.5, 0.4, 0.3, 0.2, 0.3}},
    {"\xd0\xb3\xd0\xbe\xd1\x80\xd0\xbb\xd0\xbe",               /* горло */
                    {0.6, 0.2, 0.3, 0.5, 0.0, 0.3}},
    {"\xd0\xb6\xd0\xb8\xd0\xb2\xd0\xbe\xd1\x82",               /* живот */
                    {0.5, 0.1, 0.3, 0.6, 0.1, 0.3}},
    {"\xd1\x87\xd0\xb5\xd0\xbb\xd1\x8e\xd1\x81\xd1\x82\xd1\x8c", /* челюсть */
                    {0.2, 0.0, 0.9, 0.1, 0.0, 0.2}},
    {"\xd0\xba\xd1\x83\xd0\xbb\xd0\xb0\xd0\xba\xd0\xb8",       /* кулаки */
                    {0.1, 0.0, 0.9, 0.0, 0.0, 0.2}},
    {"\xd0\xbf\xd0\xbe\xd0\xb7\xd0\xb2\xd0\xbe\xd0\xbd\xd0\xbe\xd1\x87\xd0\xbd\xd0\xb8\xd0\xba", /* позвоночник */
                    {0.7, 0.0, 0.2, 0.2, 0.1, 0.4}},
    {"\xd0\xb2\xd0\xb8\xd1\x81\xd0\xba\xd0\xb8",               /* виски */
                    {0.4, 0.0, 0.3, 0.3, 0.0, 0.6}},
    {NULL, {0}}
};

static const SomWord SEED_FR[] = {
    {"pouls",       {0.4, 0.0, 0.8, 0.0, 0.3, 0.2}},
    {"frisson",     {0.7, 0.1, 0.1, 0.2, 0.1, 0.4}},
    {"chaleur",     {0.1, 0.7, 0.5, 0.0, 0.4, 0.1}},
    {"froid",       {0.7, 0.0, 0.0, 0.5, 0.0, 0.2}},
    {"vertige",     {0.6, 0.0, 0.0, 0.6, 0.2, 0.4}},
    {"douleur",     {0.3, 0.0, 0.4, 0.5, 0.0, 0.4}},
    {"lourdeur",    {0.2, 0.0, 0.1, 0.8, 0.0, 0.2}},
    {"tension",     {0.5, 0.0, 0.5, 0.3, 0.0, 0.4}},
    {"pression",    {0.4, 0.0, 0.5, 0.5, 0.0, 0.4}},
    {"picotement",  {0.3, 0.1, 0.1, 0.0, 0.4, 0.6}},
    {"tremblement", {0.7, 0.0, 0.3, 0.1, 0.0, 0.3}},
    {"suffocation", {0.8, 0.0, 0.2, 0.5, 0.0, 0.3}},
    {"poitrine",    {0.4, 0.5, 0.4, 0.3, 0.2, 0.3}},
    {"gorge",       {0.6, 0.2, 0.3, 0.5, 0.0, 0.3}},
    {"ventre",      {0.5, 0.1, 0.3, 0.6, 0.1, 0.3}},
    {"estomac",     {0.5, 0.0, 0.3, 0.7, 0.0, 0.3}},
    {"\x63\x68\x61\x69\x72",  /* chair */
                    {0.3, 0.3, 0.2, 0.2, 0.2, 0.4}},
    {NULL, {0}}
};

static const SomWord SEED_HE[] = {
    {"\xd7\x93\xd7\x95\xd7\xa4\xd7\xa7",           /* דופק */
                    {0.4, 0.0, 0.8, 0.0, 0.3, 0.2}},
    {"\xd7\xa8\xd7\xa2\xd7\x93",                     /* רעד */
                    {0.8, 0.0, 0.2, 0.2, 0.0, 0.3}},
    {"\xd7\x97\xd7\x95\xd7\x9d",                     /* חום */
                    {0.2, 0.4, 0.6, 0.0, 0.2, 0.2}},
    {"\xd7\xa7\xd7\x95\xd7\xa8",                     /* קור */
                    {0.7, 0.0, 0.0, 0.5, 0.0, 0.2}},
    {"\xd7\x91\xd7\x97\xd7\x99\xd7\x9c\xd7\x94",   /* בחילה */
                    {0.5, 0.0, 0.2, 0.7, 0.0, 0.3}},
    {"\xd7\x9b\xd7\x90\xd7\x91",                     /* כאב */
                    {0.3, 0.0, 0.4, 0.5, 0.0, 0.4}},
    {"\xd7\x9b\xd7\x95\xd7\x91\xd7\x93",           /* כובד */
                    {0.2, 0.0, 0.1, 0.8, 0.0, 0.2}},
    {"\xd7\x9c\xd7\x97\xd7\xa5",                     /* לחץ */
                    {0.4, 0.0, 0.5, 0.5, 0.0, 0.4}},
    {"\xd7\x91\xd7\x95\xd7\xa2\xd7\xa8",           /* בוער */
                    {0.3, 0.1, 0.9, 0.0, 0.1, 0.2}},
    {"\xd7\x97\xd7\x96\xd7\x94",                     /* חזה */
                    {0.4, 0.5, 0.4, 0.3, 0.2, 0.3}},
    {"\xd7\x92\xd7\xa8\xd7\x95\xd7\x9f",           /* גרון */
                    {0.6, 0.2, 0.3, 0.5, 0.0, 0.3}},
    {"\xd7\x91\xd7\x98\xd7\x9f",                     /* בטן */
                    {0.5, 0.1, 0.3, 0.6, 0.1, 0.3}},
    {"\xd7\x9c\xd7\xa1\xd7\xaa",                     /* לסת */
                    {0.2, 0.0, 0.9, 0.1, 0.0, 0.2}},
    {NULL, {0}}
};

static const SomWord *SEEDS[] = { SEED_EN, SEED_RU, SEED_FR, SEED_HE };

/* ================================================================
 *  BPE ENCODE / DECODE
 * ================================================================ */

static int bpe_encode(const Klaus *k, const char *text, int *out, int maxn) {
    const unsigned char *p = (const unsigned char *)text;
    int n = 0;
    while (*p && n < maxn) {
        int best_id = -1, best_len = 0;
        for (int v = 0; v < BPE_VOCAB; v++) {
            int pl = k->bpe[v].len;
            if (pl > best_len && pl <= (int)strlen((const char *)p))
                if (memcmp(p, k->bpe[v].text, pl) == 0)
                    { best_len = pl; best_id = v; }
        }
        if (best_id >= 0) { out[n++] = best_id; p += best_len; }
        else { out[n++] = (int)*p; p++; }
    }
    return n;
}

static void bpe_decode(const Klaus *k, const int *tokens, int n, char *out, int maxlen) {
    int pos = 0;
    for (int i = 0; i < n && pos < maxlen - 1; i++) {
        int id = tokens[i];
        if (id < 0 || id >= BPE_VOCAB) continue;
        int pl = k->bpe[id].len;
        if (pos + pl >= maxlen) break;
        memcpy(out + pos, k->bpe[id].text, pl);
        pos += pl;
    }
    out[pos] = '\0';
}

/* ================================================================
 *  LANGUAGE DETECTION
 * ================================================================ */

enum { LANG_EN = 0, LANG_RU = 1, LANG_FR = 2, LANG_HE = 3 };

static int detect_lang(const char *text) {
    const unsigned char *p = (const unsigned char *)text;
    int cy = 0, he = 0, acc = 0;
    while (*p) {
        if (p[0] >= 0xD0 && p[0] <= 0xD3 && p[1] >= 0x80) { cy++; p += 2; }
        else if (p[0] == 0xD7 && p[1] >= 0x80) { he++; p += 2; }
        else if (p[0] == 0xC3 && p[1]) { acc++; p += 2; }
        else { if (*p >= 0x80) { p++; while (*p && (*p & 0xC0) == 0x80) p++; } else p++; }
    }
    if (he > 2) return LANG_HE;
    if (cy > 2) return LANG_RU;
    if (acc > 1) return LANG_FR;
    const char *fw[] = {"je ","tu ","le ","la ","les ","un ","une ",
                         "suis ","est ","dans ","que ","qui ", NULL};
    for (int i = 0; fw[i]; i++)
        if (strstr(text, fw[i])) return LANG_FR;
    return LANG_EN;
}

static const char *LNAME[] = {"EN", "RU", "FR", "HE"};
static const char *CNAME[] = {"FEAR", "LOVE", "RAGE", "VOID", "FLOW", "CMPLX"};

/* ================================================================
 *  SOMATIC AFFINITY INIT
 *  After weights loaded (BPE table available):
 *  For each seed word -> BPE encode -> mark tokens with affinities.
 * ================================================================ */

static void init_somatic_affinity(Klaus *k) {
    memset(k->somatic_aff, 0, sizeof(k->somatic_aff));
    for (int lang = 0; lang < N_LANGS; lang++) {
        const SomWord *sw = SEEDS[lang];
        for (int w = 0; sw[w].word; w++) {
            int toks[32];
            int nt = bpe_encode(k, sw[w].word, toks, 32);
            for (int t = 0; t < nt; t++) {
                int id = toks[t];
                if (id < 0 || id >= BPE_VOCAB) continue;
                for (int c = 0; c < N_CHAMBERS; c++) {
                    float cur = k->somatic_aff[id * N_CHAMBERS + c];
                    float nw  = sw[w].aff[c];
                    /* max -- strongest affinity wins */
                    if (nw > cur) k->somatic_aff[id * N_CHAMBERS + c] = nw;
                }
            }
        }
    }
    int nsoma = 0;
    for (int v = 0; v < BPE_VOCAB; v++) {
        float sum = 0;
        for (int c = 0; c < N_CHAMBERS; c++)
            sum += k->somatic_aff[v * N_CHAMBERS + c];
        if (sum > 0.01f) nsoma++;
    }
    fprintf(stderr, "somatic seed: %d/%d tokens have affinity\n", nsoma, BPE_VOCAB);
}

/* ================================================================
 *  RRPRAM -- Rhythmic resonance from BPE merge structure
 *  "The tokenizer IS the training." (postgpt)
 *
 *  For each piece C in vocab, if C = concat(A, B) where A and B
 *  are also vocab pieces, then (A, B) is a merge pair.
 *  Tokens that merge together flow together.
 *  At generation: candidate token scores higher if it merges
 *  with recent context tokens. Position decay = closer = stronger.
 * ================================================================ */

static void init_rrpram(Klaus *k) {
    k->merges.n = 0;
    for (int c = 0; c < BPE_VOCAB; c++) {
        int cl = k->bpe[c].len;
        if (cl < 2) continue;
        /* try every split point of piece C */
        for (int sp = 1; sp < cl && sp < 32; sp++) {
            /* find piece A matching prefix [0..sp) */
            int a = -1;
            for (int v = 0; v < BPE_VOCAB; v++) {
                if (k->bpe[v].len == sp &&
                    memcmp(k->bpe[v].text, k->bpe[c].text, sp) == 0)
                    { a = v; break; }
            }
            if (a < 0) continue;
            /* find piece B matching suffix [sp..cl) */
            int b = -1;
            int sl = cl - sp;
            for (int v = 0; v < BPE_VOCAB; v++) {
                if (k->bpe[v].len == sl &&
                    memcmp(k->bpe[v].text, k->bpe[c].text + sp, sl) == 0)
                    { b = v; break; }
            }
            if (b < 0) continue;
            /* record merge pair */
            if (k->merges.n < MAX_MERGES) {
                k->merges.a[k->merges.n] = (uint16_t)a;
                k->merges.b[k->merges.n] = (uint16_t)b;
                k->merges.n++;
            }
        }
    }
    fprintf(stderr, "rrpram: %d merge pairs from BPE vocabulary\n", k->merges.n);
}

/* How well does candidate token flow after recent context? */
static float rrpram_score(const Klaus *k, const int *recent, int nrecent, int cand) {
    float score = 0.0f;
    for (int r = 0; r < nrecent; r++) {
        int ctx = recent[r];
        float pos_w = 1.0f / (float)(nrecent - r); /* closer = stronger */
        for (int m = 0; m < k->merges.n; m++) {
            if (k->merges.a[m] == ctx && k->merges.b[m] == cand) {
                score += pos_w;
                break;
            }
        }
    }
    return score;
}

/* ================================================================
 *  TRANSFORMER FORWARD
 *  Decoder-only, causal attention.
 * ================================================================ */

static void xf_forward(const Transformer *xf, const int *tok, int slen,
                        float *h /* slen * DIM */) {
    float nr[DIM], fo[DIM];
    float q[MAX_SEQ * DIM], kk[MAX_SEQ * DIM], v[MAX_SEQ * DIM];
    float att[N_HEADS * MAX_SEQ * MAX_SEQ];
    float buf[MAX_SEQ * DIM];
    float g[MAX_SEQ * HDIM], u[MAX_SEQ * HDIM];

    for (int t = 0; t < slen; t++)
        for (int d = 0; d < DIM; d++)
            h[t * DIM + d] = xf->tok_emb[tok[t] * DIM + d]
                            + xf->pos_emb[t * DIM + d];

    for (int l = 0; l < N_LAYERS; l++) {
        for (int t = 0; t < slen; t++) {
            rmsnorm(nr, &h[t * DIM], xf->layers[l].attn_norm, DIM);
            matmul(&q[t * DIM],  nr, xf->layers[l].wq, DIM, DIM);
            matmul(&kk[t * DIM], nr, xf->layers[l].wk, DIM, DIM);
            matmul(&v[t * DIM],  nr, xf->layers[l].wv, DIM, DIM);
        }
        for (int hd = 0; hd < N_HEADS; hd++) {
            for (int i = 0; i < slen; i++) {
                for (int j = 0; j < slen; j++) {
                    if (j > i) {
                        att[hd * MAX_SEQ * MAX_SEQ + i * MAX_SEQ + j] = -1e9f;
                    } else {
                        float sc = 0.0f;
                        for (int d = 0; d < HEAD_DIM; d++)
                            sc += q[i * DIM + hd * HEAD_DIM + d]
                                * kk[j * DIM + hd * HEAD_DIM + d];
                        att[hd * MAX_SEQ * MAX_SEQ + i * MAX_SEQ + j] = sc / sqrtf((float)HEAD_DIM);
                    }
                }
                softmax(&att[hd * MAX_SEQ * MAX_SEQ + i * MAX_SEQ], slen);
            }
        }
        memset(buf, 0, slen * DIM * sizeof(float));
        for (int hd = 0; hd < N_HEADS; hd++)
            for (int i = 0; i < slen; i++)
                for (int j = 0; j <= i; j++) {
                    float a = att[hd * MAX_SEQ * MAX_SEQ + i * MAX_SEQ + j];
                    for (int d = 0; d < HEAD_DIM; d++)
                        buf[i * DIM + hd * HEAD_DIM + d] += a * v[j * DIM + hd * HEAD_DIM + d];
                }
        for (int t = 0; t < slen; t++) {
            matmul(fo, &buf[t * DIM], xf->layers[l].wo, DIM, DIM);
            for (int d = 0; d < DIM; d++) h[t * DIM + d] += fo[d];
        }
        for (int t = 0; t < slen; t++) {
            rmsnorm(nr, &h[t * DIM], xf->layers[l].ffn_norm, DIM);
            matmul(&g[t * HDIM], nr, xf->layers[l].w1, DIM, HDIM);
            matmul(&u[t * HDIM], nr, xf->layers[l].w2, DIM, HDIM);
            for (int d = 0; d < HDIM; d++)
                g[t * HDIM + d] = swish(g[t * HDIM + d]) * u[t * HDIM + d];
            matmul(fo, &g[t * HDIM], xf->layers[l].w3, HDIM, DIM);
            for (int d = 0; d < DIM; d++) h[t * DIM + d] += fo[d];
        }
    }
    for (int t = 0; t < slen; t++)
        rmsnorm(&h[t * DIM], &h[t * DIM], xf->final_norm, DIM);
}

/* lm_head: h[DIM] -> logits[BPE_VOCAB] via tok_emb^T */
static void lm_head_logits(const Transformer *xf, const float *hidden, float *logits) {
    for (int v = 0; v < BPE_VOCAB; v++) {
        float s = 0.0f;
        for (int d = 0; d < DIM; d++)
            s += hidden[d] * xf->tok_emb[v * DIM + d];
        logits[v] = s;
    }
}

/* mean-pool -> project to resonance */
static void hidden_to_resonance(const Transformer *xf, const float *h,
                                 int slen, float *res) {
    float pool[DIM] = {0};
    for (int t = 0; t < slen; t++)
        for (int d = 0; d < DIM; d++)
            pool[d] += h[t * DIM + d];
    for (int d = 0; d < DIM; d++) pool[d] /= slen;
    matmul(res, pool, xf->res_proj, DIM, RES_DIM);
}

/* ================================================================
 *  CHAMBERS + CROSS-FIRE  (from haze/cloud)
 * ================================================================ */

static void chambers_forward(const Chambers *ch, const float *res,
                              float *act, const float *decay) {
    float h1[128], h2[64], h3[32];
    for (int c = 0; c < N_CHAMBERS; c++) {
        matmul_bias(h1, res, ch->ch[c].w1, ch->ch[c].b1, RES_DIM, 128);
        for (int i = 0; i < 128; i++) h1[i] = swish(h1[i]);
        matmul_bias(h2, h1, ch->ch[c].w2, ch->ch[c].b2, 128, 64);
        for (int i = 0; i < 64; i++) h2[i] = swish(h2[i]);
        matmul_bias(h3, h2, ch->ch[c].w3, ch->ch[c].b3, 64, 32);
        for (int i = 0; i < 32; i++) h3[i] = swish(h3[i]);
        float out = ch->ch[c].b4[0];
        for (int i = 0; i < 32; i++) out += h3[i] * ch->ch[c].w4[i];
        act[c] = sigmoid(out);
    }
    /* Linear coupling, decay once (fixed from sin+decay^5) */
    for (int iter = 0; iter < XFIRE_ITERS; iter++) {
        float old[N_CHAMBERS];
        for (int i = 0; i < N_CHAMBERS; i++) old[i] = act[i];
        for (int i = 0; i < N_CHAMBERS; i++)
            for (int j = 0; j < N_CHAMBERS; j++)
                if (i != j)
                    act[i] += XFIRE_K * ch->coupling[i * N_CHAMBERS + j]
                            * (old[j] - old[i]);
        for (int i = 0; i < N_CHAMBERS; i++)
            act[i] = clampf(act[i], 0.0f, 1.0f);
    }
    for (int i = 0; i < N_CHAMBERS; i++) act[i] *= decay[i];
}

/* ================================================================
 *  CO-OCCURRENCE (Hebbian, from dario.c)
 *  Grows through conversation. No backprop.
 * ================================================================ */

static void cooc_update(CoocField *cf, int src, int dst, float w) {
    for (int i = 0; i < cf->n; i++)
        if (cf->src[i] == src && cf->dst[i] == dst)
            { cf->count[i] += w; return; }
    if (cf->n < MAX_COOC) {
        cf->src[cf->n] = src;
        cf->dst[cf->n] = dst;
        cf->count[cf->n] = w;
        cf->n++;
    }
}

static void cooc_ingest(CoocField *cf, const int *toks, int n) {
    for (int i = 0; i < n; i++) {
        int lo = (i - COOC_WINDOW > 0) ? i - COOC_WINDOW : 0;
        int hi = (i + COOC_WINDOW < n) ? i + COOC_WINDOW : n;
        for (int j = lo; j < hi; j++) {
            if (j == i) continue;
            float w = 1.0f / (float)(abs(i - j));
            cooc_update(cf, toks[i], toks[j], w);
        }
    }
    /* slow decay so old pairs fade */
    for (int i = 0; i < cf->n; i++)
        cf->count[i] *= COOC_DECAY;
}

static float cooc_score(const CoocField *cf, int ctx_tok, int cand) {
    for (int i = 0; i < cf->n; i++)
        if (cf->src[i] == ctx_tok && cf->dst[i] == cand)
            return cf->count[i];
    return 0.0f;
}

/* ================================================================
 *  DARIO EQUATION -- logit injection
 *
 *  logits[v] = base[v]
 *            + alpha * T[v]     (somatic boost from chambers)
 *            + beta  * B[v]     (bigram from previous token)
 *            + gamma * H[v]     (Hebbian co-occurrence)
 *
 *  alpha, beta, gamma modulated by chamber activations:
 *    alpha grows with dominant chamber strength
 *    beta grows with FLOW (sequential body logic)
 *    gamma grows with COMPLEX (emergent associations)
 * ================================================================ */

static void dario_inject(float *logits, Klaus *k, const float *act,
                          int prev_tok, const int *ctx, int ctx_len) {
    /* coefficient modulation (from dario.c) */
    float alpha = ALPHA_BASE * clampf(
        1.0f + 0.4f * act[CH_RAGE] + 0.3f * act[CH_FEAR]
             + 0.3f * act[CH_VOID] + 0.2f * act[CH_LOVE]
             - 0.1f * act[CH_FLOW], 0.5f, 3.0f);

    float beta = BETA_BASE * clampf(
        1.0f + 0.4f * act[CH_FLOW] + 0.2f * act[CH_LOVE]
             - 0.2f * act[CH_VOID], 0.3f, 2.0f);

    float gamma = GAMMA_BASE * clampf(
        1.0f + 0.5f * act[CH_COMPLEX] + 0.2f * act[CH_FLOW]
             - 0.1f * act[CH_RAGE], 0.3f, 2.0f);

    float delta = DELTA_BASE * clampf(
        1.0f + 0.3f * act[CH_FLOW] + 0.2f * act[CH_LOVE]
             + 0.1f * act[CH_COMPLEX], 0.3f, 2.0f);

    /* RRPRAM context window */
    int rr_start = (ctx_len > RRPRAM_WIN) ? ctx_len - RRPRAM_WIN : 0;
    int rr_n = ctx_len - rr_start;
    const int *rr_ctx = &ctx[rr_start];

    for (int v = 0; v < BPE_VOCAB; v++) {
        /* T: somatic boost */
        float boost = 0.0f;
        for (int c = 0; c < N_CHAMBERS; c++)
            boost += act[c] * k->somatic_aff[v * N_CHAMBERS + c];

        /* B: bigram from previous token */
        float bigram = 0.0f;
        if (prev_tok >= 0)
            bigram = cooc_score(&k->cooc, prev_tok, v);

        /* H: Hebbian from context window */
        float hebbian = 0.0f;
        int hstart = (ctx_len > COOC_WINDOW) ? ctx_len - COOC_WINDOW : 0;
        for (int c = hstart; c < ctx_len; c++) {
            float dist_w = 1.0f / (float)(ctx_len - c);
            hebbian += cooc_score(&k->cooc, ctx[c], v) * dist_w;
        }

        /* R: RRPRAM rhythmic flow */
        float rhythm = rrpram_score(k, rr_ctx, rr_n, v);

        logits[v] += alpha * boost + beta * bigram + gamma * hebbian + delta * rhythm;
    }
}

/* ================================================================
 *  MEMBRANE
 * ================================================================ */

static void mem_store(Klaus *k, const float *res, const float *act) {
    int idx = k->mem_ptr % MEM_SLOTS;
    for (int i = 0; i < MEM_SLOTS; i++) k->mem_w[i] *= MEM_DECAY;
    memcpy(k->mem_res[idx], res, RES_DIM * sizeof(float));
    memcpy(k->mem_ch[idx], act, N_CHAMBERS * sizeof(float));
    k->mem_w[idx] = 1.0f;
    k->mem_ptr++;
    if (k->mem_n < MEM_SLOTS) k->mem_n++;
}

static void mem_modulate(const Klaus *k, float *res) {
    if (k->mem_n == 0) return;
    float avg[RES_DIM] = {0};
    float ws = 0.0f;
    for (int i = 0; i < k->mem_n; i++) {
        for (int d = 0; d < RES_DIM; d++)
            avg[d] += k->mem_res[i][d] * k->mem_w[i];
        ws += k->mem_w[i];
    }
    if (ws < 0.01f) return;
    for (int d = 0; d < RES_DIM; d++)
        res[d] = (1.0f - MEM_BLEND) * res[d] + MEM_BLEND * (avg[d] / ws);
}

static float mem_trend(const Klaus *k) {
    if (k->mem_n < 2) return 0.0f;
    int curr = (k->mem_ptr - 1) % MEM_SLOTS;
    int prev = (k->mem_ptr - 2) % MEM_SLOTS;
    float t = 0.0f;
    for (int i = 0; i < N_CHAMBERS; i++)
        t += k->mem_ch[curr][i] - k->mem_ch[prev][i];
    return t / N_CHAMBERS;
}

/* ================================================================
 *  SAMPLING
 * ================================================================ */

static int sample_top_k(float *logits, int vocab, float temp, int topk) {
    for (int i = 0; i < vocab; i++) logits[i] /= temp;
    int idx[TOP_K_SAMP];
    int n = (topk < vocab) ? topk : vocab;
    for (int t = 0; t < n; t++) {
        float best = -1e30f; int bi = 0;
        for (int i = 0; i < vocab; i++) {
            int skip = 0;
            for (int j = 0; j < t; j++) if (idx[j] == i) skip = 1;
            if (!skip && logits[i] > best) { best = logits[i]; bi = i; }
        }
        idx[t] = bi;
    }
    float probs[TOP_K_SAMP];
    float mx = logits[idx[0]], s = 0.0f;
    for (int i = 0; i < n; i++) { probs[i] = expf(logits[idx[i]] - mx); s += probs[i]; }
    for (int i = 0; i < n; i++) probs[i] /= s;
    float r = (float)rand() / (float)RAND_MAX, cum = 0.0f;
    for (int i = 0; i < n; i++) { cum += probs[i]; if (r <= cum) return idx[i]; }
    return idx[n - 1];
}

/* ================================================================
 *  GENERATE
 *
 *  1. Transformer reads input (causal) -> hidden states
 *  2. Mean-pool -> resonance -> chambers -> cross-fire
 *  3. Generate tokens with Dario equation logit injection
 *  4. Each step: base logits + somatic boost + bigram + Hebbian
 *  5. Generated tokens fed into co-occurrence (Hebbian grows)
 * ================================================================ */

static int generate(Klaus *k, int lang, const int *input, int inlen,
                    float *act_out, int *output, int maxout) {
    const Transformer *xf = &k->xf[lang];
    float *h = calloc(MAX_SEQ * DIM, sizeof(float));
    float logits[BPE_VOCAB];
    float res[RES_DIM], act[N_CHAMBERS];

    if (!h) { fprintf(stderr, "alloc failed\n"); return 0; }

    /* phase 1: transformer reads input */
    xf_forward(xf, input, inlen, h);

    /* phase 2: resonance -> chambers */
    hidden_to_resonance(xf, h, inlen, res);
    mem_modulate(k, res);
    chambers_forward(&k->chambers, res, act, k->decay);
    mem_store(k, res, act);
    memcpy(act_out, act, N_CHAMBERS * sizeof(float));

    /* phase 3: generate with Dario equation */
    int seq[MAX_SEQ];
    int slen = inlen;
    if (slen >= MAX_SEQ) slen = MAX_SEQ - 1;
    memcpy(seq, input, slen * sizeof(int));

    int ngen = 0;
    int prev_tok = (slen > 0) ? seq[slen - 1] : -1;

    for (int step = 0; step < maxout && slen < MAX_SEQ; step++) {
        /* base logits from transformer */
        lm_head_logits(xf, &h[(slen - 1) * DIM], logits);

        /* Dario equation injection */
        dario_inject(logits, k, act, prev_tok, seq, slen);

        int next = sample_top_k(logits, BPE_VOCAB, GEN_TEMP, TOP_K_SAMP);
        if (next == 0) break;

        output[ngen++] = next;
        prev_tok = next;
        seq[slen++] = next;

        /* re-run transformer for next prediction */
        if (slen < MAX_SEQ)
            xf_forward(xf, seq, slen, h);
    }

    /* Hebbian: ingest generated tokens into co-occurrence */
    if (ngen > 0)
        cooc_ingest(&k->cooc, output, ngen);

    free(h);
    return ngen;
}

/* ================================================================
 *  LOAD WEIGHTS
 *
 *  Binary format:
 *    magic     u32  "KLS2" = 0x32534C4B
 *    version   u32  2
 *    config    u32 x 8  (vocab,dim,heads,layers,hdim,seq,res,chambers)
 *    BPE       [vocab entries: u8 len + bytes]
 *    4 transformers  [tok_emb, pos_emb, layers, final_norm, res_proj]
 *    chambers        [6 MLPs + coupling]
 * ================================================================ */

#define MAGIC_KLS2 0x32534C4B

#define RD(ptr, sz, n, f) do { if (fread(ptr, sz, n, f) != (size_t)(n)) { \
    fprintf(stderr, "truncated at offset %ld\n", ftell(f)); \
    fclose(f); return -1; } } while(0)

static int load_weights(Klaus *k, const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", path); return -1; }

    uint32_t magic, version;
    RD(&magic, 4, 1, f);
    if (magic != MAGIC_KLS2) {
        fprintf(stderr, "bad magic 0x%08X\n", magic);
        fclose(f); return -1;
    }
    RD(&version, 4, 1, f);
    fprintf(stderr, "KLS2 v%u\n", version);

    uint32_t cfg[8];
    RD(cfg, 4, 8, f);
    if (cfg[0] != BPE_VOCAB || cfg[1] != DIM || cfg[2] != N_HEADS ||
        cfg[3] != N_LAYERS || cfg[4] != HDIM || cfg[5] != MAX_SEQ ||
        cfg[6] != RES_DIM || cfg[7] != N_CHAMBERS) {
        fprintf(stderr, "config mismatch\n");
        fclose(f); return -1;
    }

    for (int v = 0; v < BPE_VOCAB; v++) {
        uint8_t len;
        RD(&len, 1, 1, f);
        if (len >= 64) { fprintf(stderr, "bpe %d too long\n", v); fclose(f); return -1; }
        RD(k->bpe[v].text, 1, len, f);
        k->bpe[v].text[len] = '\0';
        k->bpe[v].len = len;
    }

    for (int l = 0; l < N_LANGS; l++) {
        Transformer *xf = &k->xf[l];
        RD(xf->tok_emb,  4, BPE_VOCAB * DIM, f);
        RD(xf->pos_emb,  4, MAX_SEQ * DIM, f);
        for (int ly = 0; ly < N_LAYERS; ly++) {
            RD(xf->layers[ly].attn_norm, 4, DIM, f);
            RD(xf->layers[ly].wq, 4, DIM * DIM, f);
            RD(xf->layers[ly].wk, 4, DIM * DIM, f);
            RD(xf->layers[ly].wv, 4, DIM * DIM, f);
            RD(xf->layers[ly].wo, 4, DIM * DIM, f);
            RD(xf->layers[ly].ffn_norm, 4, DIM, f);
            RD(xf->layers[ly].w1, 4, DIM * HDIM, f);
            RD(xf->layers[ly].w2, 4, DIM * HDIM, f);
            RD(xf->layers[ly].w3, 4, HDIM * DIM, f);
        }
        RD(xf->final_norm, 4, DIM, f);
        RD(xf->res_proj,   4, DIM * RES_DIM, f);
    }

    for (int c = 0; c < N_CHAMBERS; c++) {
        RD(k->chambers.ch[c].w1, 4, RES_DIM * 128, f);
        RD(k->chambers.ch[c].b1, 4, 128, f);
        RD(k->chambers.ch[c].w2, 4, 128 * 64, f);
        RD(k->chambers.ch[c].b2, 4, 64, f);
        RD(k->chambers.ch[c].w3, 4, 64 * 32, f);
        RD(k->chambers.ch[c].b3, 4, 32, f);
        RD(k->chambers.ch[c].w4, 4, 32, f);
        RD(k->chambers.ch[c].b4, 4, 1, f);
    }
    RD(k->chambers.coupling, 4, N_CHAMBERS * N_CHAMBERS, f);

    fclose(f);

    float d[] = {0.90f, 0.93f, 0.85f, 0.97f, 0.88f, 0.94f};
    memcpy(k->decay, d, sizeof(d));
    k->mem_ptr = 0; k->mem_n = 0;
    memset(k->mem_w, 0, sizeof(k->mem_w));
    k->cooc.n = 0;

    /* build somatic affinity table from seed + loaded BPE */
    init_somatic_affinity(k);

    /* build RRPRAM merge table from BPE structure */
    init_rrpram(k);

    fprintf(stderr, "loaded: 4 decoders x %d layers, 6 chambers, Dario equation, RRPRAM\n", N_LAYERS);
    return 0;
}

#undef RD

/* ================================================================
 *  PING
 * ================================================================ */

typedef struct {
    int   lang;
    float act[N_CHAMBERS];
    float trend;
    char  somatic[2048];
} KlausResponse;

static void ping(Klaus *k, const char *text, KlausResponse *resp) {
    resp->lang = detect_lang(text);

    int tokens[MAX_SEQ];
    int slen = bpe_encode(k, text, tokens, MAX_SEQ - MAX_GEN);
    if (slen == 0) { resp->somatic[0] = '\0'; return; }

    /* ingest input into Hebbian field */
    cooc_ingest(&k->cooc, tokens, slen);

    int gen[MAX_GEN];
    int ngen = generate(k, resp->lang, tokens, slen, resp->act, gen, MAX_GEN);
    resp->trend = mem_trend(k);
    bpe_decode(k, gen, ngen, resp->somatic, sizeof(resp->somatic));
}

/* ================================================================
 *  MAIN
 * ================================================================ */

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr,
            "klaus -- somatic engine\n"
            "usage: %s <weights.bin> [--interactive]\n\n"
            "  4 decoders, 6 chambers, Dario equation, Hebbian field.\n"
            "  Ancestors: haze/cloud, dario.c, postgpt.\n\n", argv[0]);
        return 1;
    }

    srand(time(NULL));

    Klaus *k = calloc(1, sizeof(Klaus));
    if (!k) { fprintf(stderr, "alloc failed (%zu bytes)\n", sizeof(Klaus)); return 1; }

    fprintf(stderr, "loading %s...\n", argv[1]);
    if (load_weights(k, argv[1]) != 0) { free(k); return 1; }

    int interactive = (argc > 2 && strcmp(argv[2], "--interactive") == 0);
    char line[4096];

    if (interactive)
        fprintf(stderr, "\nready. type text, get somatic response. 'quit' to exit.\n\n");

    while (1) {
        if (interactive) { printf("klaus> "); fflush(stdout); }
        if (!fgets(line, sizeof(line), stdin)) break;

        int len = strlen(line);
        while (len > 0 && (line[len - 1] == '\n' || line[len - 1] == '\r'))
            line[--len] = '\0';
        if (len == 0) continue;
        if (strcmp(line, "quit") == 0 || strcmp(line, "exit") == 0) break;

        KlausResponse resp;
        ping(k, line, &resp);

        printf("[%s] \"%s\"\n", LNAME[resp.lang], line);
        printf("  chambers:");
        for (int i = 0; i < N_CHAMBERS; i++)
            if (resp.act[i] > 0.05f)
                printf(" %s:%.0f%%", CNAME[i], resp.act[i] * 100.0f);
        printf("\n");
        printf("  \xe2\x86\x92 %s\n", resp.somatic);
        if (resp.trend > 0.05f || resp.trend < -0.05f)
            printf("  trajectory: %s (%.3f)\n",
                   resp.trend > 0 ? "RISING" : "FALLING", resp.trend);
        printf("\n");
    }

    free(k);
    return 0;
}
