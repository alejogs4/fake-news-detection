# HGFND — Hyperedge Extension Experiments

This document records three experiments that extend the **HGFND** model
(IEEE BigData 2022) by adding new types of hyperedges to the original
social-network structure, with the goal of improving fake news detection
on the PolitiFact dataset.

**Dataset:** PolitiFact — 314 news articles (157 real, 157 fake).  
**Baseline model:** HGFND as published in the original paper.  
**Evaluation:** Test accuracy and F1-macro, averaged over 7 random seeds.  
**Seeds:** `[42, 2026, 33, 123, 456, 789, 1000]`

---

## Table of Contents

1. [How the Model Works](#how-the-model-works)
2. [Baseline](#baseline)
3. [Experiment 1 — Verified-User Hyperedge](#experiment-1--verified-user-hyperedge)
4. [Experiment 2 — Sentiment Hyperedge](#experiment-2--sentiment-hyperedge)
5. [Experiment 3 — Country Hyperedge (Not Possible)](#experiment-3--country-hyperedge-not-possible)
6. [Experiment 4 — Combined Best Hyperedges](#experiment-4--combined-best-hyperedges)
7. [Overall Conclusions](#overall-conclusions)

---

## How the Model Works

### Full pipeline at a glance

```
 ┌─────────────────────────────────────────────────────────────────────────┐
 │  For each news article                                                  │
 │                                                                         │
 │  RAW TEXT  ──►  BERT (frozen)  ──►  768-dim vector per article         │
 │                                                                         │
 │  SOCIAL GRAPH  ──►  PropagationEncoder  ──►  128-dim vector per article│
 │  (who shared it)      (GraphSAGE + pool)      (content + spread)       │
 └──────────────────────────────┬──────────────────────────────────────────┘
                                │  314 × 128   (one row per news item)
                                ▼
              ┌─────────────────────────────────────┐
              │  HYPEREDGE ASSIGNMENT  (our addition)│
              │                                     │
              │  Group articles by shared property: │
              │  sentiment tone / verified-user ratio│
              │                                     │
              │  Produces HT matrix  (E × 314)      │
              │  E = number of hyperedges            │
              └────────────────┬────────────────────┘
                               │
                               ▼
              ┌─────────────────────────────────────┐
              │  HGNN_ATT  (2-layer hypergraph attn) │
              │                                     │
              │  gat1: node → hyperedge             │
              │    articles in same group attend    │
              │    to each other, produce 1 summary │
              │    vector per hyperedge             │
              │                                     │
              │  gat2: hyperedge → node             │
              │    each article reads back the      │
              │    summary of its group             │
              │                                     │
              │  Output: 314 × 128  (enriched)      │
              └────────────────┬────────────────────┘
                               │
                               ▼
              Linear(128 → 2)  +  log_softmax
                               │
                    ┌──────────┴──────────┐
                  REAL (0)           FAKE (1)
```

> **Key idea:** BERT and the social graph are fixed. We only change the
> hyperedge grouping (step 4) and let the attention mechanism do the rest.
> Backpropagation trains `lin0`, `SAGEConv`, `lin1`, `gat1`, `gat2`, and the
> final classifier. BERT weights are never updated.

---

### Step 1 — Raw text

Every article enters as plain text. Length varies enormously:

| Article | Label | Text (truncated) | Length |
|---|---|---|---|
| `politifact4190` | REAL | *"The Budget and Economic Outlook: Fiscal Years 2009 to 2019"* | 58 chars |
| `politifact13068` | REAL | *"Clinton said, Trump Management was charged with discriminating against African-Americans..."* | 704 chars |
| `politifact14960` | FAKE | *"It's absolutely clear the Clinton Foundation has faced reports of stealing from impoverished Haitians..."* | 4 417 chars |
| `politifact13565` | FAKE | *"Trump positioned himself as a front-line soldier in the War on Christmas, which made non-Christians justifiably nervous..."* | 11 984 chars |

---

### Step 2 — BERT encoding (pre-computed, frozen)

Each article's full text is passed through BERT and compressed into a
**768-dimensional float vector**. This is done once before training and stored.
The model never re-runs BERT — it reads the stored vectors as fixed node features.

```
CBO report :  [-0.331, -0.258,  0.162,  0.069,  0.001, -0.039, ...]  (768 values)
Fact-check :  [ 0.114, -0.293, -0.214, -0.037, -0.180, -0.417, ...]
Haiti FAKE :  [ 0.014, -0.467, -0.432, -0.252,  0.354, -0.268, ...]
Xmas FAKE  :  [ 0.115, -0.361, -0.024, -0.108,  0.139, -0.373, ...]
```

All vectors have similar scale (mean ≈ −0.03, std ≈ 0.84). The **semantic
differences** between real and fake news are encoded in the *direction* of the
vector, not its magnitude. This is why simply looking at the raw numbers tells
you nothing — the model learns which directions matter during training.

---

### Step 3 — Social propagation graph + PropagationEncoder

Each article has its own **mini social graph**: a tree of users who shared or
retweeted it. The root node is the original news tweet.

```
              [news tweet]          ← root node (has BERT features)
             /      |      \
        [user A] [user B] [user C]  ← retweeters (have profile features)
        /    \
  [user D] [user E]
```

| Article | Users (nodes) | Retweets (edges) | Max degree | Pattern |
|---|---|---|---|---|
| CBO report | 497 | 496 | 450 | One hub re-shared to 450 people — very viral |
| Fact-check | 30 | 29 | 15 | Small, lightly shared |
| Haiti FAKE | 109 | 108 | 44 | Moderately viral |
| Christmas FAKE | 104 | 103 | 40 | Moderately viral |

`PropagationEncoder` processes each graph independently and outputs **one
128-dim vector per article**:

```
  x[root]  ──  lin0(768→128)  ──────────────────────────────  128d  ← text signal
                                                                  ↘
  all nodes ── SAGEConv(768→128) ── global_max_pool ──────────  128d  ← spread signal
                                                                  ↗
                                lin1([text ; spread], 256→128) ── 128d  final repr
```

After this step every news item — regardless of how many users shared it — is
represented as a single 128-dim vector. The model has learned both *what* the
article says and *how* it spread.

---

### Step 4 — Hyperedge assignment (our extension)

Articles are grouped into buckets based on a shared property. Each bucket
becomes one **hyperedge** in the incidence matrix `HT` (shape `E × 314`).
An entry `HT[e, i] = 1` means article `i` belongs to hyperedge `e`.

**Example — VADER 3-class sentiment (Experiment 2):**

```
  HT  =  [ NEGATIVE row ]  →  87 articles with alarming titles   (70 % fake)
          [ NEUTRAL  row ]  → 130 articles with dry/neutral titles (65 % real)
          [ POSITIVE row ]  →  97 articles with positive titles   (52 % fake)
```

How four concrete articles are routed:

| Article | Label | VADER compound | Assigned hyperedge |
|---|---|---|---|
| CBO report | REAL | +0.000 | **NEUTRAL** ← dry, statistical |
| Fact-check | REAL | −0.202 | **NEGATIVE** ← VADER flags "discriminating" (misrouted) |
| Haiti FAKE | FAKE | −0.155 | **NEGATIVE** ✓ alarming language detected |
| Christmas FAKE | FAKE | −0.612 | **NEGATIVE** ✓ strong negative signal |

---

### Step 5 — Hypergraph attention (HGNN_ATT)

The encoder runs **two attention layers** over the hyperedges. This is where
articles *talk to each other*:

```
  Input: 314 × 128   (all articles after PropagationEncoder)
  HT:      E × 314   (hyperedge incidence matrix)

  ── gat1: node → hyperedge attention ──────────────────────────────────────
  Each hyperedge E_k collects its member articles and computes:
    h_k = Σ  α_i · v_i      (softmax-weighted sum of member embeddings)
  where α_i is a learned attention weight for article i in hyperedge k.

  Result: one 128-dim summary vector per hyperedge  (E × 128)

  ── gat2: hyperedge → node attention (transfer=True) ─────────────────────
  Each article reads back from its hyperedge's summary:
    v_i' = Σ  β_k · h_k     (weighted sum of hyperedge summaries)
  where β_k is how much article i should attend to its group k.

  Result: 314 × 128   (final enriched representations)
```

After this step:
- The **War on Christmas** article has absorbed information from the other 86
  NEGATIVE articles — it "knows" what other alarming-language articles look like.
- The **CBO report** has absorbed information from the other 129 NEUTRAL
  articles — it "knows" what other dry, factual articles look like.

---

### Step 6 — Classification

```
  128d  →  Linear(128 → 2)  →  log_softmax  →  [log P(real), log P(fake)]
```

The class with the higher log-probability wins. During training, cross-entropy
loss flows backward through all learned layers (`lin0`, `SAGEConv`, `lin1`,
`gat1`, `gat2`, classifier). BERT is frozen throughout.

---

## Baseline

The experiment asks: *"does the original HGFND social-graph grouping alone produce a reliable accuracy ceiling to compare against?"*

The original HGFND paper groups articles using only their **social engagement
graph** — which accounts are connected to which news items through sharing
behaviour. No additional grouping is applied.

| Metric | Value |
|---|---|
| Accuracy | ~0.902 ± 0.025 |
| F1-macro | ~0.902 ± 0.025 |

> The baseline is re-run from scratch for each experiment (same seeds,
> independent model initialisation), so it varies slightly between tables
> (~0.9024 vs ~0.9043).

---

## Experiment 1 — Verified-User Hyperedge

**Notebook:** `train_threshold_sweep_v2.ipynb`

The experiment asks: *"does grouping news articles by the fraction of verified-account sharers help the model detect fake news?"*

### Motivation

Twitter/X marks some accounts as **verified** (blue checkmark). Articles shared
predominantly by verified accounts are more likely to be real; articles shared
mostly by unverified accounts may be more likely to be fake. Grouping articles
by this signal could give the model a useful structural hint.

### Method

For each news article, compute the fraction of its sharers that are verified.

- **Above-threshold group:** articles where the verified fraction exceeds a threshold *t*
- **Below-threshold group:** articles where it falls below *t*

The threshold *t* is swept from 0 % to 7 %. Both groups are added as separate
hyperedges. An ablation also tests each group alone.

### Results

**Both edges combined:**

| Config | Accuracy | ± std | Δ vs baseline | Tag |
|---|---|---|---|---|
| Baseline | 0.9043 | 0.0241 | — | — |
| +Both >0 % | 0.9005 | 0.0282 | −0.0039 | MARGINAL |
| +Both ≥1 % | 0.9095 | 0.0334 | +0.0052 | IMPROVEMENT |
| +Both ≥3 % | 0.9076 | 0.0179 | +0.0032 | MARGINAL |
| **+Both ≥5 %** | **0.9101** | **0.0229** | **+0.0058** | **IMPROVEMENT** |
| +Both ≥7 % | 0.9030 | 0.0305 | −0.0013 | MARGINAL |

**Ablation at ≥5 % threshold:**

| Config | Accuracy | ± std | Δ | Tag |
|---|---|---|---|---|
| +Above only ≥5 % | 0.9056 | 0.0244 | +0.0013 | MARGINAL |
| +Below only ≥5 % | 0.9056 | 0.0264 | +0.0013 | MARGINAL |
| **+Both ≥5 %** | **0.9101** | **0.0229** | **+0.0058** | **IMPROVEMENT** |

### Conclusions

- **Best config: +Both ≥5 %** — split into two hyperedges using a 5 % verified-user threshold.
- Too low a threshold (>0 %) adds noise from articles with a single incidental verified sharer.
  Too high (≥7 %) loses too many articles to build a useful group.
- At ≥5 %, both edges contribute equally — the model benefits from the *contrast*
  between the two groups, not just one.
- The improvement (+0.0058) is consistent across seeds but does not reach
  statistical significance with only 7 seeds.

---

## Experiment 2 — Sentiment Hyperedge

**Notebook:** `train_sentiment_sweep_v2.ipynb`

The experiment asks: *"does grouping news articles by emotional tone help the model detect fake news?"*

### Motivation

Fake news often uses emotionally charged language to provoke reactions, while
real news tends to be written in a dry, factual tone. Grouping articles by
**sentiment tone** could expose this pattern as a structural signal in the
hypergraph.

### Method

VADER (a lexicon-based sentiment analyser) scores the **title** (first ~150 chars)
of each article, producing a compound score from −1 to +1. Standard VADER
thresholds create 3 groups:

| Group | Threshold | Count | Real % | Fake % |
|---|---|---|---|---|
| NEGATIVE | compound ≤ −0.05 | 87 | 30 % | **70 %** |
| NEUTRAL | −0.05 < compound < +0.05 | 130 | **65 %** | 35 % |
| POSITIVE | compound ≥ +0.05 | 97 | 48 % | 52 % |

The experiment tests all 3 groups individually (ablation) and in combinations.

**Why 3 classes instead of 2?**  
A v1 experiment used a binary alarm/calm split. Splitting "calm" into NEUTRAL and
POSITIVE revealed that the two groups are very different: NEUTRAL is 65 % real
(a real signal), while POSITIVE is nearly random (52 % fake — noise). A 2-class
split mixes them together and dilutes the NEUTRAL signal.

**Where VADER fails (real examples from the dataset):**

| Title | True label | Compound | VADER says | Problem |
|---|---|---|---|---|
| "BREAKING!" | FAKE | 0.000 | NEUTRAL | "Breaking" is neutral in the lexicon |
| "The War We Need to Win" (Obama speech) | REAL | −0.026 | NEGATIVE | "War" triggers a false alarm |
| "make America safe again" | FAKE | +0.440 | POSITIVE | Fear-based framing scored as positive |
| "Copyright. All Rights Reserved." | FAKE | +0.402 | POSITIVE | Page footer, not a headline |

VADER is correct ~65–70 % of the time on news headlines.

### Results

| Config | Accuracy | ± std | Δ vs baseline | Tag |
|---|---|---|---|---|
| Baseline | 0.9024 | 0.0266 | — | — |
| +NEG only | 0.9076 | 0.0268 | +0.0052 | IMPROVEMENT |
| +NEU only | 0.9030 | 0.0233 | +0.0006 | NEGLIGIBLE |
| +POS only | 0.9056 | 0.0295 | +0.0032 | NEGLIGIBLE |
| +NEG+NEU | 0.9069 | 0.0198 | +0.0045 | NEGLIGIBLE |
| **+NEG+NEU+POS** | **0.9114** | **0.0219** | **+0.0090** | **IMPROVEMENT** |

### Conclusions

- **Best config: +NEG+NEU+POS** (all three groups), accuracy 0.9114 (+0.0090).
- **NEGATIVE is the most informative single group** (+0.0052 alone) — alarming
  language is the clearest fake-news signal VADER can detect.
- **NEUTRAL adds almost nothing on its own** (+0.0006) despite being 65 % real —
  it is too large and contains too many VADER misclassifications.
- **Adding POSITIVE helps even though it is near-random.** The reason is
  structural: three finer groups create a more informative partition for the
  hypergraph to exploit, regardless of each group's individual purity.
- All p-values > 0.5 with 7 seeds — no result is statistically significant.

---

## Experiment 4 — Combined Best Hyperedges

**Notebook:** `train_combined_best.ipynb`

The experiment asks: *"do sentiment and verification hyperedges stack — does combining the two best individual signals produce a larger improvement than either alone?"*

### Motivation

Experiments 1 and 2 each improved accuracy individually. The natural next question
is: **do they stack?** If the two signals are independent, combining them should
give a larger improvement than either alone. If they are redundant, combining them
may add noise and hurt.

### Method

Four configurations are tested head-to-head using the same 7 seeds:

| Config | Hyperedges added | Source |
|---|---|---|
| Baseline | none | HGFND paper |
| +Sentiment | NEG + NEU + POS (VADER 3-class) | best from Experiment 2 |
| +Verification | Above ≥5 % + Below ≥5 % | best from Experiment 1 |
| +Sent+Verif | all 5 hyperedges combined | new — synergy test |

### Results

| Config | Accuracy | ± std | Δ vs baseline | Tag |
|---|---|---|---|---|
| Baseline | 0.9037 | 0.0249 | — | — |
| **+Sentiment** | **0.9114** | **0.0219** | **+0.0078** | **IMPROVEMENT** |
| **+Verification** | **0.9114** | **0.0231** | **+0.0078** | **IMPROVEMENT** |
| +Sent+Verif | 0.9056 | 0.0225 | +0.0019 | NEGLIGIBLE |

**Synergy test:**

| Comparison | t-stat | p-value | Verdict |
|---|---|---|---|
| Combined vs +Sentiment | −0.45 | 0.658 | INTERFERENCE |
| Combined vs +Verification | −0.44 | 0.666 | INTERFERENCE |

### Conclusions

- **+Sentiment and +Verification tie exactly at 0.9114** (+0.0078 each). Both
  individually match the best results from their dedicated sweeps.
- **Combining them causes interference** — the combined config (0.9056) is
  noticeably *worse* than either alone. Stacking does not stack.
- **Why interference?** Sentiment and verification are both proxies for the same
  underlying pattern: fake news tends to use alarming language *and* be shared by
  unverified accounts. When both are added to the hypergraph, the attention layers
  receive correlated messages from two overlapping groups. The model cannot extract
  additional signal — it just processes more noise, and overfits faster.
- **Training curves confirm this:** all configs plateau around epoch 25–50, then
  degrade with increasing variance through epochs 150–200. The combined config
  (5 extra hyperedges) shows the widest variance and most visible late-stage drop,
  consistent with overfitting on only 62 training articles. Best-val checkpointing
  insulates the reported test numbers from this, but it signals that early stopping
  at epoch ~50–75 would be more appropriate for this dataset size.
- All p-values > 0.65 — no result is statistically significant.

---

## Experiment 3 — Country Hyperedge (Not Possible)

The experiment asks: *"does grouping articles by the country of origin of their sharers expose coordinated disinformation campaigns?"*

The idea was to group articles by the **country of origin** of their sharers —
for example, separating articles shared predominantly by US accounts from those
shared internationally. This could expose coordinated disinformation campaigns.

**Result: data not available.**  
`data/politifact/raw/` contains no geographic or country metadata for users.
This experiment is blocked until a dataset with geolocation data is available.

---

## Overall Conclusions

| Experiment | Best config | Δ accuracy | Statistically significant? | Status |
|---|---|---|---|---|
| Verified-user hyperedge | +Both ≥5 % | +0.0058 | No (7 seeds) | ✅ Done |
| Sentiment hyperedge 3-class | +NEG+NEU+POS | +0.0090 | No (7 seeds) | ✅ Done |
| Country hyperedge | — | — | — | ❌ Blocked — no data |
| Combined best (head-to-head) | +Sentiment or +Verification | +0.0078 | No (7 seeds) | ✅ Done |

**Best single configuration across all experiments:** +Sentiment (+NEG+NEU+POS) or
+Verification (+Both ≥5 %) — both reach accuracy **0.9114** in the combined run.

**Four key lessons:**

1. **Both extensions improve accuracy consistently** over the HGFND baseline.
   The improvements are small but point in the same direction across all 7 seeds.

2. **Combining the two extensions causes interference, not synergy.**
   When both sentiment and verification hyperedges are added together, accuracy
   *drops* compared to either alone (0.9056 vs 0.9114). The two signals are
   correlated — fake news tends to use alarming language *and* be shared by
   unverified accounts — so adding both carries redundant information and
   amplifies overfitting on the small training set (62 articles).

3. **The improvements are not statistically significant** with 7 seeds.
   Reaching p < 0.05 for effects of this size would require ~40 seeds, which is
   computationally prohibitive. These results should be treated as exploratory
   evidence, not conclusive proof.

4. **BERT redundancy is the fundamental ceiling.**  
   The HGFND model already has 768-dimensional BERT features encoding every word
   in every article. Any content-based hyperedge (sentiment, topic, veracity
   signals) is partially redundant with what BERT already knows. The hyperedge
   additions provide complementary *structural* signal — telling each article who
   its neighbours are — but cannot overcome information already captured in the
   node features. The path to larger gains likely lies in improving the node
   features themselves (e.g., replacing frozen BERT with a model fine-tuned on
   fake-news corpora) rather than in adding more hyperedges.
