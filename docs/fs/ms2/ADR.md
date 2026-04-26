# MS2 Model Architectures — Detailed Design Specification

- **Ticket:** `MS2-DESIGN-01`
- **Milestone:** MS2 (RNN vs. Transformer, both from scratch)
- **Status:** Design freeze — implementation contract for `MS2-MODEL-A` and `MS2-MODEL-B`
- **Scope:** Full architectural specification of Model A and Model B, including shapes, equations, parameter budgets, training and inference protocols, and TF/Keras implementation notes.

---

## 0. Document Conventions

- All shapes are written `(dim_1, dim_2, ...)` with `B` = batch size.
- All sequence-length symbols are runtime-variable per batch; concrete bounds are documented in §1.4.
- Pseudocode is illustrative; the binding contract is the shape annotations and equations.
- Equations use `⊙` for element-wise (Hadamard) product and `·` for matrix multiplication.
- "From scratch" means **no pretrained weights, no pretrained embeddings, no distillation from any pretrained model**. Tokenizers trained on our own corpus are permitted (the rubric forbids pretrained _models_, not pretrained _tokenizers_).
- All Keras layers below are `tf.keras.layers.*` unless noted. Implementation uses the **Subclassing API** for both models.

---

## 1. Shared Infrastructure (Both Models)

This section is binding for both Model A and Model B. Anything that differs between models is documented in §2 and §3.

### 1.1 Tokenizer

- **Algorithm:** Byte-Pair Encoding (BPE) trained from scratch via SentencePiece.
- **Vocabulary size:** `V = 4096`.
- **Training corpus:** The full transcript corpus from MS1's processed dataset (transcript text only — _not_ QA pairs, to avoid information leakage from the supervised set into the tokenizer).
- **Special tokens (reserved IDs 0–4):**
  - `<pad>` (id 0) — padding
  - `<unk>` (id 1) — unknown
  - `<bos>` (id 2) — beginning of sequence (decoder start token)
  - `<eos>` (id 3) — end of sequence
  - `<sep>` (id 4) — separator between question and context
- **Character coverage:** 1.0 (cover all characters present in the corpus, including Latin characters from code-switching).
- **Output artifact:** `data/processed/ms2/ms2_tokenizer_bpe_4k_v001.model` (SentencePiece model file).

### 1.2 Sequence Formatting

For both models, every training example is a triple `(question, context, answer)` from the QA pairs.

**Encoder input** is the concatenation:

```
[<bos>] question_tokens [<sep>] context_tokens [<eos>]
```

**Decoder input** (teacher-forced during training):

```
[<bos>] answer_tokens
```

**Decoder target** (shifted by one):

```
answer_tokens [<eos>]
```

This is the standard seq2seq teacher-forcing setup. Loss is computed on the decoder target with `<pad>` positions masked out.

### 1.3 Length Bounds and Truncation

Determined empirically from the MS1 length distribution. The numbers below are **placeholders** to be confirmed in `MS2-DATA-01` against the actual percentiles of the cleaned corpus.

| Symbol  | Meaning              | Cap (placeholder) | Truncation Policy                          |
| ------- | -------------------- | ----------------- | ------------------------------------------ |
| `L_q`   | question length      | 32                | Right-truncate (rare; questions are short) |
| `L_c`   | context length       | 384               | Window around the answer span (see below)  |
| `L_enc` | encoder input length | 420               | `L_q + L_c + 4` (specials)                 |
| `L_dec` | decoder length       | 64                | Right-truncate                             |

**Context windowing:** Because some transcripts are very long but answer spans are local, training context is a **window centered on the gold answer span** of length `L_c`, with random jitter during training (jitter range = ±20% of `L_c`) to prevent the model from learning that the answer is always in the middle of the window. At inference, when the gold span is unknown, the full transcript is fed and (if longer than `L_c`) split into overlapping windows with stride `L_c / 2`; the answer is generated per window and the highest-confidence generation is selected by length-normalized log-probability.

### 1.4 Batch Construction

- **Bucket-by-length** via `tf.data.Dataset.bucket_by_sequence_length` on `L_enc`.
- **Bucket boundaries:** [128, 192, 256, 320, 420].
- **Per-bucket batch size:** scaled inversely with length so total tokens per batch ≈ 16,384.
- **Shuffle buffer:** 4096 examples.
- **Cache tokenized dataset to TFRecord** to avoid re-tokenization across epochs.

### 1.5 Mixed Precision

- Global policy: `mixed_float16` via `tf.keras.mixed_precision.set_global_policy('mixed_float16')`.
- Loss is computed in `float32` for numerical stability.
- Optimizer is wrapped in `LossScaleOptimizer`.

### 1.6 Optimizer (Both Models)

- **Family:** AdamW (`tf.keras.optimizers.AdamW`).
- `β_1 = 0.9`, `β_2 = 0.98`, `ε = 1e-9`, `weight_decay = 0.01`.
- Learning rate **schedule differs per model** — see §2.7 and §3.7.

### 1.7 Loss

- **Categorical cross-entropy** with label smoothing `ε_ls = 0.1`.
- Computed only on non-`<pad>` decoder target positions (mask applied before reduction).
- Reduction: mean over unmasked target tokens.

### 1.8 Metrics (Computed on Dev/Test)

All metrics are computed _after_ applying a unified Arabic post-normalization step (alif/hamza unification, ya/alif-maqsura unification, diacritic stripping) to both predictions and references — same normalization for both models.

- **Exact Match (EM):** percentage of predictions equal to the reference after normalization.
- **Token-F1 (SQuAD-style):** F1 over the bag of post-normalized tokens.
- **Character edit distance (normalized):** Levenshtein / max(len(pred), len(ref)).
- **BLEU-1:** unigram BLEU as a soft secondary metric.

### 1.9 Reproducibility

- All seeds set via `tf.keras.utils.set_random_seed(seed)` with `seed ∈ {13, 42, 91}` for the three runs.
- `tf.config.experimental.enable_op_determinism()` is enabled for the seeded runs.
- Tokenizer training is deterministic (single-threaded, fixed seed).

---

## 2. Model A — Tri-Encoder Gated Fusion seq2seq with FiLM-Conditioned Decoder

### 2.1 Design Intent

Model A is constrained to use **no attention mechanism anywhere**. The architectural challenge is therefore: how does a decoder generate an answer conditioned on a question and a context, without ever attending over encoder positions?

The answer is a three-part architecture:

1. **Three parallel encoder branches**, each with a distinct inductive bias, capturing different aspects of the input.
2. **A gated merge block** that performs **per-position mixture-of-experts routing** over the three branches, producing a single fused representation.
3. **A FiLM-conditioned LSTM decoder** that receives a global summary of the fused encoder output as a feature-wise affine modulation at every layer.

The decoder never queries the encoder over positions. The encoder's influence flows through (a) the decoder's initial state and (b) FiLM modulation. Both are **global** signals — neither involves any position-pairwise computation.

### 2.2 Hyperparameters

| Symbol         | Meaning                                  | Value |
| -------------- | ---------------------------------------- | ----- |
| `V`            | vocabulary size                          | 4096  |
| `d_tok`        | BPE token embedding dim                  | 128   |
| `d_char`       | character embedding dim                  | 32    |
| `d_charcnn`    | char-CNN output dim per token            | 64    |
| `d_b1`         | Branch 1 hidden (per direction)          | 96    |
| `d_b2`         | Branch 2 hidden (per direction)          | 96    |
| `d_b3`         | Branch 3 hidden (per direction)          | 96    |
| `d_fuse`       | common projection dim in gated merge     | 192   |
| `d_refine`     | refinement Bi-GRU hidden (per direction) | 96    |
| `d_enc_out`    | final encoder output dim                 | 192   |
| `d_dec`        | decoder LSTM hidden                      | 256   |
| `d_film_h`     | FiLM generator hidden dim                | 128   |
| `n_dec_layers` | decoder LSTM layers                      | 2     |
| `dropout_emb`  | embedding dropout                        | 0.2   |
| `dropout_rnn`  | inter-layer RNN dropout                  | 0.3   |
| `dropout_ff`   | dropout on fused features                | 0.3   |

Final encoder output dim is `2 × d_refine = 192`.

### 2.3 Token Embedding Layer

- BPE token embedding: `Embedding(V, d_tok)`, weights shared with the decoder output projection (tied embeddings).
- Initialization: `tf.keras.initializers.RandomNormal(stddev=d_tok ** -0.5)`.
- Padding mask: positions where input id is `<pad>` (id 0) are masked throughout the network.

### 2.4 Branch 1 — Question Encoder (Bi-LSTM, fine-grained)

**Input:** question token IDs of shape `(B, L_q)`.

**Architecture:**

```
question_ids       : (B, L_q)
        │
        ▼   Embedding(V, d_tok)
question_emb       : (B, L_q, d_tok)
        │
        ▼   Dropout(dropout_emb)
        │
        ▼   Bidirectional(LSTM(d_b1, return_sequences=True))
b1_seq             : (B, L_q, 2·d_b1)        # 2·d_b1 = 192
        │
        ▼   structured pooling (see below)
b1_pooled          : (B, 2·d_b1)             # = 192
```

**Structured pooling (attention-free):**

```
scores_i  = w_p · tanh(W_p · b1_seq_i + b_p)        # scalar per position
α_i       = softmax_i(scores_i)                     # over question positions
b1_pooled = Σ_i α_i · b1_seq_i
```

where `W_p ∈ ℝ^{d_p × 2·d_b1}`, `w_p ∈ ℝ^{d_p}`, `d_p = 64`.

**Defense in eval:** "This is a learned weighted sum over a _single_ sequence's positions. There is no query and no key — there is one parameter vector `w_p` shared across the dataset. Attention requires position-pairwise compatibility scoring; this does not compute pairwise scores."

### 2.5 Branch 2 — Context Encoder (Bi-GRU, wide)

**Input:** context token IDs of shape `(B, L_c)`.

**Architecture:**

```
context_ids        : (B, L_c)
        │
        ▼   Embedding (shared with Branch 1)
context_emb        : (B, L_c, d_tok)
        │
        ▼   Dropout(dropout_emb)
        │
        ▼   Bidirectional(GRU(d_b2, return_sequences=True))
        │
        ▼   Dropout(dropout_rnn)
        │
        ▼   Bidirectional(GRU(d_b2, return_sequences=True))
b2_seq             : (B, L_c, 2·d_b2)        # = 192
```

Two stacked Bi-GRU layers (this is what makes Branch 2 the workhorse of long-range narrative context).

### 2.6 Branch 3 — Joint Encoder (Char-CNN + BPE + Bi-LSTM)

**Input:** the joint sequence `[<bos>] question [<sep>] context [<eos>]` of shape `(B, L_enc)`, plus the per-token character matrix `(B, L_enc, L_char_max)` where `L_char_max = 16` (chars per BPE token are right-truncated; pad with character `<pad>`).

**Character embedding and CNN:**

```
char_ids           : (B, L_enc, L_char_max)
        │
        ▼   Embedding(V_char, d_char)
char_emb           : (B, L_enc, L_char_max, d_char)
        │
        ▼   3 parallel Conv1D(filters=d_charcnn/3, kernel_size=k, padding='valid')
        │   for k ∈ {2, 3, 4}, applied along the L_char_max axis
        │
        ▼   GlobalMaxPool1D over chars (per-token)
        │
        ▼   concatenate the three filter outputs
char_feat          : (B, L_enc, d_charcnn)   # 64
```

**Joint token embedding:**

```
joint_token_ids    : (B, L_enc)
        │
        ▼   Embedding (shared)
joint_token_emb    : (B, L_enc, d_tok)

joint_input        = concat([joint_token_emb, char_feat], axis=-1)
                   : (B, L_enc, d_tok + d_charcnn)   # 192
        │
        ▼   Dropout(dropout_emb)
        │
        ▼   Bidirectional(LSTM(d_b3, return_sequences=True))
b3_seq             : (B, L_enc, 2·d_b3)      # = 192
```

`V_char` is the character vocabulary, set during preprocessing (~150 for Arabic + Latin + digits + punctuation).

**Why char-CNN here specifically:** Branch 3 is the only branch that sees question and context jointly. Character-level features help with: (a) dialectal variants of the same lemma, (b) inconsistent transliteration of named entities, and (c) Latin-script tokens from code-switching that BPE may fragment unhelpfully. This is the linguistic motivation we will state in the report.

### 2.7 Alignment

The three branches produce sequences of three different lengths and we need to align them to a common axis (the **context** axis, length `L_c`) before fusion.

```
b1_aligned : tile b1_pooled across L_c             → (B, L_c, 2·d_b1)
b2_aligned : b2_seq                                 → (B, L_c, 2·d_b2)
b3_aligned : slice b3_seq to context positions only → (B, L_c, 2·d_b3)
```

**Slicing Branch 3:** the joint sequence layout is `[<bos>] [Q tokens] [<sep>] [C tokens] [<eos>]`. We index out positions corresponding to the context tokens — i.e., starting at offset `L_q + 2` (after `<bos>` and `<sep>`) for length `L_c`. Computed per-example using the runtime question length.

All three aligned tensors have shape `(B, L_c, 192)`.

### 2.8 Gated Merge Block (Per-Position Mixture-of-Experts)

This is the centerpiece of Model A. For every context position `i`, we compute three scalar gates that route between the three branches.

**Step 1 — Project to common dim:**

```
h1_i = W_1 · b1_aligned_i + b_1     (W_1: 2·d_b1 → d_fuse)
h2_i = W_2 · b2_aligned_i + b_2     (W_2: 2·d_b2 → d_fuse)
h3_i = W_3 · b3_aligned_i + b_3     (W_3: 2·d_b3 → d_fuse)
```

All `h*_i ∈ ℝ^{d_fuse}` where `d_fuse = 192`.

**Step 2 — Compute gates:**

```
concat_i  = [h1_i ; h2_i ; h3_i]                  ∈ ℝ^{3·d_fuse}
hidden_i  = GELU(W_g · concat_i + b_g)            ∈ ℝ^{d_fuse}
logits_i  = w_out · hidden_i + b_out              ∈ ℝ^{3}
gates_i   = softmax(logits_i)                     ∈ ℝ^{3}, sums to 1
```

**Step 3 — Mix:**

```
fused_i = gates_i[0] · h1_i + gates_i[1] · h2_i + gates_i[2] · h3_i
fused   : (B, L_c, d_fuse)
```

**Step 4 — Dropout and residual normalization:**

```
fused = LayerNorm(Dropout(fused, dropout_ff))
```

**Defense in eval:** "Attention computes pairwise compatibility scores between positions — `softmax(QK^T / √d)` produces an `L × L` matrix. This block computes **three** scalars per position via a local MLP, with no comparison to other positions. The softmax is over **branch identity**, not over positions. This is provably not attention; it is mixture-of-experts gating with three fixed experts."

### 2.9 Refinement Bi-GRU

A single Bi-GRU layer over the fused sequence to smooth and propagate locally:

```
fused              : (B, L_c, d_fuse)
        │
        ▼   Bidirectional(GRU(d_refine, return_sequences=True, return_state=True))
encoder_output     : (B, L_c, 2·d_refine)        # = 192 = d_enc_out
final_fwd_state    : (B, d_refine)
final_bwd_state    : (B, d_refine)
encoder_summary    = concat([final_fwd_state, final_bwd_state])  : (B, d_enc_out)
```

`encoder_summary` is the global vector `z` used by the FiLM generator and by the decoder's initial state.

### 2.10 FiLM Generator (Option 1 — Static Global)

A single pair `(γ, β)` is generated once per example and applied at the output of every decoder LSTM layer at every timestep.

```
z          = encoder_summary                          : (B, d_enc_out)        # 192
hidden     = GELU(W_f1 · z + b_f1)                    : (B, d_film_h)         # 128
γ_β        = W_f2 · hidden + b_f2                     : (B, 2·d_dec)          # 512
γ, β       = split(γ_β, axis=-1)                      : (B, d_dec), (B, d_dec)
```

**Identity initialization (critical):**

- `W_f2` is initialized with a small standard deviation (`stddev = 1e-3`).
- `b_f2` is split into `(b_γ, b_β)`. We initialize `b_γ = 1` (vector of ones, length `d_dec`) and `b_β = 0`. Therefore at step 0 of training, `γ ≈ 1` and `β ≈ 0`, which makes FiLM the identity transformation. The model has to _learn_ to use the conditioning signal.

This convention is project-wide: any future FiLM block in MS2 or MS3 follows the γ-near-1, β-near-0 init.

### 2.11 Decoder

A 2-layer unidirectional LSTM with FiLM modulation at the output of each layer.

**Initial state:** Both LSTM layers' initial `(h, c)` states are derived from the encoder summary by a small per-layer projection:

```
h_0^{(l)} = tanh(W_h^{(l)} · z + b_h^{(l)})        : (B, d_dec)
c_0^{(l)} = tanh(W_c^{(l)} · z + b_c^{(l)})        : (B, d_dec)
```

Separate projections per layer `l ∈ {1, 2}` — four projections total.

**Per-step forward (training, teacher-forced):**

Let `y_{<t}` be the decoder input tokens (shape `(B, L_dec)`), with `y_0 = <bos>`.

```
y_emb              = Embedding(y_{<t})              : (B, L_dec, d_tok)
        │
        ▼   Dropout(dropout_emb)
        │
        ▼   LSTM_layer_1 (return_sequences=True)
h1                 : (B, L_dec, d_dec)
        │
        ▼   FiLM with (γ, β) — broadcast across time:
        │   h1 = γ[:, None, :] ⊙ h1 + β[:, None, :]
        │
        ▼   Dropout(dropout_rnn)
        │
        ▼   LSTM_layer_2 (return_sequences=True)
h2                 : (B, L_dec, d_dec)
        │
        ▼   FiLM with the same (γ, β):
        │   h2 = γ[:, None, :] ⊙ h2 + β[:, None, :]
        │
        ▼   Output projection (tied to embedding)
logits             : (B, L_dec, V)
```

**Output projection (tied embeddings):**

```
logits = h2 · E^T + b_out     where E is the token embedding matrix
```

`b_out` is a learned bias of shape `(V,)` (the embedding matrix is shared in weight, but the output bias is not part of the embedding).

A linear projection `W_proj: d_dec → d_tok` is applied before the tied multiplication if `d_dec ≠ d_tok` (here `d_dec = 256 ≠ d_tok = 128`, so the projection is required):

```
h2_proj = W_proj · h2
logits  = h2_proj · E^T + b_out
```

`W_proj` is **not** tied to anything; it's a standalone learned matrix (`d_tok × d_dec`).

### 2.12 Training Schedule (Model A)

- **LR schedule:** Cosine decay with warmup. Peak LR `3e-4`, warmup over the first 5% of training steps, decay to `1e-5` over remaining steps.
- **Total training tokens:** target ~75 minutes wall-clock on the available GPU. Approximate epoch count to be set in `MS2-MODEL-A-03`.
- **Gradient clipping:** global norm clipping at `1.0`.
- **Weight decay:** applied to all weights _except_ embeddings and LayerNorm parameters.
- **Teacher forcing ratio:** 1.0 (always teacher-force during training). Scheduled sampling is an explicit ablation in §2.15.

### 2.13 Inference

- **Greedy decoding** as the headline number.
- **Beam search** with beam width 4, length normalization with `α = 0.6`, as a secondary number.
- **Maximum decode length:** `L_dec = 64`. Stop on `<eos>` or max length.
- Encoder is run once per example; encoder output, encoder summary, FiLM `(γ, β)`, and decoder initial states are cached and reused across decoding steps.

### 2.14 Parameter Budget (Estimate)

| Component                                                    | Params (approx.)  |
| ------------------------------------------------------------ | ----------------- |
| BPE token embedding (tied with output)                       | 4096 × 128 = 524k |
| Character embedding                                          | ~150 × 32 = 5k    |
| Char-CNN (3 filter sizes, ~21 filters each, kernel ≤ 4 × 32) | ~10k              |
| Branch 1 Bi-LSTM                                             | ~170k             |
| Branch 2 Bi-GRU (2 layers)                                   | ~330k             |
| Branch 3 Bi-LSTM                                             | ~250k             |
| Structured pooling MLP (Branch 1)                            | ~12k              |
| Gated merge projections + gate MLP                           | ~150k             |
| Refinement Bi-GRU                                            | ~170k             |
| FiLM generator                                               | ~90k              |
| Decoder LSTM (2 layers, d_dec=256)                           | ~790k             |
| Decoder initial-state projections                            | ~200k             |
| Output projection W_proj                                     | ~33k              |
| Output bias                                                  | 4k                |
| **Total**                                                    | **≈ 2.7M**        |

This is on the lean end of our 4–6M target. We accept this — we'd rather under-parameterize and avoid catastrophic overfitting on 3.9k examples than match Model B's count just for symmetry. We'll match training compute, not parameter count.

### 2.15 Ablations (Model A)

Each ablation changes exactly one thing relative to the spec above:

1. **No-FiLM:** remove FiLM modulation; decoder receives only the initial state from `z`. Tests whether FiLM is doing real work.
2. **Mean-merge:** replace the gated softmax in §2.8 with a uniform mean over the three branches (`gates_i ≡ [1/3, 1/3, 1/3]`). Tests whether the gating is doing real work.
3. **Plain Branch 3:** replace the char-CNN+BPE concat in Branch 3 with plain BPE embedding. Tests whether character features are doing real work.

Each ablation reuses the same training schedule, three seeds, same evaluation protocol.

### 2.16 TF/Keras Implementation Notes (Model A)

- Use the **Subclassing API** (`tf.keras.Model` subclass with explicit `call(inputs, training)`).
- The character-CNN block is a `tf.keras.layers.Layer` subclass. Reshape `(B, L_enc, L_char_max, d_char) → (B·L_enc, L_char_max, d_char)` before the Conv1D, then reshape back.
- Use `tf.keras.layers.LSTM` and `tf.keras.layers.GRU` (which dispatch to cuDNN under mixed precision when `activation='tanh'`, `recurrent_activation='sigmoid'`, no projection, default kernel init — keep these defaults).
- `Bidirectional` wrapper around the recurrent layers; concatenation of forward/backward is the default.
- FiLM modulation is a `Layer` subclass (`call(self, x, gamma, beta)`); it broadcasts `gamma` and `beta` across the time axis.
- Padding masks: derive once at the entry point from `tf.cast(input_ids != 0, tf.float32)`. Pass mask explicitly to the structured pooling layer (the softmax must be masked before normalization to ignore padding positions).
- Tied embeddings: implement the output projection as a custom `Layer` that holds a _reference_ to the embedding's `embeddings` variable, not a copy. Test that `model.trainable_variables` does not double-count the embedding.

---

## 3. Model B — RoPE Encoder–Decoder Transformer

### 3.1 Design Intent

Model B is a small encoder–decoder transformer trained from scratch, with **Rotary Positional Embeddings (RoPE)** instead of the classical additive sinusoidal positional encoding. The remainder of the architecture is intentionally minimal — clean multi-head attention, pre-LayerNorm residuals, a standard FFN — so that the comparison with Model A isolates the architectural variable (recurrence + gating vs. attention + RoPE) rather than confounding it with auxiliary tricks.

### 3.2 Hyperparameters

| Symbol          | Meaning                   | Value                    |
| --------------- | ------------------------- | ------------------------ |
| `V`             | vocabulary size           | 4096                     |
| `d_model`       | model dimension           | 192                      |
| `d_ff`          | FFN hidden dim            | 512                      |
| `n_heads`       | number of attention heads | 4                        |
| `d_head`        | dim per head              | 48 (= d_model / n_heads) |
| `n_enc_layers`  | encoder layers            | 3                        |
| `n_dec_layers`  | decoder layers            | 3                        |
| `θ_base`        | RoPE base frequency       | 10000                    |
| `dropout_attn`  | attention dropout         | 0.1                      |
| `dropout_resid` | residual dropout          | 0.1                      |
| `dropout_emb`   | embedding dropout         | 0.1                      |

### 3.3 Token Embedding Layer

- Same BPE-4k vocabulary as Model A.
- `Embedding(V, d_model)` with `RandomNormal(stddev=d_model ** -0.5)` initialization.
- Embeddings are scaled by `√d_model` after lookup (the classical "Attention Is All You Need" scaling — without it, attention output norms grow as `√d_model` while embeddings have norm ~1, causing early-training instability).
- Embeddings are tied with the output projection.
- **Encoder and decoder share the same embedding matrix** (single vocabulary, monolingual setup — sharing is standard).

### 3.4 Rotary Positional Embeddings (RoPE)

This subsection is the mathematical contract that the RoPE implementation must satisfy. It is also the answer-key for whiteboard questions during evaluation.

#### 3.4.1 Conceptual statement

Sinusoidal positional encoding is **added** to the token embedding:

```
x_m = embedding(token_m) + sinusoidal(m)
```

This means position information lives in the same vector space as semantic information and competes with it for representational capacity.

RoPE instead **rotates** the query and key vectors in 2D subspaces by an angle proportional to position. The dot product `<q_m, k_n>` then naturally becomes a function of the relative position `(m - n)` — relative position falls out of the math without us having to engineer it.

#### 3.4.2 The rotation

Let `d = d_head = 48`. We split each query / key vector into 24 consecutive 2D pairs:

```
q = [q^{(1)}; q^{(2)}; ...; q^{(d/2)}]    where each q^{(i)} ∈ ℝ^2
```

For each pair `i ∈ {1, ..., d/2}`, define a frequency:

```
θ_i = θ_base ^ (-2(i-1) / d) = 10000 ^ (-2(i-1)/48)
```

The pair frequencies range from `θ_1 = 1.0` (fastest rotation) to `θ_{24} ≈ 10000^(-46/48) ≈ 1.16e-4` (slowest).

For a token at position `m`, we rotate each pair by angle `m · θ_i`:

```
RoPE(q^{(i)}, m) = R(m · θ_i) · q^{(i)}

where R(φ) = [ cos(φ)  -sin(φ) ]
             [ sin(φ)   cos(φ) ]
```

#### 3.4.3 The relative-position property

For queries at position `m` and keys at position `n`:

```
<RoPE(q, m), RoPE(k, n)> = Σ_i <R(m·θ_i)·q^{(i)}, R(n·θ_i)·k^{(i)}>
                         = Σ_i <q^{(i)}, R(-m·θ_i)·R(n·θ_i)·k^{(i)}>
                         = Σ_i <q^{(i)}, R((n-m)·θ_i)·k^{(i)}>
```

The dot product depends only on `(n - m)`. **This is the property we want.** Defense in eval: derive the line above starting from the rotation matrix definition.

#### 3.4.4 Efficient implementation

Direct construction of `d/2` separate 2×2 matrices is wasteful. The standard implementation uses element-wise ops:

Given `q ∈ ℝ^{d}` viewed as `d/2` pairs, define:

```
q_even = q[..., 0::2]                   # the first elements of each pair
q_odd  = q[..., 1::2]                   # the second elements of each pair
```

Precompute (depending only on position `m` and the head dim, **not** on the actual query/key values):

```
freqs   = [θ_1, θ_2, ..., θ_{d/2}]          # constant per head
angles  = m · freqs                          # for position m
cos_m   = cos(angles)                        # shape (d/2,)
sin_m   = sin(angles)                        # shape (d/2,)
```

Then RoPE applied to `q` is:

```
q_rot_even = q_even · cos_m  -  q_odd  · sin_m
q_rot_odd  = q_even · sin_m  +  q_odd  · cos_m
q_rot      = interleave(q_rot_even, q_rot_odd)
```

In TF, the cleanest implementation uses `tf.stack` and `tf.reshape` to interleave:

```python
def apply_rope(x, cos, sin):
    # x:   (B, n_heads, L, d_head)
    # cos: (1, 1, L, d_head//2)
    # sin: (1, 1, L, d_head//2)
    x_even = x[..., 0::2]
    x_odd  = x[..., 1::2]
    rot_even = x_even * cos - x_odd * sin
    rot_odd  = x_even * sin + x_odd * cos
    # interleave back
    x_rot = tf.stack([rot_even, rot_odd], axis=-1)
    x_rot = tf.reshape(x_rot, tf.shape(x))
    return x_rot
```

The `cos`/`sin` tables are precomputed once for `L = max(L_enc, L_dec)` and indexed/sliced per batch.

#### 3.4.5 Where RoPE is applied

| Site                            | Apply RoPE? | Why                                                                                                                                                                                                                  |
| ------------------------------- | ----------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Encoder self-attention Q        | ✅          | Encoder positions are well-defined                                                                                                                                                                                   |
| Encoder self-attention K        | ✅          | Encoder positions are well-defined                                                                                                                                                                                   |
| Encoder self-attention V        | ❌          | RoPE only acts through the dot product `<Q, K>`; applying it to V breaks the algebra                                                                                                                                 |
| Decoder masked self-attention Q | ✅          | Decoder positions are well-defined                                                                                                                                                                                   |
| Decoder masked self-attention K | ✅          | Same                                                                                                                                                                                                                 |
| Decoder masked self-attention V | ❌          | Same as above                                                                                                                                                                                                        |
| Decoder cross-attention Q       | ❌          | Q comes from decoder positions; K from encoder positions; "relative position" between them is not meaningful — applying RoPE here would inject spurious position-relative bias between two different position spaces |
| Decoder cross-attention K       | ❌          | Same                                                                                                                                                                                                                 |
| Decoder cross-attention V       | ❌          | Same                                                                                                                                                                                                                 |

The cross-attention table row is the key one to defend: cross-attention queries the encoder, but the decoder's position `m` and the encoder's position `n` live in different position frames, so `(m - n)` has no consistent meaning. Cross-attention therefore uses **no positional information** — it relies entirely on the encoder having already encoded position into its Q/K/V via RoPE during encoder self-attention.

### 3.5 Encoder

3 stacked encoder layers. Pre-LayerNorm variant.

**Single encoder layer:**

```
x : (B, L_enc, d_model)
        │
        ▼   y = LayerNorm(x)
        │   q, k, v = Wq·y, Wk·y, Wv·y          (each: (B, L_enc, d_model))
        │   reshape to (B, n_heads, L_enc, d_head)
        │   q = apply_rope(q, cos_enc, sin_enc)
        │   k = apply_rope(k, cos_enc, sin_enc)
        │   scores = q · k^T / √d_head
        │   apply padding mask (set to -inf for pad keys)
        │   attn = softmax(scores)
        │   attn = Dropout(attn, dropout_attn)
        │   z = attn · v
        │   reshape back to (B, L_enc, d_model)
        │   z = Wo · z
        │
        ▼   x = x + Dropout(z, dropout_resid)        # residual
        │
        ▼   y = LayerNorm(x)
        │   z = W2 · GELU(W1 · y + b1) + b2          # FFN: d_model → d_ff → d_model
        │
        ▼   x = x + Dropout(z, dropout_resid)        # residual
```

W1 has shape `(d_model, d_ff) = (192, 512)`, W2 has shape `(d_ff, d_model) = (512, 192)`.

**Final encoder output:** after the 3rd encoder layer, apply one more `LayerNorm` (this is the standard pre-LN convention to ensure the encoder output has clean statistics).

```
encoder_output : (B, L_enc, d_model)
```

### 3.6 Decoder

3 stacked decoder layers. Pre-LayerNorm variant.

**Single decoder layer:** (let `x` be the decoder input at this layer, shape `(B, L_dec, d_model)`)

```
─── Sub-layer 1: Masked self-attention with RoPE ────────────────
        y = LayerNorm(x)
        q, k, v = Wq·y, Wk·y, Wv·y
        reshape to (B, n_heads, L_dec, d_head)
        q = apply_rope(q, cos_dec, sin_dec)
        k = apply_rope(k, cos_dec, sin_dec)
        scores = q · k^T / √d_head
        apply causal mask (upper-triangular = -inf) + padding mask
        attn = softmax(scores)
        attn = Dropout(attn, dropout_attn)
        z = attn · v
        z = Wo_self · reshape(z)
        x = x + Dropout(z, dropout_resid)

─── Sub-layer 2: Cross-attention (NO RoPE) ──────────────────────
        y = LayerNorm(x)
        q = Wq_x · y                         # from decoder
        k = Wk_x · encoder_output            # from encoder
        v = Wv_x · encoder_output            # from encoder
        reshape Q to (B, n_heads, L_dec, d_head),
        reshape K, V to (B, n_heads, L_enc, d_head)
        # NO apply_rope here
        scores = q · k^T / √d_head           # (B, n_heads, L_dec, L_enc)
        apply encoder padding mask on K positions
        attn = softmax(scores)
        attn = Dropout(attn, dropout_attn)
        z = attn · v
        z = Wo_x · reshape(z)
        x = x + Dropout(z, dropout_resid)

─── Sub-layer 3: FFN ────────────────────────────────────────────
        y = LayerNorm(x)
        z = W2 · GELU(W1 · y + b1) + b2
        x = x + Dropout(z, dropout_resid)
```

After the 3rd decoder layer, apply one more `LayerNorm`, then the output projection (tied to the input embedding):

```
decoder_output : (B, L_dec, d_model)
        │
        ▼   logits = decoder_output · E^T + b_out
logits : (B, L_dec, V)
```

If `d_model ≠ d_tok`, a projection is needed before tying — but here `d_tok = d_model = 192` (we are deliberately matching them in Model B to enable simple tying), so no projection is needed.

Note: in Model A, `d_tok = 128` ≠ `d_dec = 256`, so a projection was needed. In Model B, we set `d_model = d_tok = 192` to keep the output projection clean.

**Wait — discrepancy between Model A token dim (128) and Model B token dim (192):** This is an intentional asymmetry. Model A's BPE embedding lives in a 128-dim space and is then projected up to the 192-dim "shared dimension" via the gated merge projections. Model B's BPE embedding lives in a 192-dim space directly. The vocabulary is shared; the embedding matrices are _not_ shared between models (they are trained independently from scratch as part of each model). This is documented to flag the asymmetry but is not a problem — each model is internally consistent.

### 3.7 Training Schedule (Model B)

- **LR schedule:** Noam-style warmup (the original transformer schedule). `lr(step) = d_model^(-0.5) · min(step^(-0.5), step · warmup^(-1.5))` with `warmup = 1000` steps.
- **Total training time:** target ~75 minutes wall-clock, matching Model A's compute budget. Epochs may differ; we report wall-clock as the primary axis.
- **Gradient clipping:** global norm clipping at `1.0`.
- **Weight decay:** applied to all weights _except_ embeddings, LayerNorm parameters, and biases.
- **Teacher forcing ratio:** 1.0.

### 3.8 Inference

- Greedy decoding for the headline number.
- Beam search width 4, length normalization `α = 0.6` for the secondary number.
- Maximum decode length `L_dec = 64`.
- Encoder is run once per example. **Decoder K/V caching** during autoregressive decoding: at step `t`, only the new query position is computed; cached keys and values from previous steps are reused. This reduces inference complexity from `O(L_dec^2)` to `O(L_dec)` per step.

### 3.9 Parameter Budget (Estimate)

| Component                                         | Params (approx.)                                         |
| ------------------------------------------------- | -------------------------------------------------------- |
| Token embedding (tied; shared enc/dec)            | 4096 × 192 = 786k                                        |
| Encoder × 3 layers                                | 3 × (4·192² + 2·192·512 + LayerNorms) ≈ 3 × 433k = 1.30M |
| Decoder × 3 layers (self-attn + cross-attn + FFN) | 3 × (8·192² + 2·192·512 + LayerNorms) ≈ 3 × 580k = 1.74M |
| Final LayerNorms + output bias                    | ~5k                                                      |
| **Total**                                         | **≈ 3.8M**                                               |

This is heavier than Model A's ~2.7M. We do **not** force them to be equal; we report both numbers honestly and discuss the implications. The training compute budget is matched (~75 min wall-clock); the parameter count is allowed to differ because forcing them equal would distort one of the architectures.

### 3.10 Ablations (Model B)

1. **Sinusoidal-PE:** replace RoPE with classical additive sinusoidal positional encoding, otherwise identical. Direct test of RoPE's contribution.
2. **No positional info:** drop both RoPE and any positional signal (sanity check — should significantly degrade).
3. **Shared enc/dec layers** (parameter sharing across the 3 encoder layers, separately across the 3 decoder layers — tests whether depth or parameter count matters more).

Ablation 3 is optional, run only if compute allows.

### 3.11 TF/Keras Implementation Notes (Model B)

- Use the Subclassing API.
- Implement `MultiHeadSelfAttention`, `MultiHeadCrossAttention`, and `RoPE` as standalone `Layer` subclasses with unit tests.
- **Unit tests for RoPE:**
  - Identity: `apply_rope(x, cos=ones, sin=zeros) == x` (within fp32 tolerance).
  - Relative-position invariance: for random `q, k` and shift `δ`, verify `<RoPE(q, m), RoPE(k, n)> == <RoPE(q, m+δ), RoPE(k, n+δ)>`.
  - Norm preservation: `||RoPE(x, m)|| == ||x||` for any `m`.
- Precompute `cos`/`sin` tables once at model build time for `L = max(L_enc, L_dec)`. Slice during forward pass.
- **Do not use `tf.keras.layers.MultiHeadAttention`** — its position-encoding hooks are not flexible enough to inject RoPE cleanly. Implement attention from `tf.matmul` and `tf.nn.softmax` directly.
- Causal mask: precompute as a `(L_dec, L_dec)` upper-triangular boolean tensor at build time. Apply via `tf.where(mask, scores, -1e9)`.
- Padding masks for encoder and decoder are derived from the input IDs; pass through the model as auxiliary inputs.
- For mixed precision: keep softmax in `float32` (cast scores up before softmax, cast attention weights back down before the V multiplication). The standard pattern.

---

## 4. Comparison Protocol (Both Models, Side-by-Side)

The point of having two models is the comparison. Everything in this section is binding for both.

### 4.1 Shared Across Both Models

- Same tokenizer (BPE-4k), same vocabulary, same special tokens.
- Same data splits, same seeds (`{13, 42, 91}`).
- Same context-windowing policy and length caps.
- Same loss function (label-smoothed cross-entropy with `ε_ls = 0.1`).
- Same optimizer family (AdamW), same gradient clipping, same weight-decay rules.
- Same wall-clock training budget (~75 min on the target GPU).
- Same evaluation metrics (EM, F1, char-edit-dist, BLEU-1).
- Same Arabic post-normalization in evaluation.
- Same decoding protocol (greedy + beam-4 with `α=0.6`).

### 4.2 Allowed to Differ

- LR schedule (Model A: cosine with warmup; Model B: Noam). Forcing both to share a schedule handicaps one of them; we tune each on dev independently and freeze before final runs.
- Parameter count (Model A ~2.7M, Model B ~3.8M). We document this and discuss it in the report.
- Per-architecture dropout values (already specified above).

### 4.3 Headline Comparison Table (Final Report)

| Metric                            | Model A (3 seeds, mean ± std) | Model B (3 seeds, mean ± std) |
| --------------------------------- | ----------------------------- | ----------------------------- |
| Exact Match (random split)        | —                             | —                             |
| Token-F1 (random split)           | —                             | —                             |
| Char edit distance (random split) | —                             | —                             |
| BLEU-1 (random split)             | —                             | —                             |
| Exact Match (leave-2-videos-out)  | —                             | —                             |
| Token-F1 (leave-2-videos-out)     | —                             | —                             |
| Parameter count                   | ~2.7M                         | ~3.8M                         |
| Training wall-clock               | ~75 min                       | ~75 min                       |
| Inference time / example (greedy) | —                             | —                             |
| Peak GPU memory (training)        | —                             | —                             |

### 4.4 Required Diagnostic Plots

1. **Train/dev loss curves**, all 6 runs (2 models × 3 seeds), one figure.
2. **Token-F1 vs. question→answer token distance**, both models, with confidence bands across seeds.
3. **Token-F1 vs. noise rate**, one figure per noise type (5 noise types × 1 figure).
4. **Token-F1 by difficulty bucket** (easy / medium / hard), bar chart.
5. **Attention/conditioning visualizations** on 3–4 cherry-picked examples (Model A: gated-merge weights and FiLM γ values; Model B: encoder self-attention and decoder cross-attention).

---

## 5. Defensible Architectural Claims (Whiteboard Cheat-Sheet)

Each row is a claim the report makes that an evaluator might probe. The "whiteboard answer" is the 30-second response.

| Claim                                            | Whiteboard Answer                                                                                                                                                                                                                                                                                                   |
| ------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| "Model A uses no attention mechanism."           | The structured pooling in Branch 1 is a learned weighted sum over a single sequence; there are no queries or keys. The gated merge softmax is over branch identity (3 elements), not over positions. The decoder reads the encoder via initial state and FiLM only — both are global, position-independent signals. |
| "FiLM is initialized as identity."               | `γ` is initialized at 1, `β` at 0, by setting the bias of the final FiLM-generator layer to `[1; 0]` and giving it small-weight init. At step 0, the modulation does nothing; the model learns to use it.                                                                                                           |
| "RoPE encodes relative position."                | Show that `<R(mθ)q, R(nθ)k> = <q, R((n-m)θ)k>` — the dot product depends only on `(n-m)`. This is the property that makes RoPE special.                                                                                                                                                                             |
| "Cross-attention uses no positional encoding."   | Decoder positions and encoder positions live in different frames; `(m-n)` has no consistent meaning across frames. The encoder has already encoded position into its output via RoPE in self-attention; cross-attention reads that encoded representation.                                                          |
| "Embeddings are scaled by `√d_model` (Model B)." | Without scaling, embeddings have norm ~1 but attention outputs have norm ~`√d_model`, causing gradient imbalance early in training. Scaling brings them onto the same magnitude.                                                                                                                                    |
| "We tie input and output embeddings."            | (a) Saves ~786k params on Model B / ~524k on Model A. (b) Acts as a regularizer — output projection cannot drift from input semantics. (c) Standard practice since Press & Wolf 2017.                                                                                                                               |
| "Branches in Model A are not redundant."         | Each has a distinct inductive bias: Branch 1 is fine-grained question-only with structured pooling; Branch 2 is a wide, deep recurrent processor of context only; Branch 3 is character-aware and processes question+context jointly. The gated merge learns which branch to trust at each context position.        |

---

## 6. Open Questions / Decisions Deferred to Implementation

These are items where we expect to discover the right value during implementation, not freeze them now:

- Exact placement of dropout in the transformer FFN (post-GELU vs. post-W2). Both are common; pick one and stick with it.
- Whether to use `gelu` or `gelu(approximate=True)` (latter is slightly faster on Tesla GPUs, numerically very close).
- Beam search length penalty α — we've set 0.6 as a starting point, may tune on dev.
- Whether to apply RoPE to all heads or only some — current spec is all heads. A "partial RoPE" variant exists in some literature but is out of scope here.

---

## 7. Document Status

- **Author:** MS2 design lead.
- **Reviewers:** Full team before implementation kickoff.
- **Frozen:** No, awaiting team review.
- **Supersedes:** None.
- **Next document:** `ms2_design_eval_protocol_v001.md` (specifies the noise battery, long-dependency analysis, and scoring scripts in detail).
