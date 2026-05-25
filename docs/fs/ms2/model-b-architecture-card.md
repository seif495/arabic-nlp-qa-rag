# Model B Architecture Card

Model B implements ADR §3: a small encoder-decoder Transformer trained from scratch with RoPE.

Layer summary:

- Shared BPE embedding: `4096 x 192`, scaled by `sqrt(192)`, shared between encoder, decoder, and tied output projection.
- Encoder: 3 pre-LN layers; each layer has RoPE self-attention with 4 heads of 48 dims and FFN `192 -> 552 -> 192`; final LayerNorm.
- Decoder: 3 pre-LN layers; masked self-attention uses RoPE on Q/K, cross-attention uses no RoPE, FFN `192 -> 552 -> 192`; final LayerNorm.
- Output projection: `decoder_output @ E^T + bias`, no extra projection because `d_model = d_tok = 192`.
- Inference: `KVCache` stores per-layer self-attention K/V and cross-attention K/V for step decoding.

Parameter budget: ADR §3.9 target is approximately 3.8M, accepted range `[3.4M, 4.2M]`. The concrete Keras implementation with FFN 512 audited at 3.315M, below the ticket's hard range. The FFN is therefore set to 552 as the smallest documented deviation that brings the implemented model into the accepted audit band without changing attention, RoPE, tying, depth, heads, or hidden size.
