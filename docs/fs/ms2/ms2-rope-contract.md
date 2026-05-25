# MS2 RoPE Contract

RoPE follows ADR §3.4 exactly.

- `d_head = 48`, split into 24 consecutive 2D pairs.
- Frequencies are `theta_i = 10000 ** (-2(i-1)/48)` for `i in 1..24`.
- For position `m`, each pair is rotated by `m * theta_i`.
- Tables for `cos` and `sin` are precomputed at layer initialization and sliced during calls.

Application table:

| Site | Apply RoPE? |
| --- | --- |
| Encoder self-attention Q | yes |
| Encoder self-attention K | yes |
| Encoder self-attention V | no |
| Decoder masked self-attention Q | yes |
| Decoder masked self-attention K | yes |
| Decoder masked self-attention V | no |
| Decoder cross-attention Q/K/V | no |
