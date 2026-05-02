# MS2 Length Analysis Contract

The frozen MS2 length caps are produced by `MS2-DATA-01` and consumed by `RunConfig`.

| Axis | p95 | p99 | max | chosen cap | coverage |
| --- | ---: | ---: | ---: | ---: | ---: |
| question | 0 | 0 | 0 | 32 | 1.000 |
| context | 0 | 0 | 0 | 384 | 1.000 |
| encoder | 0 | 0 | 0 | 420 | 1.000 |
| decoder | 0 | 0 | 0 | 64 | 1.000 |

Artifacts:

- `experiments/ms2/ms2_length_distribution_v001.json`
- `docs/fs/artifacts/ms2/length_proxy_*.png`
- `docs/fs/artifacts/ms2/length_post_bpe_*.png`
