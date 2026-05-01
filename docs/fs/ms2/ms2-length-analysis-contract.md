# MS2 Length Analysis Contract

The frozen MS2 length caps are produced by `MS2-DATA-01` and consumed by `RunConfig`.

| Axis | p95 | p99 | max | chosen cap | coverage |
| --- | ---: | ---: | ---: | ---: | ---: |
| question | 16 | 17 | 17 | 17 | 1.000 |
| context | 16048 | 16048 | 16048 | 16048 | 1.000 |
| encoder | 16062 | 16066 | 16066 | 420 | 0.000 |
| decoder | 16 | 18 | 23 | 18 | 0.995 |

Artifacts:

- `experiments/ms2/ms2_length_distribution_v001.json`
- `docs/fs/artifacts/ms2/length_proxy_*.png`
- `docs/fs/artifacts/ms2/length_post_bpe_*.png`
