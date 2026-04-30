# MS2 Length Analysis Contract

`MS2-DATA-01` owns the frozen length caps consumed by `src.ms2.schemas.RunConfig`.

The executable entrypoint is:

```bash
uv run python -m src.cli.ms2 analyze-lengths
```

The command reads `data/processed/ms1/ms1_dataset_processed_v001.jsonl`, computes both whitespace-proxy and post-tokenizer distributions, writes `experiments/ms2/ms2_length_distribution_v001.json`, and refreshes this contract table with p95, p99, max, chosen cap, and coverage.

The frozen defaults are:

| Field | Value |
| --- | ---: |
| `L_q` | 32 |
| `L_c` | 384 |
| `L_enc` | 420 |
| `L_dec` | 64 |

Histogram snapshots are emitted under `docs/fs/artifacts/ms2/length_<pass>_<axis>.png` for the proxy and post-tokenizer passes.
