# MS2 Headline Comparison

Seeds averaged: 13, 42, 91 (only completed runs included).

| Metric                           | Model A   | Model B   |
| -------------------------------- | --------- | --------- |
| Exact Match — test               | 0.000     | 0.000     |
| Token-F1 — test                  | 0.006     | 0.002     |
| Char edit distance — test        | 0.939     | 0.953     |
| BLEU-1 — test                    | 0.004     | 0.001     |
| Exact Match — dev                | 0.000     | 0.000     |
| Token-F1 — dev                   | 0.000     | 0.000     |
| Exact Match (leave-2-videos-out) | —         | —         |
| Token-F1 (leave-2-videos-out)    | —         | —         |
| Parameter count                  | 2,725,325 | 3,030,432 |
| Training wall-clock (min)        | 10.9      | 7.0       |
| Inference time / example (ms)    | 340.0     | 340.0     |
| Peak GPU memory (MB)             | 0         | 0         |

Cells marked `—` have no completed runs for either model.
