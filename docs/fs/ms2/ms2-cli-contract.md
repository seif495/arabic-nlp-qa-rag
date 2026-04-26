# MS2 CLI Contract

Source ticket: `MS2-INFRA-03`.

All Milestone 2 commands are reachable via:

```bash
python -m src.cli.ms2 <command>
```

The installed console script is registered as:

```toml
ms2 = "src.cli.ms2:main"
```

## Output Contract

Every successful command emits one line matching the MS1 smoke-test pattern:

```text
[<command>] status=ok output=<path>
```

Errors emit the same command prefix with `status=error` on stderr and return a non-zero exit code.

## Commands

| Command | Purpose | User-facing flags |
| --- | --- | --- |
| `analyze-lengths` | Analyze MS1 output lengths and freeze MS2 caps. | none |
| `prep-data` | Train tokenizer, build char vocab, and cache TFRecords. | `--target-model {a,b}` |
| `train` | Train one model on one seed from a RunConfig JSON file. | `--model {a,b}`, `--seed {13,42,91}`, `--config <path>` |
| `infer` | Run greedy or beam decoding for a trained run. | `--model {a,b}`, `--seed {13,42,91}`, `--split {dev,test}`, `--decoding {greedy,beam}` |
| `evaluate` | Compute EM, F1, char edit distance, and BLEU-1. | `--model {a,b}`, `--seed {13,42,91}`, `--split {dev,test}` |
| `evaluate-protocol` | Run the full MS2 evaluation protocol battery. | none |
| `ablate` | Run an ablation variant. | `--variant {...}`, `--seed {13,42,91}` |
| `compare` | Build the MS2 comparison table and diagnostic plots. | none |
| `run-all` | Run all available MS2 orchestration commands in dependency order. | none |

## Run-All Order

`run-all` executes commands in this order:

1. `analyze-lengths`
2. `prep-data`
3. `train`
4. `infer`
5. `evaluate`
6. `evaluate-protocol`
7. `ablate`
8. `compare`

## Snapshots

- Help output: `docs/fs/artifacts/ms2/ms2-cli-help-output.txt`
- Example execution log: `docs/fs/artifacts/ms2/ms2-cli-example-execution-log.txt`
