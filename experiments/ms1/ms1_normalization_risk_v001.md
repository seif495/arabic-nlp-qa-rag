# MS1 Normalization Risk Analysis

## High-Confidence (Low-Risk) Rules

| Rule              | Risk Level | Justification                                                                            |
| ----------------- | ---------- | ---------------------------------------------------------------------------------------- |
| Tatweel removal   | Low        | Pure cosmetic; no lexical or semantic role                                               |
| Diacritic removal | Low        | Consistent with unvocalised Arabic; models trained on raw web text do not expect harakat |

## Moderate-Risk Rules

| Rule               | Risk     | Mitigation                                                                                                                                                        |
| ------------------ | -------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Alef normalisation | Moderate | Rare loan words or proper nouns may rely on hamza placement for disambiguation. Effect is negligible in the predominantly colloquial ElDa7ee7 corpus.             |
| Alef maqsura → ya  | Moderate | In classical Arabic, ى at word-end is semantically distinct from ي. In Egyptian colloquial Arabic (this corpus), the distinction is rarely maintained in writing. |

## Extractive QA Risk

The dataset uses **extractive** QA: the answer is a direct span from the
transcript. If normalization is applied to **both** the transcript context
and the QA answers, alignment is preserved. The export pipeline (INFRA-04)
must apply identical normalization to both fields.

## Recommendation

Apply all four rules uniformly to transcripts and QA text. Keep the raw
`data/external/` files unmodified so that the original alignment can be
recovered if needed.
