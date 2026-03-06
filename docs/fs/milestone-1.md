# 6. Milestone 1 Functional Specification

## 6.1 Objective

Milestone 1 focuses on understanding and preparing the Arabic dataset before neural modeling. The objective is to build a deep understanding of linguistic structure and dataset challenges, then produce a cleaned and prepared version suitable for later architectures.

## 6.2 Inputs / Outputs

**Inputs**

- `Transcript/`: 13 raw Arabic transcript `.txt` files from ElDa7ee7 Season 8.
- `QA/`: 13 UTF-8 CSV files (300 extractive QA pairs per video), aligned with transcripts.

**Outputs**

- Dataset-level analysis artifacts describing textual distributions and hidden patterns.
- Identified noise and linguistic irregularities.
- Normalized Arabic text and corrected spelling inconsistencies where needed.
- Prepared data ready for neural architectures in subsequent milestones.

## 6.3 Modules

- Data profiling and distribution analysis module.
- Pattern discovery and irregularity detection module.
- Arabic normalization and optional spelling-correction module.
- Data preparation/export module for downstream modeling.

## 6.4 Functional Requirements

The milestone implementation must support the following capabilities:

- Analyze textual distributions.
- Identify hidden patterns in the data.
- Detect noise and linguistic irregularities.
- Normalize Arabic text.
- Correct spelling inconsistencies, if needed.
- Prepare the data for neural architectures.

## 6.5 Evaluation Artifacts

- Milestone 1 branch on GitHub with code submission.
- Technical report (`.md`, 2 pages) including design reasoning, insights, output analysis, and limitations.
- One-to-one discussion evaluation.
- Evaluation snapshot taken from the last commit before the Milestone 1 deadline (14 March 2026, 11:59 pm).
