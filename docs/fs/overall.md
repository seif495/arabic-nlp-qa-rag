# 1. Project Overview

This project covers the full lifecycle of an Arabic NLP system, starting from raw web-scraped text and ending with a Retrieval-Augmented Generation (RAG) system that solves a global task. The work is team-based (three members), with explicit contribution mapping and tracked progress through GitHub commits and documentation.

## 2. Global Objective

The global objective is to design, implement, and justify an end-to-end Arabic NLP pipeline while demonstrating clear understanding of design choices, system behavior, and trade-offs. Evaluation emphasizes conceptual understanding, ability to reason about outputs, and ability to modify the codebase during discussion-based assessment (with documentation access, without external AI tools).

## 3. Dataset and Inputs

The project data consists of Arabic content from ElDa7ee7 Season 8 and is organized into two aligned folders:

- `Transcript/`: 13 raw Arabic `.txt` files (one per video), preserving spoken format, timestamp markers, dialectal expressions (MSA + Egyptian), code-switching, and conversational structure.
- `QA/`: 13 UTF-8 CSV files (one per video), each containing 300 extractive QA pairs derived strictly from the corresponding transcript.

CSV schema:

```csv
video_id, video_title, question_id, question, answer, difficulty
```

Language and data characteristics include MSA, Egyptian dialect, conversational style, rhetorical/narrative structure, Arabic-English code-switching, named entities, and historical/scientific references. Realistic challenges include dialectal variation, orthographic inconsistency, informal punctuation, segmentation inconsistency, variable named-entity spelling, long narrative passages, and topic shifts.

## 4. Repository and Branching Strategy

- Maintain an active GitHub repository; evaluation includes commit history and documentation.
- Use separate branches for `Milestone 1`, `Milestone 2`, and `Milestone 3`.
- For each milestone, submit:
  1. Code on GitHub.
  2. A technical report (`.md`, 2 pages) on the milestone branch explaining design choices and reasoning, insights, output analysis, and framework limitations.
  3. One-to-one discussion evaluation.
- Submission evaluation is based on the last commit before each deadline.
- Team deadline and milestone deadlines follow the provided timeline in the project description.

## 5. End-to-End Architecture

The system evolves through three connected milestones:

1. Milestone 1: Understand and prepare the Arabic dataset through analysis, noise/irregularity detection, normalization, and data preparation.
2. Milestone 2: Build and compare two models from scratch (RNN-variant vs simple Transformer) to analyze behavior, stability, and trade-offs.
3. Milestone 3: Build a RAG system over the cleaned data, add semantic caching, and run structured prompt/context experiments.

This progression moves from raw data understanding, to neural modeling, to retrieval-augmented generation and system-level experimentation.

## 6. Execution Model

All pipeline functionality must be accessible through CLI entrypoints.

No milestone logic should rely on ad hoc scripts or notebooks as the primary execution interface.

Canonical commands:

python -m src.cli.ms1 ...
python -m src.cli.ms2 ...
python -m src.cli.ms3 ...

This ensures:

- reproducibility
- consistent usage
- simplified debugging
- standardized experiment execution
