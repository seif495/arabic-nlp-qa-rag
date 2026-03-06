# 8. Milestone 3 Functional Specification

## 6.1 Objective

Design and implement a Retrieval-Augmented Generation (RAG) system over the cleaned Arabic dataset as an end-to-end solution. In addition to vanilla RAG, include a semantic caching mechanism that reuses prior responses when a new query is sufficiently similar to cached queries.

## 6.2 Inputs / Outputs

**Inputs**

- Cleaned Arabic dataset from earlier milestones.
- User queries.
- Cached question/response entries with embeddings.
- Configured similarity threshold for cache lookup.

**Outputs**

- RAG-generated responses.
- Cache-served responses for high-similarity queries.
- Reported cache metrics and prompt/context experiment results.

## 6.3 Modules

- Retrieval module for relevant context selection.
- Generation module for final response construction.
- Semantic cache module (store prior Q/A with embeddings and perform threshold-based matching).
- Prompt experimentation module (system prompt variants and no-system baseline).
- Context window strategy module (full history, sliding window, strict truncation, summarized history).
- Evaluation and analysis module.

## 6.4 Functional Requirements

**The milestone implementation must:**

- Implement vanilla RAG on the cleaned Arabic dataset.
- Implement semantic caching where previously answered questions and embeddings are stored.
- Return cached responses when query similarity exceeds a defined threshold, instead of invoking generation.
- Measure and report:
  - cache hit rate
  - saved model calls
  - threshold sensitivity
- Conduct structured prompt engineering experiments by:
  - constructing and evaluating multiple system prompts
  - comparing system-guided responses against responses without system instructions
  - analyzing quality, controllability, hallucination behavior, and token efficiency
- Reflect Arabic linguistic characteristics and dialectal variability in prompt design where relevant.
- Experiment with context window strategies:
  - full-history context
  - sliding window
  - strict truncation
  - summarized-history
- Evaluate context strategy trade-offs in token consumption, latency, coherence, and response accuracy.

## 6.5 Evaluation Artifacts

- Milestone 3 branch on GitHub with code submission.
- Technical report (`.md`, 2 pages) including design reasoning, insights, output analysis, and limitations.
- One-to-one discussion evaluation.
- Evaluation snapshot taken from the last commit before the Milestone 3 deadline (16 May 2026, 11:59 pm).
