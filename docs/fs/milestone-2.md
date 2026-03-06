# 7. Milestone 2 Functional Specification

## 6.1 Objective

Build and compare two neural architectures from scratch (without pretrained models):

- Model A: an RNN-based model (variant chosen by the team).
- Model B: a simple Transformer model.

The objective is to evaluate, compare, and interpret differences in model behavior, with emphasis on understanding rather than raw performance.

## 6.2 Inputs / Outputs

**Inputs**

- Prepared Arabic dataset from Milestone 1.
- Milestone 2 modeling and training configuration.

**Outputs**

- Trained Model A (RNN-variant) and Model B (simple Transformer), both trained from scratch.
- Comparative evaluation results covering required metrics and behavior dimensions.
- Interpretation of model strengths, weaknesses, and trade-offs.

## 6.3 Modules

- Model A module (RNN-variant selected from options such as RNN/LSTM/GRU/Bi-LSTM).
- Model B module (simple Transformer, no pretrained weights).
- Training and experiment execution module.
- Comparative analysis module for metrics, stability, and generalization interpretation.

## 6.4 Functional Requirements

**Model A must include at least:**

- Embedding layer trained from scratch.
- Two recurrent layers.
- Task-appropriate output layer.

**Model B must include at least:**

- Token embeddings.
- Positional encoding.
- At least one attention layer (encoder or decoder).
- Task-appropriate output layer.

**The milestone must compare the two models on:**

- Performance metrics.
- Convergence speed.
- Parameter count.
- Training stability.
- Sensitivity to noisy input.
- Generalization ability.

**The analysis/report must address:**

- Why each model struggles or succeeds.
- Whether attention improves representation.
- Long dependency handling comparison.
- Overfitting behavior.
- Computational trade-offs.

## 6.5 Evaluation Artifacts

- Milestone 2 branch on GitHub with code submission.
- Technical report (`.md`, 2 pages) including design reasoning, insights, output analysis, and limitations.
- One-to-one discussion evaluation.
- Evaluation snapshot taken from the last commit before the Milestone 2 deadline (22 April 2026, 11:59 pm).
