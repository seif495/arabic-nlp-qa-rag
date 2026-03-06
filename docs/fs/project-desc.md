# NLP Project Description Spring 2026

Deadlines are listed below

## 1 Overview

This project is designed to guide you through the complete lifecycle of an Arabic NLP system, starting from raw data and ending with a Retrieval-Augmented Generation (RAG) system capable of solving a global task.

You will start with a raw Arabic dataset collected via web scraping. This simulates real-world industrial NLP settings where data is noisy, dialectal, inconsistent, and unstructured. You will work in teams of three while specifying the work done by each team member and their contribution.

Link for submitting the teams: Form  
The deadline for submitting the team is Tuesday 3rd of March, 2026 at 11:59 pm. Please refer to Section 3 for all the deadlines.

### General Notes:

- Reference all used resources.
- Maintain an active GitHub repository (evaluation includes commit history and documentation). Kindly make sure your repo is visible by adding me, **mayaarosama**
- Each milestone should have a separate branch on the repo; **Milestone 1, Milestone 2, and Milestone 3**.
- Each milestone requires:

1. Code submission (via GitHub)
2. Technical report, **2 pages .md file** on the milestone branch. The report should contain an explanation and reasoning for the choice of design, add your insights, analyze the output, and discuss the limitations of your framework.
3. One-to-one discussion evaluation

---

# 2 Project Requirements and Milestones

## 2.1 NLP Problems

The objective of this course is for you to take the lead with your choice of architecture and overall design. You are allowed to use different resources and tools to implement your project while keeping in mind the following:

1. You should have a solid understanding of your task and implementation, as you will be asked conceptual and design questions during the evaluation.
2. You have to reference all the resources in your report.
3. During the evaluation, you can modify your code with access to documentation(s), **without the help of any tools, e.g., ChatGPT, Copilot, etc.**
4. You are aware of the content in your report and understand the output of your system and how to change it based on the task.

Your work and submission will be monitored through your GitHub repo commits. The deadlines are announced in advance to give you the flexibility to work at your own pace as long as you meet the given deadlines.

---

## 2.2 Milestone 1: Understanding and Preparing Arabic Text

The dataset¹ consists of **13 Arabic YouTube transcripts from ElDa7ee7 Season 8**. From these transcripts, we generated **3900 extractive question-answer pairs (300 per video)**, the language reflects a natural mixture of:

- Modern Standard Arabic (MSA)
- Egyptian dialect
- Conversational expressions
- Rhetorical and narrative structures
- Arabic–English code-switching
- Named entities (people, places, organizations)
- Historical and scientific references

Although the QA pairs are structured, the original transcripts are naturally occurring web-scraped Arabic text. As a result, the dataset exhibits realistic linguistic challenges, including:

- Dialectal variation
- Informal punctuation and inconsistent sentence segmentation
- Orthographic inconsistencies
- Variation in named entity spelling (especially transliterated foreign names)
- Long narrative passages with embedded dialogue
- Topic shifts within the same document

The dataset is organized into **two main folders**:

### 1) Transcript/

- Contains **13 raw Arabic text files (.txt)**
- Each file represents the **full, unprocessed transcript of one video**
- Transcripts preserve the original spoken format, including **timestamp markers, dialectal expressions (MSA + Egyptian), code-switching, and conversational structure**

These files represent the **raw input data**.

### 2) QA/

- Contains **13 UTF-8 encoded CSV files (one per video)**
- Each file includes **300 extractive QA pairs** derived strictly from its corresponding transcript

The CSV schema is:

```csv
video_id, video_title, question_id, question, answer, difficulty
```

Each row represents **one QA pair**. The answer is always a span taken directly from the transcript without introducing external knowledge.

This parallel structure ensures full traceability between the raw transcript and the supervised QA data. It supports preprocessing and analysis in **MS1**, sequence-to-sequence modeling in **MS2**, and retrieval-based evaluation (**RAG**) in **MS3**.

In **MS1**, your task is to understand and prepare the data before modeling. You will be evaluated on your ability to:

- Analyze textual distributions
- Identify hidden patterns in the data
- Detect noise and linguistic irregularities
- Normalize Arabic text
- Correct spelling inconsistencies, if needed
- Prepare the data for neural architectures

Milestone 1 focuses on building a deep understanding of the dataset’s linguistic structure and challenges before moving to neural modeling in later milestones.

¹ The data is available on the CMS

---

## 2.3 Milestone 2: RNN vs Transformer

Build and compare **two neural architectures from scratch (no pre-trained models)**:

- **Model A:** an **RNN-based model**
- **Model B:** a **simple Transformer model**

You must evaluate, compare, and interpret findings.

### Model A

Your design choice of an **RNN-variant network**:

- RNN
- LSTM
- GRU
- Bi-LSTM
- etc.

Your network must include at least:

- an **embedding layer (trained from scratch)**
- **two recurrent layers**
- **an output layer (task appropriate)**

### Model B

A **simple transformer architecture**, **NO pretrained weights allowed**.

The model should include at least:

- **token embeddings**
- **positional encoding**
- **at least one attention layer (encoder, decoder)**
- **an output layer (task appropriate)**

You must compare the two models with:

- performance metrics
- convergence speed
- parameter count
- training stability
- sensitivity to noisy input
- generalization ability

Your report must include:

- Why does each model struggle or succeed?
- if attention improves representation
- Long dependency handling comparison
- Overfitting behavior
- Computational trade-offs

You are evaluated on **understanding, not raw performance**.

---

## 2.4 Milestone 3: Prompt Engineering and RAG System

You are required to design and implement a **Retrieval-Augmented Generation (RAG) system** that utilizes the cleaned Arabic dataset into an intelligent end-to-end solution.

In addition to the **vanilla RAG framework**, you are required to implement a **semantic caching mechanism** whereby previously answered questions are stored alongside their embeddings. If a new query exceeds a defined similarity threshold relative to cached entries, the stored response should be returned instead of invoking the generation model.

The system must measure and report:

- cache hit rate
- saved model calls
- threshold sensitivity

Moreover, you must conduct **structured prompt engineering experiments**.

This includes:

- constructing and evaluating multiple system prompts
- comparing system-guided responses against responses generated without system instructions
- analyzing differences in:
  - quality
  - controllability
  - hallucination behavior
  - token efficiency

The prompt design must reflect awareness of **Arabic linguistic characteristics and dialectal variability where relevant**.

You should also experiment with **context window strategies**, comparing:

- full-history context
- sliding window approaches
- strict truncation
- summarized-history

These experiments should evaluate trade-offs between:

- token consumption
- latency
- coherence
- response accuracy

---

## 2.5 Weight Distribution

Each **Milestone is worth 8% of your course grade**, and would be graded with an evaluation with the team, which is conducted in a discussion manner.

The exact time and date for the evaluations will be announced later during the semester based on your quizzes and other evaluation schedules.

**In-class tasks and participation are worth 6%.**

---

# 3 Timeline

Kindly note that the submission should be done on your **GitHub submitted repo**; the **last commit before the deadline** is the one that your milestone will be evaluated on.

- Deadline for team submission is **Tuesday 3rd of March, 2026 at 11:59 pm**
- Teams announcement **Sunday 8th of March, 2026**
- **Milestone 1 deadline:** 14th of March, 2026 at 11:59pm
- **Milestone 2 deadline:** 22nd of April, 2026 at 11:59pm
- **Milestone 3 deadline:** 16th of May, 2026 at 11:59pm
