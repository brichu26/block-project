# Context-Aware Summarization Toolkit

This project implements and evaluates four advanced summarization techniques for long conversations, designed to address **retention loss** under limited context windows like those imposed by models such as `facebook/bart-large-cnn`.

---

## Overview

This project implemented and compared four distinct **context management strategies**:

1. **Topic Segmentation Summarization**
2. **Graph-Based Summarization**
3. **Memory-Augmented Summarization**
4. **QA-Based Summarization** (discarded due to poor performance)

Each method was evaluated using an **LLM-based evaluation framework** that scores summaries on Relevance, Coherence, Conciseness, and Information Preservation.

---

## Strategies Implemented

### 1. Topic Segmentation Summarization
- **How it works:** Embeds sentences → clusters them via Agglomerative Clustering → summarizes each cluster individually.
- **Enhancements:** 
  - Dynamic cluster sizing
  - Keyword injection per cluster
  - Transition sentence generation
  - Short-sentence filtering
- **Strength:** Strong coherence within topics

---

### 2. Graph-Based Summarization
- **How it works:** Builds a sentence similarity graph using cosine similarity → applies PageRank → selects top N ranked sentences.
- **Enhancements:**
  - Token-safe truncation
  - Filtering short (<3 word) sentences
  - Post-hoc transition smoothing
- **Strength:** Globally important sentences in a concise form

---

### 3. Memory-Augmented Summarization
- **How it works:** Extracts top 3 central sentences → compresses into keyphrases → prepends them to input text before summarization.
- **Enhancements:**
  - Memory phrase compression
  - Dynamic token budgeting
  - Strict enforcement of 1024-token safety for BART
- **Strength:** Best retention of critical facts and context

---

### 4. QA-Based Summarization *(Not Recommended)*
- **How it worked:** Used T5 model to generate question–answer pairs from the full conversation.
- **Problems:** Generated generic or repetitive questions, missed key content, and fragmented coherence.

---

## Evaluation Method

Evaluations are conducted using GPT-4o (`gpt-4o-mini`) with a structured scoring rubric. It scores each summary on:
- Relevance
- Coherence
- Conciseness
- Information Preservation

Each strategy is scored out of 10 and explained via natural language rationale.

---

## Setup

```bash
# Clone the repository
git clone https://github.com/yourname/context-summarizer.git
cd context-summarizer

# Create a virtual environment
python3 -m venv venv
source venv/bin/activate

# Install requirements
pip install -r requirements.txt
