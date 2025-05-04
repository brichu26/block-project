# block-project

Report: https://docs.google.com/document/d/1p_hvUwoghSF8KtujiapYRGyuVN6qKoVGCVuKmtbmS0s/edit?usp=sharing

## Setup

1. Clone the repository and navigate to the project directory.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Ensure you have an OpenAI API key and set it in your environment (the script sets it automatically for demo purposes).

## Code: benchmark_rag.py

- Loads a small test corpus from the TIGER-Lab/LongRAG dataset.
- Benchmarks three RAG frameworks under a 1028-token limit: LongRAG, MemoRAG, and HippoRAG.
- Uses OpenAI GPT-4 for answer generation to ensure a consistent LLM backend.
- **LongRAG**: Concatenates top-4 retrieved passages as context.
- **MemoRAG**: Uses sentence-transformers (all-mpnet-base-v2) to embed context and query, selects top-3 most similar context chunks for the query.
- **HippoRAG**: Uses the same embedding model, but applies a simple personalized PageRank-inspired weighting to select top-3 relevant context chunks.
- All responses are evaluated on:
  - Relevance (main topic/key points)
  - Coherence (structure/understandability)
  - Conciseness (compactness)
  - Information Preservation (important details)
  - Overall (average of above)
- Results are printed and (for MemoRAG and HippoRAG) saved as images for easy comparison.

### Why these models/techniques?
- **all-mpnet-base-v2**: Chosen for its strong performance in semantic similarity and retrieval tasks, and wide support in the open-source community.
- **OpenAI GPT-4**: Used for answer generation to ensure a high-quality, consistent LLM backend across all frameworks.
- **Token limit**: 1028 tokens to match the LongRAG evaluation and ensure fair comparison.
- **Personalized PageRank (HippoRAG)**: Inspired by HippoRAG's use of knowledge graphs and PPR for relevance, approximated here for simplicity.

## Results

### LongRAG
```
Relevance: 0.0
Coherence: 1.0
Conciseness: 1.0
Information_preservation: 0.5
Overall: 0.62
Time taken: 3.83 seconds
```

### MemoRAG
```
Relevance:0.0
Coherence:1.0
Conciseness:0.7
Information preservation:0.5
Overall:0.55
Time taken:2.83 seconds
```

### HippoRAG
```
Relevance:0.0
Coherence:1.0
Conciseness:0.7
Information preservation:0.5
Overall:0.55
Time taken:2.83 seconds
```