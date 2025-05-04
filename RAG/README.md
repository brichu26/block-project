# block-project

## Report

https://docs.google.com/document/d/1p_hvUwoghSF8KtujiapYRGyuVN6qKoVGCVuKmtbmS0s/edit?usp=sharing

## Code: benchmark_rag.py

- Loads a small test corpus from the TIGER-Lab/LongRAG dataset
- Benchmarks three RAG frameworks under a 1028-token limit: LongRAG, MemoRAG, and HippoRAG
- Uses OpenAI GPT-4 for answer generation to ensure a high-quality and consistent LLM backend across all frameworks
- LongRAG: Concatenates top-4 retrieved passages as context
- MemoRAG: Uses sentence-transformers (all-mpnet-base-v2) to embed context and query, selects top-3 most similar context chunks for the query
- HippoRAG: Uses the same embedding model, but applies a simple personalized PageRank-inspired weighting to select top-3 relevant context chunks. Using the personalized PageRank is actually inspired by HippoRAG's use of knowledge graphs and PPR, but it's just simplified in our code for more efficient benchmarking. 
- All responses are evaluated on:
  - Relevance (main topic/key points)
  - Coherence (structure/understandability)
  - Conciseness (compactness)
  - Information Preservation (important details)
  - Overall (average of above)
 
More detailed comments in code. 

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
