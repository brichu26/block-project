import numpy as np
import networkx as nx
from utils.text_helpers import split_sentences, get_sentence_embeddings, add_transitions, extract_keywords
from transformers import pipeline
from utils.io import truncate_text, count_tokens

# Load summarization model (BART)
summarizer = pipeline("summarization", model="facebook/bart-large-cnn", framework="pt", device=-1)

def graph_summarize(text, top_n=5, token_limit=3000):

    # Extract key sentences from the input text, return summary with the top-ranked sentences, keeping total output under token_limit.
    
    text = truncate_text(text, token_limit)

    # Split text into sentences + compute embeddings 
    sentences = split_sentences(text)
    sentences = [s for s in sentences if len(s.split()) > 3]
    if len(sentences) == 0:
        return "No sentences found to summarize."
    
    embeddings = get_sentence_embeddings(sentences)

    # Build similarity matrix using cosine similarity, build graph 
    similarity_matrix = np.zeros((len(sentences), len(sentences)))
    for i in range(len(sentences)):
        for j in range(len(sentences)):
            if i != j:
                sim = np.dot(embeddings[i], embeddings[j]) / (
                    np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j])
                )
                similarity_matrix[i][j] = sim
    graph = nx.from_numpy_array(similarity_matrix)

    # Apply PageRank
    scores = nx.pagerank(graph)

    # Rank sentences by score
    ranked_sentences = sorted(((scores[i], s, i) for i, s in enumerate(sentences)), reverse=True)

    # Select top_n sentences, sorted by original order
    top_sentences = sorted(ranked_sentences[:top_n], key=lambda x: x[2])
    selected_text = " ".join([s[1] for s in top_sentences])

    # Inject keyword preview before summarization
    keywords = extract_keywords(selected_text, top_k=5)
    injected = f"This section discusses: {', '.join(keywords)}. " + selected_text
    max_length = token_limit // top_n
    min_length = max(30, int(max_length * 0.4))

    summary = summarizer(injected, max_length=max_length, min_length=min_length, do_sample=False)[0]['summary_text']

    # (Imporvement) Add transitions between ideas
    return add_transitions(summary)
