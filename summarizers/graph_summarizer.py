import numpy as np
import networkx as nx
from utils.text_helpers import split_sentences, get_sentence_embeddings

def graph_summarize(text, top_n=5):
    """
    Uses graph-based TextRank-like algorithm to extract key sentences from the input text.
    Returns a summary with the top-ranked sentences.
    """
    # 1. Split text into sentences
    sentences = split_sentences(text)
    sentences = [s for s in sentences if len(s.split()) > 3]
    if len(sentences) == 0:
        return "No sentences found to summarize."

    # 2. Compute embeddings
    embeddings = get_sentence_embeddings(sentences)

    # 3. Build similarity matrix using cosine similarity
    similarity_matrix = np.zeros((len(sentences), len(sentences)))
    for i in range(len(sentences)):
        for j in range(len(sentences)):
            if i != j:
                sim = np.dot(embeddings[i], embeddings[j]) / (
                    np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j])
                )
                similarity_matrix[i][j] = sim

    # 4. Build similarity graph
    graph = nx.from_numpy_array(similarity_matrix)

    # 5. Apply PageRank
    scores = nx.pagerank(graph)

    # 6. Rank sentences by score
    ranked_sentences = sorted(((scores[i], s, i) for i, s in enumerate(sentences)), reverse=True)

    # 7. Select top_n sentences, sorted by original order
    top_sentences = sorted(ranked_sentences[:top_n], key=lambda x: x[2])
    summary = " ".join([s[1] for s in top_sentences])

    return summary