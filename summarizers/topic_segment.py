# summarizers/topic_segment.py

from utils.text_helpers import split_sentences, get_sentence_embeddings
from transformers import pipeline
from sklearn.cluster import KMeans
import numpy as np

# Load summarization model (BART) with PyTorch backend
summarizer = pipeline("summarization", model="facebook/bart-large-cnn", framework="pt")

def summarize_chunk(chunk, max_tokens=150):
    """Generate a summary for a given list of sentences."""
    joined = " ".join(chunk)
    return summarizer(joined, max_length=max_tokens, min_length=30, do_sample=False)[0]['summary_text']

def segment_and_summarize(text):
    """
    Segments the conversation into semantically grouped topics and summarizes each group.
    Returns a structured multi-topic summary.
    """
    # 1. Split conversation into individual sentences
    sentences = split_sentences(text)

    # 2. Filter out low-information or very short sentences
    sentences = [s for s in sentences if len(s.split()) > 3]

    if not sentences:
        return "No meaningful sentences to summarize."

    # 3. Convert sentences into semantic embeddings
    embeddings = get_sentence_embeddings(sentences)

    # 4. Dynamically decide number of clusters (topics)
    num_sentences = len(sentences)
    dynamic_k = min(5, max(1, num_sentences // 5))

    # 5. Cluster similar sentences using KMeans
    kmeans = KMeans(n_clusters=min(dynamic_k, num_sentences), random_state=42)
    labels = kmeans.fit_predict(embeddings)

    # 6. Group sentences by their assigned topic cluster
    clustered = {}
    for label, sentence in zip(labels, sentences):
        clustered.setdefault(label, []).append(sentence)

    # 7. Generate a summary for each topic cluster
    summaries = []
    for topic_id in sorted(clustered.keys()):
        chunk = clustered[topic_id]
        summary = summarize_chunk(chunk)
        summaries.append(f"Topic {topic_id + 1}: {summary}")

    return "\n\n".join(summaries)
