from utils.text_helpers import split_sentences, get_sentence_embeddings, add_transitions, extract_keywords
from transformers import pipeline
from sklearn.cluster import AgglomerativeClustering
import numpy as np

# Summarization model = (BART)
summarizer = pipeline("summarization", model="facebook/bart-large-cnn", framework="pt", device=-1)

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def summarize_chunk(chunk_sentences, chunk_embeddings, max_tokens=150):
    # Summarize list of important sentences, injecting keywords
    if not chunk_sentences:
        return "No content."

    # Rank sentences by similarity to cluster centroid
    centroid = np.mean(chunk_embeddings, axis=0)
    ranked = sorted(
        zip(chunk_sentences, chunk_embeddings),
        key=lambda x: cosine_similarity(x[1], centroid),
        reverse=True
    )
    top_sentences = [s for s, _ in ranked][:min(10, len(ranked))]  # Keep only top 10 or fewer

    # Join important sentences
    joined_text = " ".join(top_sentences)

    # Smarter keyword extraction
    keywords = extract_keywords(joined_text, top_k=min(3, len(top_sentences)//2))
    keyword_prefix = f"This section discusses: {', '.join(keywords)}. " if keywords else ""

    # Inject keywords and summarize
    prompt = keyword_prefix + joined_text
    min_len = max(30, int(max_tokens * 0.4))
    return summarizer(prompt, max_length=max_tokens, min_length=min_len, do_sample=False)[0]['summary_text']

def segment_and_summarize(text, output_token_limit=3000):
    # Segments text into topic groups and summarizes each group separately
    sentences = split_sentences(text)
    sentences = [s for s in sentences if len(s.split()) > 3]  # Filter short sentences
    if not sentences:
        return "No meaningful sentences to summarize."

    embeddings = get_sentence_embeddings(sentences)

    num_sentences = len(sentences)
    dynamic_k = min(5, max(1, num_sentences // 5))  # 1 cluster per 5 sentences roughly

    # Agglomerative clustering for more natural groupings
    clustering = AgglomerativeClustering(n_clusters=min(dynamic_k, num_sentences))
    labels = clustering.fit_predict(embeddings)

    clustered = {}
    clustered_embeddings = {}
    for idx, label in enumerate(labels):
        clustered.setdefault(label, []).append(sentences[idx])
        clustered_embeddings.setdefault(label, []).append(embeddings[idx])

    num_clusters = len(clustered)
    token_budget_per_chunk = max(100, output_token_limit // num_clusters)

    summaries = []
    for topic_id in sorted(clustered.keys()):
        chunk_sentences = clustered[topic_id]
        chunk_embeddings = clustered_embeddings[topic_id]
        summary = summarize_chunk(chunk_sentences, chunk_embeddings, max_tokens=token_budget_per_chunk)
        summaries.append(f"Topic {topic_id + 1}: {summary}")

    return add_transitions("\n\n".join(summaries))
