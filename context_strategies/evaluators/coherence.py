from sentence_transformers import SentenceTransformer, util
import nltk
import numpy as np

nltk.download("punkt")

model = SentenceTransformer("all-MiniLM-L6-v2")

def coherence_score(summary_text: str) -> float:
    """
    Measures the average similarity between adjacent sentences.
    
    Args:
        summary_text: the summary to evaluate

    Returns:
        average cosine similarity between adjacent sentence embeddings
    """
    from nltk.tokenize import sent_tokenize
    sents = sent_tokenize(summary_text)

    if len(sents) < 2:
        return 0.0

    embeddings = model.encode(sents, convert_to_tensor=True)
    sims = []

    for i in range(len(sents) - 1):
        sim = util.pytorch_cos_sim(embeddings[i], embeddings[i + 1]).item()
        sims.append(sim)

    return round(float(np.mean(sims)), 2)
