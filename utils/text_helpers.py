import nltk
from sentence_transformers import SentenceTransformer
nltk.download('punkt')
from keybert import KeyBERT
import re


kw_model = KeyBERT()

sent_embedder = SentenceTransformer("all-MiniLM-L6-v2")

def split_sentences(text):
    from nltk.tokenize import sent_tokenize
    return sent_tokenize(text)

def get_sentence_embeddings(sentences):
    return sent_embedder.encode(sentences)

def add_transitions(summary):

    # Add transition phrases between topic or paragraph blocks to improve flow.
    import random
    transitions = [
        "Moving on,",
        "Next,",
        "Let’s now explore another point:",
        "Another important point is:",
        "Here’s something else to consider:",
    ]
    parts = summary.strip().split("\n\n")
    if len(parts) <= 1:
        return summary  # No need for transitions

    smoothed = [parts[0]]
    for i in range(1, len(parts)):
        transition = random.choice(transitions)
        smoothed.append(transition)
        smoothed.append(parts[i])
    return "\n\n".join(smoothed)

def extract_keywords(text, top_k=5):
    # Extract top-k keywords from the input text using KeyBERT.
    keywords = kw_model.extract_keywords(text, top_n=top_k)
    return [kw[0] for kw in keywords]


def clean_sentence(s):
    """
    Removes hallucination-triggering patterns like emails, URLs, and social filler.
    """
    s = re.sub(r'\S+@\S+\.\S+', '', s)  # Remove emails
    s = re.sub(r'https?://\S+', '', s)  # Remove URLs
    s = re.sub(r'\b(click here|back to the page|see more|follow me on|read more|view more)\b', '', s, flags=re.IGNORECASE)
    s = re.sub(r'\b\d{4}-\d{2}-\d{2}\b', '', s)  # Remove date-like patterns
    s = re.sub(r'[^\w\s.,!?]', '', s)  # Remove stray symbols
    return s.strip()