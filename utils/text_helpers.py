import nltk
from sentence_transformers import SentenceTransformer
nltk.download('punkt')

sent_embedder = SentenceTransformer("all-MiniLM-L6-v2")

def split_sentences(text):
    from nltk.tokenize import sent_tokenize
    return sent_tokenize(text)

def get_sentence_embeddings(sentences):
    return sent_embedder.encode(sentences)
