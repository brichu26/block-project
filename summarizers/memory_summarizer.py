from transformers import pipeline, AutoTokenizer
import numpy as np
from utils.text_helpers import split_sentences, get_sentence_embeddings, clean_sentence
from utils.io import count_tokens, truncate_text

# Load summarizer (BART)
summarizer = pipeline("summarization", model="facebook/bart-large-cnn", framework="pt", device=-1)
tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")


def extract_memory_keywords(text, num_points=3):
    """
    Extracts key phrases (not full sentences) to serve as memory anchors.
    """
    sentences = split_sentences(text)
    if not sentences:
        return []

    embeddings = get_sentence_embeddings(sentences)
    centroid = np.mean(embeddings, axis=0)

    similarities = [
        (np.dot(embed, centroid) / (np.linalg.norm(embed) * np.linalg.norm(centroid)), sent)
        for embed, sent in zip(embeddings, sentences)
    ]
    top_sentences = sorted(similarities, key=lambda x: x[0], reverse=True)[:num_points]

    keywords = []
    for _, sentence in top_sentences:
        cleaned = clean_sentence(sentence)
        words = cleaned.split()
        phrase = " ".join(words[:5])  # First 5 words of cleaned sentence
        keywords.append(phrase.strip())

    return keywords


def memory_augmented_summary(text, max_tokens=3000, memory_points=3):
    """
    Generates a memory-augmented summary that guides the summarizer using important concepts.
    """
    memory = extract_memory_keywords(text, num_points=memory_points)

    memory_prompt = (
        f"The following summary should cover all the main ideas in the conversation.\n"
        f"Pay particular attention to these themes: {', '.join(memory)}.\n\n"
        f"Conversation:\n{text}"
    )

    # Tokenize and truncate input if needed (BART max = 1024)
    input_ids = tokenizer(memory_prompt, truncation=True, max_length=1024, return_tensors="pt")["input_ids"]
    safe_prompt = tokenizer.decode(input_ids[0], skip_special_tokens=True)

    # Set length bounds
    input_token_count = input_ids.shape[1]
    target_max_length = min(300, input_token_count)
    target_min_length = max(30, int(target_max_length * 0.3))

    summary = summarizer(
        safe_prompt,
        max_length=target_max_length,
        min_length=target_min_length,
        do_sample=False
    )[0]['summary_text']

    return summary
