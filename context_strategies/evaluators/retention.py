from sentence_transformers import SentenceTransformer, util
import numpy as np

model = SentenceTransformer("all-MiniLM-L6-v2")

def retention_score(summary_text: str, questions: list[str], threshold: float = 0.6) -> float:
    """
    Returns a score from 0.0 to 1.0 based on how many QA-style questions
    are semantically answerable from the summary.

    Args:
        summary_text: the generated summary
        questions: list of questions (from QA summary)
        threshold: cosine similarity threshold to count a match

    Returns:
        fraction of questions retained
    """
    summary_embedding = model.encode(summary_text, convert_to_tensor=True)
    count = 0

    for question in questions:
        question_embedding = model.encode(question, convert_to_tensor=True)
        sim = util.pytorch_cos_sim(summary_embedding, question_embedding).item()
        if sim >= threshold:
            count += 1

    return round(count / len(questions), 2) if questions else 0.0
