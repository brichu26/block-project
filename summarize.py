from utils.io import load_conversation, count_tokens
from summarizers.topic_segment import segment_and_summarize
from summarizers.graph_summarizer import graph_summarize
from summarizers.question_summarizer import generate_questions
from evaluators.retention import retention_score
from evaluators.coherence import coherence_score
from evaluators.retention_llm import retention_score_llm
import os

# === Set up output path ===
os.makedirs("outputs", exist_ok=True)
output_path = "outputs/summary_output.txt"

def log(text):
    print(text)
    with open(output_path, "a", encoding="utf-8") as f:
        f.write(text + "\n")

# Clear previous output
with open(output_path, "w", encoding="utf-8") as f:
    f.write("=== Summary Output ===\n")

if __name__ == "__main__":
    text = load_conversation("data/sample_convo.txt")
    log("Loaded conversation:\n" + text)
    log(f"Token count: {count_tokens(text)}")

    # === Topic Summary ===
    log("\n=== Topic-Aware Summary ===")
    topic_summary = segment_and_summarize(text)
    log(topic_summary)

    # === Graph Summary ===
    log("\n=== Graph-Based Summary ===")
    graph_summary = graph_summarize(text, top_n=5)
    log(graph_summary)

    # === QA Summary ===
    log("\n=== QA-Based Summary ===")
    qa_summary = generate_questions(text, max_questions=5)
    log(qa_summary)

    # === Evaluation: Retention + Coherence ===
    questions = [
        "What is deep learning?",
        "What is the Pomodoro Technique?",
        "How does compound interest work?",
        "What apps help with time management?",
        "What should you know about AI privacy?"
    ]

    summaries = {
        "Topic-Based": topic_summary,
        "Graph-Based": graph_summary,
        "QA-Based": qa_summary
    }

    for label, summary_text in summaries.items():
        log(f"\n=== {label} Evaluation ===")
        log(f"Retention: {retention_score(summary_text, questions)}")
        log(f"Coherence: {coherence_score(summary_text)}")

    # === LLM Retention (with answers) ===
    qa_pairs = [
        ("What is deep learning?", "It is a subset of machine learning that uses neural networks with many layers."),
        ("What is the Pomodoro Technique?", "It involves working for 25 minutes, then taking a 5-minute break."),
        ("How does compound interest work?", "You earn interest on your original deposit and also on previously earned interest."),
        ("What apps help with time management?", "Focus Keeper, Forest, and Habitica."),
        ("What should you know about AI privacy?", "Avoid sharing sensitive info; some tools store data unless privacy mode is on.")
    ]

    for label, summary_text in summaries.items():
        log(f"\n=== {label} LLM-Based Retention ===")
        score = retention_score_llm(summary_text, qa_pairs)
        log(f"LLM Retention Score: {round(score, 2)}")
