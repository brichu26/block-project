from utils.io import load_conversation, count_tokens, truncate_text_safe
from summarizers.topic_segment import segment_and_summarize
from summarizers.graph_summarizer import graph_summarize
from summarizers.memory_summarizer import memory_augmented_summary
from evaluators.evaluate_llm import evaluate_summary_llm
from datetime import datetime

MAX_INPUT_TOKENS = 1024
SUMMARY_TOKEN_LIMIT = 1024
OUTPUT_FILE = "outputs/summary_output.txt"

if __name__ == "__main__":
    output_lines = []

    # Header
    output_lines.append(f"Summary Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    output_lines.append("=" * 50)

    # Load conversation
    text = load_conversation("data/sample_convo.txt")
    output_lines.append(f"\nToken count: {count_tokens(text)}")

    if count_tokens(text) > MAX_INPUT_TOKENS:
        output_lines.append(f"\nTruncating input to {MAX_INPUT_TOKENS} tokens...")
        text = truncate_text_safe(text, MAX_INPUT_TOKENS)
        output_lines.append(f"After truncation token count: {count_tokens(text)}")

    if not text.strip():
        raise ValueError("Input conversation is empty after truncation. Cannot summarize.")

    input_tokens = count_tokens(text)
    max_summary_length = max(64, min(int(0.3 * input_tokens), 512))
    min_summary_length = max(30, int(max_summary_length * 0.4))
    output_lines.append(f"\nDynamic summary length settings: max_length={max_summary_length}, min_length={min_summary_length}")

    # Summarizations
    summaries = {}
    output_lines.append("\n" + "=" * 50)
    output_lines.append("\n=== Topic-Based Summary ===")
    topic_summary = segment_and_summarize(text, output_token_limit=SUMMARY_TOKEN_LIMIT)
    output_lines.append(topic_summary)
    summaries["Topic-Based"] = topic_summary

    output_lines.append("\n" + "=" * 50)
    output_lines.append("\n=== Graph-Based Summary ===")
    graph_summary = graph_summarize(text, top_n=5, token_limit=SUMMARY_TOKEN_LIMIT)
    output_lines.append(graph_summary)
    summaries["Graph-Based"] = graph_summary

    output_lines.append("\n" + "=" * 50)
    output_lines.append("\n=== Memory-Augmented Summary ===")
    memory_summary = memory_augmented_summary(text, max_tokens=SUMMARY_TOKEN_LIMIT, memory_points=5)
    output_lines.append(memory_summary)
    summaries["Memory-Augmented"] = memory_summary

    # Evals
    for label, summary_text in summaries.items():
        output_lines.append("\n" + "=" * 50)
        output_lines.append(f"\n=== {label} Evaluation ===")
        evaluation = evaluate_summary_llm(text, summary_text)
        if "evaluation" in evaluation:
            eval_data = evaluation["evaluation"]
            for metric, details in eval_data.items():
                output_lines.append(f"{metric.capitalize()}: {details['score']}/5 — {details['justification']}")
            output_lines.append("Overall Comment: " + evaluation.get('overall_comment', "No comment."))
        else:
            output_lines.append(f"Evaluation failed: {evaluation}")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write("\n\n".join(output_lines))

    print(f"\nSummaries and evaluations saved to {OUTPUT_FILE}")
