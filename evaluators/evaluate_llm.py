# evaluators/evaluate_llm.py

from openai import OpenAI
import os
import json

# --- Configuration ---
EVALUATOR_MODEL = "gpt-4o"  # or gpt-3.5-turbo if you prefer faster/cheaper
client = OpenAI()

def evaluate_summary_llm(reference_text, generated_summary):
    """
    Evaluates the quality of a generated summary compared to the original text using an LLM as a judge.
    """
    try:
        prompt = (
            "You are evaluating a generated summary.\n\n"
            "Original Text:\n"
            f"{reference_text}\n\n"
            "Generated Summary:\n"
            f"{generated_summary}\n\n"
            "Evaluation Instructions:\n"
            "Rate the summary from 1 to 10 based on:\n"
            "- Relevance: Does it cover all core points from the conversation?\n"
            "- Coherence: Does the summary flow logically without abrupt or disjointed jumps?\n"
            "- Precision: Does it avoid hallucinated or irrelevant information?\n"
            "- Conciseness: Is it tight and avoids redundancy?\n"
            "Then explain your reasoning in 1-2 sentences.\n\n"
            "Format your response strictly like this:\n"
            "Score: <number>\n"
            "Explanation: <your explanation>\n"
        )

        response = client.chat.completions.create(
            model=EVALUATOR_MODEL,
            messages=[
                {"role": "system", "content": "You are a critical but fair evaluator of summaries."},
                {"role": "user", "content": prompt},
            ],
            temperature=0,
            max_tokens=500,
        )

        reply = response.choices[0].message.content

        # Simple parsing
        lines = reply.split("\n")
        score_line = next((line for line in lines if line.lower().startswith("score:")), None)
        explanation_line = next((line for line in lines if line.lower().startswith("explanation:")), None)

        if score_line and explanation_line:
            score = int(score_line.split(":")[1].strip())
            explanation = explanation_line.split(":", 1)[1].strip()
            return {
                "score": score,
                "explanation": explanation
            }
        else:
            return {
                "error": "Failed to parse evaluation output",
                "raw_output": reply
            }

    except Exception as e:
        return {
            "error": str(e)
        }
