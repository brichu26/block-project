import os
import openai

# Get API key
openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY env variable.")

def ask_gpt_if_retained(summary, question, answer):
    system_msg = "You are a strict evaluator. Only respond YES if the summary clearly answers the question."

    prompt = f"""Given the following summary, question, and correct answer, does the summary contain enough information to answer the question accurately?

QUESTION: {question}
ANSWER: {answer}

SUMMARY:
{summary}

Reply with YES or NO only."""

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": prompt}
            ],
            max_tokens=10,
            temperature=0.0,
        )
        return response["choices"][0]["message"]["content"].strip().upper()
    except Exception as e:
        print("Error calling GPT:", e)
        return "ERROR"

def retention_score_llm(summary, qa_pairs):
    yes_count = 0
    total = len(qa_pairs)

    for question, answer in qa_pairs:
        verdict = ask_gpt_if_retained(summary, question, answer)
        print(f"\nQ: {question}\nA: {answer}\nLLM Verdict: {verdict}")
        if verdict == "YES":
            yes_count += 1

    return yes_count / total if total > 0 else 0
