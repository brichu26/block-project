import os
import time
import tiktoken
import openai
from datasets import load_dataset

# === Setup ===
print("[LongRAG Demo] Loading test corpus...")
corpus = load_dataset("TIGER-Lab/LongRAG", "nq_corpus", split="train[:5%]")

# === Retrieve a long unit (simulate LongRAG unit) ===
long_units = [doc['text'] for doc in corpus if 'text' in doc][:4]  # simulate top-4 retrievals
long_context = "\n\n".join(long_units)
print(f"[Info] Combined context from {len(long_units)} units.")

# === Token management (Enforce 1028 token limit) ===
encoding = tiktoken.get_encoding("cl100k_base")
context_tokens = len(encoding.encode(long_context))

MAX_TOTAL_TOKENS = 1028
MAX_OUTPUT_TOKENS = 256

if context_tokens + MAX_OUTPUT_TOKENS > MAX_TOTAL_TOKENS:
    token_budget = MAX_TOTAL_TOKENS - MAX_OUTPUT_TOKENS
    print(f"[Truncate] Context tokens before truncation: {context_tokens}")
    words = long_context.split()
    while len(encoding.encode(" ".join(words))) > token_budget:
        words = words[:-100]
    long_context = " ".join(words)
    context_tokens = len(encoding.encode(long_context))
    print(f"[Truncate] Truncated context tokens: {context_tokens}")

# === User query ===
query = "What is the main idea covered in these documents?"

# === OpenAI Completion ===
openai.api_key = os.getenv("OPENAI_API_KEY")

print("[Query] Sending request to OpenAI...")
start = time.time()
response = openai.ChatCompletion.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": f"{long_context}\n\nQuestion: {query}"}
    ],
    max_tokens=MAX_TOTAL_TOKENS - context_tokens,
    temperature=0
)
end = time.time()

output = response['choices'][0]['message']['content']
print("\n=== Response ===")
print(output)

# === Benchmarking ===
def evaluate(response, keywords):
    score = {
        "coherence": 1.0 if response[0].isupper() and response.strip().endswith('.') else 0.5,
        "relevance": 1.0 if any(k in response.lower() for k in keywords) else 0.5,
        "efficiency": 1.0 if len(response.split()) <= 120 else 0.7
    }
    score["overall"] = round(sum(score.values()) / 3, 2)
    return score

reference_keywords = ["article", "person", "history", "politics", "science", "explains", "research", "data"]
results = evaluate(output, reference_keywords)

print("\n=== Evaluation Scores ===")
for key, val in results.items():
    print(f"{key.capitalize()}: {val}")
print(f"Elapsed time: {round(end - start, 2)} seconds")
