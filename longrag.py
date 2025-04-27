import os
import requests
import time
import tiktoken
import openai
from memorag import MemoRAG

# Document
url = 'https://raw.githubusercontent.com/qhjqhj00/MemoRAG/main/examples/harry_potter.txt'
print("Downloading sample document...")
response = requests.get(url)
content = response.text

# Use about 50k words (~67k tokens) to simulate a long file
small_part = " ".join(content.split()[:50000])
print(f"Loaded document: {len(small_part.split())} words.")

# Initialize MemoRAG
print("Initializing MemoRAG...")
pipe = MemoRAG(
    mem_model_name_or_path="TommyChien/memorag-qwen2-7b-inst",
    ret_model_name_or_path="BAAI/bge-m3",
    beacon_ratio=16,
    load_in_4bit=True,
    enable_flash_attn=False  # Safer for most local GPUs
)

# Load pre-cached memory if available (or you can call memorize if you have compute)
memory_path = "./harry_potter_qwen2_ratio16"
if os.path.exists(memory_path):
    print(f"Loading pre-cached memory from {memory_path}...")
    pipe.load(memory_path, print_stats=True)
else:
    print("Warning: No memory cache found. You should run `pipe.memorize` if needed.")

# -- Step 3: Set up Benchmark Parameters
query = "What is the main theme of the book?"
token_limit = 1028  # OpenAI token budget constraint

# -- Step 4: Retrieval and Benchmarking
print("Starting retrieval...")
start_time = time.time()

retrieved_passages = pipe._retrieve([query])
retrieved_context = " ".join(retrieved_passages[:2])  # use top-2 passages

retrieval_time = time.time() - start_time
print(f"Retrieved {len(retrieved_passages)} passages in {retrieval_time:.2f} seconds.")

# Count tokens in the retrieved context
encoding = tiktoken.get_encoding("cl100k_base")
retrieved_tokens = len(encoding.encode(retrieved_context))
print(f"Retrieved context uses {retrieved_tokens} tokens.")

# Truncate if necessary
if retrieved_tokens > 800:
    print("Truncating retrieved context to fit under token budget...")
    retrieved_context = " ".join(retrieved_context.split()[:750])

# OpenAI LLM
print("Sending request to OpenAI...")
openai.api_key = os.getenv("OPENAI_API_KEY")  # Make sure your key is set in the terminal

completion = openai.ChatCompletion.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": "You are a helpful AI assistant."},
        {"role": "user", "content": f"{retrieved_context}\n\nQuestion: {query}"}
    ],
    max_tokens=token_limit - retrieved_tokens,
    temperature=0
)

# Print
generated_answer = completion["choices"][0]["message"]["content"]

print("\n=== Benchmark Results ===")
print(f"Query: {query}")
print(f"Retrieved context tokens: {retrieved_tokens}")
print(f"Time to retrieve: {retrieval_time:.2f} seconds")
print(f"Generated answer:\n{generated_answer}")
print("==========================\n")
