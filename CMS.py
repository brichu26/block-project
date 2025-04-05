# summarization_token_limit_test.py
import os
import tiktoken
import openai
import nltk
import numpy as np
from transformers import pipeline
from keybert import KeyBERT
from rouge import Rouge
from sentence_transformers import SentenceTransformer
import sys
import argparse

# Set up argument parser
parser = argparse.ArgumentParser(description='Content Management System with summarization capabilities')
parser.add_argument('input_file', help='Input conversation file to process')
parser.add_argument('--api-key', help='OpenAI API key')
parser.add_argument('--token-limit', type=int, default=2048, help='Maximum token limit for summaries')
parser.add_argument('--input-token-limit', type=int, default=8192, help='Maximum token limit for input text')
parser.add_argument('--check-tokens', action='store_true', help='Only check token count of input file')
args = parser.parse_args()

# Download required NLTK data
nltk.download("punkt")

# Set OpenAI API key from environment variable or command line argument
api_key = args.api_key or os.getenv("OPENAI_API_KEY")
if not api_key:
    print("Error: OpenAI API key not found")
    print("Please either:")
    print("1. Set OPENAI_API_KEY environment variable: export OPENAI_API_KEY='your-api-key'")
    print("2. Provide the API key as an argument: python CMS.py test_conversation.txt --api-key 'your-api-key'")
    sys.exit(1)
openai.api_key = api_key

# Load text
def load_conversation(filepath):
    with open(filepath, 'r') as f:
        return f.read()

# Tokenizer for simulation (OpenAI GPT-3.5 or GPT-4 as example)
def num_tokens_from_string(string, model_name="gpt-3.5-turbo"):
    encoding = tiktoken.encoding_for_model(model_name)
    return len(encoding.encode(string))

# Check token count
def check_token_count(filepath):
    text = load_conversation(filepath)
    tokens = num_tokens_from_string(text)
    print(f"Input file token count: {tokens}")
    print(f"Approximate word count: {len(text.split())}")
    return tokens

# Summarization using BART
def abstractive_summary(text, max_tokens=1024):
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    # Calculate appropriate max_length based on input length
    input_length = len(text.split())
    max_length = min(max_tokens // 0.75, input_length // 2)  # Take half of input length or max_tokens, whichever is smaller
    min_length = min(30, max_length // 3)  # Take 1/3 of max_length or 30, whichever is smaller
    
    return summarizer(text, 
                     max_length=int(max_length), 
                     min_length=int(min_length), 
                     do_sample=False)[0]['summary_text']

# ChatGPT-4 summarization
def chatgpt_summary(prompt, model="gpt-4", max_tokens=1024):
    # Using the new OpenAI API format
    response = openai.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful summarization assistant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=max_tokens,
        temperature=0.3,
    )
    return response.choices[0].message.content

# Truncate input text by token limit
def truncate_text(text, token_limit, model="gpt-3.5-turbo"):
    encoding = tiktoken.encoding_for_model(model)
    tokens = encoding.encode(text)
    truncated = encoding.decode(tokens[:token_limit])
    return truncated

# Evaluation metric: Retention via keyword overlap
def retention_score(summary, original_text):
    kw_model = KeyBERT()
    original_keywords = [kw[0] for kw in kw_model.extract_keywords(original_text, top_n=10)]
    summary_keywords = [kw[0] for kw in kw_model.extract_keywords(summary, top_n=10)]
    retained = set(original_keywords) & set(summary_keywords)
    return len(retained) / len(original_keywords) if original_keywords else 0

# Coherence: sentence similarity
def coherence_score(summary):
    sentences = nltk.sent_tokenize(summary)
    if len(sentences) < 2:
        return 0
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(sentences)
    sim_scores = [
        np.dot(embeddings[i], embeddings[i+1]) / (np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[i+1]))
        for i in range(len(embeddings)-1)
    ]
    return np.mean(sim_scores)

# Main
def main(convo_file, token_limit=2048, input_token_limit=8192):
    try:
        full_text = load_conversation(convo_file)
        input_tokens = num_tokens_from_string(full_text)
        print(f"Original token count: {input_tokens}")
        
        # Truncate input if it exceeds the input token limit
        if input_tokens > input_token_limit:
            print(f"Input exceeds {input_token_limit} tokens. Truncating to first {input_token_limit} tokens.")
            full_text = truncate_text(full_text, input_token_limit)
            input_tokens = num_tokens_from_string(full_text)
            print(f"Truncated input token count: {input_tokens}")

        # Truncate
        truncated = truncate_text(full_text, token_limit)
        print(f"\nTruncated summary token count: {num_tokens_from_string(truncated)}")
        
        # Generate abstractive summary
        print("\nGenerating abstractive summary...")
        abstractive = abstractive_summary(full_text, max_tokens=token_limit)
        print(f"Abstractive summary token count: {num_tokens_from_string(abstractive)}")
        
        # Generate ChatGPT summary
        print("\nGenerating ChatGPT summary...")
        chatgpt_prompt = f"Summarize the following text:\n\n{truncated}"
        chatgpt = chatgpt_summary(chatgpt_prompt, model="gpt-4", max_tokens=token_limit)
        print(f"ChatGPT summary token count: {num_tokens_from_string(chatgpt)}")

        # Evaluate
        methods = {
            "Truncated": truncated,
            "Abstractive": abstractive,
            "ChatGPT-4": chatgpt,
        }

        results = {}
        for method, summary in methods.items():
            tokens = num_tokens_from_string(summary)
            retention = retention_score(summary, full_text)
            coherence = coherence_score(summary)
            results[method] = {
                "tokens": tokens,
                "retention": retention,
                "coherence": coherence,
            }

        for method, metrics in results.items():
            print(f"\n=== {method} Summary ===")
            print(f"Tokens: {metrics['tokens']}")
            print(f"Retention: {metrics['retention']:.2f}")
            print(f"Coherence: {metrics['coherence']:.2f}")
            print(methods[method][:500] + "...\n")
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    if args.check_tokens:
        check_token_count(args.input_file)
    else:
        main(args.input_file, token_limit=args.token_limit, input_token_limit=args.input_token_limit)
