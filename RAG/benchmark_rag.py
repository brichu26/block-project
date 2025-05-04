import os
import time
import tiktoken
from openai import OpenAI
import json
from typing import Dict, List, Tuple
from datasets import load_dataset
import requests
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch

# Constants
MAX_TOTAL_TOKENS = 1028
MAX_OUTPUT_TOKENS = 256
ENCODING = tiktoken.get_encoding("cl100k_base")

# Set OpenAI API key
os.environ["OPENAI_API_KEY"] = "key"
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class RAGEvaluator:
    def __init__(self):
        # Initialize MemoRAG components
        self.memorag_tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-mpnet-base-v2")
        self.memorag_model = AutoModel.from_pretrained("sentence-transformers/all-mpnet-base-v2")
        
        # Initialize HippoRAG components
        self.hipporag_tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-mpnet-base-v2")
        self.hipporag_model = AutoModel.from_pretrained("sentence-transformers/all-mpnet-base-v2")

    def evaluate_response(self, response: str, reference_keywords: List[str]) -> Dict[str, float]:
        """Evaluate RAG response based on multiple criteria."""
        # Relevance: Check if key topics are covered
        relevance = sum(1 for keyword in reference_keywords if keyword.lower() in response.lower()) / len(reference_keywords)
        
        # Coherence: Check structure and readability
        coherence = 1.0 if (response[0].isupper() and 
                          response.strip().endswith('.') and 
                          len(response.split('\n')) <= 5) else 0.5
        
        # Conciseness: Check if response is compact but meaningful
        word_count = len(response.split())
        conciseness = 1.0 if 50 <= word_count <= 120 else 0.7
        
        # Information Preservation: Check if important details are retained
        info_preservation = 1.0 if any(keyword in response.lower() for keyword in reference_keywords) else 0.5
        
        # Calculate overall score
        overall = (relevance + coherence + conciseness + info_preservation) / 4
        
        return {
            "relevance": round(relevance, 2),
            "coherence": round(coherence, 2),
            "conciseness": round(conciseness, 2),
            "information_preservation": round(info_preservation, 2),
            "overall": round(overall, 2)
        }

    def run_longrag(self, context: str, query: str) -> Tuple[str, float]:
        """Run LongRAG evaluation."""
        start_time = time.time()
        
        # Token management
        context_tokens = len(ENCODING.encode(context))
        if context_tokens + MAX_OUTPUT_TOKENS > MAX_TOTAL_TOKENS:
            token_budget = MAX_TOTAL_TOKENS - MAX_OUTPUT_TOKENS
            words = context.split()
            while len(ENCODING.encode(" ".join(words))) > token_budget:
                words = words[:-100]
            context = " ".join(words)
        
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": f"{context}\n\nQuestion: {query}"}
            ],
            max_tokens=MAX_TOTAL_TOKENS - len(ENCODING.encode(context)),
            temperature=0
        )
        
        elapsed_time = time.time() - start_time
        return response.choices[0].message.content, elapsed_time

    def get_embeddings(self, text: str, tokenizer, model) -> np.ndarray:
        """Get embeddings for text using the specified model."""
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).numpy()

    def run_memorag(self, context: str, query: str) -> Tuple[str, float]:
        """Run MemoRAG evaluation."""
        start_time = time.time()
        
        # Get embeddings for context and query
        context_emb = self.get_embeddings(context, self.memorag_tokenizer, self.memorag_model)
        query_emb = self.get_embeddings(query, self.memorag_tokenizer, self.memorag_model)
        
        # Calculate similarity scores
        similarity_scores = np.dot(context_emb, query_emb.T).flatten()
        
        # Select most relevant parts based on similarity
        relevant_indices = np.argsort(similarity_scores)[-3:]  # Top 3 most relevant parts
        relevant_parts = [context.split('\n')[i] for i in relevant_indices]
        relevant_context = '\n'.join(relevant_parts)
        
        # Generate response using OpenAI
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": f"{relevant_context}\n\nQuestion: {query}"}
            ],
            max_tokens=MAX_OUTPUT_TOKENS,
            temperature=0
        )
        
        elapsed_time = time.time() - start_time
        return response.choices[0].message.content, elapsed_time

    def run_hipporag(self, context: str, query: str) -> Tuple[str, float]:
        """Run HippoRAG evaluation."""
        start_time = time.time()
        
        # Get embeddings for context and query
        context_emb = self.get_embeddings(context, self.hipporag_tokenizer, self.hipporag_model)
        query_emb = self.get_embeddings(query, self.hipporag_tokenizer, self.hipporag_model)
        
        # Calculate similarity scores with personalized PageRank
        similarity_scores = np.dot(context_emb, query_emb.T).flatten()
        ppr_scores = similarity_scores * 0.85 + 0.15  # Simple PPR approximation
        
        # Select most relevant parts based on PPR scores
        relevant_indices = np.argsort(ppr_scores)[-3:]  # Top 3 most relevant parts
        relevant_parts = [context.split('\n')[i] for i in relevant_indices]
        relevant_context = '\n'.join(relevant_parts)
        
        # Generate response using OpenAI
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": f"{relevant_context}\n\nQuestion: {query}"}
            ],
            max_tokens=MAX_OUTPUT_TOKENS,
            temperature=0
        )
        
        elapsed_time = time.time() - start_time
        return response.choices[0].message.content, elapsed_time

def main():
    # Load test corpus
    print("[Benchmark] Loading test corpus...")
    corpus = load_dataset("TIGER-Lab/LongRAG", "nq_corpus", split="train[:5%]")
    
    # Prepare test data
    long_units = [doc['text'] for doc in corpus if 'text' in doc][:4]
    context = "\n\n".join(long_units)
    query = "What is the main idea covered in these documents?"
    
    # Reference keywords for evaluation
    reference_keywords = ["article", "person", "history", "politics", 
                         "science", "explains", "research", "data"]
    
    evaluator = RAGEvaluator()
    
    # Run evaluations
    print("\n=== Running LongRAG Evaluation ===")
    longrag_response, longrag_time = evaluator.run_longrag(context, query)
    longrag_scores = evaluator.evaluate_response(longrag_response, reference_keywords)
    
    print("\n=== Running MemoRAG Evaluation ===")
    memorag_response, memorag_time = evaluator.run_memorag(context, query)
    memorag_scores = evaluator.evaluate_response(memorag_response, reference_keywords)
    
    print("\n=== Running HippoRAG Evaluation ===")
    hipporag_response, hipporag_time = evaluator.run_hipporag(context, query)
    hipporag_scores = evaluator.evaluate_response(hipporag_response, reference_keywords)
    
    # Print results
    print("\n=== Evaluation Results ===")
    print("\nLongRAG:")
    print(f"Response: {longrag_response}")
    print("Scores:")
    for metric, score in longrag_scores.items():
        print(f"{metric.capitalize()}: {score}")
    print(f"Time taken: {round(longrag_time, 2)} seconds")
    
    print("\nMemoRAG:")
    print(f"Response: {memorag_response}")
    print("Scores:")
    for metric, score in memorag_scores.items():
        print(f"{metric.capitalize()}: {score}")
    print(f"Time taken: {round(memorag_time, 2)} seconds")
    
    print("\nHippoRAG:")
    print(f"Response: {hipporag_response}")
    print("Scores:")
    for metric, score in hipporag_scores.items():
        print(f"{metric.capitalize()}: {score}")
    print(f"Time taken: {round(hipporag_time, 2)} seconds")

if __name__ == "__main__":
    main() 