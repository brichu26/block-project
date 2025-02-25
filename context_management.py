import random
import time
from transformers import pipeline
from collections import deque

# Sample conversation logs
conversation_data = [
    ("User: What's the weather like today?", "AI: It's sunny with a high of 75°F."),
    ("User: Can you summarize our last conversation?", "AI: You asked about the weather, and I told you it's sunny."),
    ("User: Tell me about quantum computing.", "AI: Quantum computing leverages superposition and entanglement to perform computations exponentially faster than classical computers."),
    ("User: What are some key challenges in AI research?", "AI: Key challenges include interpretability, bias, efficiency, and ethical considerations."),
]

# Truncation Strategies
def sliding_window_truncation(conversation, window_size=2):
    return conversation[-window_size:]

def fixed_length_truncation(conversation, char_limit=200):
    truncated = "".join(conversation)
    return truncated[-char_limit:]

def relevance_based_truncation(conversation):
    return [msg for msg in conversation if "important" in msg[0].lower() or "key" in msg[0].lower()]

# Summarization Strategy
summarizer = pipeline("summarization")
def summarization_strategy(conversation):
    joined_text = " ".join([msg[0] for msg in conversation])
    summary = summarizer(joined_text, max_length=50, min_length=20, do_sample=False)
    return summary[0]['summary_text']

# Memory-based Strategy (Retrieval-Augmented Memory)
class MemorySystem:
    def __init__(self, capacity=3):
        self.memory = deque(maxlen=capacity)
    
    def store(self, entry):
        self.memory.append(entry)
    
    def retrieve(self):
        return list(self.memory)

memory = MemorySystem()

def evaluate_strategies():
    print("Evaluating Context Handling Strategies...")
    
    # Truncation Methods
    truncated_sliding = sliding_window_truncation(conversation_data)
    truncated_fixed = fixed_length_truncation(conversation_data)
    truncated_relevance = relevance_based_truncation(conversation_data)
    
    # Summarization
    summary = summarization_strategy(conversation_data)
    
    # Memory-based
    for conv in conversation_data:
        memory.store(conv)
    memory_retrieved = memory.retrieve()
    
    print("Sliding Window Truncation:", truncated_sliding)
    print("Fixed-Length Truncation:", truncated_fixed)
    print("Relevance-Based Truncation:", truncated_relevance)
    print("Summarization Strategy:", summary)
    print("Memory-Based Retrieval:", memory_retrieved)

# Run evaluation
evaluate_strategies()
