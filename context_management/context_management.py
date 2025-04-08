import random
import time
import numpy as np
import torch
from transformers import pipeline
from collections import deque
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Tuple, Dict

# Sample conversation logs
conversation_data = [
    ("User: What's the weather like today?", "AI: It's sunny with a high of 75°F."),
    ("User: Can you summarize our last conversation?", "AI: You asked about the weather, and I told you it's sunny."),
    ("User: Tell me about quantum computing.", "AI: Quantum computing leverages superposition and entanglement to perform computations exponentially faster than classical computers."),
    ("User: What are some key challenges in AI research?", "AI: Key challenges include interpretability, bias, efficiency, and ethical considerations."),
]

class ContextStrategy:
    def __init__(self, name: str):
        self.name = name
        self.scores = {
            'coherence': 0.0,
            'relevance': 0.0,
            'efficiency': 0.0,
            'overall': 0.0
        }
    
    def evaluate(self, original: List[Tuple[str, str]], processed: List[Tuple[str, str]]) -> Dict[str, float]:
        # Calculate coherence score (how well the processed conversation flows)
        coherence = self._calculate_coherence(processed)
        
        # Calculate relevance score (how well it maintains important information)
        relevance = self._calculate_relevance(original, processed)
        
        # Calculate efficiency score (compression ratio while maintaining quality)
        efficiency = self._calculate_efficiency(original, processed)
        
        # Calculate overall score
        overall = (coherence + relevance + efficiency) / 3
        
        self.scores = {
            'coherence': coherence,
            'relevance': relevance,
            'efficiency': efficiency,
            'overall': overall
        }
        return self.scores
    
    def _calculate_coherence(self, processed: List[Tuple[str, str]]) -> float:
        # Simple coherence check based on conversation flow
        if len(processed) < 2:
            return 1.0
        return 0.8  # Placeholder - could be enhanced with more sophisticated metrics
    
    def _calculate_relevance(self, original: List[Tuple[str, str]], processed: List[Tuple[str, str]]) -> float:
        # Check if key terms from original are preserved
        original_terms = set(term.lower() for msg in original for term in msg[0].split())
        processed_terms = set(term.lower() for msg in processed for term in msg[0].split())
        preserved_terms = original_terms.intersection(processed_terms)
        return len(preserved_terms) / len(original_terms) if original_terms else 0.0
    
    def _calculate_efficiency(self, original: List[Tuple[str, str]], processed: List[Tuple[str, str]]) -> float:
        # Calculate compression ratio while maintaining quality
        original_length = sum(len(msg[0]) for msg in original)
        processed_length = sum(len(msg[0]) for msg in processed)
        return 1 - (processed_length / original_length) if original_length > 0 else 0.0

# Truncation Strategies
class SlidingWindowStrategy(ContextStrategy):
    def __init__(self, window_size=2):
        super().__init__("Sliding Window")
        self.window_size = window_size
    
    def process(self, conversation: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
        return conversation[-self.window_size:]

class FixedLengthStrategy(ContextStrategy):
    def __init__(self, char_limit=200):
        super().__init__("Fixed Length")
        self.char_limit = char_limit
    
    def process(self, conversation: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
        result = []
        current_length = 0
        for msg in reversed(conversation):
            msg_length = len(msg[0])
            if current_length + msg_length > self.char_limit:
                break
            result.insert(0, msg)
            current_length += msg_length
        return result

class RelevanceStrategy(ContextStrategy):
    def __init__(self):
        super().__init__("Relevance Based")
    
    def process(self, conversation: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
        return [msg for msg in conversation if "important" in msg[0].lower() or "key" in msg[0].lower()]

# Summarization Strategy
class SummarizationStrategy(ContextStrategy):
    def __init__(self):
        super().__init__("Summarization")
        # Force CPU for summarization to avoid MPS issues
        device = "cpu"
        self.summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=device)
    
    def process(self, conversation: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
        joined_text = " ".join([msg[0] for msg in conversation])
        try:
            summary = self.summarizer(joined_text, max_length=100, min_length=30, do_sample=False)
            return [("Summary", summary[0]['summary_text'])]
        except Exception as e:
            print(f"Summarization failed: {str(e)}")
            return conversation[-1:]  # Return last message as fallback

# Memory-based Strategy
class MemoryStrategy(ContextStrategy):
    def __init__(self, capacity=3):
        super().__init__("Memory Based")
        self.memory = deque(maxlen=capacity)
    
    def process(self, conversation: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
        for msg in conversation:
            self.memory.append(msg)
        return list(self.memory)

def evaluate_strategies():
    print("Evaluating Context Handling Strategies...")
    
    strategies = [
        SlidingWindowStrategy(window_size=2),
        FixedLengthStrategy(char_limit=200),
        RelevanceStrategy(),
        SummarizationStrategy(),
        MemoryStrategy(capacity=3)
    ]
    
    results = {}
    for strategy in strategies:
        print(f"\nEvaluating {strategy.name} Strategy:")
        try:
            processed = strategy.process(conversation_data)
            scores = strategy.evaluate(conversation_data, processed)
            
            print(f"Processed conversation: {processed}")
            print("Scores:")
            for metric, score in scores.items():
                print(f"{metric.capitalize()}: {score:.2f}")
            
            results[strategy.name] = scores
        except Exception as e:
            print(f"Error evaluating {strategy.name}: {str(e)}")
            continue
    
    if results:
        # Print comparison
        print("\nStrategy Comparison:")
        print("-" * 80)
        print(f"{'Strategy':<20} {'Coherence':<10} {'Relevance':<10} {'Efficiency':<10} {'Overall':<10}")
        print("-" * 80)
        for name, scores in results.items():
            print(f"{name:<20} {scores['coherence']:<10.2f} {scores['relevance']:<10.2f} "
                  f"{scores['efficiency']:<10.2f} {scores['overall']:<10.2f}")
        print("-" * 80)
        
        # Find best strategy
        best_strategy = max(results.items(), key=lambda x: x[1]['overall'])
        print(f"\nBest performing strategy: {best_strategy[0]} (Overall score: {best_strategy[1]['overall']:.2f})")

if __name__ == "__main__":
    try:
        evaluate_strategies()
    except Exception as e:
        print(f"An error occurred: {str(e)}")
