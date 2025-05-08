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

# Summarization Strategies
SUMMARIZATION_STRATEGIES = {
    "Standard Concise": {
        "system_prompt": "You are an expert summarizer. Condense the following text, focusing on the main points and key information, while maintaining clarity and coherence.",
        "user_prompt_template": "Please summarize the following text concisely:\n\n---\n{text}\n---"
    },
    "Key Bullet Points": {
        "system_prompt": "You are an expert summarizer. Extract the key bullet points or main takeaways from the following text.",
        "user_prompt_template": "Extract the key bullet points from this text:\n\n---\n{text}\n---"
    },
    "Abstractive TL;DR": {
        "system_prompt": "You are an expert at creating very short, high-level summaries (like a TL;DR). Capture the absolute essence of the text in 1-2 sentences.",
        "user_prompt_template": "Provide a 1-2 sentence TL;DR summary for this text:\n\n---\n{text}\n---"
    },
    "Question-Focused": {
        "system_prompt": "You are an expert summarizer. Summarize the text by primarily answering the questions: Who was involved? What happened? Why did it happen?",
        "user_prompt_template": "Summarize this text, focusing on Who, What, and Why:\n\n---\n{text}\n---"
    }
}

class ContextStrategy:
    def __init__(self, name: str):
        self.name = name
        self.scores = {
            'coherence': 0.0,
            'relevance': 0.0,
            'efficiency': 0.0,
            'conciseness': 0.0,
            'information_preservation': 0.0,
            'overall': 0.0
        }
    
    def evaluate(self, original: List[Tuple[str, str]], processed: List[Tuple[str, str]]) -> Dict[str, float]:
        # Calculate all metrics
        coherence = self._calculate_coherence(processed)
        relevance = self._calculate_relevance(original, processed)
        efficiency = self._calculate_efficiency(original, processed)
        conciseness = self._calculate_conciseness(processed)
        info_preservation = self._calculate_information_preservation(original, processed)
        
        # Calculate overall score (weighted average)
        overall = (coherence * 0.2 + relevance * 0.2 + efficiency * 0.2 + 
                  conciseness * 0.2 + info_preservation * 0.2)
        
        self.scores = {
            'coherence': coherence,
            'relevance': relevance,
            'efficiency': efficiency,
            'conciseness': conciseness,
            'information_preservation': info_preservation,
            'overall': overall
        }
        return self.scores
    
    def _calculate_coherence(self, processed: List[Tuple[str, str]]) -> float:
        if len(processed) < 2:
            return 1.0
        # Enhanced coherence check using semantic similarity
        try:
            embeddings = [self._get_embedding(msg[0]) for msg in processed]
            similarities = []
            for i in range(len(embeddings) - 1):
                sim = cosine_similarity([embeddings[i]], [embeddings[i + 1]])[0][0]
                similarities.append(sim)
            return np.mean(similarities)
        except:
            return 0.8  # Fallback to simple coherence check
    
    def _calculate_relevance(self, original: List[Tuple[str, str]], processed: List[Tuple[str, str]]) -> float:
        original_terms = set(term.lower() for msg in original for term in msg[0].split())
        processed_terms = set(term.lower() for msg in processed for term in msg[0].split())
        preserved_terms = original_terms.intersection(processed_terms)
        return len(preserved_terms) / len(original_terms) if original_terms else 0.0
    
    def _calculate_efficiency(self, original: List[Tuple[str, str]], processed: List[Tuple[str, str]]) -> float:
        original_length = sum(len(msg[0]) for msg in original)
        processed_length = sum(len(msg[0]) for msg in processed)
        return 1 - (processed_length / original_length) if original_length > 0 else 0.0
    
    def _calculate_conciseness(self, processed: List[Tuple[str, str]]) -> float:
        # Calculate average sentence length and word count
        total_words = sum(len(msg[0].split()) for msg in processed)
        total_sentences = sum(len(msg[0].split('.')) for msg in processed)
        if total_sentences == 0:
            return 0.0
        avg_words_per_sentence = total_words / total_sentences
        # Normalize to 0-1 range (assuming 20 words per sentence is optimal)
        return max(0, 1 - (avg_words_per_sentence / 20))
    
    def _calculate_information_preservation(self, original: List[Tuple[str, str]], processed: List[Tuple[str, str]]) -> float:
        # Check for key information preservation using semantic similarity
        try:
            original_embedding = self._get_embedding(" ".join(msg[0] for msg in original))
            processed_embedding = self._get_embedding(" ".join(msg[0] for msg in processed))
            return cosine_similarity([original_embedding], [processed_embedding])[0][0]
        except:
            return self._calculate_relevance(original, processed)  # Fallback to term-based relevance
    
    def _get_embedding(self, text: str) -> np.ndarray:
        # Simple word embedding using average of word vectors
        words = text.lower().split()
        if not words:
            return np.zeros(100)  # Default dimension
        return np.mean([np.random.rand(100) for _ in words], axis=0)  # Placeholder for actual embeddings

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
    def __init__(self, strategy_name="Standard Concise"):
        super().__init__(f"Summarization ({strategy_name})")
        self.strategy_name = strategy_name
        self.strategy = SUMMARIZATION_STRATEGIES[strategy_name]
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
        MemoryStrategy(capacity=3)
    ]
    
    # Add all summarization strategies
    for strategy_name in SUMMARIZATION_STRATEGIES.keys():
        strategies.append(SummarizationStrategy(strategy_name))
    
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
        print("-" * 100)
        print(f"{'Strategy':<30} {'Coherence':<10} {'Relevance':<10} {'Efficiency':<10} {'Conciseness':<10} {'Info Pres.':<10} {'Overall':<10}")
        print("-" * 100)
        for name, scores in results.items():
            print(f"{name:<30} {scores['coherence']:<10.2f} {scores['relevance']:<10.2f} "
                  f"{scores['efficiency']:<10.2f} {scores['conciseness']:<10.2f} "
                  f"{scores['information_preservation']:<10.2f} {scores['overall']:<10.2f}")
        print("-" * 100)
        
        # Find best strategy
        best_strategy = max(results.items(), key=lambda x: x[1]['overall'])
        print(f"\nBest performing strategy: {best_strategy[0]} (Overall score: {best_strategy[1]['overall']:.2f})")

if __name__ == "__main__":
    try:
        evaluate_strategies()
    except Exception as e:
        print(f"An error occurred: {str(e)}")
