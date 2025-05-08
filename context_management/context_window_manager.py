import openai
from openai import OpenAI
import tiktoken
from typing import List, Dict, Optional
from datetime import datetime
import os

class ContextWindowManager:
    def __init__(
        self,
        api_key: str, # Added api_key parameter
        model_name: str = "gpt-4o", # Defaulting to gpt-4o
        max_context_length: int = 8000, # Reasonable default, adjustable in UI
        summary_length: int = 1000,
        summary_ratio: float = 0.1 # This ratio isn't actively used in current summary logic but kept for potential future use
    ):
        # Validate API Key presence (basic check)
        if not api_key:
            raise ValueError("OpenAI API key is required.")

        self.model_name = model_name
        self.max_context_length = max_context_length
        self.summary_length = summary_length
        self.summary_ratio = summary_ratio
        
        # Initialize OpenAI client
        try:
            self.client = OpenAI(api_key=api_key)
            # Test connection with a simple call (optional, but good practice)
            # self.client.models.list() 
        except Exception as e:
            raise ConnectionError(f"Failed to initialize OpenAI client: {e}") from e
        
        # Initialize tokenizer based on the selected model
        try:
            self.tokenizer = tiktoken.encoding_for_model(model_name)
        except KeyError:
            print(f"Warning: No exact tokenizer found for model '{model_name}'. Falling back to cl100k_base.")
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
        
        # Initialize context tracking
        self.context_history: List[Dict] = []
        self.current_context: str = ""
        self.current_token_count: int = 0
        
    def _count_tokens(self, text: str) -> int:
        """Count tokens in text using tiktoken"""
        return len(self.tokenizer.encode(text))
    
    def _generate_summary(self, text_to_summarize: str) -> str:
        """Generate summary using the configured OpenAI model"""
        if not text_to_summarize.strip():
            return "" # Return empty if nothing to summarize

        # Estimate tokens for the prompt itself to avoid exceeding limits
        system_prompt = "You are a helpful assistant that creates concise summaries while preserving key information."
        user_prompt_template = f"Please summarize the following text in a way that preserves the most important information. Target length: {self.summary_length} tokens.\n\n{{text}}"
        prompt_overhead_tokens = self._count_tokens(system_prompt + user_prompt_template.format(text=""))

        # Calculate available tokens for the actual text to be summarized
        # Use a buffer (e.g., 100 tokens) to be safe
        available_tokens_for_text = self.max_context_length - self.summary_length - prompt_overhead_tokens - 100
        if available_tokens_for_text <= 0:
             raise ValueError("Not enough token space for summarization prompt and output. Adjust max_context or summary_length.")

        # Truncate input text if it exceeds available tokens for the API call
        input_tokens = self.tokenizer.encode(text_to_summarize)
        if len(input_tokens) > available_tokens_for_text:
            print(f"Warning: Input text for summarization ({len(input_tokens)} tokens) exceeds calculated available space ({available_tokens_for_text} tokens). Truncating.")
            truncated_tokens = input_tokens[:available_tokens_for_text]
            text_to_summarize = self.tokenizer.decode(truncated_tokens)

        user_prompt = user_prompt_template.format(text=text_to_summarize)

        try:
            print(f"\n--- Summarizing --- \nContext length (tokens): {self._count_tokens(self.current_context)}\nTarget summary length (tokens): {self.summary_length}\nModel: {self.model_name}\n---")
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=self.summary_length, # Max tokens for the *completion*
                temperature=0.3 # Low temperature for more deterministic summaries
            )
            summary = response.choices[0].message.content.strip()
            print(f"--- Summary Generated ({self._count_tokens(summary)} tokens) ---")
            return summary
        
        except openai.RateLimitError as e:
             print(f"OpenAI Rate Limit Error during summarization: {e}. Falling back.")
             # Fallback: return the last N tokens of the original context
             num_fallback_tokens = min(self.summary_length, self._count_tokens(self.current_context))
             fallback_tokens = self.tokenizer.encode(self.current_context)[-num_fallback_tokens:]
             return self.tokenizer.decode(fallback_tokens)
        except Exception as e:
            print(f"Error during summarization API call: {e}. Falling back.")
            # Fallback: return the last N tokens of the original context
            num_fallback_tokens = min(self.summary_length, self._count_tokens(self.current_context))
            fallback_tokens = self.tokenizer.encode(self.current_context)[-num_fallback_tokens:]
            return self.tokenizer.decode(fallback_tokens)

    
    def add_to_context(self, text: str) -> None:
        """Add new text to the context and manage window size"""
        if not text.strip():
            print("Skipping empty input.")
            return

        # Count tokens in new text
        new_token_count = self._count_tokens(text)
        print(f"Adding text ({new_token_count} tokens). Current context: {self.current_token_count} tokens.")

        # Check if the *new text alone* exceeds the limit
        if new_token_count > self.max_context_length:
            print(f"Warning: Input text ({new_token_count} tokens) is larger than max context length ({self.max_context_length}). Truncating input.")
            # Truncate the input text itself
            truncated_input_tokens = self.tokenizer.encode(text)[:self.max_context_length]
            text = self.tokenizer.decode(truncated_input_tokens)
            new_token_count = self._count_tokens(text)
            # If even the truncated input is too large for an empty context (shouldn't happen with check above, but safety), error out
            if new_token_count > self.max_context_length:
                 raise ValueError("Truncated input still exceeds max context length.")

        # Check if adding new text would exceed context limit
        if self.current_token_count + new_token_count > self.max_context_length:
            print(f"Context limit ({self.max_context_length}) exceeded ({self.current_token_count + new_token_count} tokens). Triggering summarization.")
            self._summarize_context()
            # After summarization, re-check if there's enough space for the new text
            if self.current_token_count + new_token_count > self.max_context_length:
                # This might happen if the summary + new text is still too long.
                # A more robust strategy might be needed here, like summarizing again or dropping older info.
                # For now, we'll print a warning and potentially truncate the new text again if needed.
                print(f"Warning: Even after summarization, adding the new text ({new_token_count} tokens) exceeds the limit ({self.max_context_length} tokens). Context: {self.current_token_count} tokens.")
                available_space = self.max_context_length - self.current_token_count
                if available_space < new_token_count and available_space > 0:
                    print(f"Truncating new text to fit remaining space ({available_space} tokens).")
                    truncated_input_tokens = self.tokenizer.encode(text)[:available_space]
                    text = self.tokenizer.decode(truncated_input_tokens)
                    new_token_count = self._count_tokens(text)
                elif available_space <= 0:
                    print("Error: No space left even after summarization. Cannot add new text.")
                    return # Cannot add the text

        # Add the potentially truncated text to current context
        # Add a newline for separation, count its token too if necessary (usually 1 token)
        separator = "\n\n"
        separator_tokens = self._count_tokens(separator)
        self.current_context += text + separator
        self.current_token_count += new_token_count + separator_tokens
        
        # Add original (or initially truncated if it was huge) text to history
        self.context_history.append({
            "text": text, # Log the text that was actually added
            "timestamp": datetime.now(),
            "token_count": new_token_count # Log the token count of the added text
        })
        print(f"Text added. New context token count: {self.current_token_count}")
    
    def _summarize_context(self) -> None:
        """Summarize the current context when it exceeds the limit"""
        if not self.current_context.strip():
            print("Attempted to summarize empty context.")
            return
        
        original_token_count = self.current_token_count
        
        try:
            # Generate summary
            summary = self._generate_summary(self.current_context)
            summary_token_count = self._count_tokens(summary)
            
            # Update context with summary
            self.current_context = summary + "\n\n" # Add separator after summary
            self.current_token_count = summary_token_count + self._count_tokens("\n\n")
            
            # Add summary entry to history
            self.context_history.append({
                "text": f"[SUMMARY] {summary}", # Keep prefix for history identification
                "timestamp": datetime.now(),
                "token_count": summary_token_count,
                "is_summary": True
            })
            print(f"Summarization complete. Context reduced from {original_token_count} to {self.current_token_count} tokens.")
            
        except Exception as e:
            # Catch potential errors from _generate_summary's fallback or other issues
            print(f"Error during summarization process: {e}")
            # Fallback: Drastic trim - keep only the last N tokens matching summary length target
            print(f"Applying fallback: Trimming context from {original_token_count} tokens.")
            fallback_tokens_to_keep = min(self.summary_length, original_token_count)
            context_tokens = self.tokenizer.encode(self.current_context)
            trimmed_tokens = context_tokens[-fallback_tokens_to_keep:]
            self.current_context = self.tokenizer.decode(trimmed_tokens) + "\n\n"
            self.current_token_count = self._count_tokens(self.current_context)
            print(f"Fallback trim complete. Context reduced to {self.current_token_count} tokens.")
            # Add a history entry indicating fallback occurred
            self.context_history.append({
                "text": f"[FALLBACK TRIM] Context reduced to {self.current_token_count} tokens due to summarization error.",
                "timestamp": datetime.now(),
                "token_count": 0, # Indicate this isn't a content entry
                "is_summary": False, # Or maybe a different flag?
                "is_fallback": True
            })

    
    def get_current_context(self) -> str:
        """Get the current effective context (usually the last summary or recent messages)"""
        return self.current_context
    
    def get_context_history(self) -> List[Dict]:
        """Get the full context history including messages and summaries"""
        return self.context_history
    
    def clear_context(self) -> None:
        """Clear the current context and history"""
        self.current_context = ""
        self.current_token_count = 0
        self.context_history = []
        print("Context and history cleared.")
    
    def get_token_count(self) -> int:
        """Get the current token count of the effective context"""
        return self.current_token_count
    
    def get_context_usage_ratio(self) -> float:
        """Get the ratio of used context tokens to the max limit"""
        if self.max_context_length == 0: return 0.0
        return min(1.0, self.current_token_count / self.max_context_length) # Cap at 100%

# Remove the old example usage block if it exists
# if __name__ == "__main__":
#    ... (old example code using local models) ... 