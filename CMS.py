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
import json
from openai import OpenAI

import os

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY environment variable not set")

# Set up argument parser
parser = argparse.ArgumentParser(description='Content Management System with summarization capabilities')
parser.add_argument('input_file', help='Input conversation file to process')
parser.add_argument('--api-key', help='OpenAI API key')
parser.add_argument('--token-limit', type=int, default=500, help='Maximum token limit for summaries (simulating context window limit)')
parser.add_argument('--input-token-limit', type=int, default=1000, help='Maximum token limit for input text (simulating context window limit)')
parser.add_argument('--check-tokens', action='store_true', help='Only check token count of input file')
args = parser.parse_args()

# Download required NLTK data
nltk.download("punkt")

# Set OpenAI API key from environment variable or command line argument
api_key = args.api_key or os.getenv("OPENAI_API_KEY")
if not api_key:
    print("Error: OpenAI API key not found")
    print("Please either:")
    print("1. Set OPENAI_API_KEY environment variable: export OPENAI_API_KEY='your-api-key-here'")
    print("2. Provide the API key as an argument: python CMS.py test_conversation.txt --api-key 'your-api-key'")
    sys.exit(1)
openai.api_key = api_key

EVALUATION_PROMPT_TEMPLATE = """
You are an expert evaluator assessing the quality of a text summary.
Compare the 'Generated Summary' against the 'Original Text'.

Please evaluate the summary based on the following criteria, providing a score from 1 (Poor) to 5 (Excellent) for each, along with a brief justification:

1.  **Relevance:** Does the summary accurately capture the main topic and key points of the original text? (Score 1-5)
2.  **Coherence:** Is the summary well-organized, easy to understand, and does it flow logically? (Score 1-5)
3.  **Conciseness:** Does the summary effectively condense the original text without unnecessary jargon or repetition? (Is it significantly shorter?) (Score 1-5)
4.  **Information Preservation:** Does the summary retain the most critical information from the original text without introducing inaccuracies or misinterpretations? (Score 1-5)

Provide your evaluation in JSON format like this example:

```json
{{
  "evaluation": {{
    "relevance": {{"score": 4, "justification": "Captures most key points but missed one minor detail."}},
    "coherence": {{"score": 5, "justification": "The summary is clear and flows well."}},
    "conciseness": {{"score": 5, "justification": "Significantly shorter while retaining core message."}},
    "information_preservation": {{"score": 3, "justification": "Omitted a key statistic mentioned in the original."}}
  }},
  "overall_comment": "A good summary, but could improve on preserving specific details."
}}
```

**Original Text:**
---
{{original_text}}
---

**Generated Summary:**
---
{{summary_text}}
---

Your JSON Evaluation:
"""

# Load text
def load_conversation(filepath):
    with open(filepath, 'r', encoding='utf-8-sig') as f:
        return f.read()

# Tokenizer for simulation (OpenAI GPT-3.5 or GPT-4 as example)
def num_tokens_from_string(string, model_name="gpt-3.5-turbo"):
    try:
        encoding = tiktoken.encoding_for_model(model_name)
        # Ensure the input is a string
        if not isinstance(string, str):
            string = str(string)
        # Attempt to encode, handling potential errors
        tokens = encoding.encode(string)
        return len(tokens)
    except UnicodeEncodeError as e:
        print(f"UnicodeEncodeError in num_tokens_from_string for model {model_name}: {e}")
        print("Problematic string snippet (first 100 chars):")
        try:
            print(string[:100])
        except UnicodeEncodeError:
            print("[Could not display snippet due to encoding issue]")
        # Fallback: return an estimated token count based on characters, or 0
        return len(string) // 3 # Rough estimate
    except Exception as e:
        print(f"Error in num_tokens_from_string for model {model_name}: {e}")
        # Fallback: return an estimated token count based on characters, or 0
        return len(string) // 3 # Rough estimate

# Check token count
def check_token_count(filepath):
    text = load_conversation(filepath)
    # Use a simple estimation method that doesn't require API key
    words = text.split()
    estimated_tokens = len(words) * 1.3  # Rough estimate: 1.3 tokens per word
    print(f"Input file estimated token count: {int(estimated_tokens)}")
    print(f"Word count: {len(words)}")
    return int(estimated_tokens)

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
    try:
        # Aggressive sanitization: Force ASCII encoding, ignoring errors
        sanitized_prompt_bytes = prompt.encode('ascii', errors='ignore')
        sanitized_prompt = sanitized_prompt_bytes.decode('ascii')
        
        # Using the new OpenAI API format
        response = openai.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful summarization assistant."},
                {"role": "user", "content": sanitized_prompt} # Use aggressively sanitized prompt
            ],
            max_tokens=max_tokens,
            temperature=0.3,
        )
        # Attempt to access content safely and ensure UTF-8
        content = response.choices[0].message.content
        if isinstance(content, bytes):
            # If it's bytes, decode assuming UTF-8
            return content.decode('utf-8', errors='replace')
        elif isinstance(content, str):
            # If it's already a string, return as is (Python 3 strings are Unicode)
             return content
        else:
            # Handle unexpected type
            print(f"Warning: Unexpected type for message content: {type(content)}")
            return str(content) # Attempt to convert to string
            
    except UnicodeDecodeError as e:
        print(f"Error decoding API response in chatgpt_summary: {e}")
        return "[Error decoding API response]"
    except Exception as e:
        # Print sanitized exception message
        sanitized_error = str(e).encode('ascii', errors='replace').decode('ascii')
        print(f"Error during ChatGPT API call: {sanitized_error}")
        # It might be helpful to see the raw response if possible
        # print(f"Raw response object: {response}")
        return "[Error during API call]"

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

# LLM-based evaluation
def llm_evaluate_summary(original_text, summary_text, api_key):
    """Evaluate a summary using LLM (GPT-4) based on relevance, coherence, conciseness, and information preservation."""
    try:
        # Sanitize inputs
        original_text = original_text.encode('ascii', errors='ignore').decode('ascii')
        summary_text = summary_text.encode('ascii', errors='ignore').decode('ascii')

        prompt = f"""Evaluate the following summary based on the original text. Provide a detailed evaluation in JSON format.

Original Text:
{original_text}

Summary:
{summary_text}

Evaluation Criteria:
1. Relevance: Does the summary capture the main topics and themes of the original text?
2. Coherence: Is the summary well-structured and logically organized?
3. Conciseness: Is the summary concise while retaining essential information?
4. Information Preservation: Does the summary preserve key details and facts from the original text?

Provide a score (0-10) and justification for each criterion, and an overall comment. Return the evaluation in the following JSON format:

{{
  "evaluation": {{
    "relevance": {{"score": <score>, "justification": "<justification>"}},
    "coherence": {{"score": <score>, "justification": "<justification>"}},
    "conciseness": {{"score": <score>, "justification": "<justification>"}},
    "information_preservation": {{"score": <score>, "justification": "<justification>"}}
  }},
  "overall_comment": "<overall_comment>"
}}"""

        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that evaluates summaries."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=1000
        )

        evaluation_text = response.choices[0].message.content.strip()

        # Remove markdown formatting
        evaluation_text = evaluation_text.replace("```json", "").replace("```", "").replace("---", "").strip()

        try:
            evaluation = json.loads(evaluation_text)
            return evaluation
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON from evaluation response: {e}")
            print(f"Raw response: {evaluation_text}")
            return None

    except Exception as e:
        error_message = str(e).encode('ascii', errors='replace').decode('ascii')
        print(f"Error evaluating summary using LLM: {error_message}")
        return None

# Hierarchical summarization
def hierarchical_summary(text, max_tokens=500, chunk_size=1000):
    """Two-stage summarization that first breaks text into chunks, summarizes each chunk,
    then combines and summarizes the chunk summaries."""
    try:
        # First stage: Split text into chunks and summarize each
        chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
        chunk_summaries = []
        
        for chunk in chunks:
            # Use BART for first-stage summarization
            chunk_summary = abstractive_summary(chunk, max_tokens=max_tokens//2)
            chunk_summaries.append(chunk_summary)
        
        # Second stage: Combine chunk summaries and create final summary
        combined_summaries = " ".join(chunk_summaries)
        final_summary = chatgpt_summary(
            f"Create a coherent summary from these section summaries, maintaining key information and logical flow:\n\n{combined_summaries}",
            model="gpt-4",
            max_tokens=max_tokens
        )
        
        return final_summary
    except Exception as e:
        print(f"Error in hierarchical summarization: {str(e)}")
        return None

# Main
def main(convo_file, token_limit=500, input_token_limit=1000):
    try:
        full_text = load_conversation(convo_file)
        input_tokens = num_tokens_from_string(full_text)
        print(f"Original token count: {input_tokens}")
        
        # Create output filename based on input filename
        output_filename = os.path.splitext(convo_file)[0] + "_summary_results.txt"
        
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
        
        # Generate hierarchical summary
        print("\nGenerating hierarchical summary...")
        hierarchical = hierarchical_summary(full_text, max_tokens=token_limit)
        print(f"Hierarchical summary token count: {num_tokens_from_string(hierarchical)}")
        
        # Generate ChatGPT summary
        print("\nGenerating ChatGPT summary...")
        chatgpt_prompt = f"Summarize the following text in {token_limit} tokens or less:\n\n{truncated}"
        chatgpt = chatgpt_summary(chatgpt_prompt, model="gpt-4", max_tokens=token_limit)
        try:
            print(f"ChatGPT summary token count: {num_tokens_from_string(chatgpt)}")
        except UnicodeEncodeError:
            print("ChatGPT summary token count: [Could not display due to encoding issue]")

        # Evaluate
        methods = {
            "Truncated": truncated,
            "Abstractive": abstractive,
            "Hierarchical": hierarchical,
            "ChatGPT-4": chatgpt,
        }

        results = {}
        for method, summary in methods.items():
            tokens = num_tokens_from_string(summary)
            if method == "ChatGPT-4":
                print(f"\nEvaluating ChatGPT-4 summary using LLM (gpt-4) ...")
                evaluation_result = llm_evaluate_summary(full_text, summary, api_key)
                results[method] = {
                    "tokens": tokens,
                    "llm_evaluation": evaluation_result
                }
            else:
                 # Use original metrics for other methods
                retention = retention_score(summary, full_text)
                coherence = coherence_score(summary)
                results[method] = {
                    "tokens": tokens,
                    "retention": retention,
                    "coherence": coherence,
                }

        # Write results to file
        with open(output_filename, 'w', encoding='utf-8') as f:
            f.write("=== Summary Results ===\n\n")
            
            for method, metrics in results.items():
                f.write(f"\n=== {method} Summary ===\n")
                f.write(f"Tokens: {metrics['tokens']}\n")
                
                if "llm_evaluation" in metrics:
                    eval_data = metrics["llm_evaluation"]
                    if "error" in eval_data:
                        f.write(f"LLM Evaluation Error: {eval_data['error']}\n")
                    elif "evaluation" in eval_data:
                        f.write("LLM Evaluation:\n")
                        for criterion, details in eval_data["evaluation"].items():
                            f.write(f"  - {criterion.capitalize()}: {details['score']}\n")
                            f.write(f"    Justification: {details['justification']}\n")
                        f.write(f"  Overall Comment: {eval_data.get('overall_comment', 'N/A')}\n")
                    else:
                        f.write("LLM Evaluation response format incorrect.\n")
                        f.write(f"Raw LLM Response: {eval_data}\n")
                else:
                    # Write original metrics for other methods with justifications
                    retention = metrics.get('retention', 'N/A')
                    coherence = metrics.get('coherence', 'N/A')
                    f.write(f"Retention Score: {retention:.2f}\n")
                    f.write(f"  Justification: This score measures how much of the original content is preserved in the summary.\n")
                    f.write(f"  A score of {retention:.2f} indicates that {int(retention * 100)}% of the key information from the original text is maintained.\n")
                    
                    f.write(f"Coherence Score: {coherence:.2f}\n")
                    f.write(f"  Justification: This score measures how well the summary flows and maintains logical connections.\n")
                    f.write(f"  A score of {coherence:.2f} indicates the level of semantic coherence and readability of the summary.\n")
                
                # Write the summary itself
                summary_text = methods[method]
                f.write("\nSummary Text:\n")
                f.write(summary_text + "\n")
                f.write("\n" + "="*50 + "\n")

        print(f"\nResults have been saved to: {output_filename}")

        # Print results to console as before
        for method, metrics in results.items():
            print(f"\n=== {method} Summary ===")
            print(f"Tokens: {metrics['tokens']}")
            if "llm_evaluation" in metrics:
                eval_data = metrics["llm_evaluation"]
                if "error" in eval_data:
                     print(f"LLM Evaluation Error: {eval_data['error']}")
                elif "evaluation" in eval_data:
                    print("LLM Evaluation:")
                    for criterion, details in eval_data["evaluation"].items():
                        print(f"  - {criterion.capitalize()}: {details['score']} ({details['justification']})")
                    print(f"  Overall Comment: {eval_data.get('overall_comment', 'N/A')}")
                else:
                    print("LLM Evaluation response format incorrect.")
                    print(f"Raw LLM Response: {eval_data}") # Print raw data for debugging
            else:
                # Print original metrics for other methods
                print(f"Retention: {metrics.get('retention', 'N/A'):.2f}")
                print(f"Coherence: {metrics.get('coherence', 'N/A'):.2f}")
            
            # Print a snippet of the summary itself
            summary_text = methods[method]
            try:
                 print(summary_text[:500] + ("..." if len(summary_text) > 500 else "") + "\n")
            except UnicodeEncodeError:
                print("[Summary snippet could not be displayed due to encoding issue]\n")

        def get_overall_score(metrics):
            # For LLM-evaluated summaries
            if "llm_evaluation" in metrics and "evaluation" in metrics["llm_evaluation"]:
                eval_scores = metrics["llm_evaluation"]["evaluation"]
                return sum([v["score"] for v in eval_scores.values()]) / len(eval_scores)
            # For other summaries
            elif "retention" in metrics and "coherence" in metrics:
                return (metrics["retention"] + metrics["coherence"]) / 2
            else:
                return 0

        # Compute scores
        technique_scores = {method: get_overall_score(metrics) for method, metrics in results.items()}

        # Rank
        ranked = sorted(technique_scores.items(), key=lambda x: x[1], reverse=True)

        print("Ranking of summarization techniques:")
        for i, (method, score) in enumerate(ranked, 1):
            print(f"{i}. {method} (Score: {score:.2f})")

        # Detailed explanation for the best technique
        best_method, best_score = ranked[0]
        best_metrics = results[best_method]
        explanation_lines = [f"\nWhy '{best_method}' was the best summarization technique:"]
        if "llm_evaluation" in best_metrics and "evaluation" in best_metrics["llm_evaluation"]:
            eval_data = best_metrics["llm_evaluation"]["evaluation"]
            for criterion, details in eval_data.items():
                explanation_lines.append(f"- {criterion.capitalize()}: Scored {details['score']}/5. Justification: {details['justification']}")
            overall_comment = best_metrics["llm_evaluation"].get("overall_comment", "")
            if overall_comment:
                explanation_lines.append(f"- Overall Comment: {overall_comment}")
        else:
            retention = best_metrics.get('retention', 0)
            coherence = best_metrics.get('coherence', 0)
            explanation_lines.append(f"- Retention: Scored {retention:.2f}. This measures how much of the original content was preserved in the summary.")
            explanation_lines.append(f"- Coherence: Scored {coherence:.2f}. This measures the logical flow and readability of the summary.")
            if retention > coherence:
                explanation_lines.append("- The high retention score indicates that this summary preserved more key information from the original text than the others.")
            elif coherence > retention:
                explanation_lines.append("- The high coherence score indicates that this summary was especially well-structured and easy to follow compared to the others.")
            else:
                explanation_lines.append("- This summary balanced both retention and coherence well, making it the top choice overall.")
        explanation = "\n".join(explanation_lines)
        print(explanation)

        # Write ranking and explanation to file
        with open(output_filename, 'a', encoding='utf-8') as f:
            f.write("\nRanking of summarization techniques:\n")
            for i, (method, score) in enumerate(ranked, 1):
                f.write(f"{i}. {method} (Score: {score:.2f})\n")
            f.write(explanation + "\n")

    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    if args.check_tokens:
        check_token_count(args.input_file)
    else:
        main(args.input_file, token_limit=args.token_limit, input_token_limit=args.input_token_limit)
