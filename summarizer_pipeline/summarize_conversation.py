import os
from dotenv import load_dotenv
import openai
from typing import List, Dict
import json
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import nltk
import numpy as np

# Load environment variables
load_dotenv()

# Set up OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

# Download required NLTK data
nltk.download('punkt')

def read_conversation_log(file_path: str) -> str:
    """Read the conversation log file."""
    with open(file_path, 'r') as f:
        return f.read()

def generate_summary(text: str, prompt_type: str) -> str:
    """Generate a summary using GPT-3.5 with different prompts."""
    prompts = {
        "concise": "Please provide a concise, bullet-point summary of this conversation, focusing on the main topics and key information.",
        "detailed": "Please provide a detailed summary of this conversation, including specific examples and explanations.",
        "thematic": "Please identify the main themes of this conversation and summarize how they are discussed throughout."
    }
    
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that summarizes conversations effectively."},
                {"role": "user", "content": f"{prompts[prompt_type]}\n\n{text}"}
            ],
            max_tokens=500,  # Increased max_tokens limit
            temperature=0.7
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error generating summary: {str(e)}"

def main():
    # Read the conversation log
    conversation = read_conversation_log('conversation_log.txt')
    
    # Generate summaries using different prompts
    print("\n=== Concise Summary ===")
    concise_summary = generate_summary(conversation, "concise")
    print(concise_summary)
    
    print("\n=== Detailed Summary ===")
    detailed_summary = generate_summary(conversation, "detailed")
    print(detailed_summary)
    
    print("\n=== Thematic Summary ===")
    thematic_summary = generate_summary(conversation, "thematic")
    print(thematic_summary)
    
    # Save all summaries to a file
    summaries = {
        'concise_summary': concise_summary,
        'detailed_summary': detailed_summary,
        'thematic_summary': thematic_summary
    }
    
    with open('summaries.json', 'w') as f:
        json.dump(summaries, f, indent=2)

if __name__ == "__main__":
    main() 