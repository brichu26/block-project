# Content Management System with Summarization Capabilities

This system provides multiple summarization techniques for text content, with evaluation metrics to compare their effectiveness. A key feature is the implementation of an LLM-based evaluation pipeline that provides detailed, human-like assessment of summary quality.

## Prerequisites

- Python 3.x
- OpenAI API key
- Required Python packages (install using `pip install -r requirements.txt`):
  - tiktoken
  - openai
  - nltk
  - numpy
  - transformers
  - keybert
  - rouge
  - sentence-transformers

## Installation

1. Clone this repository
2. Create a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Set your OpenAI API key:
   ```bash
   export OPENAI_API_KEY='your-api-key-here'
   ```

## Usage

Run the script with an input text file:
```bash
python CMS.py input_file.txt
```

Optional arguments:
- `--api-key`: Your OpenAI API key (if not set as environment variable)
- `--token-limit`: Maximum token limit for summaries (default: 500)
- `--input-token-limit`: Maximum token limit for input text (default: 1000)
- `--check-tokens`: Only check token count of input file

Example:
```bash
python CMS.py conversation_log.txt --token-limit 300 --input-token-limit 800
```

## Expected Results

The script will generate a file named `input_file_summary_results.txt` containing:

1. **Four Types of Summaries:**
   - Truncated: Simple token-based truncation
   - Abstractive: Using BART model
   - Hierarchical: Two-stage summarization
   - ChatGPT-4: Using GPT-4 model

2. **Evaluation Metrics for Each Summary:**
   - Token count
   - For ChatGPT-4 summary:
     - Relevance score (0-5)
     - Coherence score (0-5)
     - Conciseness score (0-5)
     - Information preservation score (0-5)
   - For other summaries:
     - Retention score (0-1)
     - Coherence score (0-1)

3. **Ranking of Summarization Techniques:**
   - Ordered list of techniques by overall score
   - Detailed explanation of why the best technique performed well

## Code Structure

### Main Components

1. **Text Processing Functions:**
   - `load_conversation()`: Loads text from file
   - `num_tokens_from_string()`: Counts tokens in text
   - `truncate_text()`: Truncates text to token limit

2. **Summarization Methods:**
   - `abstractive_summary()`: Uses BART model
   - `chatgpt_summary()`: Uses GPT-4
   - `hierarchical_summary()`: Two-stage summarization

3. **Evaluation Functions:**
   - `retention_score()`: Measures keyword overlap
   - `coherence_score()`: Measures sentence similarity
   - `llm_evaluate_summary()`: GPT-4-based evaluation

### Output Format

The results file contains sections for each summarization method:
```
=== Method Name Summary ===
Tokens: [number]
[Evaluation Metrics]
Summary Text:
[actual summary]
==================
```

Followed by:
- Ranking of techniques
- Detailed explanation of the best technique

## Interpreting Results

1. **Token Counts:**
   - Lower is better (more concise)
   - Should be within specified token limit

2. **Evaluation Scores:**
   - ChatGPT-4 scores: 0-5 scale (higher is better)
   - Other methods: 0-1 scale (higher is better)

3. **Ranking:**
   - Based on combined scores
   - Best technique is explained in detail

## LLM Evaluation Pipeline

The system implements a sophisticated LLM-based evaluation pipeline using GPT-4 to assess summary quality. This pipeline:

1. **Evaluation Criteria:**
   - Relevance (0-5): How well the summary captures main topics and themes
   - Coherence (0-5): How well-structured and logically organized the summary is
   - Conciseness (0-5): How effectively the summary condenses information
   - Information Preservation (0-5): How well key details and facts are maintained

2. **Evaluation Process:**
   - Uses GPT-4 to analyze both original text and summary
   - Provides detailed justifications for each score
   - Generates an overall assessment of summary quality
   - Returns structured JSON output for easy parsing

3. **Benefits:**
   - More nuanced evaluation than traditional metrics
   - Human-like assessment of summary quality
   - Detailed feedback on strengths and weaknesses
   - Consistent scoring across different types of content

## Notes

- The script simulates context window limitations
- All summaries are evaluated for quality and effectiveness
- Results are saved in both console output and text file
- UTF-8 encoding is used throughout
- LLM evaluation provides detailed, human-like assessment of summary quality

