# Conversation Summarizer

A Python tool that analyzes conversation logs and generates different types of summaries using OpenAI's GPT-3.5.

## Features

- Process conversation logs from text files
- Generate three types of summaries:
  - **Concise**: Bullet-point summary focusing on main topics
  - **Detailed**: Extended summary with specific examples
  - **Thematic**: Analysis of the main themes in the conversation
- Save summaries to a JSON file for further processing or reference

## Requirements

- Python 3.6+
- OpenAI API key
- Required Python packages (see below)

## Installation

1. Clone this repository or download the files
2. Install required packages:
```
pip install openai python-dotenv nltk scikit-learn numpy
```
3. Create a `.env` file in the project directory with your OpenAI API key:
```
OPENAI_API_KEY=your_api_key_here
```

## Usage

1. Place your conversation transcript in `conversation_log.txt`
2. Run the summarization script:
```
python summarize_conversation.py
```
3. View the generated summaries in the terminal
4. Find the saved summaries in `summaries.json`

## Files

- `summarize_conversation.py`: Main script for generating summaries
- `conversation_log.txt`: Sample conversation log
- `summaries.json`: Output file containing generated summaries (created upon running)

## Customization

You can modify the prompts used for summary generation by editing the `prompts` dictionary in the `generate_summary` function. 