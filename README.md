# Model Context Protocol and Evaluation Toolkit

This repository contains a collection of tools for implementing the Model Context Protocol (MCP), conversation summarization, and model comparison/evaluation pipelines.

## Project Structure

### Context Management (MCP Implementation)

- **needle_in_haystack_evaluator.py**: Evaluates how well different summarization strategies preserve critical information within texts using the Model Context Protocol. Uses OpenAI API to generate summaries and evaluate them based on retention of key information.

- **context_management.py**: Core implementation of Model Context Protocol strategies for handling large text contexts in LLM applications.

- **context_visualizer.py**: Visualization tools for analyzing context windows, displaying how content is prioritized and managed within the MCP framework.

- **context_window_manager.py**: Manages context windows for LLM inputs, implementing various strategies for context prioritization according to MCP principles.

- **summary_evaluator_app.py**: Application for evaluating the quality of different summarization approaches.

- **.env**: Environment variables file containing API keys (OPENAI_API_KEY).

### Conversation Summarizer

- **conversation_log.txt**: Sample conversation logs for testing summarization techniques.

- **summarize_conversation.py**: Script for summarizing conversation logs using various techniques.

### Model Comparison Pipeline (MCP)

- **mcp_security_eval.py**: Evaluates models based on security criteria.

- **mcp_documentation_eval.py**: Assesses model outputs based on documentation quality.

- **mcp_performance_eval.py**: Benchmarks model performance metrics.

- **mcp_popularity_eval.py**: Analyzes model popularity and usage statistics.

- **mcp_scoring.py**: Core scoring system for the model comparison pipeline.

## Setup Instructions

### Prerequisites

- Python 3.8 or higher
- OpenAI API key

### Installation

1. Clone the repository:
   ```
   git clone [repository-url]
   cd [repository-name]
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
   
   (Key dependencies include: openai, python-dotenv, matplotlib, pandas)

3. Set up environment variables:
   - Create a `.env` file in the context_management directory
   - Add your OpenAI API key:
     ```
     OPENAI_API_KEY=your_openai_api_key_here
     ```

### Running the Tools

#### Needle in Haystack Evaluator
```
cd context_management
python needle_in_haystack_evaluator.py
```
This will evaluate different summarization strategies on sample conversations.

#### Context Management Visualizer
```
cd context_management
python context_visualizer.py
```

#### Conversation Summarizer
```
cd conversation_summarizer
python summarize_conversation.py
```

#### Model Context Protocol
```
cd mcp
python mcp_scoring.py
```

## Component Details

### Needle in Haystack Evaluator
Evaluates how well different summarization strategies retain key information. Implements a scoring system that awards full points for exact matches and partial points for context-preserving summaries.

### Context Window Manager
Implements various strategies for managing context windows for LLMs according to the Model Context Protocol (MCP), including prioritization, truncation, and summarization approaches.

### Model Comparison Pipeline
A comprehensive framework for evaluating and comparing different language models across security, documentation quality, performance, and popularity metrics.

## Note

This project requires an OpenAI API key to function properly. Make sure to set up the `.env` file with a valid API key before running any of the tools.

