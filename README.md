# Multi-Agent System Exploration Project

This project explores different architectures for building multi-agent systems (MAS), primarily focusing on tasks involving large documents and complex reasoning.

## Architectures Explored

1.  **Hierarchical Agent System (`multiagents/`)**
    *   **Description:** This system implements a manager-worker hierarchy using LangGraph. It's designed to process large documents that exceed a single LLM's context window. The document is split into smaller chunks (using Algorithm 2 from the Chain-of-Agents paper). Each chunk is assigned to a 'worker' agent. Workers first assess the relevance of their chunk to the user's query and then extract key information if relevant. A central 'manager' agent then synthesizes the relevant information gathered by the workers into a final, comprehensive answer.
    *   **Key Features:** Parallel processing of document chunks, context window management for large texts, explicit relevance assessment.
    *   **File:** `multiagents/hierarchicaltest.py`

2.  **Chain-of-Agents (`chain-of-agents/`)**
    *   **Description:** Implements the Chain-of-Agents (CoA) framework for long-context reasoning with LLMs. In this system, a long document is split into manageable chunks, each processed sequentially by a worker agent. Each worker builds on the previous agent's output, creating a chain of communication units (CUs) that aggregate evidence and reasoning. A manager agent then synthesizes these CUs into a final, comprehensive answer. This approach enables efficient context window management and deep reasoning across large texts. The implementation supports multiple task types (summarization, QA, code), uses OpenAI models, and is fully configurable from the command line.
    *   **Key Features:** Sequential multi-agent processing, chain-of-communication, robust chunking (Algorithm 2 from CoA paper), flexible prompts, manager synthesis, PDF input, CLI interface, and environment variable support.
    *   **Main File:** `chain-of-agents/main.py`

3.  **Multi-Agent System for RAG (`masRAG/`)**
    *   **Description:** Implements a modular, multi-agent Retrieval-Augmented Generation (masRAG) system using LangGraph. The workflow starts with a manager agent routing queries to a RAG retriever, which fetches relevant code/document chunks from a vector database. Specialized worker agents process these chunks in parallel, extracting relevant information. The manager agent then aggregates and synthesizes the worker outputs into a final response. The system is designed for codebase and document QA, supports dynamic agent/task assignment, and is extensible for new roles or evaluation pipelines. The evaluation pipeline integrates Anthropic Claude for automated LLM-based assessment.
    *   **Key Features:** Multi-agent orchestration with LangGraph, RAG pipeline, parallel worker execution, dynamic task assignment, manager aggregation, codebase/document support, vector DB integration, and LLM-based evaluation.
    *   **Main File:** `masRAG/main.py` (run pipeline)

## Setup

1.  **Clone the repository:**
    ```bash
    git clone <your-repo-url>
    cd block-project
    ```
2.  **Create a virtual environment python 3.11 (recommended):**
    ```bash
    python3 -m venv venv
    source venv/bin/activate # On Windows use `venv\Scripts\activate`
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **API Keys:** Create a `.env` file in the root directory (`block-project/`) and add your OpenAI API key:
    ```
    OPENAI_API_KEY='your_openai_api_key_here'
    ```

## Running the Systems

Make sure your virtual environment is activated (`source venv/bin/activate` or `source .venv/bin/activate`).

1.  **Hierarchical Agent System:**
    *   Place the required PDF document (e.g., `coa.pdf`) inside the `multiagents/` directory.
    *   Run the script from the root directory:
        ```bash
        python multiagents/hierarchicaltest.py
        ```

2.  **Chain-of-Agents:**
    *   All code for Chain-of-Agents is now in the `chain-of-agents/` directory (no subdirectory).
    *   Ensure your `.env` file (with your `OPENAI_API_KEY`) is in the project root (`block-project/.env`). The script will look for `.env` in both the `chain-of-agents/` directory and its parent directory.
    *   To run the system, use:
        ```bash
        cd chain-of-agents
        python main.py <pdf_path> "<query>" --task_type summarization
        ```
        Example:
        ```bash
        python main.py coa.pdf "Summarize the chain of agents workflow and any insights from the paper" --task_type summarization
        ```
    *   **Arguments:**
        - `pdf_path`: Path to the PDF file to process (relative to `chain-of-agents/`)
        - `query`: The question or prompt for the system
        - `--task_type`: (Optional) Task type: `summarization`, `qa`, or `code` (default: `summarization`)
        - `--worker_model`: (Optional) LLM model for worker agents (default: `gpt-4o`)
        - `--manager_model`: (Optional) LLM model for manager agent (default: `gpt-4o`)
        - `--chunk_size`: (Optional) Chunk size for splitting the document (default: 8000)
    *   **Note:** The old `run.sh` bash script is no longer needed. All logic is now in `main.py`.
    *   All imports in the codebase have been updated for the flat structure (e.g., `from utils import ...`).

3.  **masRAG:**
    *   Run the main script from the root directory (`block-project/`):
        ```bash
        python masRAG/main.py --query "<your_query>"
        ```
    *   *(Check `masRAG/main.py --help` for additional arguments and options.)*

**General Notes:**
- Make sure all dependencies are installed from the project root:
    ```bash
    pip install -r requirements.txt
    ```
- The `.env` file should contain your OpenAI API key:
    ```env
    OPENAI_API_KEY=sk-...
    ```
