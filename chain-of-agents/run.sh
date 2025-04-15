#!/bin/bash

# Exit on error
set -e

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install requirements
echo "Installing requirements..."
pip install -r requirements.txt

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo "Creating .env file..."
    echo "OPENAI_API_KEY=" > .env
    echo "Please add your OpenAI API key to the .env file"
    exit 1
fi

# Run example
echo "Running Chain of Agents example..."
python3 - << EOF
from chain_of_agents import ChainOfAgents
from chain_of_agents.utils import read_pdf, split_into_chunks
from chain_of_agents.agents import WorkerAgent, ManagerAgent
import os
from dotenv import load_dotenv
import pathlib
import sys

# Load environment variables
env_path = pathlib.Path('.') / '.env'
load_dotenv(dotenv_path=env_path)

# Verify API key is loaded
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY not found in environment variables")

# Initialize Chain of Agents
coa = ChainOfAgents(
    worker_model="gpt-4o",
    manager_model="gpt-4-turbo-preview",
    chunk_size=8000,  # Increased chunk size for OpenAI models
    task_type="summarization"
)

# Read PDF file
pdf_path = "coa.pdf"  # Updated to your PDF file
if not os.path.exists(pdf_path):
    print(f"Error: PDF file not found at {pdf_path}")
    sys.exit(1)

input_text = read_pdf(pdf_path)
query = "Summarize, in depth, how Chain of Agents works, and how multiple agents can be useful for context window management and reasoning."

# Process the text
print("\nProcessing document with Chain of Agents...\n")

chunks = split_into_chunks(input_text, coa.chunk_size)
worker_outputs = []
previous_cu = None

print("=" * 80)
print("WORKER RESPONSES")
print("=" * 80 + "\n")

for i, chunk in enumerate(chunks):
    print(f"\n{'='*30} Worker {i+1}/{len(chunks)} {'='*30}")
    worker = WorkerAgent(coa.worker_model, coa.worker_prompt)
    output = worker.process_chunk(chunk, query, previous_cu, coa.task_type)
    worker_outputs.append(output)
    previous_cu = output
    print(f"\n{output}\n")

print("\n" + "=" * 80)
print("MANAGER SYNTHESIS")
print("=" * 80 + "\n")

manager = ManagerAgent(coa.manager_model, coa.manager_prompt)
final_output = manager.synthesize(previous_cu, query, coa.task_type)
print(final_output)

print("\n" + "=" * 80)
EOF

# Deactivate virtual environment
deactivate

echo "Done!"
