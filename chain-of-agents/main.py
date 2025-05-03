from typing import Optional, Iterator, Dict
from agents import WorkerAgent, ManagerAgent
from utils import chunk_text_algorithm2, get_task_prompts, read_pdf
import logging
import json
import os
import sys
import argparse
from dotenv import load_dotenv
import pathlib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ChainOfAgents:
    """Main class for the Chain of Agents implementation."""
    
    def __init__(
        self,
        worker_model: str = "gpt-4o-mini",
        manager_model: str = "gpt-4o",
        chunk_size: int = 8000,  # Adjusted for token count rather than words
        task_type: str = "qa",
        worker_prompt: Optional[str] = None,
        manager_prompt: Optional[str] = None
    ):
        """
        Initialize the Chain of Agents.
        
        Args:
            worker_model: Model to use for worker agents
            manager_model: Model to use for manager agent
            chunk_size: Maximum tokens per chunk
            task_type: Type of task (qa, summarization, code)
            worker_prompt: Custom system prompt for workers
            manager_prompt: Custom system prompt for manager
        """
        worker_prompt_default, manager_prompt_default = get_task_prompts(task_type)
        
        self.worker_prompt = worker_prompt or worker_prompt_default
        self.manager_prompt = manager_prompt or manager_prompt_default
        self.chunk_size = chunk_size
        self.worker_model = worker_model
        self.manager_model = manager_model
        self.task_type = task_type
        
        logger.info(f"Initialized Chain of Agents with {worker_model} workers and {manager_model} manager for {task_type} task")
    
    def process(self, input_text: str, query: str) -> str:
        """
        Process a long text input using the Chain of Agents.
        
        Args:
            input_text: The long input text to process
            query: The user's query about the text
            
        Returns:
            str: The final response from the manager agent
        """
        # Split text into chunks using Algorithm 2
        chunks = chunk_text_algorithm2(
            source_text=input_text,
            query=query,
            instruction=self.worker_prompt,
            window_size=self.chunk_size
        )
        
        # Process chunks with worker agents in sequence
        previous_cu = None
        
        for i, chunk in enumerate(chunks):
            logger.info(f"Processing chunk {i+1}/{len(chunks)}")
            worker = WorkerAgent(self.worker_model, self.worker_prompt)
            output = worker.process_chunk(chunk, query, previous_cu, self.task_type)
            previous_cu = output
        
        # Final CU is the output from the last worker
        final_cu = previous_cu
        
        # Synthesize results with manager agent
        manager = ManagerAgent(self.manager_model, self.manager_prompt)
        final_output = manager.synthesize(final_cu, query, self.task_type)
        
        return final_output
    
    def process_stream(self, input_text: str, query: str) -> Iterator[Dict[str, str]]:
        """Process text with streaming - yields worker and manager messages."""
        chunks = chunk_text_algorithm2(
            source_text=input_text,
            query=query,
            instruction=self.worker_prompt,
            window_size=self.chunk_size
        )
        total_chunks = len(chunks)
        
        # Debug logging for metadata
        metadata_message = {
            "type": "metadata",
            "content": json.dumps({
                "total_chunks": total_chunks,
                "total_pages": getattr(input_text, 'total_pages', 0)
            })
        }
        logger.info(f"Sending metadata: {metadata_message}")
        yield metadata_message
        
        previous_cu = None
        
        for i, chunk in enumerate(chunks):
            logger.info(f"Processing chunk {i+1}/{total_chunks}")
            worker = WorkerAgent(self.worker_model, self.worker_prompt)
            output = worker.process_chunk(chunk, query, previous_cu, self.task_type)
            previous_cu = output
            
            # Send worker progress
            worker_message = {
                "type": "worker",
                "content": output,
                "progress": {
                    "current": i + 1,
                    "total": total_chunks
                }
            }
            logger.info(f"Sending worker message: {worker_message}")
            yield worker_message
        
        # Final CU is the output from the last worker
        final_cu = previous_cu
        
        logger.info("Processing manager synthesis")
        manager = ManagerAgent(self.manager_model, self.manager_prompt)
        final_output = manager.synthesize(final_cu, query, self.task_type)
        
        yield {
            "type": "manager",
            "content": final_output
        }

def main():
    parser = argparse.ArgumentParser(description="Run Chain of Agents on a PDF document.")
    parser.add_argument("pdf_path", help="Path to the PDF file to process.")
    parser.add_argument("query", help="Query to ask the agents.")
    parser.add_argument("--task_type", default="summarization", choices=["summarization", "qa", "code"], help="Task type for the agents.")
    parser.add_argument("--worker_model", default="gpt-4o", help="OpenAI model for worker agents.")
    parser.add_argument("--manager_model", default="gpt-4o", help="OpenAI model for manager agent.")
    parser.add_argument("--chunk_size", type=int, default=8000, help="Chunk size for splitting the document.")
    args = parser.parse_args()

    # Load environment variables
    env_path = pathlib.Path('.') / '.env'
    parent_env_path = pathlib.Path('..') / '.env'
    if env_path.exists():
        load_dotenv(dotenv_path=env_path)
    elif parent_env_path.exists():
        load_dotenv(dotenv_path=parent_env_path)
    else:
        print(f"No .env file found in {env_path.resolve()} or {parent_env_path.resolve()}. Please create one with your OpenAI API key.")
        sys.exit(1)

    # Verify API key is loaded
    if not os.getenv("OPENAI_API_KEY"):
        print("OPENAI_API_KEY not found in environment variables")
        sys.exit(1)

    # Read PDF file
    if not os.path.exists(args.pdf_path):
        print(f"Error: PDF file not found at {args.pdf_path}")
        sys.exit(1)
    input_text = read_pdf(args.pdf_path)

    # Get prompts
    worker_prompt, manager_prompt = get_task_prompts(args.task_type)

    # Create Chain of Agents instance
    chain = ChainOfAgents(
        worker_model=args.worker_model,
        manager_model=args.manager_model,
        chunk_size=args.chunk_size,
        task_type=args.task_type,
        worker_prompt=worker_prompt,
        manager_prompt=manager_prompt
    )

    # Process text
    final_output = chain.process(input_text, args.query)

    print("\n" + "=" * 80)
    print("FINAL RESPONSE")
    print("=" * 80 + "\n")

    print(final_output)
    print("\n" + "=" * 80)

if __name__ == "__main__":
    main()