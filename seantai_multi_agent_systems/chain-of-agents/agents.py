from typing import List, Optional, Iterator, Dict
import openai
import os

class WorkerAgent:
    """Worker agent that processes individual chunks of text."""
    
    def __init__(self, model: str, system_prompt: str):
        """
        Initialize a worker agent.
        
        Args:
            model: The LLM model to use (e.g., "gpt-4o")
            system_prompt: The system prompt that defines the worker's role
        """
        self.model = model
        self.system_prompt = system_prompt
        self.client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    
    def process_chunk(self, chunk: str, query: str, previous_cu: Optional[str] = None, task_type: str = "qa") -> str:
        """
        Process a single chunk of text.
        
        Args:
            chunk: The text chunk to process
            query: The user's query
            previous_cu: The previous Communication Unit (CU) if any
            task_type: Type of task (qa, summarization, code)
            
        Returns:
            str: The processed output for this chunk
        """
        # Format the user message based on the paper's approach
        if previous_cu:
            user_content = f"""Here is the current source text:
{chunk}

Here is the summary of the previous source text:
{previous_cu}

Question: {query}

You need to read current source text and summary of previous source text and generate a summary to include them both. 
This summary will be used for other agents to answer the question, so please write the summary that can include the evidence for answering the question."""
        else:
            user_content = f"""Here is the current source text:
{chunk}

Question: {query}

You need to read current source text and generate a summary that includes evidence for answering the question."""
        
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_content}
        ]
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0,  # Paper uses temperature=0 for deterministic outputs
            max_tokens=1024
        )
        
        return response.choices[0].message.content
    
    async def process_chunk_stream(self, chunk: str, query: str, previous_cu: Optional[str] = None, task_type: str = "qa") -> Iterator[str]:
        """Process a chunk with streaming."""
        # Format the user message based on the paper's approach
        if previous_cu:
            user_content = f"""Here is the current source text:
{chunk}

Here is the summary of the previous source text:
{previous_cu}

Question: {query}

You need to read current source text and summary of previous source text and generate a summary to include them both. 
This summary will be used for other agents to answer the question, so please write the summary that can include the evidence for answering the question."""
        else:
            user_content = f"""Here is the current source text:
{chunk}

Question: {query}

You need to read current source text and generate a summary that includes evidence for answering the question."""
            
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_content}
        ]
        
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0,
            stream=True
        )
        
        async for chunk in response:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

class ManagerAgent:
    """Manager agent that synthesizes outputs from worker agents."""
    
    def __init__(self, model: str, system_prompt: str):
        """
        Initialize a manager agent.
        
        Args:
            model: The LLM model to use (e.g., "gpt-4")
            system_prompt: The system prompt that defines the manager's role
        """
        self.model = model
        self.system_prompt = system_prompt
        self.client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    
    def synthesize(self, final_cu: str, query: str, task_type: str = "qa") -> str:
        """
        Synthesize the final communication unit into a response.
        
        Args:
            final_cu: The final communication unit from the last worker
            query: The original user query
            task_type: Type of task (qa, summarization, code)
            
        Returns:
            str: The final synthesized response
        """
        user_content = f"""The following is a summary of a long document:
{final_cu}

Question: {query}

Based on the summary, please provide a comprehensive answer to the question."""
        
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_content}
        ]
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0,
            max_tokens=1024
        )
        
        return response.choices[0].message.content
    
    async def synthesize_stream(self, final_cu: str, query: str, task_type: str = "qa") -> Iterator[str]:
        """Synthesize with streaming."""
        user_content = f"""The following is a summary of a long document:
{final_cu}

Question: {query}

Based on the summary, please provide a comprehensive answer to the question."""
        
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_content}
        ]
        
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0,
            stream=True
        )
        
        async for chunk in response:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content