from typing import List, Tuple
import logging
import fitz  # PyMuPDF
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def read_pdf(pdf_path: str) -> str:
    """
    Read text content from a PDF file.
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        str: Extracted text from the PDF
    """
    try:
        text = []
        with fitz.open(pdf_path) as doc:
            logger.info(f"Processing PDF with {len(doc)} pages")
            for page in doc:
                text.append(page.get_text())
        
        return "\n".join(filter(None, text))  # Filter out empty strings
    except Exception as e:
        logger.error(f"Error reading PDF: {str(e)}")
        raise

def split_into_chunks(text: str, chunk_size: int) -> List[str]:
    """
    Split text into chunks based on sentence boundaries.
    
    Args:
        text: The input text to split
        chunk_size: Maximum number of tokens per chunk
        
    Returns:
        List[str]: List of text chunks
    """
    # Split by paragraphs first to maintain context
    paragraphs = text.split('\n\n')
    chunks = []
    current_chunk = []
    current_size = 0
    
    for paragraph in paragraphs:
        paragraph = paragraph.strip()
        if not paragraph:
            continue
            
        # Split paragraph into sentences
        sentences = re.split(r'(?<=[.!?])\s+', paragraph)
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            # Approximate token count (words + punctuation)
            sentence_size = len(re.findall(r'\b\w+\b', sentence)) + len(re.findall(r'[.,!?;:]', sentence))
            
            # If adding this sentence exceeds chunk size, save current chunk and start new one
            if current_size + sentence_size > chunk_size and current_chunk:
                chunks.append(' '.join(current_chunk))
                current_chunk = [sentence]
                current_size = sentence_size
            else:
                current_chunk.append(sentence)
                current_size += sentence_size
    
    # Add any remaining text
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    logger.info(f"Split text into {len(chunks)} chunks")
    return chunks

def get_task_prompts(task_type: str = "qa") -> Tuple[str, str]:
    """
    Get task-specific system prompts for worker and manager agents.
    
    Args:
        task_type: Type of task (qa, summarization, code)
        
    Returns:
        tuple[str, str]: (worker_prompt, manager_prompt)
    """
    if task_type == "qa":
        worker_prompt = """You are a worker agent in a Chain-of-Agents system analyzing a document to answer a question.
Your task is to identify key information related to the user's query and create a Communication Unit (CU) that contains:
1. Relevant evidence from your chunk of text
2. Any reasoning steps that connect to previous evidence
3. Important context that might be needed to answer the question

Focus on being comprehensive but concise. Your output will be used by other agents to continue the reasoning chain."""

        manager_prompt = """You are a manager agent in a Chain-of-Agents system.
Your task is to synthesize the final Communication Unit (CU) into a coherent, comprehensive answer to the user's query.
The CU contains accumulated evidence and reasoning from multiple worker agents who have processed different parts of a long document.
Provide a direct, factual answer based solely on the information in the CU."""

    elif task_type == "summarization":
        worker_prompt = """You are a worker agent in a Chain-of-Agents system analyzing a document to create a summary.
Your task is to create a Communication Unit (CU) that:
1. Summarizes the key points from your chunk of text
2. Integrates with the summary from previous chunks
3. Maintains a coherent narrative flow

Focus on capturing the most important information while maintaining continuity with previous content."""

        manager_prompt = """You are a manager agent in a Chain-of-Agents system.
Your task is to synthesize the final Communication Unit (CU) into a coherent, comprehensive summary.
The CU contains a progressive summary built by multiple worker agents who have processed different parts of a long document.
Create a well-structured final summary that captures the key points and maintains a logical flow."""

    elif task_type == "code":
        worker_prompt = """You are a worker agent in a Chain-of-Agents system analyzing code to complete or explain it.
Your task is to create a Communication Unit (CU) that:
1. Identifies key functions, classes, and variables from your chunk
2. Builds on the understanding from previous chunks
3. Notes any patterns or dependencies relevant to the query

Focus on technical accuracy and maintaining the logical structure of the code."""

        manager_prompt = """You are a manager agent in a Chain-of-Agents system.
Your task is to synthesize the final Communication Unit (CU) into a coherent code completion or explanation.
The CU contains accumulated code understanding from multiple worker agents who have processed different parts of a codebase.
Provide a technically accurate and well-structured response that directly addresses the query."""

    else:
        # Default generic prompts
        worker_prompt = """You are a worker agent in a Chain-of-Agents system.
Your task is to analyze your portion of a document and create a Communication Unit (CU) that builds on previous information.
Identify key information related to the user's query and provide clear, concise analysis."""

        manager_prompt = """You are a manager agent in a Chain-of-Agents system.
Your task is to synthesize the final Communication Unit (CU) into a coherent, comprehensive response.
The CU contains accumulated information from multiple worker agents who have processed different parts of a long document.
Provide a direct response that addresses the user's query."""

    return worker_prompt, manager_prompt
