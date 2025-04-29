# Utility functions for codebase reading, chunking, vector DB interactions
# print("masRAG utils.py loaded") # Comment out print statement
# We will populate this with functions from the plan

import os
import nltk
import tiktoken
from typing import List, Dict
import logging
# Added imports for RAG/VectorDB
import uuid
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document

# Load environment variables (e.g., OPENAI_API_KEY)
load_dotenv() # Looks for .env in the current or parent directories

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Ensure nltk punkt tokenizer is downloaded (run once)
def download_nltk_punkt():
    try:
        nltk.data.find('tokenizers/punkt')
        logger.info("NLTK 'punkt' tokenizer already downloaded.")
    except nltk.downloader.DownloadError:
        logger.info("Downloading NLTK 'punkt' tokenizer data...")
        nltk.download('punkt')
        logger.info("NLTK 'punkt' tokenizer downloaded successfully.")

download_nltk_punkt() # Ensure it's available when module is loaded


def read_codebase(directory: str, extensions: List[str] = None) -> Dict[str, str]:
    """Reads all files with specified extensions from a directory and its subdirectories.

    Args:
        directory: The path to the root directory of the codebase.
        extensions: A list of file extensions to include (e.g., ['.py', '.js']). 
                    If None, reads all files.

    Returns:
        A dictionary mapping absolute file paths to their content.
    """
    logger.info(f"Reading codebase from: {directory} for extensions: {extensions}")
    codebase_content = {}
    # Convert extensions to lowercase and add dot if missing for comparison
    processed_extensions = None
    if extensions:
        processed_extensions = {ext.lower() if ext.startswith('.') else '.' + ext.lower() for ext in extensions}

    for root, _, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            should_include = False
            if processed_extensions:
                _, ext = os.path.splitext(file)
                if ext.lower() in processed_extensions:
                    should_include = True
            else:
                should_include = True # Include all files if no extensions specified
                
            if should_include:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        codebase_content[file_path] = f.read()
                        # logger.debug(f"Read file: {file_path}")
                except Exception as e:
                    logger.warning(f"Could not read file {file_path}: {e}")
                    
    total_chars = sum(len(content) for content in codebase_content.values())
    logger.info(f"Finished reading codebase. Read {len(codebase_content)} files. Total length: {total_chars} characters.")
    return codebase_content


def get_tokenizer(model_name="gpt-4o"):
    """Gets a tokenizer for token counting."""
    try:
        return tiktoken.encoding_for_model(model_name)
    except KeyError:
        logger.warning(f"Warning: Model {model_name} not found for tiktoken. Using cl100k_base.")
        return tiktoken.get_encoding("cl100k_base")

# Initialize tokenizer globally or pass it around
# Global initialization might be simpler for this script structure
tokenizer = get_tokenizer()

def count_tokens(text: str) -> int:
    """Counts tokens using the initialized tokenizer."""
    return len(tokenizer.encode(text))

# Provided Chunking Algorithm (integrated into utils.py)
def chunk_text_algorithm2(
    source_text: str,
    query: str,
    instruction: str,
    window_size: int,
    # tokenizer=None, # Use the globally defined tokenizer
    buffer_tokens: int = 100
) -> List[str]:
    """
    Chunk text according to Algorithm 2 from the Chain-of-Agents paper appendix.
    Args:
        source_text: The input document text
        query: The user query
        instruction: The worker instruction string
        window_size: The agent's window size (max tokens)
        buffer_tokens: Optional safety buffer for prompt overhead
    Returns:
        List[str]: List of text chunks
    """
    # Use the globally defined tokenizer from this module
    global tokenizer
    # if tokenizer is None:
    #     try:
    #         tokenizer = tiktoken.encoding_for_model("gpt-4o")
    #     except Exception:
    #         tokenizer = tiktoken.get_encoding("cl100k_base")

    # Calculate budget
    query_tokens = count_tokens(query)
    instruction_tokens = count_tokens(instruction)
    budget = window_size - query_tokens - instruction_tokens - buffer_tokens
    if budget <= 0:
        logger.warning(f"Window size {window_size} too small for query ({query_tokens}) and instruction ({instruction_tokens}). Budget is {budget}. Adjusting budget to 50 tokens.")
        budget = 50 # Ensure minimum chunk size possible

    # Code-aware chunking by lines
    lines = source_text.splitlines()
    chunks = []
    current_chunk = ""
    current_tokens = 0

    for line in lines:
        line = line.rstrip()
        if not line:
            continue
        line_tokens = count_tokens(line)
        if current_tokens > 0 and (current_tokens + line_tokens) > budget:
            chunks.append(current_chunk.rstrip())
            current_chunk = line + "\n"
            current_tokens = line_tokens
        elif line_tokens > budget:
            if current_chunk:
                chunks.append(current_chunk.rstrip())
            chunks.append(line)
            current_chunk = ""
            current_tokens = 0
        else:
            current_chunk += line + "\n"
            current_tokens += line_tokens + count_tokens("\n")

    if current_chunk.strip():
        chunks.append(current_chunk.rstrip())

    logger.info(f"[Algorithm2] Chunked text into {len(chunks)} chunks (budget={budget} tokens per chunk)")
    return chunks


# --- Vector DB / RAG Related Utilities --- 

# Global variable for embeddings
embeddings = None

def initialize_embeddings(model_name="text-embedding-ada-002"):
    """Initializes the OpenAI embeddings model."""
    global embeddings
    try:
        # Ensure API key is available
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY not found in environment variables. Please ensure it's set in your .env file.")
        embeddings = OpenAIEmbeddings(model=model_name)
        logger.info(f"Initialized OpenAI Embeddings with model: {model_name}")
        return embeddings
    except Exception as e:
        logger.error(f"Failed to initialize OpenAI Embeddings: {e}")
        raise

def initialize_vectorstore(persist_directory: str = "./chroma_db") -> Chroma:
    """Initializes the Chroma vector store, loading if it exists.
    
    Returns:
        Chroma: The initialized vector store instance.
    """
    local_vectorstore = None # Use a local variable
    if embeddings is None:
        initialize_embeddings() # Ensure embeddings are initialized first
    
    # Check if persist_directory exists
    if os.path.exists(persist_directory):
        logger.info(f"Loading existing Chroma vector store from: {persist_directory}")
        local_vectorstore = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
    else:
        logger.info(f"Creating new Chroma vector store at: {persist_directory}")
        # Initialize empty and it will be saved on first add with persist_directory set
        local_vectorstore = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
        # We need to add at least one document for the directory to be created and persisted correctly initially
        # Add a dummy doc to ensure persistence directory is created immediately
        # vectorstore.add_documents([Document(page_content="Initial document", metadata={"dummy": True})])
        # vectorstore.delete(ids=vectorstore.get(where={"dummy": True})['ids']) # Clean up dummy - maybe better to just let add_chunks handle first save?
        # Let's rely on the first call to add_chunks_to_vectorstore to create and persist.
        
    logger.info(f"Chroma vector store initialized. Collection name: {local_vectorstore._collection.name}")
    return local_vectorstore # Return the created instance

def add_chunks_to_vectorstore(chunks: List[str], vectorstore_instance: Chroma, file_path: str = "N/A", initial_add: bool = False) -> List[str]:
    """Adds text chunks to the specified vectorstore instance.
    
    Args:
        chunks: List of text chunks to add.
        vectorstore_instance: The initialized Chroma instance.
        file_path: The original file path the chunks belong to.
        initial_add: Flag indicating if this is the first add operation (not currently used).
        
    Returns:
        List of generated chunk IDs.
    """
    if vectorstore_instance is None: # Explicitly check for None
        logger.error("Vectorstore is not initialized.")
        return []
        
    docs = []
    chunk_ids = []
    for i, chunk in enumerate(chunks):
        chunk_id = str(uuid.uuid4()) # Generate unique ID for each chunk
        metadata = {
            'source': file_path, # Store the original file path
            'chunk_index': i,    # Store the index within the file (optional)
            'chunk_id': chunk_id # Store the unique ID itself (redundant but potentially useful)
        }
        docs.append(Document(page_content=chunk, metadata=metadata))
        chunk_ids.append(chunk_id)

    if docs:
        logger.info(f"Adding {len(docs)} chunks to vectorstore from {file_path}...")
        try:
            vectorstore_instance.add_documents(docs)
            # Chroma with persist_directory should save automatically on add/update.
            # If not using persist_directory, explicit persistence might be needed.
            # vectorstore_instance.persist() # Only needed if persist_directory wasn't set at init or auto-persist fails
            logger.info(f"Successfully added {len(docs)} chunks.")
        except Exception as e:
            logger.error(f"Error adding documents to Chroma: {e}")
            # Depending on the error, you might want to retry or handle differently
            return [] # Return empty list on failure
    else:
        logger.info("No documents to add.")

    return chunk_ids



# Example Usage (Can be uncommented for testing)
# if __name__ == "__main__":
#     print("Testing utils...")
#     test_folder = "../test_codebase" # Assuming utils.py is in masRAG
#     code_text = read_codebase(test_folder)
#     print(f"Read codebase text (first 500 chars):\n{code_text[:500]}...")

#     test_query = "Describe the main function"
#     test_instruction = "Analyze the provided code snippet"
#     test_window_size = 100 # Use a small window for testing chunking

#     chunks = chunk_text_algorithm2(
#         source_text=code_text,
#         query=test_query,
#         instruction=test_instruction,
#         window_size=test_window_size
#     )

#     print(f"\nGenerated {len(chunks)} chunks with window size {test_window_size}:")
#     for i, chunk in enumerate(chunks):
#         print(f"--- Chunk {i+1} (Tokens: {count_tokens(chunk)}) ---")
#         print(chunk)
#         print("---")
