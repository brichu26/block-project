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
import re
from pypdf import PdfReader

# --- Constants ---
DEFAULT_EMBEDDING_MODEL = "text-embedding-ada-002"
DEFAULT_TOKENIZER_MODEL = "gpt-4o"
DEFAULT_CHROMA_PERSIST_DIR = "./chroma_db"
DEFAULT_CHUNK_BUFFER_TOKENS = 100
# -----------------

# Load environment variables (e.g., OPENAI_API_KEY)
load_dotenv() # Looks for .env in the current or parent directories

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Ensure nltk punkt tokenizer is downloaded (run once)
def setup_nltk():
    """Downloads necessary NLTK data if not already present."""
    try:
        nltk.data.find('tokenizers/punkt')
        logger.info("NLTK 'punkt' tokenizer data found.")
    except LookupError:
        logger.info("Downloading NLTK 'punkt' tokenizer data...")
        nltk.download('punkt', quiet=True)

setup_nltk()

def split_into_sentences(text: str) -> List[str]:
    """Splits text into sentences using NLTK."""
    return nltk.sent_tokenize(text)

def get_tokenizer(model_name=DEFAULT_TOKENIZER_MODEL):
    """Gets a tokenizer for token counting."""
    try:
        return tiktoken.encoding_for_model(model_name)
    except KeyError:
        logger.warning(f"Model {model_name} not found for tiktoken. Using cl100k_base.")
        return tiktoken.get_encoding("cl100k_base")

def count_tokens(text: str, tokenizer=None) -> int:
    """Counts tokens using the provided or default tokenizer."""
    if tokenizer is None:
        tokenizer = get_tokenizer()
    return len(tokenizer.encode(text))

def chunk_text_algorithm2(
    source_text: str,
    query: str,
    instruction: str,
    window_size: int,
    tokenizer=None,
    buffer_tokens: int = DEFAULT_CHUNK_BUFFER_TOKENS
) -> List[str]:
    """
    Chunk text according to Algorithm 2 from the Chain-of-Agents paper appendix.
    Args:
        source_text: The input document text
        query: The user query
        instruction: The worker instruction string
        window_size: The agent's window size (max tokens)
        tokenizer: A tiktoken encoding instance (if None, will use cl100k_base)
        buffer_tokens: Optional safety buffer for prompt overhead
    Returns:
        List[str]: List of text chunks
    """
    if tokenizer is None:
        tokenizer = get_tokenizer()
    query_tokens = len(tokenizer.encode(query))
    instruction_tokens = len(tokenizer.encode(instruction))
    budget = window_size - query_tokens - instruction_tokens - buffer_tokens
    if budget <= 0:
        raise ValueError("Window size too small for query and instruction.")
    try:
        sentences = nltk.sent_tokenize(source_text)
    except Exception as e:
        logger.error(f"NLTK sentence tokenization failed: {e}. Ensure 'punkt' data is downloaded.")
        sentences = source_text.split('\n')
        logger.warning("Falling back to newline splitting for chunking.")
    chunks = []
    current_chunk = ""
    current_tokens = 0
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        sentence_tokens = len(tokenizer.encode(sentence))
        if current_tokens + sentence_tokens > budget and current_chunk:
            chunks.append(current_chunk.strip())
            current_chunk = sentence
            current_tokens = sentence_tokens
        else:
            if current_chunk:
                current_chunk += " " + sentence
                current_tokens += sentence_tokens
            else:
                current_chunk = sentence
                current_tokens = sentence_tokens
    if current_chunk:
        chunks.append(current_chunk.strip())
    logger.info(f"[Algorithm2] Chunked text into {len(chunks)} chunks (budget={budget})")
    return chunks

def read_pdf(pdf_path: str) -> str:
    """Read text content from a PDF file."""
    try:
        text = []
        reader = PdfReader(pdf_path)
        logger.info(f"Processing PDF with {len(reader.pages)} pages")
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text.append(page_text)
        return "\n".join(filter(None, text))
    except Exception as e:
        logger.error(f"Error reading PDF: {str(e)}")
        raise

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

# --- Vector DB / RAG Related Utilities --- 

def initialize_embeddings(model_name=DEFAULT_EMBEDDING_MODEL):
    """Initializes the OpenAI embeddings model.

    Args:
        model_name (str): The name of the OpenAI embedding model to use.

    Returns:
        OpenAIEmbeddings: The initialized embeddings instance, or None if API key is missing.
    """
    # Check if API key is available (Langchain checks this internally, but explicit check is clearer)
    if not os.getenv("OPENAI_API_KEY"):
        logger.error("OPENAI_API_KEY environment variable not set.")
        return None
    try:
        logger.info(f"Initializing OpenAI embeddings model: {model_name}")
        embeddings_instance = OpenAIEmbeddings(model=model_name)
        logger.info("Embeddings model initialized successfully.")
        return embeddings_instance
    except Exception as e:
        logger.error(f"Failed to initialize OpenAI embeddings: {e}")
        return None

def initialize_vectorstore(embeddings_instance: OpenAIEmbeddings, persist_directory: str = DEFAULT_CHROMA_PERSIST_DIR):
    """Initializes the Chroma vector store, loading if it exists.

    Args:
        embeddings_instance (OpenAIEmbeddings): The initialized embeddings model.
        persist_directory (str): The directory to persist the vector store.

    Returns:
        Chroma: The initialized vector store instance, or None on failure.
    """
    if not embeddings_instance:
        logger.error("Cannot initialize vectorstore without a valid embeddings instance.")
        return None

    logger.info(f"Initializing Chroma vector store in directory: {persist_directory}")
    # Check if the directory exists
    if os.path.exists(persist_directory):
        logger.info(f"Loading existing Chroma vector store from: {persist_directory}")
        local_vectorstore = Chroma(persist_directory=persist_directory, embedding_function=embeddings_instance)
    else:
        logger.info(f"Creating new Chroma vector store at: {persist_directory}")
        # Initialize empty and it will be saved on first add with persist_directory set
        local_vectorstore = Chroma(persist_directory=persist_directory, embedding_function=embeddings_instance)
        
    logger.info(f"Chroma vector store initialized/loaded. Collection name: {local_vectorstore._collection.name}")
    return local_vectorstore # Return the created instance

def add_chunks_to_vectorstore(chunks: List[str], vectorstore_instance: Chroma, file_path: str = "N/A") -> List[str]:
    """Adds text chunks to the specified vectorstore instance.

    Args:
        chunks: List of text chunks to add.
        vectorstore_instance: The initialized Chroma instance.
        file_path: The original file path the chunks belong to.

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

def merge_worker_outputs(existing, updates):
     result = dict(existing)
     if isinstance(updates, list):
         for update in updates:
             if isinstance(update, dict):
                 result.update(update)
     elif isinstance(updates, dict):
         result.update(updates)
     return result

def escape_curly_braces(text: str) -> str:
     return text.replace('{', '{{').replace('}', '}}')
