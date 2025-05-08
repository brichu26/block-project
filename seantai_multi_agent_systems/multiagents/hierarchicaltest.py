import os
import operator
from typing import TypedDict, List, Annotated
from functools import partial
import tiktoken
import nltk
from pypdf import PdfReader # To read the PDF text directly if needed
from dotenv import load_dotenv
import logging

# Langchain/LangGraph imports
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END, START

# --- Configuration Constants ---
PDF_PATH = "coa.pdf" # Path to the source PDF document
QUERY = "Summarize, in depth, how Chain of Agents works, and how multiple agents can be useful for context window management and reasoning."
WORKER_MODEL = "gpt-4o-mini"
MANAGER_MODEL = "gpt-4o"
TOKENIZER_MODEL = "gpt-4o-mini" # Model used for token counting during chunking
CHUNK_WINDOW_SIZE = 8000 # Max tokens for a worker's context (including prompt)
CHUNK_BUFFER_TOKENS = 100 # Buffer for prompt overhead in chunking
LANGGRAPH_RECURSION_LIMIT = 150 # Safety limit for graph execution

# Prompts
WORKER_JUDGE_PROMPT = """You are an expert relevance assessor.
Read the user query and the provided text chunk.
Determine if the text chunk contains any information DIRECTLY RELEVANT to answering the query.
Answer ONLY with 'yes' or 'no'.

Query: {query}
Text Chunk:
---
{chunk}
---

Relevant? (yes/no):"""

WORKER_EXTRACT_PROMPT = """You are a worker agent analyzing a portion of a document.
Read the following text chunk and the original user query.
Extract and summarize only the key information relevant to answering the original query.
Be concise and focus only on relevance to the query. Ensure your summary is self-contained.
If no information is relevant, state 'No relevant information found in this chunk.'

Original Query: {query}
Text Chunk:
---
{chunk}
---

Relevant Information Summary:"""

MANAGER_PROMPT = """You are the manager agent.
You have received summaries from several worker agents who analyzed different parts of a long document based on the original query.
Synthesize these summaries into a single, coherent, and comprehensive final answer to the query.
If no relevant summaries were provided, state that you could not find the answer in the document.

Original Query: {query}

Worker Summaries (combine these pieces of information):
---
{summaries}
---

Final Answer:"""
# --- End Configuration ---

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load Environment Variables (for API keys)
load_dotenv()

# --- 0. Helper Functions & Setup ---

def setup_nltk():
    """Downloads necessary NLTK data if not already present."""
    try:
        nltk.data.find('tokenizers/punkt')
        logger.info("NLTK 'punkt' tokenizer data found.")
    except nltk.downloader.DownloadError:
        logger.info("Downloading NLTK 'punkt' tokenizer data...")
        nltk.download('punkt', quiet=True)

# Call NLTK setup once
setup_nltk()

def split_into_sentences(text: str) -> List[str]:
    """Splits text into sentences using NLTK."""
    return nltk.sent_tokenize(text)

def get_tokenizer(model_name="gpt-4o-mini"):
    """Gets a tokenizer for token counting."""
    try:
        return tiktoken.encoding_for_model(model_name)
    except KeyError:
        logger.warning(f"Model {model_name} not found for tiktoken. Using cl100k_base.")
        return tiktoken.get_encoding("cl100k_base")

# Initialize tokenizer globally using constant
tokenizer = get_tokenizer(TOKENIZER_MODEL)

def count_tokens(text: str) -> int:
    """Counts tokens using the initialized tokenizer."""
    return len(tokenizer.encode(text))

def chunk_text_algorithm2(
    source_text: str,
    query: str,
    instruction: str, # Pass the specific instruction used (judge or extract)
    window_size: int,
    tokenizer=None,
    buffer_tokens: int = 100
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
        # Default to the global tokenizer if none provided
        tokenizer = get_tokenizer(TOKENIZER_MODEL)

    # Calculate budget based on the passed instruction
    query_tokens = len(tokenizer.encode(query))
    instruction_tokens = len(tokenizer.encode(instruction))
    budget = window_size - query_tokens - instruction_tokens - buffer_tokens
    if budget <= 0:
        raise ValueError("Window size too small for query and instruction.")
    
    # Sentence splitting
    try:
        sentences = nltk.sent_tokenize(source_text)
    except Exception as e:
        logger.error(f"NLTK sentence tokenization failed: {e}. Ensure 'punkt' data is downloaded.")
        # Fallback to simple newline splitting if NLTK fails
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
        # If adding this sentence would exceed the budget, start a new chunk
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
    
# --- 1. Define Agent State ---
class HierarchicalAgentState(TypedDict):
    query: str
    source_text: str # Add source text to state if needed by nodes
    chunks: List[str]
    relevant_communication_units: Annotated[List[str], operator.add]
    final_answer: str
    worker_error: Annotated[List[bool], operator.add] # Flag if any worker fails

# --- 2. Define LLM Clients ---
try:
    # Worker LLMs (potentially cheaper model)
    worker_llm = ChatOpenAI(model=WORKER_MODEL, temperature=0)
    # Manager LLM (potentially more capable model)
    manager_llm = ChatOpenAI(model=MANAGER_MODEL, temperature=0)
    logger.info(f"Initialized LLMs: Worker ({WORKER_MODEL}), Manager ({MANAGER_MODEL})")
except Exception as e:
    logger.error(f"Failed to initialize OpenAI LLMs: {e}. Check API key and model names.")
    exit()

# --- 3. Define Nodes (Worker and Manager) ---

def worker_node(state: HierarchicalAgentState, chunk_index: int):
    """Worker node: judges usefulness and processes chunk if useful."""
    query = state["query"]
    chunks = state["chunks"]
    chunk = chunks[chunk_index]
    node_name = f"Worker {chunk_index + 1}/{len(chunks)}"
    logger.info(f"--- {node_name}: Processing Chunk --- ")

    is_relevant = False
    summary = ""
    error_occurred = False

    try:
        # 1. Judge Relevance
        judge_prompt = WORKER_JUDGE_PROMPT.format(query=query, chunk=chunk)
        judge_response = worker_llm.invoke([HumanMessage(content=judge_prompt)])
        relevance_decision = judge_response.content.strip().lower()
        logger.info(f"{node_name}: Relevance decision = '{relevance_decision}'")

        if 'yes' in relevance_decision:
            is_relevant = True
            # 2. Extract/Summarize if Relevant
            extract_prompt = WORKER_EXTRACT_PROMPT.format(query=query, chunk=chunk)
            summary_response = worker_llm.invoke([HumanMessage(content=extract_prompt)])
            summary = summary_response.content.strip()
            # Avoid adding the placeholder if extraction actually failed
            if "No relevant information found" not in summary and summary:
                logger.info(f"{node_name}: Extracted relevant summary (length {len(summary)})." )
            else:
                logger.info(f"{node_name}: Judged relevant, but found no specific info to extract.")
                is_relevant = False # Treat as not relevant if summary is empty/placeholder
                summary = "" # Ensure empty summary if not truly relevant
        else:
            logger.info(f"{node_name}: Chunk deemed not relevant.")

    except Exception as e:
        logger.error(f"!!! Error in {node_name}: {e}")
        error_occurred = True
        # Optionally return a specific error message or placeholder
        summary = f"[Error in {node_name}]"

    # Return results: only add summary if relevant, always add error status
    return {
        "relevant_communication_units": [summary] if is_relevant else [],
        "worker_error": [error_occurred]
    }

def manager_node(state: HierarchicalAgentState):
    """Manager node: synthesizes relevant CUs into the final answer."""
    logger.info("--- Manager Node: Synthesizing Final Answer ---")
    query = state["query"]
    relevant_cus = state.get("relevant_communication_units", []) # Use .get for safety

    if not relevant_cus:
        logger.warning("Manager: No relevant information found by workers.")
        final_answer = "Based on the analysis of the document chunks, no relevant information was found to answer the query."
        return {"final_answer": final_answer}

    # Combine CUs for the manager prompt
    summaries_text = "\n\n---\n\n".join(relevant_cus)
    manager_prompt_filled = MANAGER_PROMPT.format(query=query, summaries=summaries_text)
    logger.info(f"Manager: Synthesizing {len(relevant_cus)} summaries.")

    try:
        final_response = manager_llm.invoke([HumanMessage(content=manager_prompt_filled)])
        final_answer = final_response.content
        logger.info(f"Manager: Produced final answer (length {len(final_answer)}).")
    except Exception as e:
         logger.error(f"!!! Error in Manager Node: {e}")
         final_answer = "An error occurred while synthesizing the final answer."

    return {"final_answer": final_answer}

# --- 4. Main Execution Logic ---
def main():
    """Loads data, builds and runs the LangGraph workflow."""
    logger.info("--- Starting Hierarchical Agent Workflow --- ")

    # Get PDF Text
    source_text = ""
    try:
        reader = PdfReader(PDF_PATH)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text: # Add text only if extraction was successful
                 source_text += page_text + "\n"
        logger.info(f"Successfully read {len(reader.pages)} pages from '{PDF_PATH}'. Total chars: {len(source_text)}")
    except FileNotFoundError:
        logger.error(f"Error: PDF file not found at '{PDF_PATH}'. Please place it in the script's directory.")
        return # Exit main function
    except Exception as e:
        logger.error(f"Error reading PDF '{PDF_PATH}': {e}")
        return # Exit main function

    if not source_text.strip():
        logger.error("Error: Extracted text from PDF is empty.")
        return

    # Initial chunking using the 'judge' prompt length for budget calculation initially
    # (Assuming judge prompt is representative or we accept slight overflow on extract)
    logger.info(f"Chunking text for query: '{QUERY[:50]}...' with window size {CHUNK_WINDOW_SIZE}")
    try:
        # We pass WORKER_JUDGE_PROMPT here for budget calculation, as it's likely the first step.
        # Alternatively, calculate budget based on the longer prompt if guarantees are needed.
        chunks = chunk_text_algorithm2(
            source_text=source_text,
            query=QUERY,
            instruction=WORKER_JUDGE_PROMPT, # Use judge prompt for budget estimate
            window_size=CHUNK_WINDOW_SIZE,
            tokenizer=tokenizer,
            buffer_tokens=CHUNK_BUFFER_TOKENS
        )
    except ValueError as e:
        logger.error(f"Error during chunking: {e}")
        return
    except Exception as e: # Catch unexpected chunking errors
        logger.error(f"Unexpected error during chunking: {e}")
        return

    if not chunks:
        logger.error("Error: No chunks were generated. Check window size, buffer, and input text.")
        return

    logger.info(f"Generated {len(chunks)} chunks for processing.")

    # Build Workflow Graph
    workflow = StateGraph(HierarchicalAgentState)

    # Add Manager Node
    workflow.add_node("manager", manager_node)

    # Dynamically add worker nodes based on the number of chunks
    num_workers = len(chunks)
    worker_node_names = []
    for i in range(num_workers):
        node_name = f"worker_{i+1}"
        worker_node_names.append(node_name)
        # Use functools.partial to pass the chunk_index to the node function
        workflow.add_node(node_name, partial(worker_node, chunk_index=i))
        # Workers all lead to the manager node
        workflow.add_edge(node_name, "manager")

    # Set the entry point - Workers run in parallel after START
    # Ensure entry point exists before setting
    if worker_node_names:
        # Add edges from START to all workers
        for node_name in worker_node_names:
             workflow.add_edge(START, node_name)
        # Setting the entry point isn't strictly necessary when START branches to all workers,
        # but it can be clearer. LangGraph handles the parallel start.
        # workflow.set_entry_point(worker_node_names[0]) # Optional: officially marks one as first
    else:
        logger.error("No worker nodes created, cannot build graph.")
        return

    # Manager node leads to the end
    workflow.add_edge("manager", END)

    # Compile the graph
    try:
        app = workflow.compile()
        logger.info("LangGraph workflow compiled successfully.")
    except Exception as e:
        logger.error(f"Failed to compile LangGraph workflow: {e}")
        return

    # --- 5. Run the Graph ---
    initial_state = {
        "query": QUERY,
        "source_text": source_text, # Pass source_text if needed later
        "chunks": chunks,
        "relevant_communication_units": [], # Initialize as empty list
        "final_answer": "",
        "worker_error": [], # Initialize as empty list
    }

    logger.info("--- Invoking LangGraph Workflow ---")
    final_state = None
    try:
        # Invoke the graph
        final_state = app.invoke(initial_state, {"recursion_limit": LANGGRAPH_RECURSION_LIMIT})
        logger.info("--- LangGraph Workflow Invocation Complete ---")
    except Exception as e:
        logger.error(f"LangGraph invocation failed: {e}")
        # Attempt to log partial state if available
        if isinstance(initial_state, dict):
             logger.error(f"State at time of error (partial): { {k: v[:100] + '...' if isinstance(v, str) and len(v) > 100 else v for k,v in initial_state.items()} }")


    # Print the final result
    print("\n" + "="*20 + " FINAL ANSWER " + "="*20)
    if final_state and 'final_answer' in final_state:
        print(final_state['final_answer'])
    else:
        print("No final answer was generated or an error occurred during execution.")

    # Report any worker errors
    if final_state and any(final_state.get('worker_error', [])):
        error_indices = [i+1 for i, err in enumerate(final_state['worker_error']) if err]
        logger.warning(f"\nWarning: Worker node(s) {error_indices} encountered errors during execution.")
    elif final_state:
         logger.info("All worker nodes completed without reported errors.")

if __name__ == "__main__":
    main()