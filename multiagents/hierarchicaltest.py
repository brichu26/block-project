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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()# --- 0. Helper Functions & Setup ---

# Download sentence tokenizer data
print("Downloading NLTK 'punkt' tokenizer data...")
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('punkt_tab')
nltk.data.find('tokenizers/punkt')


def split_into_sentences(text: str) -> List[str]:
    """Splits text into sentences using NLTK."""
    return nltk.sent_tokenize(text)

def get_tokenizer(model_name="gpt-4o-mini"):
    """Gets a tokenizer for token counting."""
    try:
        return tiktoken.encoding_for_model(model_name)
    except KeyError:
        print(f"Warning: Model {model_name} not found for tiktoken. Using cl100k_base.")
        return tiktoken.get_encoding("cl100k_base")

# Initialize tokenizer globally
tokenizer = get_tokenizer()

def count_tokens(text: str) -> int:
    """Counts tokens using the initialized tokenizer."""
    return len(tokenizer.encode(text))

def chunk_text_algorithm2(
    source_text: str,
    query: str,
    instruction: str,
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
        try:
            tokenizer = tiktoken.encoding_for_model("gpt-4o")
        except Exception:
            tokenizer = tiktoken.get_encoding("cl100k_base")
    
    # Calculate budget
    query_tokens = len(tokenizer.encode(query))
    instruction_tokens = len(tokenizer.encode(instruction))
    budget = window_size - query_tokens - instruction_tokens - buffer_tokens
    if budget <= 0:
        raise ValueError("Window size too small for query and instruction.")
    
    # Sentence splitting
    sentences = nltk.sent_tokenize(source_text)
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
    
# --- 2. Define Agent State ---
class HierarchicalAgentState(TypedDict):
    query: str
    chunks: List[str]
    # Store results from workers that deemed their chunk useful
    relevant_communication_units: Annotated[List[str], operator.add]
    final_answer: str
    worker_error: Annotated[List[bool], operator.add] # Flag if any worker fails

# --- 3. Define Nodes (Worker and Manager) ---

# LLM Initialization - Requires OPENAI_API_KEY in environment
try:
    # Worker LLMs (cheaper model)
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    judgement_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    # Manager LLM (higher-quality model)
    manager_llm = ChatOpenAI(model="gpt-4o", temperature=0)
except Exception as e:
    print(f"Failed to initialize OpenAI models. Ensure OPENAI_API_KEY is set. Error: {e}")
    exit()


# Define prompts (adapt from paper's Appendix Table 11/12 and add judgement logic)
WORKER_JUDGEMENT_PROMPT = """
Given the following text chunk and a query, determine if the chunk contains information potentially relevant to answering the query. Answer only with 'YES' or 'NO'.

Query: {query}
Text Chunk:
---
{chunk}
---

Contains Relevant Information (YES/NO):"""

WORKER_PROCESS_PROMPT = """
You are a worker agent analyzing a portion of a document. Read the following text chunk. Extract and summarize only the key information relevant to answering the original query. Be concise and focus only on relevance to the query. If no information is relevant, say 'No relevant information found in this chunk.'

Original Query: {query}
Text Chunk:
---
{chunk}
---

Relevant Information Summary:"""

MANAGER_PROMPT = """
You are the manager agent. You have received summaries from several worker agents who analyzed different parts of a long document based on the original query. Synthesize these summaries into a single, coherent, and comprehensive final answer to the query. If no relevant summaries were provided, state that you could not find the answer in the document.

Original Query: {query}

Worker Summaries (combine these pieces of information):
---
{summaries}
---

Final Answer:"""

def worker_node(state: HierarchicalAgentState, chunk_index: int):
    """Worker node: judges usefulness and processes chunk if useful."""
    query = state["query"]
    chunk = state["chunks"][chunk_index]
    print(f"--- Worker {chunk_index+1} Processing Chunk ({len(chunk)} chars) ---")

    communication_unit = ""
    is_useful = False
    try:
        # a) Judge Usefulness
        judgement_prompt_filled = WORKER_JUDGEMENT_PROMPT.format(query=query, chunk=chunk)
        judgement_response = judgement_llm.invoke([HumanMessage(content=judgement_prompt_filled)])
        judgement_text = judgement_response.content.strip().upper()
        is_useful = "YES" in judgement_text

        print(f"Worker {chunk_index+1} judged chunk useful: {is_useful} (Response: '{judgement_text}')")

        if is_useful:
            # b) Process Chunk if Useful
            process_prompt_filled = WORKER_PROCESS_PROMPT.format(query=query, chunk=chunk)
            process_response = llm.invoke([HumanMessage(content=process_prompt_filled)])
            communication_unit = process_response.content
            # Filter out empty or placeholder responses
            if "no relevant information" in communication_unit.lower() or len(communication_unit) < 10:
                 print(f"Worker {chunk_index+1} produced non-substantive CU: '{communication_unit[:100]}...' - Treating as not useful.")
                 is_useful = False # Override usefulness if processing yields nothing relevant
                 communication_unit = "" # Clear it
            else:
                 print(f"Worker {chunk_index+1} produced CU: {communication_unit[:150]}...") # Print snippet

    except Exception as e:
        print(f"!!! Error in Worker {chunk_index+1}: {e}")
        # Decide how to handle errors, e.g., return empty or raise exception
        # Returning empty lets the process continue without this worker's input
        return {"relevant_communication_units": [], "worker_error": [True]} # Flag error as list

    # Return the CU if deemed useful *after* processing, otherwise empty list
    return {"relevant_communication_units": [communication_unit] if is_useful else [], "worker_error": [False]}


def manager_node(state: HierarchicalAgentState):
    """Manager node: synthesizes relevant CUs into the final answer."""
    print("\n--- Manager Node Synthesizing ---")
    query = state["query"]
    relevant_cus = state["relevant_communication_units"]

    if not relevant_cus:
        print("Manager: No relevant information found by workers.")
        final_answer = "Based on the analysis of the document chunks, no relevant information was found to answer the query."
        return {"final_answer": final_answer}

    # Combine CUs for the manager prompt
    summaries_text = "\n\n---\n\n".join(relevant_cus)
    manager_prompt_filled = MANAGER_PROMPT.format(query=query, summaries=summaries_text)

    try:
        final_response = manager_llm.invoke([HumanMessage(content=manager_prompt_filled)])
        final_answer = final_response.content
        print(f"Manager produced final answer: {final_answer[:150]}...") # Print snippet
    except Exception as e:
         print(f"!!! Error in Manager Node: {e}")
         final_answer = "An error occurred while synthesizing the final answer."

    return {"final_answer": final_answer}

# --- 4. Build the Graph ---

# Get PDF Text (using the provided coa.pdf)
pdf_path = "coa.pdf" # Make sure this file is in the same directory
source_text = ""
try:
    reader = PdfReader(pdf_path)
    for page in reader.pages:
        source_text += page.extract_text() + "\n" # Add newline between pages
    print(f"Successfully read {len(reader.pages)} pages from {pdf_path}.")
except FileNotFoundError:
    print(f"Error: PDF file not found at '{pdf_path}'. Please place it in the script's directory.")
    exit()
except Exception as e:
    print(f"Error reading PDF: {e}")
    exit()

# Hardcoded Query and Parameters
query = "Summarize, in depth, how Chain of Agents works, and how multiple agents can be useful for context window management and reasoning."
worker_instruction = "You are a worker agent analyzing a portion of a document. Read the following text chunk. Extract and summarize only the key information relevant to answering the original query. Be concise and focus only on relevance to the query. If no information is relevant, say 'No relevant information found in this chunk." # Simplified instruction
# Use a smaller window for testing if needed, but paper used 8k
window_size = 8000

# Initial chunking
try:
    chunks = chunk_text_algorithm2(
        source_text=source_text,
        query=query,
        instruction=worker_instruction,
        window_size=window_size,
        tokenizer=tokenizer
    )
except ValueError as e:
    print(f"Error during chunking: {e}")
    exit()

if not chunks:
    print("Error: No chunks were generated. Check window size and input text.")
    exit()

# Workflow Graph
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
if worker_node_names:
    workflow.set_entry_point(worker_node_names[0])
for node_name in worker_node_names:
    workflow.add_edge(START, node_name)

# Manager node leads to the end
workflow.add_edge("manager", END)

# Compile the graph
app = workflow.compile()

# --- 5. Run the Graph ---
initial_state = {
    "query": query,
    "chunks": chunks,
    "relevant_communication_units": [], # Initialize as empty list
    "final_answer": "",
    "worker_error": [], # Initialize as empty list
}

print("\n--- Invoking LangGraph Workflow ---")
# Invoke the graph
# Add recursion limit for safety with potentially many chunks/nodes
final_state = app.invoke(initial_state, {"recursion_limit": 100}) # Adjust limit if needed

print("\n--- Final Answer ---")
print(final_state.get('final_answer', 'No final answer generated.'))

if any(final_state.get('worker_error', [])):
    print("\nWarning: One or more worker nodes encountered an error during execution.")