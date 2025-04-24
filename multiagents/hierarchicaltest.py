import os
import operator
from typing import TypedDict, List, Annotated
from functools import partial
import tiktoken
import nltk
from pypdf import PdfReader # To read the PDF text directly if needed
from dotenv import load_dotenv
# Langchain/LangGraph imports
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END, START

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

# --- 1. Chunking Function (Based on Paper's Algorithm 2) ---
def chunk_text(source_text: str, query: str, instruction: str, window_size: int) -> List[str]:
    """Chunks the source text based on Algorithm 2."""
    sentences = split_into_sentences(source_text)

    # Calculate budget
    query_tokens = count_tokens(query)
    instruction_tokens = count_tokens(instruction)
    # Add a buffer for safety margin with tokenization differences and prompts
    budget = window_size - query_tokens - instruction_tokens - 100

    if budget <= 0:
        raise ValueError("Window size too small for query and instruction.")

    chunks = []
    current_chunk = ""
    current_chunk_tokens = 0

    for sentence in sentences:
        sentence_tokens = count_tokens(sentence)
        tokens_with_space = sentence_tokens + (1 if current_chunk else 0) # Approx token for space

        # Check if adding the sentence exceeds budget OR if sentence itself is too long
        if sentence_tokens > budget:
             print(f"Warning: Skipping sentence longer than budget: '{sentence[:100]}...'")
             continue # Skip sentence that's too long on its own

        if current_chunk_tokens + tokens_with_space > budget:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence
            current_chunk_tokens = sentence_tokens
        else:
            # Add sentence to current chunk
            if current_chunk:
                current_chunk += " " + sentence
                current_chunk_tokens += tokens_with_space
            else:
                current_chunk = sentence
                current_chunk_tokens = sentence_tokens


    # Add the last chunk if it's not empty
    if current_chunk:
        chunks.append(current_chunk.strip())

    print(f"Chunked text into {len(chunks)} chunks.")
    # Optional: Print token count per chunk for verification
    # for i, chunk in enumerate(chunks):
    #    print(f"Chunk {i+1} tokens: {count_tokens(chunk)}")
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
    chunks = chunk_text(source_text, query, worker_instruction, window_size)
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