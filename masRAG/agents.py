# Agent node functions for the LangGraph
# print("masRAG agents.py loaded") # Comment out print statement
# We will populate this with manager, worker, and RAG node functions

import os
import logging
from typing import Dict, Any, List
from langgraph.graph import END

# Project specific imports
from .state import AgentState # Import the state definition from the new file
from .utils import ( 
    initialize_vectorstore, 
    chunk_text_algorithm2, 
    add_chunks_to_vectorstore,
    read_codebase, 
)
# LangChain imports (for eventual LLM integration)
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Constants and Configuration ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found in environment variables.")

DEFAULT_MODEL_NAME = "gpt-4o"
MANAGER_MODEL_NAME = "gpt-4o"
WORKER_MODEL_NAME = "gpt-4o" # Or a smaller/faster model if desired
TEMPERATURE = 0.1
MAX_TOKENS = 4000

# --- Global Variables (consider encapsulating in a class later) ---
worker_llm = None
manager_llm = None

# --- Initialization Function ---
def initialize_models():
    """Initializes the LLM models."""
    global worker_llm, manager_llm
    try:
        worker_llm = ChatOpenAI(
            model=WORKER_MODEL_NAME, 
            temperature=TEMPERATURE, 
            max_tokens=MAX_TOKENS,
            api_key=OPENAI_API_KEY
        )
        manager_llm = ChatOpenAI(
            model=MANAGER_MODEL_NAME, 
            temperature=TEMPERATURE, 
            max_tokens=MAX_TOKENS, 
            api_key=OPENAI_API_KEY
        )
        logger.info(f"Initialized OpenAI models.")
    except Exception as e:
        logger.error(f"Error initializing OpenAI models: {e}")
        raise

# --- Node Functions --- 

def manager_node_function(state: AgentState) -> Dict[str, Any]:
    """Initial manager node. Always routes to RAG retrieval first."""
    logger.info("--- Manager Node ---")
    user_query = state['user_query']
    logger.info(f"Manager processing query: {user_query}")
    # Always retrieve context first
    return {"next": "rag_retriever"}

def rag_retriever_node_function(state: AgentState) -> Dict[str, Any]:
    """Retrieves relevant chunks from the vector store."""
    logger.info("--- RAG Retriever Node ---")
    user_query = state['user_query']
    vectorstore_instance = state['vectorstore_instance']
    
    if not vectorstore_instance:
        logger.error("Vector store not initialized.")
        return {"relevant_chunk_ids": [], "relevant_chunk_contents": []} 

    try:
        # Use Chroma retriever to get top 5 relevant documents
        retriever = vectorstore_instance.as_retriever(search_kwargs={"k": 5})
        relevant_docs = retriever.invoke(user_query)
        chunk_ids = []
        chunk_contents = []
        for doc in relevant_docs:
            chunk_id = doc.metadata.get('chunk_id', None)
            if chunk_id is not None:
                chunk_ids.append(chunk_id)
                chunk_contents.append(doc.page_content)
        logger.info(f"Retrieved {len(chunk_ids)} relevant chunks.")
        return {"relevant_chunk_ids": chunk_ids, "relevant_chunk_contents": chunk_contents}
    except Exception as e:
        logger.error(f"Error during RAG retrieval: {e}")
        return {"relevant_chunk_ids": [], "relevant_chunk_contents": []}

def manager_after_rag_node_function(state: AgentState) -> Dict[str, Any]:
    """Manager node after RAG. Assigns tasks to workers based on retrieved chunks."""
    logger.info("--- Manager After RAG ---")
    relevant_chunk_ids = state.get('relevant_chunk_ids', [])
    
    if not relevant_chunk_ids:
        logger.warning("No relevant chunks found. Cannot assign tasks to workers.")
        # Decide how to proceed - maybe directly to final aggregation?
        return {"subtasks": [], "next": "manager_after_workers"} # Skip workers if no chunks

    num_chunks = len(relevant_chunk_ids)
    logger.info(f"Manager assigning tasks for {num_chunks} relevant chunks.")

    # Simple assignment: one worker per chunk for now
    subtasks = []
    worker_names = []
    for i, chunk_id in enumerate(relevant_chunk_ids):
        worker_name = f"worker_{i}"
        task_description = (
            f"Analyze this code chunk (ID: {{chunk_id}}) regarding the query: '{{state['user_query']}}'. "
            f"Focus on relevance and key details. If the query asks for code implementation or modification, "
            f"propose specific, relevant code changes or additions based *only* on this chunk's context. "
            f"If the chunk is irrelevant, state that clearly.\n\nCode Chunk:\n{{chunk_content}}"
        )
        prompt = ChatPromptTemplate.from_messages([
            ("system", f"You are a helpful code analysis assistant."),
            ("human", task_description)
        ])
        subtasks.append({"worker_name": worker_name, "task": prompt, "chunk_id": chunk_id})
        worker_names.append(worker_name)

    logger.info(f"Assigned tasks: {worker_names}")
    return {"subtasks": subtasks, "next": "workers"}

def worker_node_function(state: AgentState, task_info: Dict[str, Any]) -> Dict[str, Any]:
    """Worker node function. Processes a specific task related to a chunk."""
    worker_name = task_info['worker_name']
    task = task_info['task']
    chunk_id = task_info['chunk_id']
    logger.info(f"--- Worker Node {worker_name} ---")
    logger.info(f"Worker {worker_name} processing chunk {chunk_id} for task: {task}")

    # Find the content for the assigned chunk_id
    chunk_content = ""
    try:
        chunk_index = state['relevant_chunk_ids'].index(chunk_id)
        chunk_content = state['relevant_chunk_contents'][chunk_index]
    except (ValueError, IndexError):
        logger.error(f"Worker {worker_name} could not find content for chunk_id {chunk_id}")
        return {worker_name: {"status": "error", "output": "Chunk content not found."}}

    if not worker_llm:
        logger.error("Worker LLM not initialized.")
        return {worker_name: {"status": "error", "output": "LLM not available."}}

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful AI assistant analyzing code chunks. Your goal is to assess the relevance of the provided code chunk to the user's query and, if the query involves coding, propose specific, relevant code changes or additions based *only* on this chunk's context. If the chunk is irrelevant, state that clearly."),
        ("human", "Analyze this code chunk (ID: {chunk_id}) regarding the query: '{user_query}'. Focus on relevance and key details. If the query asks for code implementation or modification, propose specific, relevant code changes or additions based *only* on this chunk's context. If the chunk is irrelevant, state that clearly.\n\nCode Chunk:\n{chunk_content}")
    ])
    
    chain = prompt | worker_llm | StrOutputParser()
    
    try:
        response = chain.invoke({
            "chunk_content": chunk_content,
            "chunk_id": chunk_id,
            "user_query": state["user_query"]
        })
        logger.info(f"Worker {worker_name} LLM response received (length: {len(response)}).")
        return {"worker_outputs": {worker_name: {"status": "success", "output": response}}}
    except Exception as e:
        logger.error(f"Error during worker {worker_name} LLM call: {e}")
        return {"worker_outputs": {worker_name: {"status": "error", "output": f"LLM Error: {e}"}}}

def manager_after_workers_node_function(state: AgentState) -> Dict[str, Any]:
    """Manager node after workers. Aggregates results and synthesizes final response."""
    logger.info("--- Manager After Workers (Aggregation) ---")
    worker_outputs = state.get('worker_outputs') or {}
    if not isinstance(worker_outputs, dict):
        worker_outputs = {}
    user_query = state['user_query']

    if not worker_outputs:
        logger.warning("No worker outputs received.")
        # Handle cases: No chunks found initially, or workers failed
        if not state.get('relevant_chunk_ids'):
            final_response = "I couldn't find any relevant information in the codebase for your query."
        else:
             final_response = "There was an issue processing the information. Please try again."
        return {"final_response": final_response, "next": "__end__"}

    # --- Aggregate Worker Outputs ---
    successful_outputs = []
    code_suggestions = []
    analysis_parts = []
    has_errors = False

    from masRAG.escape_utils import escape_curly_braces
    for worker_name, result in worker_outputs.items():
        if result.get('status') == 'success':
            output_text = result.get('output', '')
            output_text_escaped = escape_curly_braces(output_text)
            successful_outputs.append(f"Output from {worker_name}:\n{output_text_escaped}")
            # Basic check for code blocks or keywords suggesting code
            # TODO: Improve this detection - maybe use LLM to classify output?
            if "```" in output_text or any(kw in output_text.lower() for kw in ["def ", "class ", "function(", "=> {", "import ", "const ", "let ", "var "]): 
                 code_suggestions.append(output_text_escaped)
            else:
                analysis_parts.append(output_text_escaped)
        else:
            logger.error(f"Worker {worker_name} failed: {result.get('output')}")
            has_errors = True

    if not successful_outputs:
        logger.error("All workers failed or produced no output.")
        return {"final_response": "Failed to process your request due to worker errors.", "next": "__end__"}

    # --- Synthesize Final Response --- 
    final_response = ""
    if code_suggestions:
        logger.info(f"Aggregating {len(code_suggestions)} code suggestions from workers.")
        if not manager_llm:
             logger.error("Manager LLM not initialized for synthesis.")
             final_response = "Error: Cannot synthesize code, manager LLM unavailable.\n\nRaw Suggestions:\n" + "\n---\n".join(code_suggestions)
        else:
            synthesis_prompt = ChatPromptTemplate.from_messages([
                ("system", "You are an expert software developer. Your task is to synthesize code suggestions from multiple workers into a single, coherent, and correct code implementation or modification based on the original user query. Combine relevant parts, resolve conflicts, ensure correctness, and format the final code clearly. If suggestions are contradictory or irrelevant, point it out. ONLY output the final synthesized code or explanation."),
                ("human", f"Original User Query: {user_query}\n\nWorker Code Suggestions:\n" + "\n---\n".join(code_suggestions) + "\n\nSynthesized Code/Response:")
            ])
            synthesis_chain = synthesis_prompt | manager_llm | StrOutputParser()
            try:
                final_response = synthesis_chain.invoke({})
                logger.info("Manager successfully synthesized code suggestions.")
            except Exception as e:
                logger.error(f"Error during manager synthesis LLM call: {e}")
                final_response = f"Error during synthesis: {e}\n\nRaw Suggestions:\n" + "\n---\n".join(code_suggestions)
                
    elif analysis_parts:
        logger.info("Aggregating analysis from workers.")
        # Simple concatenation for now, could be improved with summarization LLM call
        final_response = "\n\n".join(analysis_parts)
    else:
        # This case might happen if outputs were successful but deemed neither code nor analysis by the basic check
        logger.warning("Workers provided output, but no clear code suggestions or analysis identified.")
        final_response = "I analyzed the relevant parts of the codebase, but couldn't determine specific code changes or a direct answer based on the workers' outputs. Here's a summary of what they found:\n\n" + "\n\n".join(successful_outputs)

    if has_errors:
         final_response += "\n\nWarning: Some processing steps encountered errors."

    # Ensure final_response is never empty
    if not final_response:
        final_response = "I have processed your request but could not generate a specific response."
        logger.warning("Final response was empty after aggregation, setting default message.")

    return {"final_response": final_response, "next": "__end__"}

# --- Utility to create worker nodes (if needed for dynamic graph building) ---
# def create_worker_node(worker_name):
#     def node(state: AgentState):
#         task_info = next(t for t in state['subtasks'] if t['worker_name'] == worker_name)
#         return worker_node_function(state, task_info)
#     return node
