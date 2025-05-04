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
    escape_curly_braces
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

MANAGER_MODEL_NAME = "gpt-4o"
WORKER_MODEL_NAME = "gpt-4o"
TEMPERATURE = 0.1
MAX_TOKENS = 4000
RETRIEVER_K = 5 # Number of chunks to retrieve

# Prompts
WORKER_SYSTEM_PROMPT = (
    "You are an expert software developer assistant. Analyze the provided code snippet (chunk) in the context of the user's query and the overall goal. "
    "Focus ONLY on the provided chunk. Your goal is to provide a detailed analysis or a specific code modification suggestion for THIS CHUNK based on the query. "
    "Keep your response concise and focused on the task for this specific chunk. "
    "If the chunk is relevant, explain how and suggest changes or extract information. "
    "If the chunk seems irrelevant to the query, state that clearly. "
    "Output format should be clear. If suggesting code, use markdown code blocks. "
    "Context: {file_path}"
)
MANAGER_SYNTHESIS_SYSTEM_PROMPT = (
    "You are an expert software developer. Your task is to synthesize analysis and code suggestions from multiple workers into a single, coherent, and correct final response or code implementation based on the original user query. "
    "Combine relevant parts, resolve conflicts, ensure correctness, and format the final code clearly using markdown. "
    "If suggestions are contradictory or irrelevant, point it out. If no worker provided useful information, state that. "
    "Address the user's original query directly in your final output."
)
MANAGER_AGGREGATION_SYSTEM_PROMPT = (
    "You are an aggregation assistant. Combine the key points from the following worker analyses into a single, concise summary that directly addresses the user's query. "
    "Do not include greetings or conversational filler. Focus on presenting the findings clearly. If workers reported errors or irrelevance, summarize that too."
)

# --- Initialization Function ---
def initialize_models():
    """Initializes the LLM models and returns them."""
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
        logger.info(f"Initialized OpenAI models: Worker={WORKER_MODEL_NAME}, Manager={MANAGER_MODEL_NAME}")
        return worker_llm, manager_llm
    except Exception as e:
        logger.error(f"Error initializing OpenAI models: {e}")
        return None, None

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
        # Use Chroma retriever to get top K relevant documents
        retriever = vectorstore_instance.as_retriever(search_kwargs={"k": RETRIEVER_K})
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
        return {"next": "manager_after_workers", "subtasks": []}

    # Assign each chunk to a worker
    subtasks = []
    for i, chunk_id in enumerate(relevant_chunk_ids):
        subtasks.append({
            "worker_name": f"worker_{i}",
            "chunk_id": chunk_id,
            "chunk_content": state.get('relevant_chunk_contents', [])[i] if i < len(state.get('relevant_chunk_contents', [])) else "",
            "user_query": state['user_query'],
            "file_path": state.get('file_path', 'N/A')
        })
    logger.info(f"Assigned {len(subtasks)} subtasks to workers.")
    return {"next": "workers", "subtasks": subtasks}

def worker_node_function(state: AgentState, task_info: Dict[str, Any]) -> Dict[str, Any]:
    """Worker node function. Processes a specific task related to a chunk."""
    logger.info(f"--- Worker Node: {task_info.get('worker_name', 'unknown')} ---")
    worker_llm = state.get('worker_llm')
    if not worker_llm:
        logger.error("Worker LLM not found in state.")
        return {task_info.get('worker_name', 'unknown'): {"status": "error", "output": "Worker LLM not available."}}

    user_query = task_info.get('user_query', '')
    chunk_content = task_info.get('chunk_content', '')
    file_path = task_info.get('file_path', 'N/A')

    prompt = ChatPromptTemplate.from_messages([
        ("system", WORKER_SYSTEM_PROMPT.format(file_path=file_path)),
        ("human", f"User Query: {user_query}\n\nCode Chunk:\n```\n{chunk_content}\n```\n\nAnalysis/Suggestion:")
    ])
    chain = prompt | worker_llm | StrOutputParser()
    try:
        result = chain.invoke({})
        logger.info(f"Worker {task_info.get('worker_name', 'unknown')} completed task.")
        return {task_info.get('worker_name', 'unknown'): {"status": "success", "output": result}}
    except Exception as e:
        logger.error(f"Error in worker node: {e}")
        return {task_info.get('worker_name', 'unknown'): {"status": "error", "output": str(e)}}

def manager_after_workers_node_function(state: AgentState) -> Dict[str, Any]:
    """Manager node after workers. Aggregates results and synthesizes final response."""
    logger.info("--- Manager After Workers ---")
    manager_llm = state.get('manager_llm')
    worker_outputs = state.get('worker_outputs', {})
    user_query = state['user_query']

    successful_outputs = []
    code_suggestions = []
    analysis_parts = []
    has_errors = False

    for worker_name, result in worker_outputs.items():
        if result.get('status') == 'success':
            output_text = result.get('output', '')
            output_text_escaped = escape_curly_braces(output_text)
            successful_outputs.append(f"Output from {worker_name}:\n{output_text_escaped}")
            # Basic check for code blocks or keywords suggesting code
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
                ("system", MANAGER_SYNTHESIS_SYSTEM_PROMPT),
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
        logger.warning("Workers provided output, but no clear code suggestions or analysis identified.")
        final_response = "I analyzed the relevant parts of the codebase, but couldn't determine specific code changes or a direct answer based on the workers' outputs. Here's a summary of what they found:\n\n" + "\n\n".join(successful_outputs)

    if has_errors:
         final_response += "\n\nWarning: Some processing steps encountered errors."

    # Ensure final_response is never empty
    if not final_response:
        final_response = "I have processed your request but could not generate a specific response."
        logger.warning("Final response was empty after aggregation, setting default message.")

    return {"final_response": final_response, "next": "__end__"}
