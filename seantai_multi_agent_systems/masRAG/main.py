# Main entry point for the masRAG system
# print("masRAG main.py loaded") # Comment out print statement

import sys
import os
import logging

# Adjust path to import from parent directory if needed (running as script)
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import necessary components from the project
from .utils import (
    read_codebase,
    chunk_text_algorithm2,
    initialize_vectorstore,
    add_chunks_to_vectorstore,
    count_tokens,
    logger as utils_logger # Use logger defined in utils
)
from .state import AgentState # Import state definition from .state
from .graph import build_dynamic_graph
from .agents import logger as agents_logger, rag_retriever_node_function, manager_after_rag_node_function

# Configure logging level for main script
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Configuration ---
CODEBASE_PATH = "./masRAG/test_codebase"  # Path to the directory containing the code
VECTORSTORE_PATH = "./masRAG/chroma_db" # Path to persist the vector store
# Initial chunking parameters (adjust as needed)
# Using a smaller window for initial chunking might be okay if retrieval is good
# Or use a larger window similar to what the agent might expect
INITIAL_CHUNK_WINDOW_SIZE = 1500 # Tokens for each chunk during initial indexing
INITIAL_CHUNK_QUERY = "General Code Indexing" # Placeholder query for chunking budget
INITIAL_CHUNK_INSTRUCTION = "Chunk the following code text for retrieval."

# Sample User Query
USER_QUERY = "Please implement a new feature; add a new button while in game to forfeit, effectively ending the game upon clicking. Code the forfeit button." # Example query
# USER_QUERY = "What does the utility function in file2.py do?" # Example query 2

# --- Constants --- 
CHUNKING_INSTRUCTION = "Chunk the following code text for analysis." # Instruction for chunking budget calculation

from masRAG.agents import initialize_models

def run_masrag():
    initialize_models()
    logger.info("--- Starting masRAG System --- ")

    # 0. Clear vector store (delete all files in VECTORSTORE_PATH)
    import shutil, os
    if os.path.exists(VECTORSTORE_PATH):
        shutil.rmtree(VECTORSTORE_PATH)
        logger.info(f"Cleared vector store at: {VECTORSTORE_PATH}")
    os.makedirs(VECTORSTORE_PATH, exist_ok=True)

    # 1. Read Codebase (include .py, .js, .html, .css)
    logger.info(f"Reading codebase from: {CODEBASE_PATH}")
    try:
        codebase_dict = read_codebase(CODEBASE_PATH, extensions=['.py', '.js', '.html', '.css']) # Returns {path: content}
    except Exception as e:
        logger.error(f"Failed to read codebase: {e}")
        return

    # 2. Initialize Vector Store
    logger.info(f"Initializing vector store at: {VECTORSTORE_PATH}")
    try:
        vectorstore_instance = initialize_vectorstore(persist_directory=VECTORSTORE_PATH)
    except Exception as e:
        logger.error(f"Failed to initialize vector store: {e}. Please check API keys and dependencies. Exiting.")
        return
    
    # 3. Chunk Codebase (only if DB seems empty or force update)
    # Simple check: see if the collection is empty. A more robust check might be needed.
    try:
        collection_count = vectorstore_instance._collection.count()
    except Exception as e:
        logger.error(f"Failed to get collection count from vector store: {e}")
        # Handle case where DB exists but might be corrupt or inaccessible
        collection_count = -1 # Indicate error

    logger.info(f"Vector store collection count: {collection_count}")

    # Decide whether to re-chunk and add. For simplicity, let's add if empty.
    # A more robust approach would involve checking if codebase files have changed.
    if collection_count <= 0: # If empty or error
        logger.info("Vector store appears empty or encountered error. Chunking and adding codebase...")
        total_chunks_added = 0
        for file_path, file_content in codebase_dict.items():
            if not file_content.strip():
                logger.debug(f"Skipping empty file: {file_path}")
                continue

            logger.info(f"Chunking file: {file_path}")
            # Using Algorithm 2 (query-aware) per file
            chunks = chunk_text_algorithm2(
                source_text=file_content,
                query=USER_QUERY,
                instruction=CHUNKING_INSTRUCTION, # Add instruction
                window_size=INITIAL_CHUNK_WINDOW_SIZE
            )
            logger.info(f"  -> Generated {len(chunks)} chunks (window_size={INITIAL_CHUNK_WINDOW_SIZE})")

            if chunks:
                # Add chunks to vector store, passing the specific file_path for metadata
                added_ids = add_chunks_to_vectorstore(chunks, vectorstore_instance, file_path=file_path)
                if added_ids:
                    logger.info(f"  -> Successfully added {len(added_ids)} chunks to the vector store.")
                    total_chunks_added += len(added_ids)
                else:
                    logger.error(f"Failed to add chunks for file {file_path}. Stopping.")
                    # Decide if we should continue with other files or exit
                    return # Exit for now
            else:
                logger.warning(f"No chunks generated for file: {file_path}")
        
        if total_chunks_added > 0:
            logger.info(f"Finished adding chunks. Total chunks added: {total_chunks_added}")
        else:
            logger.warning("No chunks were added to the vector store from any file.")
            # Consider exiting if no data could be processed
            # return 
    else:
        logger.info("Vector store already contains data. Skipping chunking and adding.")

    # 5. Prepare Initial State for the Graph
    initial_state = AgentState(
        user_query=USER_QUERY,
        codebase_path=CODEBASE_PATH,
        conversation_history=[],
        vectorstore_instance=vectorstore_instance, # Pass the instance here
        # Initialize other fields that might be expected by nodes
        worker_outputs=[], # Must be an empty list for LangGraph message channel
        subtasks={}
    )
    logger.info(f"Prepared initial state for query: '{USER_QUERY}'")

    # 6. Perform RAG retrieval and dynamic graph construction
    logger.info("--- Retrieval and Routing --- ")
    config = {"recursion_limit": 15}
    try:
        # RAG retrieval step
        rag_out = rag_retriever_node_function(initial_state)
        initial_state.update(rag_out)
        # Manager routes tasks or code generation
        mgr_out = manager_after_rag_node_function(initial_state)
        initial_state.update(mgr_out)
        next_step = initial_state.get("next")
        if next_step == "code_generator":
            # Direct code generation
            code_out = code_generator_node_function(initial_state)
            initial_state["final_response"] = code_out.get("final_response")
            final_state = initial_state
        else:
            # Build and invoke dynamic graph for workers
            num_workers = len(initial_state.get("relevant_chunk_ids", []))
            dynamic_graph = build_dynamic_graph(num_workers)
            final_state = dynamic_graph.invoke(initial_state, config=config)
    except Exception as e:
        logger.error(f"Error during dynamic graph execution: {e}", exc_info=True)
        final_state = None

    # 7. Print Final Response
    logger.info("--- Graph Execution Complete --- ")
    if final_state and 'final_response' in final_state:
        logger.info("Final Response:")
        print("\n==================== FINAL RESPONSE ====================")
        print(final_state['final_response'])
        print("======================================================\n")
    elif final_state:
        logger.warning("Graph finished, but 'final_response' key not found in the final state.")
        print("\nGraph finished, but no final response was generated.")
        print(f"Final state keys: {final_state.keys()}")
    else:
        print("\nGraph execution failed or did not produce a final state.")

# --- Main Execution Guard ---
if __name__ == "__main__":
    # Ensure the script can find modules if run directly
    # This assumes main.py is inside masRAG which is inside block-project
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
        print(f"Added {project_root} to sys.path")
    
    # Now re-attempt imports relative to the project structure if needed
    # Typically, running `python -m masRAG.main` handles imports correctly.
    # If running `python masRAG/main.py`, the above sys.path adjustment helps.
    
    run_masrag()
