# Defines the shared state structure for the LangGraph

import operator # Import operator
from typing import Dict, Any, Optional, Annotated, List
from langgraph.graph.message import AnyMessage, add_messages
from langchain_community.vectorstores import Chroma # Import Chroma
from masRAG.merge_utils import merge_worker_outputs

# Using operator.add is often simpler and standard for merging dictionaries
# from parallel branches writing to the same dictionary key.

class AgentState(Dict[str, Any]):
    """Defines the structure of the state managed by the LangGraph.

    Attributes:
        user_query: The current user request.
        codebase_path: Path to the codebase being analyzed.
        codebase_text: The full text of the codebase (optional, might be large).
        chunks: List of chunk texts (optional, could be retrieved from DB if needed).
        chunk_metadata: Metadata for each chunk (e.g., original file, index, chunk_id).
        relevant_chunk_ids: List of chunk IDs retrieved by RAG for the current query.
        relevant_chunk_contents: Dictionary mapping relevant chunk IDs to their text content.
        subtasks: Dictionary mapping worker IDs to their assigned tasks (e.g., {'worker_0': {'chunk_id': 'abc', 'task_description': 'Analyze...'}}).
        worker_outputs: Dictionary mapping worker IDs to their outputs.
        final_response: The synthesized final output string for the user.
        conversation_history: List of messages, managed by LangGraph.
        vectorstore_instance: The initialized Chroma vector store instance.
        current_worker_id: Helper key to pass context to worker nodes (used internally by graph logic).
        next: Helper key to control conditional edge routing.
    """
    # Required state keys
    user_query: str
    codebase_path: str # Path to the codebase directory
    conversation_history: Annotated[List[AnyMessage], add_messages]

    # Optional / Updated during graph execution
    codebase_text: Optional[str] = None
    chunks: Optional[List[str]] = None
    chunk_metadata: Optional[List[Dict[str, Any]]] = None
    relevant_chunk_ids: Optional[List[str]] = None
    relevant_chunk_contents: Optional[Dict[str, str]] = None
    subtasks: Optional[Dict[str, Any]] = None
    # Use custom merge function for parallel worker outputs
    worker_outputs: Annotated[Dict[str, Any], merge_worker_outputs] = {}
    final_response: Optional[str] = None
    vectorstore_instance: Optional[Chroma] = None # type: ignore

    # Internal routing/helper keys (might not be explicitly defined in TypedDict if used)
    current_worker_id: Optional[str] = None
    next: Optional[str] = None
