# LangGraph state definition and graph wiring
# We will define the AgentState and build the StateGraph here

from typing import List, Dict, Any, Optional, Annotated
from langgraph.graph.message import AnyMessage, add_messages
from langgraph.graph import StateGraph, END
import logging
from functools import partial # Needed for passing worker_id to node function

# Project specific imports
from .state import AgentState # Import state definition
from .agents import (
    initialize_models,
    manager_node_function,
    rag_retriever_node_function,
    manager_after_rag_node_function,
    worker_node_function,
    manager_after_workers_node_function
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Graph Definition & Wiring --- 


# Define the routing function for conditional edges
def route_after_rag(state: AgentState) -> str | List[str]:
    """Routes based on the 'next' value set by manager_after_rag.
    
    Returns:
        str or List[str]: The name(s) of the next node(s) to run.
    """
    next_step = state.get("next")
    logger.debug(f"route_after_rag: routing based on state['next'] = {next_step}")
    
    if next_step == "manager_after_workers":
        logger.info("Routing from RAG Manager directly to Aggregation Manager.")
        return "manager_after_workers"
    elif isinstance(next_step, list):
        logger.info(f"Routing from RAG Manager to workers: {next_step}")
        return next_step # Return the list of worker node names to run in parallel
    elif next_step == END:
         logger.info("Routing from RAG Manager to END.")
         return END
    elif next_step == "handle_error":
        logger.error("Routing to error handler.")
        return END
    else:
        # Default or fallback case - should ideally not happen with current logic
        logger.warning(f"Unexpected 'next' value in route_after_rag: {next_step}. Defaulting to END.")
        return END

# --- Manager Routing ---
def route_after_manager(state: AgentState) -> str:
    """Routes based on manager_node_function's 'next' value."""
    next_step = state.get("next")
    logger.debug(f"route_after_manager: routing based on state['next'] = {next_step}")
    return next_step

# Dynamic graph builder for variable number of worker nodes
def create_worker_wrapper(worker_id: str):
    """Creates a wrapper function for a specific worker ID."""
    def worker_wrapper(state: AgentState) -> Dict[str, Any]:
        # Find the task_info for this worker_id
        task_info = next((t for t in state.get('subtasks', []) if t['worker_name'] == worker_id), None)
        if not task_info:
            logger.error(f"No task_info found for {worker_id}")
            return {worker_id: {"status": "error", "output": f"No task_info found for {worker_id}"}}
        return worker_node_function(state, task_info)
    return worker_wrapper

def build_dynamic_graph(num_workers: int):
    """Builds a StateGraph starting at manager_after_rag with dynamic workers."""
    workflow = StateGraph(AgentState)
    # Core nodes for post-RAG processing
    workflow.add_node("manager_after_rag", manager_after_rag_node_function)
    workflow.add_node("manager_after_workers", manager_after_workers_node_function)
    # Dynamic worker nodes
    worker_node_names = [f"worker_{i}" for i in range(num_workers)]
    for worker_name in worker_node_names:
        workflow.add_node(worker_name, create_worker_wrapper(worker_name))
        workflow.add_edge(worker_name, "manager_after_workers")
    # Entry and routing after RAG
    workflow.set_entry_point("manager_after_rag")
    mapping = {**{name: name for name in worker_node_names},
               "manager_after_workers": "manager_after_workers",
               END: END}
    def route_after_rag(state: AgentState):
        next_step = state.get("next")
        logger.debug(f"route_after_rag: routing based on state['next'] = {next_step}")
        if next_step == "manager_after_workers":
            logger.info("Routing from RAG Manager directly to Aggregation Manager.")
            return "manager_after_workers"
        elif next_step == "workers":
            logger.info(f"Routing from RAG Manager to workers: {worker_node_names}")
            return worker_node_names
        elif isinstance(next_step, list):
            logger.info(f"Routing from RAG Manager to workers: {next_step}")
            return next_step # Return the list of worker node names to run in parallel
        elif next_step == END:
            logger.info("Routing from RAG Manager to END.")
            return END
        else:
            logger.warning(f"Unexpected 'next' value in route_after_rag: {next_step}. Defaulting to END.")
            return END
    workflow.add_conditional_edges(
        "manager_after_rag",
        route_after_rag,
        mapping
    )
    # Final edges
    workflow.add_edge("manager_after_workers", END)
    return workflow.compile()

def create_graph():
    """Creates the LangGraph StateGraph."""
    workflow = StateGraph(AgentState)

    # Define the nodes
    workflow.add_node("manager", manager_node_function)
    workflow.add_node("rag_retriever", rag_retriever_node_function)
    workflow.add_node("manager_after_rag", manager_after_rag_node_function)
    # Note: Worker nodes are typically handled dynamically or via a specific routing mechanism
    # within the graph execution logic if their number varies.
    # For simplicity here, we assume a mechanism handles calling workers based on 'subtasks'.
    # A simple approach is a single node that iterates or maps tasks.
    
    # Placeholder node to represent the collective execution of workers
    # In a real scenario, this might use `add_node` with dynamic invocation 
    # or LangGraph's map/reduce features if applicable.
    def worker_execution_node(state: AgentState):
        logger.info("--- Worker Execution Phase ---")
        subtasks = state.get('subtasks', [])
        all_worker_outputs = {}
        if not subtasks:
            logger.warning("No subtasks found for workers.")
            return {"worker_outputs": {}}
            
        for task_info in subtasks:
            worker_output = worker_node_function(state, task_info)
            all_worker_outputs.update(worker_output)
        
        logger.info(f"Collected outputs from {len(all_worker_outputs)} worker calls.")
        return {"worker_outputs": all_worker_outputs}
    
    workflow.add_node("workers", worker_execution_node)
    workflow.add_node("manager_after_workers", manager_after_workers_node_function)

    # Define the edges (workflow connections)
    workflow.set_entry_point("manager")

    # Manager decides the first step (now always RAG)
    workflow.add_edge("manager", "rag_retriever") 
    
    # After RAG, the second manager assigns tasks
    workflow.add_edge("rag_retriever", "manager_after_rag")

    # The second manager routes either to workers or directly to aggregation if no chunks
    def route_after_rag(state: AgentState):
        if state.get('subtasks') and len(state['subtasks']) > 0:
            logger.info("Routing from RAG Manager to workers")
            return "workers"
        else:
            logger.info("Routing from RAG Manager directly to final aggregation (no tasks)")
            return "manager_after_workers"
            
    workflow.add_conditional_edges(
        "manager_after_rag",
        route_after_rag,
        {
            "workers": "workers",
            "manager_after_workers": "manager_after_workers"
        }
    )
    
    # After workers execute, aggregate results
    workflow.add_edge("workers", "manager_after_workers")

    # The final manager node decides to end the process
    workflow.add_edge("manager_after_workers", END)

    # Compile the graph
    app = workflow.compile()
    logger.info("Graph compiled successfully.")
    return app
