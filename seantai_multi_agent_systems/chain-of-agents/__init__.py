from .main import *
from .agents import *
from .utils import *

__version__ = "0.1.0"

__all__ = [
    "ChainOfAgents",
    "WorkerAgent",
    "ManagerAgent",
    "split_into_chunks",
    "get_task_prompts"
]
