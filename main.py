# Import necessary libraries
from typing import TypedDict # For defining the state schema with type hints

from rich.console import Console # For pretty-printing output
from rich.pretty import pprint # For pretty-printing Python objects

# Initialize a console for rich, formatted output
console = Console()

# Define the schema for the graph's state using TypedDict.
# This class acts as a data structure that will be passed between nodes in the graph.
# It ensures that the state has a consistent shape and provides type hints.
class State(TypedDict):
    """
    Defines the structure of the state for our joke generator workflow.

    Attributes:
        topic: The input topic for which a joke will be generated
        joke: The output field where the generated joke will be stored.
    """
    topic: str
    joke: str


def main():
    print("Hello from optimizing-langchain-ai-agents-with-contextual-engineering!")


if __name__ == "__main__":
    main()
