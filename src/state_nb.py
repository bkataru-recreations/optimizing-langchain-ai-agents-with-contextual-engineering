import marimo

__generated_with = "0.14.16"
app = marimo.App()


@app.cell
def _():
    # Import necessary libraries
    from typing import TypedDict # For defining the state schema with type hints

    from rich.console import Console # For pretty-printing output
    from rich.pretty import pprint # For pretty-printing Python objects

    # Initialize a console for rich, formatted output
    console = Console()
    return TypedDict, console, pprint


@app.cell
def _(TypedDict):
    # Define the schema for the graph's state using TypedDict.
    # This class acts as a data structure that will be passed between nodes in the graph.
    # It ensures that the state has a consistent shape and provides type hints.
    class State(TypedDict):
        """
        Defines the structure of the state for our joke generator workflow.

        Attributes:
            topic: The input topic for which a joke will be generated
            joke: The output field where the generated joke wi ll be stored.
        """
        topic: str
        joke: str
    return (State,)


@app.cell
def _():
    # Import necessary libraries for environment management, display, and LangGraph
    import getpass
    import os

    from dotenv import load_dotenv
    from langchain_nvidia_ai_endpoints import ChatNVIDIA

    # --- Environment and Model Setup ---
    load_dotenv()

    if not os.environ.get("NVIDIA_API_KEY", "").startswith("nvapi-"):
        print("NVIDIA_API_KEY not found in environment variables.")
        nvidia_api_key = getpass.getpass("Enter your NVIDIA API key: ")
        assert nvidia_api_key.startswith("nvapi-"), f"{nvidia_api_key[:5]}... is not a valid key"
        os.environ["NVIDIA_API_KEY"] = nvidia_api_key

    # Initialize the chat model to be used in the workflow
    llm = ChatNVIDIA(model="moonshotai/kimi-k2-instruct")
    return (llm,)


@app.cell
def _(State, llm):
    from langgraph.graph import END, START, StateGraph

    # --- Define Workflow Node ---
    def generate_joke(state: State) -> dict[str, str]:
        """
        A node function that generates a joke based on the topic in the current state.

        This function reads the 'topic' from the state, uses the LLM to generate a joke,
        and returns a dictionary to update the 'joke' field in the state.

        Args:
            state: The current state of the graph, which must contain a 'topic'.

        Returns:
            A dictionary with the 'joke' key to update the state.
        """
        # Read the topic from the state
        topic = state["topic"]
        print(f"Generating a joke about: {topic}")

        # Invoke the language model to generate a joke
        msg = llm.invoke(f"Write a short joke about {topic}")

        # Return the generated joke to be written back to the state
        return {"joke": msg.content}
    return END, START, StateGraph, generate_joke


@app.cell
def _(END, START, State, StateGraph, generate_joke):
    # --- Build and Compile the Graph ---
    # Initialize a new StateGraph with the predefined State schema
    workflow = StateGraph(State)

    # Add the 'generate_joke' function as a node in the graph
    workflow.add_node("generate_joke", generate_joke)

    # Define the workflow's execution path:
    # The graph starts at the START entrypoint and flows to our 'generate_joke' node.
    workflow.add_edge(START, "generate_joke")
    # After 'generate_joke' completes, the graph's execution ends.
    workflow.add_edge("generate_joke", END)

    # Compile the workflow into an executable chain
    chain = workflow.compile()
    return (chain,)


@app.class_definition
# Define mime serializable data structure for marimo to render images
class Image(object):
    def __init__(self, url: str) -> None:
        self.url = url

    def _mime_(self) -> tuple[str, str]:
        return ("image/png", self.url)


@app.cell
def _(chain):
    # --- Visualize the Graph ---
    # Display a visual representation of the compiled workflow graph
    Image(chain.get_graph().draw_mermaid_png())
    return


@app.cell
def _(chain, console, pprint):
    # --- Execute the Workflow ---
    # Invoke the compiled graph with an initial state containing the topic.
    # The `invoke` method runs the graph from the START node to the END node.
    joke_generator_state = chain.invoke({"topic": "cats"})

    # --- Display the Final State ---
    # Print the final state of the graph after execution.
    # This will show both the input 'topic' and the output 'joke' that was written to the state.
    console.print("\n[bold blue]Joke Generator State:[/bold blue]")
    pprint(joke_generator_state)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
