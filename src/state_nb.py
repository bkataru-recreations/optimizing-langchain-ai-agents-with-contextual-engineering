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
    # from langchain_nvidia_ai_endpoints import ChatNVIDIA
    from langchain_ollama import ChatOllama

    # --- Environment and Model Setup ---
    load_dotenv()

    # if not os.environ.get("NVIDIA_API_KEY", "").startswith("nvapi-"):
    #     print("NVIDIA_API_KEY not found in environment variables.")
    #     nvidia_api_key = getpass.getpass("Enter your NVIDIA API key: ")
    #     assert nvidia_api_key.startswith("nvapi-"), f"{nvidia_api_key[:5]}... is not a valid key"
    #     os.environ["NVIDIA_API_KEY"] = nvidia_api_key

    # Initialize the chat model to be used in the workflow
    # llm = ChatNVIDIA(model="moonshotai/kimi-k2-instruct")

    os.environ["OLLAMA_HOST"] = "host.docker.internal"

    llm = ChatOllama(
        model="gemma3n:e2b",
        temperature="1"
    )
    return (llm,)


@app.class_definition
# Define mime serializable data structure for marimo to render images
class Image(object):
    def __init__(self, url: str) -> None:
        self.url = url

    def _mime_(self) -> tuple[str, str]:
        return ("image/png", self.url)


@app.cell
def _(State, console, llm, pprint):
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

    # --- Visualize the Graph ---
    # Display a visual representation of the compiled workflow graph
    Image(chain.get_graph().draw_mermaid_png())

    # --- Execute the Workflow ---
    # Invoke the compiled graph with an initial state containing the topic.
    # The `invoke` method runs the graph from the START node to the END node.
    joke_generator_state = chain.invoke({"topic": "cats"})

    # --- Display the Final State ---
    # Print the final state of the graph after execution.
    # This will show both the input 'topic' and the output 'joke' that was written to the state.
    console.print("\n[bold blue]Joke Generator State:[/bold blue]")
    pprint(joke_generator_state)
    return END, START, StateGraph, joke_generator_state


@app.cell
def _(console, joke_generator_state, pprint):
    from langgraph.store.memory import InMemoryStore

    # --- Initialize Long-Term Memory Store ---
    # Create an instance of InMemoryStore, which provides a simple, non-persistent,
    # key-value storage system for use within the current session.
    store = InMemoryStore()

    # --- Define a Namespace for Organization ---
    # A namespace is used to logically group related data within the store.
    # Here, we use a tuple to represent a hierarchical namespace,
    # which could correspond to a user ID and an application context.
    namespace = ("rlm", "joke_generator")

    # --- Write Data to the Memory Store ---
    # Use the `put` method to save a key-value pair into the specified namespace.
    # This operation persists the joke generated in the previous step, making it
    # available for retrieval across different sessions or threads.
    store.put(
        namespace, # The namespace to write to
        "last_joke", # The key for the data entry
        {"joke": joke_generator_state["joke"]}, # The value to be stored
    )

    # Search the namespace to view all stored items
    stored_items = list(store.search(namespace))

    # Display the stored items with rich formatting
    console.print("\n[bold green]Stored Items in Memory:[/bold green]")
    pprint(stored_items)
    return InMemoryStore, namespace, store


@app.cell
def _(
    END,
    InMemoryStore,
    START,
    State,
    StateGraph,
    console,
    llm,
    namespace,
    pprint,
):
    from langgraph.checkpoint.memory import InMemorySaver
    from langgraph.store.base import BaseStore

    # Initialize storage components
    _checkpointer = InMemorySaver() # For thread-level state persistence
    _memory_store = InMemoryStore() # For cross-thread memory storage

    def _generate_joke(state: State, store: BaseStore) -> dict[str, str]:
        """Generate a joke with memory aweareness.

        This enhanced version checks for existing jokes in memory
        before generating new ones.

        Args:
            state: Current state containing the topic
            store: Memory store for persistent context

        Returns:
            Dictionary with the generated joke
        """
        # Check if there's an existing joke in memory
        existing_jokes = list(store.search(namespace))
        if existing_jokes:
            existing_joke = existing_jokes[0].value
            print(f"Existing joke: {existing_joke}")
        else:
            print("Existing joke: No existing joke")

        # Generate a new joke based on the topic
        msg = llm.invoke(f"Write a short joke about {state['topic']}")
        new_joke = {"joke": msg.content}

        # Store the new joke in long-term memory
        store.put(namespace, "last_joke", new_joke)

        # Return the joke to be added to state
        return new_joke


    # Build the workflow with memory capabilities
    _workflow = StateGraph(State)

    # Add the memory-aware joke generation node
    _workflow.add_node("generate_joke", _generate_joke)

    # Connect the workflow components
    _workflow.add_edge(START, "generate_joke")
    _workflow.add_edge("generate_joke", END)

    # Compile with both checkpointing and memory store
    _chain = _workflow.compile(checkpointer=_checkpointer, store=_memory_store)

    # Execute the workflow with thread-based configuration
    _config = {"configurable": {"thread_id": "1"}}
    _joke_generator_state = _chain.invoke({"topic": "cats"}, _config)

    # Display the workflow result with rich formatting
    console.print("\n[bold cyan]Workflow Result (Thread 1):[/bold cyan]")
    pprint(_joke_generator_state)

    print('=' * 50)

    # --- Retrieve and Inspect the Graph State ---
    # Use the `get_state` method to retrieve the latest state snapshot for the
    # thread specified in the `config` (in this case, thread "1"). THis is
    # possible because we compiled the graph with a checkpointer
    _latest_state = _chain.get_state(_config)

    # --- Display the State Snapshot ---
    # Print the retrieved state to the console. The StateSnapshot includes not only
    # the data ('topic', 'joke') but also execution metadata.
    console.print("\n[bold magenta]Latest Graph State (Thread 1):[/bold magenta]")
    pprint(_latest_state)

    print('=' * 50)

    # Execute the workflow with a different thread ID
    _config = {"configurable": {"thread_id": "2"}}
    _joke_generator_state = _chain.invoke({"topic": "cats"}, _config)

    # Display the result showing memory persistence across threads
    console.print("\n[bold yellow]Workflow Result (Thread 2):[/bold yellow]")
    pprint(_joke_generator_state)
    return BaseStore, InMemorySaver


@app.cell
def _(
    END,
    InMemoryStore,
    START,
    State,
    StateGraph,
    console,
    llm,
    namespace,
    pprint,
    store,
):
    def _generate_joke(state: State) -> dict[str, str]:
        """Generate an initial joke about the topic.

        Args:
            state: Current state containing the topic

        Returns:
            Dictionary with the generated joke
        """
        msg = llm.invoke(f"Write a short joke about {state['topic']}")
        return {"joke": msg.content}

    def _improve_joke(state: State) -> dict[str, str]:
        """Improve an existing joke by adding wordplay.

        This demonstrates selecting context from state - we read the existing
        joke from state and use it to generate an improved version.

        Args:
            state: Current state containing the original joke

        Returns:
            Dictionary with the improved joke
        """
        print(f"Initial joke: {state['joke']}")

        # Select the joke from state to present it to the LLM
        msg = llm.invoke(f"Make this joke funnier by adding wordplay: {state['joke']}")
        return {"improved_joke": msg.content}

    # Build the workflow with two sequential nodes
    _workflow = StateGraph(State)

    # Add both joke generation nodes
    _workflow.add_node("generate_joke", _generate_joke)
    _workflow.add_node("improve_joke", _improve_joke)

    # Connect nodes in sequence
    _workflow.add_edge(START, "generate_joke")
    _workflow.add_edge("generate_joke", "improve_joke")
    _workflow.add_edge("improve_joke", END)

    # Compile the workflow
    _chain = _workflow.compile()

    # Display the workflow visualization
    Image(_chain.get_graph().draw_mermaid_png())

    print("=" * 50)

    # Execute the workflow to see context selection in action
    _joke_generator_state = _chain.invoke({"topic": "cats"})

    # Display the final state with rich formatting
    console.print("\n[bold blue]Final Workflow State:[/bold blue]")
    pprint(_joke_generator_state)

    print("=" * 50)

    # Initialize the memory store
    _store = InMemoryStore()

    # Define namespace for organizing memories
    _namespace = ("rlm", "joke_generator")

    # Store the generated joke in memory
    _store.put(
        _namespace,     # namespace for organization
        "last_joke",   # key identifier
        {"joke": _joke_generator_state["joke"]}  # value to store
    )

    # Select (retrieve) the joke from memory
    retrieved_joke = store.get(namespace, "last_joke").value

    # Display the retrieved context
    console.print("\n[bold green]Retrieved Context from Memory:[/bold green]")
    pprint(retrieved_joke)

    return


@app.cell
def _(
    BaseStore,
    END,
    InMemorySaver,
    InMemoryStore,
    START,
    State,
    StateGraph,
    console,
    llm,
    namespace,
    pprint,
):
    # Initialize storage components
    _checkpointer = InMemorySaver()
    _memory_store = InMemoryStore()

    def _generate_joke(state: State, store: BaseStore) -> dict[str, str]:
        """Generate a joke with memory-aware context selection.

        This function demonstrates selecting context from memory before
        generating new content, ensuring consistency and avoiding duplication.

        Args:
            state: Current state containing the topic
            store: Memory store for persistent context

        Returns:
            Dictionary with the generated joke
        """
        # Select prior joke from memory if it exists
        prior_joke = store.get(namespace, "last_joke")
        if prior_joke:
            prior_joke_text = prior_joke.value["joke"]
            print(f"Prior joke: {prior_joke_text}")
        else:
            print("Prior joke: None!")

        # Generate a new joke that differs from the prior one
        prompt = (
            f"Write a short joke about {state['topic']}, "
            f"but make it different from any prior joke you've written: {prior_joke_text if prior_joke else 'None'}"
        )
        msg = llm.invoke(prompt)
        new_joke = {"joke": msg.content}

        # Store the new joke in memory for future context selection
        store.put(namespace, "last_joke", new_joke)

        return new_joke

    print('=' * 50)

    # Build the memory-aware workflow
    _workflow = StateGraph(State)
    _workflow.add_node("generate_joke", _generate_joke)

    # Connect the workflow
    _workflow.add_edge(START, "generate_joke")
    _workflow.add_edge("generate_joke", END)

    # Compile with both checkpointing and memory store
    _chain = _workflow.compile(checkpointer=_checkpointer, store=_memory_store)

    # Execute the workflow with the first thread
    _config = {"configurable": {"thread_id": "1"}}
    _joke_generator_state = _chain.invoke({"topic": "cats"}, _config)

    print('=' * 50)

    # Get the latest state of the graph
    _latest_state = _chain.get_state(_config)

    console.print("\n[bold magenta]Latest Graph State:[/bold magenta]")
    pprint(_latest_state)

    print('=' * 50)

    # Execute the workflow with a second thread to demonstrate memory persistence
    _config = {"configurable": {"thread_id": "2"}}
    _joke_generator_state = _chain.invoke({"topic": "cats"}, _config)

    print('=' * 50)

    # Get the latest state of the graph once again
    _latest_state = _chain.get_state(_config)

    console.print("\n[bold magenta]Latest Graph State:[/bold magenta]")
    pprint(_latest_state)

    print('=' * 50)
    return


@app.cell
def _(InMemoryStore, llm):
    import uuid
    import math
    import types

    from langgraph_bigtool import create_agent
    from langgraph_bigtool.utils import convert_positional_only_function_to_tool

    # Collect functions from `math` built-in
    all_tools = []
    for function_name in dir(math):
        function = getattr(math, function_name)
        if not isinstance(function, types.BuiltinFunctionType):
            continue
        # This is an idiosyncrasy of the `math` library
        if tool := convert_positional_only_function_to_tool(function):
            all_tools.append(tool)

    print('=' * 50)

    # Create registry of tools. This is a dict mapping
    # identifiers to tool instances.
    tool_registry = {
        str(uuid.uuid4()): tool for tool in all_tools
    }

    from langchain_ollama import OllamaEmbeddings

    # Index tool names and descriptions in the LangGraph
    # Store. Here we use a simple in-memory store.
    embeddings = OllamaEmbeddings(
        model="dengcao/Qwen3-Embedding-0.6B:F16"
    )

    _store = InMemoryStore(
        index={
            "embed": embeddings,
            "dims": 1024,
            "fields": ["description"],
        }
    )

    for tool_id, tool in tool_registry.items():
        _store.put(
            ("tools",),
            tool_id,
            {
                "description": f"{tool.name}: {tool.description}",    
            },
        )

    print('=' * 50)

    # Initialize agent
    _builder = create_agent(llm, tool_registry)
    _agent = _builder.compile(store=_store)

    Image(_agent.get_graph().draw_mermaid_png())
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
