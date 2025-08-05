"""
Utility functions for formatting and displaying conversation messages.

This module provides functions to render message objects in a structured
and visually appealing way in the console using the `rich` library 
"""

# Import necessary standard libraries
import json
# Import typing hints for clear function signatures
from typing import Any, List

# Import components from the rich library for enhanced console output
from rich.console import Console
from rich.panel import Panel

# Initialize a global Console object from the rich library.
# This object will be used for all styled output to the terminal.
console = Console()

def format_message_content(message: Any) -> str:
    """
    Converts the content of a message object into a displayable string.

    This function handles simple string content as well as complex list-based
    content, such as tool calls, by parsing and formatting them appropriately.

    Args:
        message: A message object that has a 'content' attribute.

    Returns:
        A formatted string representation of the message content.
    """
    # Retrieve the content from the message object.
    content = message.content

    # Check if the content is a simple string.
    if isinstance(content, str):
        # If it is, return it directly.
        return content
    # Check if the content is a list, which often indicates complex data like tool calls
    elif isinstance(content, list):
        # Initialize an empty list to hold formatted parts of the content.
        parts = []
        # Iterate over each item in the content list.
        for item in content:
            # If the item is a simple text block.
            if item.get("type") == "text":
                # Append the text directly to our parts list.
                parts.append(item["text"])
            # If the item represents
