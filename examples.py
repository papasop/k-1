#!/usr/bin/env python3
"""
examples.py – Demonstration of the ChatGPT Codex Connector.

Run this file directly to see the connector in action:

    python examples.py

You need a valid OPENAI_API_KEY environment variable (or a .env file)
before running these examples.
"""

import os
import sys
import asyncio

# ---------------------------------------------------------------------------
# Allow running from the repository root without installing the package
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(__file__))

from codex_connector import CodexConnector, Config, setup_logging

setup_logging("INFO")


# ---------------------------------------------------------------------------
# Sample code snippets used in examples
# ---------------------------------------------------------------------------

PARTIAL_FUNCTION = """\
def calculate_statistics(numbers):
    \"\"\"Return basic statistics for a list of numbers.\"\"\"
    # TODO: implement mean, median, and standard deviation
"""

BUGGY_CODE = """\
def find_max(lst):
    max_val = lst[0]
    for i in range(len(lst)):
        if lst[i] > max_val
            max_val = lst[i]
    return max_value  # wrong variable name
"""

SLOW_CODE = """\
def has_duplicates(items):
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            if items[i] == items[j]:
                return True
    return False
"""

K1_SNIPPET = """\
class InformationTimeTracker:
    def __init__(self):
        self.history = []

    def compute(self, predictions, targets, activations):
        phi = float(predictions.mean())
        H   = float(activations.std()) + 1e-8
        dt  = phi / H
        self.history.append(dt)
        return dt
"""


# ---------------------------------------------------------------------------
# Example functions
# ---------------------------------------------------------------------------

def example_generate(connector: CodexConnector) -> None:
    print("\n" + "=" * 60)
    print("EXAMPLE 1: Code Generation")
    print("=" * 60)

    description = (
        "a Python function that reads a CSV file and returns "
        "a list of dictionaries, one per row"
    )
    print(f"Task: {description}\n")
    code = connector.generate(description)
    print(code)


def example_complete(connector: CodexConnector) -> None:
    print("\n" + "=" * 60)
    print("EXAMPLE 2: Code Completion")
    print("=" * 60)
    print("Partial code:\n")
    print(PARTIAL_FUNCTION)
    print("Completed code:\n")
    result = connector.complete(PARTIAL_FUNCTION)
    print(result)


def example_explain(connector: CodexConnector) -> None:
    print("\n" + "=" * 60)
    print("EXAMPLE 3: Code Explanation")
    print("=" * 60)
    print("Code to explain:\n")
    print(K1_SNIPPET)
    print("Explanation:\n")
    explanation = connector.explain(K1_SNIPPET)
    print(explanation)


def example_fix_bugs(connector: CodexConnector) -> None:
    print("\n" + "=" * 60)
    print("EXAMPLE 4: Bug Fixing")
    print("=" * 60)
    print("Buggy code:\n")
    print(BUGGY_CODE)
    print("Fixed code:\n")
    fixed = connector.fix_bugs(BUGGY_CODE)
    print(fixed)


def example_optimize(connector: CodexConnector) -> None:
    print("\n" + "=" * 60)
    print("EXAMPLE 5: Code Optimization")
    print("=" * 60)
    print("Original code:\n")
    print(SLOW_CODE)
    print("Optimized code:\n")
    optimized = connector.optimize(SLOW_CODE, goal="performance")
    print(optimized)


async def example_async(connector: CodexConnector) -> None:
    print("\n" + "=" * 60)
    print("EXAMPLE 6: Async Code Generation")
    print("=" * 60)

    description = "an async Python function that fetches JSON from a URL using aiohttp"
    print(f"Task: {description}\n")
    code = await connector.generate_async(description)
    print(code)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main() -> None:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print(
            "ERROR: OPENAI_API_KEY environment variable is not set.\n"
            "Create a .env file from .env.example and add your key.",
            file=sys.stderr,
        )
        sys.exit(1)

    config = Config(
        api_key=api_key,
        cache_enabled=True,   # cache identical requests during the demo
    )
    connector = CodexConnector(config=config)

    example_generate(connector)
    example_complete(connector)
    example_explain(connector)
    example_fix_bugs(connector)
    example_optimize(connector)
    asyncio.run(example_async(connector))

    print("\n✅ All examples completed successfully!")


if __name__ == "__main__":
    main()
