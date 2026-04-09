from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, List

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.tools import tool


# =========================================================
# Basic configuration
# =========================================================
ROOT_DIR = r"D:\OPDA-Cases"


# =========================================================
# Utilities
# =========================================================
def normalize_name(name: str) -> str:
    """
    Normalize a patient name for folder matching.
    Examples:
        'Yao Gao'  -> 'YAO-GAO'
        'yao_gao'  -> 'YAO-GAO'
        ' yao-gao' -> 'YAO-GAO'
    """
    name = name.strip().upper()
    name = re.sub(r"[\s_]+", "-", name)
    return name


def find_patient_folder(root_dir: str, patient_name: str) -> str:
    """
    Recursively search for the patient's folder under the given root directory.
    Returns the matched folder path as a string.
    Raises an exception if not found.
    """
    root = Path(root_dir)

    if not root.exists():
        raise FileNotFoundError(f"Root directory does not exist: {root}")

    target_name = normalize_name(patient_name)

    for path in root.rglob("*"):
        if path.is_dir() and normalize_name(path.name) == target_name:
            return str(path)

    raise FileNotFoundError(f"No patient folder named '{target_name}' was found.")


# =========================================================
# LangChain tool
# =========================================================
@tool
def search_patient_folder(patient_name: str) -> str:
    """
    Search for a patient folder under the predefined root directory.
    """
    try:
        folder = find_patient_folder(ROOT_DIR, patient_name)
        return (
            f"I found the patient's folder here: {folder}. "
            f"I will now begin the PSI design and printing process."
            f"I will keep you informed at key milestones, including design completion, 3D printing initiation, and printing completion."
        )
    except Exception:
        return (
            f"I could not find the folder for {patient_name}. "
            f"Please check the patient name and try again."
        )

# =========================================================
# System prompt
# =========================================================
SYSTEM_PROMPT = """
You are OPDA, an AI assistant for PSI design.

Start the conversation naturally and briefly introduce yourself as OPDA.
Ask the user for the patient's name if it has not been provided.

Once the user provides a patient name, use the tool `search_patient_folder`.
If the folder is found, tell the user where it is and say that you will now begin the PSI design process.
Do not add unnecessary technical details.
Do not invent any path.
Keep the conversation short, natural, and professional.
Always reply in English.
"""


# =========================================================
# Agent builder
# =========================================================
def build_agent():
    """
    Build the LangChain agent.
    Make sure OPENAI_API_KEY is set in your environment before running.
    """
    if not os.getenv("OPENAI_API_KEY"):
        raise EnvironmentError(
            "OPENAI_API_KEY is not set. Please set your API key first."
        )

    model = init_chat_model("gpt-5.2")

    agent = create_agent(
        model=model,
        tools=[search_patient_folder],
        system_prompt=SYSTEM_PROMPT,
    )
    return agent


# =========================================================
# Helper: safely extract last assistant message text
# =========================================================
def extract_last_assistant_text(messages: List[Any]) -> str:
    """
    Extract readable text from the last assistant message.
    Works for common LangChain message object formats.
    """
    if not messages:
        return ""

    last_msg = messages[-1]
    content = getattr(last_msg, "content", "")

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text")
                if text:
                    parts.append(text)
            else:
                parts.append(str(item))
        return "\n".join(parts).strip()

    return str(content)


# =========================================================
# CLI chat loop
# =========================================================
def main():
    agent = build_agent()

    messages: List[Any] = [
        {
            "role": "user",
            "content": "Introduce yourself as OPDA and ask for the patient's name."
        }
    ]

    result = agent.invoke({"messages": messages})
    messages = result["messages"]

    print(f"OPDA: {extract_last_assistant_text(messages)}")

    while True:
        user_input = input("You: ").strip()

        if user_input.lower() in {"exit", "quit", "q"}:
            print("OPDA: Goodbye.")
            break

        if not user_input:
            print("OPDA: Please provide the patient's name.")
            continue

        messages.append({"role": "user", "content": user_input})
        result = agent.invoke({"messages": messages})
        messages = result["messages"]

        print(f"OPDA: {extract_last_assistant_text(messages)}")


if __name__ == "__main__":
    main()
