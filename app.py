import os
import json
from pathlib import Path
from typing import List, Dict, Any
from dotenv import load_dotenv
from pypdf import PdfReader
import gradio as gr
from openai import OpenAI

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL_NAME = os.getenv("OPENAI_MODEL_NAME", "gpt-4o-mini")

client = OpenAI(api_key=OPENAI_API_KEY)


class DynamicContextEngine:
    """Ingests profile documents and unstructured text summaries to ground LLM reasoning."""

    def __init__(self, pdf_path: str = "data/profile.pdf", summary_path: str = "data/summary.txt"):
        self.profile_text = ""
        self.summary_text = ""

        pdf_file = Path(pdf_path)
        if pdf_file.exists():
            try:
                reader = PdfReader(str(pdf_file))
                for page in reader.pages:
                    extracted = page.extract_text()
                    if extracted:
                        self.profile_text += extracted + "\n"
            except Exception as e:
                print(f"[Warning] PDF read error: {e}")

        summary_file = Path(summary_path)
        if summary_file.exists():
            try:
                with open(summary_file, "r", encoding="utf-8") as f:
                    self.summary_text = f.read()
            except Exception as e:
                print(f"[Warning] Summary read error: {e}")

    def get_system_prompt(self) -> str:
        return (
            "You are an interactive AI Technical Portfolio Agent. "
            "Your objective is to provide precise details regarding software architecture, "
            "technical skills, projects, and domain research based on the candidate context provided below.\n\n"
            f"Candidate Summary:\n{self.summary_text}\n\n"
            f"Curriculum Vitae / Credentials:\n{self.profile_text}"
        )


def verify_service_health() -> Dict[str, str]:
    """Returns current health status of the orchestrator runtime."""
    return {"status": "operational", "engine": "modular-agent-runtime-v1"}


TOOL_REGISTRY = {
    "verify_service_health": verify_service_health
}

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "verify_service_health",
            "description": "Inspect operational health and runtime configuration.",
            "parameters": {"type": "object", "properties": {}, "required": []}
        }
    }
]


def dispatch_tools(tool_calls: List[Any]) -> List[Dict[str, str]]:
    """Executes requested tool calls and formats results for conversational reinjection."""
    responses = []
    for call in tool_calls:
        fn_name = call.function.name
        fn_args = json.loads(call.function.arguments or "{}")
        func = TOOL_REGISTRY.get(fn_name)

        if func:
            output = func(**fn_args)
        else:
            output = {"error": f"Tool '{fn_name}' not found."}

        responses.append({
            "role": "tool",
            "content": json.dumps(output),
            "tool_call_id": call.id
        })
    return responses


context_engine = DynamicContextEngine()


def chat_orchestrator(message: str, history: List[Dict[str, str]]) -> str:
    """Main execution loop for context management, tool dispatching, and response generation."""
    messages = [{"role": "system", "content": context_engine.get_system_prompt()}]

    for item in history:
        messages.append({"role": item["role"], "content": item["content"]})
    messages.append({"role": "user", "content": message})

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        tools=TOOL_SCHEMAS,
        tool_choice="auto"
    )

    msg = response.choices[0].message
    if msg.tool_calls:
        messages.append(msg)
        tool_outputs = dispatch_tools(msg.tool_calls)
        messages.extend(tool_outputs)

        final_response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages
        )
        return final_response.choices[0].message.content

    return msg.content or "No response generated."


def run_interface():
    demo = gr.ChatInterface(
        fn=chat_orchestrator,
        type="messages",
        title="Agentic Workflow & Dynamic Context Ingestion Engine",
        description="Autonomous AI agent architecture featuring PDF ingestion, system prompt grounding, and automated tool calling loops."
    )
    demo.launch()


if __name__ == "__main__":
    run_interface()
