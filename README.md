# Agentic Workflow & Dynamic Context Ingestion Engine

An autonomous conversational agent runtime built in Python that pairs dynamic document ingestion (PDF/unstructured text) with OpenAI tool execution and function dispatch loops.

## Architecture
The system isolates document parsing and tool-calling runtimes into a structured pipeline:
1. **Context Extraction Layer:** Ingests candidate profile artifacts (`pypdf` extraction and plain text summaries) to construct domain-specific grounded system prompts.
2. **Autonomous Tool Dispatch Engine:** Intercepts model tool calls, validates JSON argument payloads, executes mapped registry functions, and reinjects structured payloads back into the conversational thread.
3. **Interactive UI Interface:** Delivers real-time session testing and multi-turn state persistence through a lightweight Gradio interface.

## Tech Stack
- **Language:** Python 3.11+
- **Agent Orchestrator:** OpenAI API (`gpt-4o-mini`, function calling schemas)
- **Document Processing:** PyPDF
- **Frontend / Prototyping:** Gradio
- **Configuration:** Python-Dotenv

## Repository Structure
```text
agentic-orchestration-engine/
├── data/
│   ├── profile.pdf          # Candidate resume / credentials document
│   └── summary.txt          # Profile summary text
├── docs/                    # Architecture documentation & diagrams
├── src/
│   ├── core/                # Core engine modules
│   └── tools/               # Custom tool registry implementations
├── app.py                   # Main Gradio application & agent runtime
├── .env.example             # Environment template
├── .gitignore               # Ignored secrets & virtualenvs
├── pyproject.toml           # Project metadata & packaging
├── requirements.txt         # Pinned production dependencies
└── README.md                # System documentation
```

## Quickstart

### 1. Installation
Clone the repository and set up a Python virtual environment:
```bash
git clone https://github.com/<your-username>/agentic-orchestration-engine.git
cd agentic-orchestration-engine
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# macOS/Linux:
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Environment Variables
Copy `.env.example` to `.env` and configure your API key:
```bash
cp .env.example .env
```

### 3. Running the Engine
```bash
python app.py
```
Open `http://127.0.0.1:7860` in your web browser.
