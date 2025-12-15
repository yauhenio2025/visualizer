# Visualizer MCP Server

AI-powered document analysis with 47+ engines. Generates 4K visual diagrams or text reports from PDFs and documents.

---

## Setup (2 minutes)

### Step 1: Unzip this package

Unzip to any folder, e.g., `~/visualizer-mcp-server/`

### Step 2: Tell Claude Code to set it up

Open Claude Code and say:

> "I unzipped the visualizer MCP server to ~/visualizer-mcp-server/. Please set it up for me. My Anthropic API key is sk-ant-... and my Gemini API key is AIza..."

Claude Code will:
- Create the virtual environment
- Install dependencies
- Create `.env.mcp` with your API keys
- Update your Claude Code config

### Step 3: Restart Claude Code

Quit and reopen Claude Code.

### Step 4: Test it

Say: "List the available visualizer engines"

---

## Usage

**Get recommendations for a document:**
> "Analyze /path/to/document.pdf and recommend the best engines"

**Run analysis:**
> "Analyze /path/to/document.pdf with the argument_architecture engine"

**Run a pipeline (multi-stage analysis):**
> "Run the argument_assumption_excavation pipeline on /path/to/document.pdf"

**Results** are automatically downloaded to `~/visualizer-results/`

---

## Get API Keys

- **Anthropic**: https://console.anthropic.com/
- **Gemini**: https://aistudio.google.com/apikey

---

## What's Included

| File | Purpose |
|------|---------|
| `mcp_server.py` | The MCP server |
| `requirements.txt` | Python dependencies |
| `run-mcp-server.sh` | Startup script |
| `.env.mcp.template` | API key template |

---

## Claude Code Setup Instructions

**For Claude Code to set this up, it needs to:**

1. Create virtual environment:
```bash
cd /path/to/visualizer-mcp-server
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

2. Create `.env.mcp` with user's API keys:
```
VISUALIZER_API_URL=https://visualizer-tw4i.onrender.com
ANALYZER_API_URL=https://analyzer-3wsg.onrender.com
VISUALIZER_OUTPUT_DIR=~/visualizer-results
ANTHROPIC_API_KEY=<user's anthropic key>
GEMINI_API_KEY=<user's gemini key>
```

3. Make script executable:
```bash
chmod +x run-mcp-server.sh
```

4. Add to `~/.claude.json`:
```json
{
  "mcpServers": {
    "visualizer": {
      "command": "/absolute/path/to/visualizer-mcp-server/run-mcp-server.sh"
    }
  }
}
```

Then tell the user to restart Claude Code.
