# Visualizer MCP Server - Installation Guide

This guide explains how to install and configure the Visualizer MCP Server for use with Claude Code. The server provides AI-powered document analysis with 47+ analysis engines.

## Prerequisites

- **Python 3.10+** installed
- **Claude Code** (Claude's official CLI) installed and working
- API keys for:
  - **Anthropic** (for Claude-based AI recommendations and analysis)
  - **Google Gemini** (for visual output mode - 4K image generation)

## Quick Install (5 minutes)

### Step 1: Download the MCP Server

```bash
# Clone or download the visualizer repository
git clone https://github.com/yauhenio2025/visualizer.git
cd visualizer/mcp_server
```

Or download just the `mcp_server` folder if you don't need the full repo.

### Step 2: Create Virtual Environment & Install Dependencies

```bash
# Create virtual environment
python3 -m venv venv

# Activate it
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### Step 3: Configure API Keys

Create a `.env.mcp` file in the `mcp_server` directory with your API keys:

```bash
# Visualizer MCP Server Configuration
VISUALIZER_API_URL=https://visualizer-tw4i.onrender.com
ANALYZER_API_URL=https://analyzer-3wsg.onrender.com
VISUALIZER_OUTPUT_DIR=~/visualizer-results

# YOUR API KEYS (required)
ANTHROPIC_API_KEY=sk-ant-api03-your-key-here
GEMINI_API_KEY=AIzaSy-your-key-here

# Optional: notification topic (generates unique one if not set)
# VISUALIZER_NTFY_TOPIC=my-custom-topic
```

**Tip:** You can ask Claude Code to create this file for you! Just say:
> "Create the .env.mcp file with my API keys: Anthropic key is sk-ant-... and Gemini key is AIza..."

Claude Code will create the file with the correct format.

### Step 4: Add to Claude Code

Edit your Claude Code MCP settings file:

**Location:**
- Linux/Mac: `~/.claude.json` (in your home directory)
- Or: `~/.claude/claude_desktop_config.json`

Add this server configuration:

```json
{
  "mcpServers": {
    "visualizer": {
      "command": "/absolute/path/to/visualizer/mcp_server/run-mcp-server.sh"
    }
  }
}
```

**Important:**
- Replace `/absolute/path/to/` with the actual path where you cloned/downloaded the repo
- **No need to put API keys here** - they're loaded from `.env.mcp` automatically

### Step 5: Restart Claude Code

Quit and restart Claude Code to load the new MCP server.

### Step 6: Verify Installation

In Claude Code, try:
```
Use the visualizer to list available engines
```

You should see a list of 47+ analysis engines organized by category.

---

## Usage Examples

### Get AI Recommendations for a Document

```
Analyze this PDF and recommend the best engines: /path/to/document.pdf
```

Claude will call `get_ai_recommendations()` and suggest engines, bundles, and pipelines based on your document's content.

### Submit Analysis

```
Submit analysis for /path/to/document.pdf using the recommended engines
```

Or specify engines manually:
```
Analyze /path/to/document.pdf with argument_architecture and thematic_synthesis engines
```

### Output Modes

- **Visual mode** (default): Generates 4K PNG images (requires Gemini API key)
- **Textual mode**: Generates markdown reports and text diagrams

```
Analyze /path/to/document.pdf in textual mode
```

### Check Job Status

```
Check the status of job abc123...
```

### Get Results

Results are automatically downloaded to `~/visualizer-results/` when jobs complete.

---

## Available Tools

| Tool | Description |
|------|-------------|
| `get_ai_recommendations` | AI-powered engine suggestions based on document content |
| `submit_analysis` | Submit document for analysis with selected engines |
| `check_job_status` | Check progress of a running job |
| `get_results` | Download completed job results |
| `list_available_engines` | List all 47+ analysis engines |
| `list_bundles` | List pre-configured engine bundles |
| `list_pipelines` | List multi-stage analysis pipelines |
| `submit_batch_analysis` | Process entire folders of documents |
| `submit_pipeline_analysis` | Run multi-stage pipeline analysis |
| `scan_folder` | Preview folder contents before batch processing |

---

## API Keys

### Anthropic API Key
- Required for: AI recommendations, document extraction, curation
- Get one at: https://console.anthropic.com/

### Gemini API Key
- Required for: Visual output mode (4K image generation)
- Optional for: Textual output mode
- Get one at: https://aistudio.google.com/apikey

### Where to Put Your Keys (Recommended: .env.mcp)

**Best option: Put keys in `.env.mcp`** (in the mcp_server folder)

```bash
ANTHROPIC_API_KEY=sk-ant-api03-your-actual-key-here
GEMINI_API_KEY=AIzaSy-your-actual-key-here
```

This keeps your keys in one place, separate from Claude Code config, and easy to update.

**Ask Claude Code to do it for you:**
> "Please create the .env.mcp file at /path/to/visualizer/mcp_server/.env.mcp with my Anthropic key sk-ant-... and Gemini key AIza..."

Claude Code will write the file in the correct format.

**Alternative: Per-request keys**
You can also provide keys when calling tools (useful for testing or multiple accounts):
> "Analyze document.pdf using Anthropic key sk-ant-... and Gemini key AIza..."

---

## Troubleshooting

### "File not found" errors
- Ensure the path to your document is absolute (starts with `/` or `~`)
- Check the file exists and is readable

### "API key not configured" errors
- Verify your API keys are set in `.env.mcp` or the MCP server config
- Make sure there are no extra spaces or quotes around the keys

### Jobs stuck at "pending"
- Check if the analyzer service is running: https://analyzer-3wsg.onrender.com/health
- Render free tier services may need a few seconds to wake up

### Images not downloading
- Ensure `VISUALIZER_OUTPUT_DIR` is set to a writable directory
- Check that `~/visualizer-results/` exists or can be created

### MCP server not appearing in Claude Code
- Verify the path in `claude_desktop_config.json` is absolute and correct
- Check that `run-mcp-server.sh` is executable: `chmod +x run-mcp-server.sh`
- Restart Claude Code completely (quit and reopen)

---

## Cost Estimates

Analysis costs depend on document length and engines used:

| Operation | Typical Cost |
|-----------|--------------|
| AI Recommendations | ~$0.02-0.05 |
| Single Engine Analysis | ~$0.10-0.30 |
| Bundle (5 engines) | ~$0.30-0.80 |
| Pipeline (2-3 stages) | ~$0.20-0.50 |
| Visual Output (per image) | ~$0.05-0.10 |

Costs are charged to your Anthropic and Gemini accounts based on token usage.

---

## Support

- Issues: https://github.com/yauhenio2025/visualizer/issues
- Documentation: See `README.md` in the mcp_server folder for detailed API reference

---

## Quick Reference Card

```
# Get recommendations
get_ai_recommendations("/path/to/doc.pdf")

# Submit with specific engines
submit_analysis("/path/to/doc.pdf", ["argument_architecture", "thematic_synthesis"])

# Check status
check_job_status("job_id_here")

# Get results (auto-downloads to ~/visualizer-results/)
get_results("job_id_here")

# List all engines
list_available_engines()

# List bundles
list_bundles()

# List pipelines
list_pipelines()
```
