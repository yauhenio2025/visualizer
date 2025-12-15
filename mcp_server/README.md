# Visualizer MCP Server

> **AI-Powered Document Analysis & Visualization for Claude Code**

Transform any document into deep analytical insights using 47+ specialized analysis engines with AI-powered recommendations. This MCP server brings the full power of the Visualizer platform directly into Claude Code.

---

## What It Does

The Visualizer MCP Server enables Claude Code to:

1. **Get AI Recommendations** - Upload a document and receive 4-5 tailored engine suggestions
2. **Run Analysis** - Execute visual (4K images) or textual (reports/diagrams) analysis
3. **Monitor Progress** - Track job status with real-time updates
4. **Get Notified** - Receive desktop + sound alerts when analysis completes (Linux, macOS, Windows)
5. **Auto-Download** - Results automatically saved to your local filesystem

---

## Quick Start

See **INSTALL.md** for quick setup instructions.

### TL;DR

1. Unzip to any folder
2. Tell Claude Code: "Set up the visualizer MCP server at ~/visualizer-mcp-server with my API keys..."
3. Restart Claude Code
4. Say: "Analyze ~/Documents/paper.pdf and recommend the best engines"

---

## Available Tools

### 1. `get_ai_recommendations`

**Get AI-powered engine suggestions for your document**

```python
get_ai_recommendations(
    document_path: str,              # Path to PDF, TXT, MD, etc.
    max_recommendations: int = 5,    # How many engines to suggest
    anthropic_api_key: str = None,   # Optional: Anthropic API key
    gemini_api_key: str = None       # Optional: Gemini API key
)
```

**Returns:**
```json
{
  "document": "research-paper.pdf",
  "recommendations": [
    {
      "engine_key": "argument-architecture-mapper",
      "engine_name": "Argument Architecture Mapper",
      "confidence": 0.92
    }
  ],
  "document_characteristics": {
    "type": "Academic Research Paper",
    "style": "Empirical Study"
  },
  "analysis_strategy": "Use argument mapping to visualize claims..."
}
```

---

### 2. `list_available_engines`

**Browse all 47+ analysis engines**

```python
list_available_engines(
    category: str = None  # Optional: filter by category
)
```

**Categories:**
- AI Engines
- Argument & Reasoning
- Concepts & Frameworks
- Temporal & Historical
- Power & Resources
- Evidence & Data
- Rhetoric & Language
- Scholarly Landscape

---

### 3. `submit_analysis`

**Submit document for analysis with selected engines**

```python
submit_analysis(
    document_path: str,               # Path to document
    engine_keys: List[str],           # List of engine keys to use
    output_mode: str = "visual",      # "visual" or "textual"
    anthropic_api_key: str = None,    # Optional API key
    gemini_api_key: str = None,       # Optional API key (required for visual)
    auto_monitor: bool = True         # Auto-notify when complete
)
```

**Output Modes:**
- `"visual"` - 4K PNG images via Gemini (requires Gemini API key)
- `"textual"` - Markdown reports and text diagrams

**Auto-Monitor:**
When `auto_monitor=True` (default), a background process polls the job and:
- Downloads results automatically when complete
- Shows desktop notification
- Plays sound alert

---

### 4. `check_job_status`

**Monitor analysis progress**

```python
check_job_status(
    job_id: str  # Job ID from submit_analysis
)
```

---

### 5. `get_results`

**Download completed analysis results**

```python
get_results(
    job_id: str,                      # Job ID to retrieve
    download_to: str = None           # Optional: custom download directory
)
```

Results are saved to `~/visualizer-results/` by default.

---

### 6. `list_bundles`

**View pre-configured engine bundles**

Bundles are groups of engines that work well together:
- **Argument & Reasoning Bundle** - 5 engines for logical analysis
- **Temporal & Historical Bundle** - 7 engines for timeline analysis
- **Power & Resources Bundle** - 4 engines for stakeholder analysis

---

### 7. `list_pipelines`

**View analysis pipelines**

Pipelines are sequences where engine outputs feed into subsequent engines:
- **Argument Chain**: Argument Architecture → Evidence Data → Assumption Excavator
- **Temporal Chain**: Chronology Mapper → Temporal Analysis → Historical Comparison

---

## Configuration

### Environment File (.env.mcp)

```bash
# API Endpoints (hosted on Render - always available)
VISUALIZER_API_URL=https://visualizer-tw4i.onrender.com
ANALYZER_API_URL=https://analyzer-3wsg.onrender.com

# Output directory
VISUALIZER_OUTPUT_DIR=~/visualizer-results

# API Keys
ANTHROPIC_API_KEY=your_anthropic_key_here
GEMINI_API_KEY=your_gemini_key_here
```

### API Keys

You can provide API keys in 3 ways:

1. **Per-Request** (in Claude Code conversation)
2. **Environment Variable** (`export GEMINI_API_KEY=...`)
3. **In .env.mcp file**

---

## Notifications

When jobs complete, you'll receive:
- **Desktop notification** with job summary
- **Sound alert** to get your attention

### Cross-Platform Support

| Platform | Notification | Sound |
|----------|-------------|-------|
| **Linux** | `notify-send` | `paplay` (freedesktop sounds) |
| **macOS** | Native notification | Built-in Glass sound |
| **Windows** | PowerShell toast | System sound |

Plus **ntfy.sh** for mobile notifications (optional).

---

## Directory Structure

```
visualizer-mcp-server/
├── mcp_server.py          # Main MCP server
├── job_poller.py          # Auto-download & notifications
├── run-mcp-server.sh      # Startup script
├── requirements.txt       # Python dependencies
├── .env.mcp.template      # Configuration template
├── .env.mcp               # Your configuration (created during setup)
├── venv/                  # Python virtual environment (created during setup)
├── INSTALL.md             # Quick setup guide
└── README.md              # This file
```

---

## Example Workflows

### Workflow 1: AI-Recommended Analysis

```
User: "Analyze this document: ~/research/paper.pdf"

Claude:
  1. Reads the document
  2. Calls get_ai_recommendations()
  3. Shows 4-5 recommended engines with confidence scores
  4. "I recommend these engines based on your document type..."
  5. User: "Run all of them with visual output"
  6. Calls submit_analysis() with all engines
  7. Job submitted, auto-monitoring enabled
  8. [5 minutes later] Desktop notification + sound alert
  9. Results auto-downloaded to ~/visualizer-results/
  10. "Analysis complete! Results are at ~/visualizer-results/..."
```

### Workflow 2: Specific Engine Selection

```
User: "I want to map the argument structure of this legal brief"

Claude:
  1. Calls list_available_engines(category="Argument & Reasoning")
  2. Shows engines in that category
  3. "For argument mapping, I suggest 'Argument Architecture Mapper'"
  4. User: "Perfect, run that one"
  5. Calls submit_analysis() with that engine
  6. Monitors job and notifies when complete
```

---

## Troubleshooting

### MCP Server Not Found

```bash
# Check if registered
claude mcp list

# Re-register
cd ~/visualizer-mcp-server
claude mcp add visualizer --transport stdio -- "$(pwd)/run-mcp-server.sh"
```

### API Connection Errors

```bash
# Check services are accessible
curl https://visualizer-tw4i.onrender.com/api/analyzer/engines

# Note: First request may be slow if services are sleeping (free tier)
```

### No Sound/Notification

- **Linux**: Install `libnotify-bin` and `pulseaudio-utils` if missing
- **macOS**: Should work out of box
- **Windows**: Requires PowerShell execution policy to allow scripts

### Visual Output Requires Gemini API Key

```python
submit_analysis(
    document_path="~/doc.pdf",
    engine_keys=["anomaly-detector"],
    output_mode="visual",
    gemini_api_key="your_gemini_key_here"  # Required for visual
)
```

---

## Additional Resources

- **Visualizer Web UI**: https://visualizer-tw4i.onrender.com
- **Analyzer API Docs**: https://analyzer-3wsg.onrender.com/docs
- **FastMCP Docs**: https://github.com/jlowin/fastmcp

---

## Success Checklist

After installation, verify:

- [ ] `claude mcp list` shows "visualizer"
- [ ] Test: "List the available visualizer engines"
- [ ] Test: "Get AI recommendations for ~/Documents/test.pdf"
- [ ] Notification sound plays when job completes
- [ ] Results downloaded to `~/visualizer-results/`

---

**You're ready to analyze documents with AI-powered insights!**
