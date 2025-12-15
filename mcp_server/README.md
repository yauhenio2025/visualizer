# Visualizer MCP Server

> **AI-Powered Document Analysis & Visualization for Claude Code**

Transform any document into deep analytical insights using 47+ specialized analysis engines with AI-powered recommendations. This MCP server brings the full power of the Visualizer platform directly into Claude Code.

---

## 🎯 What It Does

The Visualizer MCP Server enables Claude Code to:

1. **🤖 Get AI Recommendations** - Upload a document and receive 4-5 tailored engine suggestions
2. **📊 Run Analysis** - Execute visual (4K images) or textual (reports/diagrams) analysis
3. **⏱️ Monitor Progress** - Track job status with real-time updates
4. **🔔 Get Notified** - Receive desktop + sound alerts when analysis completes
5. **📥 Auto-Download** - Results automatically saved to your local filesystem

---

## 🚀 Quick Start

### Installation

```bash
cd /home/evgeny/projects/visualizer/mcp_server
./install-mcp-server.sh
```

This will:
- ✅ Create Python virtual environment
- ✅ Install dependencies (fastmcp, requests, pydantic)
- ✅ Register MCP server with Claude Code
- ✅ Configure notification listener with sound alerts
- ✅ Set up environment configuration

### Start Services

```bash
# 1. Start the visualizer and analyzer APIs
cd /home/evgeny/projects/visualizer
./start

# 2. Start the notification listener (in a new terminal)
cd /home/evgeny/projects/visualizer/mcp_server
./start-listener.sh
```

### Use in Claude Code

Open Claude Code and try:

```
Get AI recommendations for ~/Documents/research-paper.pdf
```

Claude will:
1. Read the document
2. Call `get_ai_recommendations()`
3. Show you 4-5 suggested engines
4. Ask which ones you want to use
5. Submit the analysis
6. Monitor progress
7. Download results when complete
8. Alert you with a sound notification

---

## 🛠️ Available Tools

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
    },
    {
      "engine_key": "evidence-data-mapper",
      "engine_name": "Evidence & Data Mapper",
      "confidence": 0.88
    }
  ],
  "document_characteristics": {
    "type": "Academic Research Paper",
    "style": "Empirical Study",
    "focus": "Quantitative Analysis"
  },
  "analysis_strategy": "Use argument mapping to visualize claims, then evidence mapping to validate data sources"
}
```

**Example Usage in Claude Code:**
```
User: "I need to analyze this document: ~/papers/climate-policy.pdf"
Claude: <calls get_ai_recommendations>
Claude: "I've analyzed your document. It's an academic policy paper.
         I recommend these 4 engines:
         1. Policy Flow Tracker (92% confidence)
         2. Stakeholder Network Mapper (88% confidence)
         3. Evidence & Data Mapper (85% confidence)
         4. Temporal Analysis Tracker (82% confidence)

         Would you like me to run all 4, or just specific ones?"
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

**Returns:**
```json
{
  "total_engines": 47,
  "engines": [
    {
      "key": "anomaly-detector",
      "name": "Anomaly Detector",
      "description": "Finds conceptual outliers, contradictions, and unexpected patterns...",
      "category": "AI Engines"
    }
  ]
}
```

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
- `"visual"` - 4K PNG images via Gemini Nano Banana (image generation)
- `"textual"` - Markdown reports and text diagrams

**Returns:**
```json
{
  "job_id": "job_abc123",
  "status": "submitted",
  "document": "climate-policy.pdf",
  "engines": ["policy-flow-tracker", "stakeholder-network-mapper"],
  "output_mode": "visual",
  "monitor_url": "http://localhost:5847/api/analyzer/jobs/job_abc123",
  "message": "Analysis job submitted successfully. Auto-monitoring enabled - you'll be notified when complete."
}
```

**Example:**
```
User: "Run all 4 engines with visual output"
Claude: <calls submit_analysis with all 4 engine keys>
Claude: "✅ Analysis started! Job ID: job_abc123

         I'm monitoring the job and will notify you when it completes.
         This typically takes 5-10 minutes for visual outputs.

         You'll hear a sound alert when results are ready."
```

---

### 4. `check_job_status`

**Monitor analysis progress**

```python
check_job_status(
    job_id: str  # Job ID from submit_analysis
)
```

**Returns:**
```json
{
  "job_id": "job_abc123",
  "status": "running",  // pending | running | completed | failed
  "created_at": "2025-12-15T16:30:00Z",
  "updated_at": "2025-12-15T16:32:00Z",
  "message": "Job is running. Check back in a few moments."
}
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

**Returns:**
```json
{
  "job_id": "job_abc123",
  "download_directory": "~/visualizer-results/job_abc123_20251215_163500",
  "files_downloaded": 4,
  "files": [
    "~/visualizer-results/job_abc123_20251215_163500/policy-flow-tracker.png",
    "~/visualizer-results/job_abc123_20251215_163500/stakeholder-network-mapper.png",
    "~/visualizer-results/job_abc123_20251215_163500/evidence-data-mapper.png",
    "~/visualizer-results/job_abc123_20251215_163500/temporal-analysis-tracker.png",
    "~/visualizer-results/job_abc123_20251215_163500/results.json"
  ]
}
```

**Automatic Download:**
When `auto_monitor: true` is set in `submit_analysis()`, results are automatically downloaded when the job completes, and you receive a notification with sound alert.

---

### 6. `list_bundles`

**View pre-configured engine bundles**

Bundles are groups of engines that work well together:
- **Argument & Reasoning Bundle** - 5 engines for logical analysis
- **Temporal & Historical Bundle** - 7 engines for timeline analysis
- **Power & Resources Bundle** - 4 engines for stakeholder analysis
- etc.

---

### 7. `list_pipelines`

**View analysis pipelines**

Pipelines are sequences where engine outputs feed into subsequent engines:
- **Chain 1**: Argument Architecture → Evidence Data → Assumption Excavator
- **Chain 2**: Chronology Mapper → Temporal Analysis → Historical Comparison
- etc.

---

## 📋 Configuration

### Environment File

Located at `/home/evgeny/projects/visualizer/mcp_server/.env.mcp`:

```bash
# API Endpoints
VISUALIZER_API_URL=http://localhost:5847
ANALYZER_API_URL=http://localhost:8847

# Notifications
VISUALIZER_NTFY_TOPIC=visualizer-evgeny-1234

# Output
VISUALIZER_OUTPUT_DIR=~/visualizer-results

# Optional: Default API Keys
# GEMINI_API_KEY=your_gemini_key_here
# ANTHROPIC_API_KEY=your_anthropic_key_here
```

### API Keys

You can provide API keys in 3 ways:

1. **Per-Request** (Recommended)
   ```
   get_ai_recommendations(
       document_path="~/doc.pdf",
       gemini_api_key="your_key",
       anthropic_api_key="your_key"
   )
   ```

2. **Environment Variable**
   ```bash
   export GEMINI_API_KEY="your_key"
   export ANTHROPIC_API_KEY="your_key"
   ```

3. **In .env.mcp file**
   ```bash
   GEMINI_API_KEY=your_key
   ANTHROPIC_API_KEY=your_key
   ```

---

## 🔔 Notifications

The notification listener subscribes to `ntfy.sh` and plays sound alerts when:

- ✅ **Analysis Complete** - 7 beeps + desktop notification
- ❌ **Analysis Failed** - 5 beeps + desktop notification
- 📊 **Status Update** - 3 beeps + desktop notification

### Start Listener

```bash
cd /home/evgeny/projects/visualizer/mcp_server
./start-listener.sh
```

### Auto-Start on Session (Optional)

Add to `~/.claude/settings.json` under `SessionStart` hooks:

```json
{
  "type": "command",
  "command": "~/.claude/hooks/active/visualizer-listener.sh"
}
```

---

## 📁 Directory Structure

```
/home/evgeny/projects/visualizer/mcp_server/
├── mcp_server.py              # Main MCP server
├── install-mcp-server.sh      # Installation script
├── visualizer-listener.sh     # Notification listener
├── start-listener.sh          # Listener starter
├── run-mcp-server.sh          # MCP server wrapper (generated)
├── .env.mcp                   # Configuration (generated)
├── venv/                      # Python virtual environment (generated)
└── README.md                  # This file
```

---

## 🔍 Example Workflows

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
  9. Calls get_results() automatically
  10. "✅ Analysis complete! Results downloaded to ~/visualizer-results/..."
  11. Shows user the file paths
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

### Workflow 3: Bundle Analysis

```
User: "Run the full temporal analysis bundle on my timeline.pdf"

Claude:
  1. Calls list_bundles()
  2. Finds "Temporal & Historical Bundle"
  3. Extracts engine keys from bundle
  4. Calls submit_analysis() with all bundle engines
  5. Notifies when complete
```

---

## 🐛 Troubleshooting

### MCP Server Not Found

```bash
# Check if registered
claude mcp list

# Re-register
cd /home/evgeny/projects/visualizer/mcp_server
./install-mcp-server.sh
```

### API Connection Errors

```bash
# Ensure services are running
cd /home/evgeny/projects/visualizer
./start

# Check status
curl http://localhost:5847/api/analyzer/engines
curl http://localhost:8847/v1/engines
```

### No Notifications

```bash
# Check listener status
ps aux | grep visualizer-listener

# Restart listener
pkill -f visualizer-listener
./start-listener.sh
```

### Visual Output Requires Gemini API Key

For `output_mode="visual"`, you **must** provide a Gemini API key:

```python
submit_analysis(
    document_path="~/doc.pdf",
    engine_keys=["anomaly-detector"],
    output_mode="visual",
    gemini_api_key="your_gemini_key_here"  # Required for visual
)
```

---

## 📚 Additional Resources

- **Visualizer Web UI**: http://localhost:5847
- **Analyzer API Docs**: http://localhost:8847/docs
- **ntfy.sh Docs**: https://ntfy.sh
- **FastMCP Docs**: https://github.com/jlowin/fastmcp

---

## 🤝 Contributing

Found a bug or want to add features?

1. Test locally first
2. Check logs in `/tmp/visualizer_listener.log`
3. Update this README if adding new tools
4. Ensure backward compatibility

---

## 📄 License

This MCP server is part of the Visualizer project.

---

## 🎉 Success Checklist

After installation, verify:

- [ ] `claude mcp list` shows "visualizer"
- [ ] Visualizer API running on port 5847
- [ ] Analyzer API running on port 8847
- [ ] Notification listener running (`ps aux | grep visualizer-listener`)
- [ ] Test: "Get AI recommendations for ~/Documents/test.pdf"
- [ ] Notification sound plays when job completes
- [ ] Results downloaded to `~/visualizer-results/`

---

**🚀 You're ready to analyze documents with AI-powered insights!**
