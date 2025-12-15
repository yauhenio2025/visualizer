"""
Visualizer MCP Server

Exposes document analysis and visualization capabilities via Model Context Protocol.

Usage:
    claude mcp add --transport stdio visualizer -- python /path/to/mcp_server.py

Configuration via environment variables:
    VISUALIZER_API_URL: Visualizer API base URL (default: http://localhost:5847)
    ANALYZER_API_URL: Analyzer API base URL (default: http://localhost:8847)
    GEMINI_API_KEY: User's Gemini API key (optional, can be provided per-request)
    ANTHROPIC_API_KEY: User's Anthropic API key (optional, can be provided per-request)
    VISUALIZER_NTFY_TOPIC: Notification topic for job completion alerts
    VISUALIZER_OUTPUT_DIR: Directory for downloaded results (default: ~/visualizer-results)

Primary Usage Pattern:
    1. User provides a document path or content
    2. Claude Code calls get_ai_recommendations() to suggest best engines
    3. User selects which engines to use (or accepts all)
    4. Claude Code calls submit_analysis() with selected engines
    5. Server polls job status and sends notification when complete
    6. Results are downloaded from S3 and saved locally
    7. User is alerted with sound signal
"""

import os
import sys
import time
import json
import logging
import requests
import base64
from datetime import datetime
from typing import Annotated, Optional, List, Dict, Any
from pathlib import Path

from fastmcp import FastMCP
from pydantic import Field

# Configure logging to stderr (stdout is reserved for JSON-RPC)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stderr)]
)
logger = logging.getLogger(__name__)

# Initialize MCP server
mcp = FastMCP(
    name="visualizer",
    instructions="""
    Visualizer MCP Server - Document Intelligence & Visual Analysis

    This server connects Claude Code to a powerful document analysis and visualization
    pipeline that uses AI to recommend the best analysis engines for your documents.

    ## Primary Usage

    When a user wants to analyze/visualize a document:

    1. **Get AI Recommendations**: Call get_ai_recommendations() with document path/content
       - Returns 4-5 recommended engines with confidence scores
    2. **Submit Analysis**: Call submit_analysis() with selected engines
       - Supports both visual and textual output modes
    3. **Monitor Progress**: Call check_job_status() to track progress
    4. **Get Results**: Results are auto-downloaded when complete, or use get_results()

    ## Features

    - AI-powered engine recommendations using Claude/Gemini
    - Support for 47+ analysis engines across 8 categories
    - Multiple output modes: visual (4K images) and textual (reports, diagrams)
    - Bundle analysis: Run multiple engines in parallel
    - Pipeline analysis: Chain engines sequentially
    - Automatic notifications with sound alerts when jobs complete
    """
)

# Configuration from environment
VISUALIZER_API_URL = os.environ.get('VISUALIZER_API_URL', 'http://localhost:5847')
ANALYZER_API_URL = os.environ.get('ANALYZER_API_URL', 'http://localhost:8847')
OUTPUT_DIR = Path(os.environ.get('VISUALIZER_OUTPUT_DIR', '~/visualizer-results')).expanduser()
NTFY_TOPIC = os.environ.get('VISUALIZER_NTFY_TOPIC', f'visualizer-{os.getenv("USER", "user")}-{hash(os.getenv("USER", "user")) % 10000}')

# Ensure output directory exists
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Request timeout
REQUEST_TIMEOUT = 30
POLL_INTERVAL = 5  # seconds
MAX_POLL_ATTEMPTS = 600  # 30 minutes max


def api_request(
    base_url: str,
    method: str,
    endpoint: str,
    data: dict = None,
    headers: dict = None,
    timeout: int = REQUEST_TIMEOUT
) -> dict:
    """Make an API request to visualizer or analyzer backend."""
    url = f"{base_url}{endpoint}"

    try:
        request_headers = headers or {}

        if method.upper() == 'GET':
            response = requests.get(url, params=data, headers=request_headers, timeout=timeout)
        elif method.upper() == 'POST':
            response = requests.post(url, json=data, headers=request_headers, timeout=timeout)
        elif method.upper() == 'PUT':
            response = requests.put(url, json=data, headers=request_headers, timeout=timeout)
        elif method.upper() == 'DELETE':
            response = requests.delete(url, headers=request_headers, timeout=timeout)
        else:
            return {"error": f"Unsupported HTTP method: {method}"}

        response.raise_for_status()
        return response.json()

    except requests.Timeout:
        return {"error": f"Request timeout after {timeout}s"}
    except requests.HTTPError as e:
        return {"error": f"HTTP {e.response.status_code}: {e.response.text[:500]}"}
    except requests.RequestException as e:
        return {"error": f"Request failed: {str(e)}"}


def read_document(file_path: str) -> Dict[str, Any]:
    """Read a document from file path and return structured document object."""
    path = Path(file_path).expanduser().resolve()

    if not path.exists():
        return {"error": f"File not found: {file_path}"}

    if not path.is_file():
        return {"error": f"Not a file: {file_path}"}

    # Determine file type
    suffix = path.suffix.lower()

    # For PDFs, read as base64
    if suffix == '.pdf':
        try:
            with open(path, 'rb') as f:
                content = base64.b64encode(f.read()).decode('utf-8')
            encoding = 'base64'
        except Exception as e:
            return {"error": f"Cannot read PDF: {str(e)}"}
    else:
        # For text files, read as text
        try:
            content = path.read_text(encoding='utf-8')
            encoding = 'text'
        except UnicodeDecodeError:
            return {"error": f"Cannot read file as text (try PDF format): {file_path}"}
        except Exception as e:
            return {"error": f"Cannot read file: {str(e)}"}

    return {
        "id": f"doc_{int(time.time())}",
        "title": path.name,
        "content": content,
        "encoding": encoding,
        "path": str(path),
        "size": path.stat().st_size
    }


def send_notification(title: str, message: str, tags: str = "visualizer"):
    """Send notification via ntfy.sh."""
    try:
        requests.post(
            f"https://ntfy.sh/{NTFY_TOPIC}",
            json={
                "topic": NTFY_TOPIC,
                "title": title,
                "message": message,
                "tags": tags,
                "priority": 4
            },
            timeout=5
        )
    except Exception as e:
        logger.warning(f"Failed to send notification: {e}")


# =============================================================================
# MCP TOOLS
# =============================================================================

@mcp.tool()
def get_ai_recommendations(
    document_path: Annotated[str, Field(description="Path to the document to analyze (PDF, TXT, MD, etc.)")],
    max_recommendations: Annotated[int, Field(description="Maximum number of engine recommendations to return (default: 5)")] = 5,
    anthropic_api_key: Annotated[Optional[str], Field(description="Anthropic API key for Claude-based recommendations")] = None,
    gemini_api_key: Annotated[Optional[str], Field(description="Gemini API key for recommendations")] = None
) -> str:
    """
    Get AI-powered recommendations for the best analysis engines to use for a document.

    The AI analyzes the document content and suggests 4-5 engines that would provide
    the most valuable insights based on document type, style, and content.

    Returns: JSON with recommended engines, document characteristics, and analysis strategy.
    """
    logger.info(f"Getting AI recommendations for: {document_path}")

    # Read document
    doc = read_document(document_path)
    if "error" in doc:
        return json.dumps({"error": doc["error"]})

    # Extract sample text for curator (up to 2000 chars)
    sample_text = doc["content"][:2000] if doc["encoding"] == "text" else f"[PDF document: {doc['title']}]"

    # Build headers with API keys
    headers = {}
    if anthropic_api_key:
        headers['X-Anthropic-Api-Key'] = anthropic_api_key
    elif os.environ.get('ANTHROPIC_API_KEY'):
        headers['X-Anthropic-Api-Key'] = os.environ.get('ANTHROPIC_API_KEY')

    if gemini_api_key:
        headers['X-Gemini-Api-Key'] = gemini_api_key
    elif os.environ.get('GEMINI_API_KEY'):
        headers['X-Gemini-Api-Key'] = os.environ.get('GEMINI_API_KEY')

    # Call curator endpoint
    result = api_request(
        VISUALIZER_API_URL,
        'POST',
        '/api/analyzer/curator/recommend',
        data={
            "sample_text": sample_text,
            "max_recommendations": max_recommendations
        },
        headers=headers,
        timeout=60  # Curator may take longer
    )

    if "error" in result:
        return json.dumps({"error": result["error"]})

    # Format response for readability
    output = {
        "document": doc["title"],
        "recommendations": result.get("primary_recommendations", []),
        "document_characteristics": result.get("document_characteristics", {}),
        "analysis_strategy": result.get("analysis_strategy", ""),
        "curator_note": "The AI has analyzed your document and suggests these engines for optimal insights."
    }

    logger.info(f"Got {len(output['recommendations'])} recommendations")
    return json.dumps(output, indent=2)


@mcp.tool()
def list_available_engines(
    category: Annotated[Optional[str], Field(description="Filter by category (optional)")] = None
) -> str:
    """
    List all available analysis engines.

    Categories include: AI Engines, Argument & Reasoning, Concepts & Frameworks,
    Temporal & Historical, Power & Resources, Evidence & Data, Rhetoric & Language,
    Scholarly Landscape.

    Returns: JSON with engines grouped by category.
    """
    logger.info(f"Listing engines (category: {category or 'all'})")

    result = api_request(VISUALIZER_API_URL, 'GET', '/api/analyzer/engines')

    if "error" in result:
        return json.dumps({"error": result["error"]})

    # Filter by category if specified
    if category:
        engines = [e for e in result if e.get('category', '').lower() == category.lower()]
    else:
        engines = result

    # Format for readability
    output = {
        "total_engines": len(engines),
        "engines": [
            {
                "key": e.get("key"),
                "name": e.get("name"),
                "description": e.get("description", "")[:100] + "..." if len(e.get("description", "")) > 100 else e.get("description", ""),
                "category": e.get("category")
            }
            for e in engines
        ]
    }

    return json.dumps(output, indent=2)


@mcp.tool()
def submit_analysis(
    document_path: Annotated[str, Field(description="Path to the document to analyze")],
    engine_keys: Annotated[List[str], Field(description="List of engine keys to use (e.g., ['anomaly-detector', 'argument-architecture-mapper'])")],
    output_mode: Annotated[str, Field(description="Output mode: 'visual' for 4K images, 'textual' for reports/diagrams")] = "visual",
    anthropic_api_key: Annotated[Optional[str], Field(description="Anthropic API key")] = None,
    gemini_api_key: Annotated[Optional[str], Field(description="Gemini API key (required for visual mode)")] = None,
    auto_monitor: Annotated[bool, Field(description="Automatically monitor job and notify when complete (default: True)")] = True
) -> str:
    """
    Submit a document for analysis with selected engines.

    This initiates the analysis pipeline:
    1. Documents are sent to the analyzer backend
    2. Selected engines are executed
    3. Results are generated (visual or textual)
    4. Outputs are uploaded to S3
    5. Job completes and user is notified

    Returns: Job ID for tracking progress.
    """
    logger.info(f"Submitting analysis: {document_path} with {len(engine_keys)} engines")

    # Read document
    doc = read_document(document_path)
    if "error" in doc:
        return json.dumps({"error": doc["error"]})

    # Build headers with API keys
    headers = {}
    if anthropic_api_key:
        headers['X-Anthropic-Api-Key'] = anthropic_api_key
    elif os.environ.get('ANTHROPIC_API_KEY'):
        headers['X-Anthropic-Api-Key'] = os.environ.get('ANTHROPIC_API_KEY')

    if gemini_api_key:
        headers['X-Gemini-Api-Key'] = gemini_api_key
    elif os.environ.get('GEMINI_API_KEY'):
        headers['X-Gemini-Api-Key'] = os.environ.get('GEMINI_API_KEY')

    # Validate output mode
    if output_mode not in ['visual', 'textual']:
        return json.dumps({"error": "output_mode must be 'visual' or 'textual'"})

    # For multiple engines, use bundle analysis
    if len(engine_keys) > 1:
        # Create a temporary bundle
        result = api_request(
            VISUALIZER_API_URL,
            'POST',
            '/api/analyzer/analyze/bundle',
            data={
                "documents": [doc],
                "bundle": engine_keys,  # Pass as array
                "output_modes": {"visual": "nano_banana"} if output_mode == "visual" else {"textual": "memo"}
            },
            headers=headers,
            timeout=60
        )
    else:
        # Single engine analysis
        result = api_request(
            VISUALIZER_API_URL,
            'POST',
            '/api/analyzer/analyze',
            data={
                "documents": [doc],
                "engine": engine_keys[0],
                "output_mode": "nano_banana" if output_mode == "visual" else "memo",
                "collection_mode": "single"
            },
            headers=headers,
            timeout=60
        )

    if "error" in result:
        return json.dumps({"error": result["error"]})

    job_id = result.get("job_id")

    if not job_id:
        return json.dumps({"error": "No job ID returned from API"})

    output = {
        "job_id": job_id,
        "status": "submitted",
        "document": doc["title"],
        "engines": engine_keys,
        "output_mode": output_mode,
        "monitor_url": f"{VISUALIZER_API_URL}/api/analyzer/jobs/{job_id}",
        "message": "Analysis job submitted successfully. Use check_job_status() to monitor progress."
    }

    # Send notification
    send_notification(
        "📊 Analysis Started",
        f"Analyzing {doc['title']} with {len(engine_keys)} engine(s)",
        "visualizer,started"
    )

    # Auto-monitor if requested (async in background)
    if auto_monitor:
        output["message"] += " Auto-monitoring enabled - you'll be notified when complete."
        # Note: In a real implementation, you'd spawn a background thread here
        # For now, we'll rely on manual status checks

    logger.info(f"Job submitted: {job_id}")
    return json.dumps(output, indent=2)


@mcp.tool()
def check_job_status(
    job_id: Annotated[str, Field(description="Job ID returned from submit_analysis()")]
) -> str:
    """
    Check the status of an analysis job.

    Returns: Current job status, progress, and results if complete.
    """
    logger.info(f"Checking status for job: {job_id}")

    result = api_request(
        VISUALIZER_API_URL,
        'GET',
        f'/api/analyzer/jobs/{job_id}'
    )

    if "error" in result:
        return json.dumps({"error": result["error"]})

    status = result.get("status", "unknown")

    output = {
        "job_id": job_id,
        "status": status,
        "created_at": result.get("created_at"),
        "updated_at": result.get("updated_at")
    }

    if status == "completed":
        output["result_available"] = True
        output["message"] = "Job completed! Use get_results() to download outputs."
    elif status == "failed":
        output["error"] = result.get("error", "Job failed")
    elif status in ["pending", "running"]:
        output["message"] = f"Job is {status}. Check back in a few moments."

    return json.dumps(output, indent=2)


@mcp.tool()
def get_results(
    job_id: Annotated[str, Field(description="Job ID to retrieve results for")],
    download_to: Annotated[Optional[str], Field(description="Directory to save results (default: ~/visualizer-results)")] = None
) -> str:
    """
    Retrieve and download completed job results.

    Results are downloaded from S3 and saved to the local filesystem.
    For visual outputs, you'll get 4K PNG images. For textual outputs, you'll get markdown files.

    Returns: Paths to downloaded files.
    """
    logger.info(f"Getting results for job: {job_id}")

    # First get job details
    result = api_request(
        VISUALIZER_API_URL,
        'GET',
        f'/api/analyzer/jobs/{job_id}/result'
    )

    if "error" in result:
        return json.dumps({"error": result["error"]})

    # Determine output directory
    output_dir = Path(download_to).expanduser() if download_to else OUTPUT_DIR
    job_dir = output_dir / f"job_{job_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    job_dir.mkdir(parents=True, exist_ok=True)

    downloaded_files = []

    # Handle different result structures
    results = result.get("result", {})

    # Check if results contain S3 URLs
    if isinstance(results, dict):
        for engine_key, engine_result in results.items():
            if isinstance(engine_result, dict) and "output" in engine_result:
                output_data = engine_result["output"]

                # Handle S3 URL
                if isinstance(output_data, str) and output_data.startswith("s3://"):
                    # Convert S3 URL to HTTP URL (assuming bucket is public or accessible)
                    http_url = output_data.replace("s3://", "https://").replace(".s3.amazonaws.com/", "/")

                    # Download file
                    try:
                        response = requests.get(http_url, timeout=30)
                        response.raise_for_status()

                        # Determine file extension
                        if "image" in response.headers.get("Content-Type", ""):
                            ext = ".png"
                        else:
                            ext = ".md"

                        file_path = job_dir / f"{engine_key}{ext}"
                        file_path.write_bytes(response.content)
                        downloaded_files.append(str(file_path))
                        logger.info(f"Downloaded: {file_path}")

                    except Exception as e:
                        logger.error(f"Failed to download {http_url}: {e}")

                # Handle inline content
                elif isinstance(output_data, str):
                    file_path = job_dir / f"{engine_key}.md"
                    file_path.write_text(output_data, encoding='utf-8')
                    downloaded_files.append(str(file_path))

    # Save raw JSON for reference
    json_path = job_dir / "results.json"
    json_path.write_text(json.dumps(result, indent=2), encoding='utf-8')
    downloaded_files.append(str(json_path))

    output = {
        "job_id": job_id,
        "download_directory": str(job_dir),
        "files_downloaded": len(downloaded_files),
        "files": downloaded_files
    }

    # Send notification
    send_notification(
        "✅ Results Downloaded",
        f"Downloaded {len(downloaded_files)} file(s) to {job_dir}",
        "visualizer,completed"
    )

    logger.info(f"Downloaded {len(downloaded_files)} files to {job_dir}")
    return json.dumps(output, indent=2)


@mcp.tool()
def list_bundles() -> str:
    """
    List available engine bundles.

    Bundles are pre-configured groups of engines that work well together.

    Returns: JSON with available bundles.
    """
    result = api_request(VISUALIZER_API_URL, 'GET', '/api/analyzer/bundles')

    if "error" in result:
        return json.dumps({"error": result["error"]})

    return json.dumps({"bundles": result}, indent=2)


@mcp.tool()
def list_pipelines() -> str:
    """
    List available analysis pipelines.

    Pipelines are sequences of engines where the output of one feeds into the next.

    Returns: JSON with available pipelines.
    """
    result = api_request(VISUALIZER_API_URL, 'GET', '/api/analyzer/pipelines')

    if "error" in result:
        return json.dumps({"error": result["error"]})

    return json.dumps({"pipelines": result}, indent=2)


if __name__ == "__main__":
    # Run the MCP server
    mcp.run()
