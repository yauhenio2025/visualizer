"""
Visualizer MCP Server

Exposes document analysis and visualization capabilities via Model Context Protocol.

Usage:
    claude mcp add --transport stdio visualizer -- python /path/to/mcp_server.py

Configuration via environment variables:
    VISUALIZER_API_URL: Visualizer API base URL (default: https://visualizer-tw4i.onrender.com)
    ANALYZER_API_URL: Analyzer API base URL (default: https://analyzer-3wsg.onrender.com)
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

# PDF support
try:
    import fitz  # pymupdf
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False
    logging.warning("pymupdf not installed - PDF text extraction disabled")

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
    - Batch analysis: Process entire folders of documents
    - Automatic notifications with sound alerts when jobs complete
    """
)

# Configuration from environment
VISUALIZER_API_URL = os.environ.get('VISUALIZER_API_URL', 'https://visualizer-tw4i.onrender.com')
ANALYZER_API_URL = os.environ.get('ANALYZER_API_URL', 'https://analyzer-3wsg.onrender.com')
OUTPUT_DIR = Path(os.environ.get('VISUALIZER_OUTPUT_DIR', '~/visualizer-results')).expanduser()
NTFY_TOPIC = os.environ.get('VISUALIZER_NTFY_TOPIC', f'visualizer-{os.getenv("USER", "user")}-{hash(os.getenv("USER", "user")) % 10000}')

# Ensure output directory exists
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Request timeout
REQUEST_TIMEOUT = 30
POLL_INTERVAL = 5  # seconds
MAX_POLL_ATTEMPTS = 600  # 30 minutes max

# Supported file extensions for batch processing
SUPPORTED_EXTENSIONS = {'.pdf', '.txt', '.md', '.markdown', '.rst', '.tex'}


def get_llm_keys(anthropic_api_key: Optional[str] = None, gemini_api_key: Optional[str] = None) -> Dict[str, str]:
    """Build llm_keys dict from provided keys or environment."""
    llm_keys = {}

    if anthropic_api_key:
        llm_keys['anthropic_api_key'] = anthropic_api_key
    elif os.environ.get('ANTHROPIC_API_KEY'):
        llm_keys['anthropic_api_key'] = os.environ.get('ANTHROPIC_API_KEY')

    if gemini_api_key:
        llm_keys['gemini_api_key'] = gemini_api_key
    elif os.environ.get('GEMINI_API_KEY'):
        llm_keys['gemini_api_key'] = os.environ.get('GEMINI_API_KEY')

    return llm_keys


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
        error_text = ""
        try:
            error_text = e.response.text[:500]
        except:
            pass
        return {"error": f"API error: {str(e)}", "details": error_text}
    except requests.RequestException as e:
        return {"error": f"Request failed: {str(e)}"}


def extract_pdf_text(file_path: Path) -> str:
    """Extract text from a PDF file using PyMuPDF."""
    if not PDF_SUPPORT:
        return "[PDF text extraction not available - install pymupdf]"

    try:
        doc = fitz.open(str(file_path))
        text_parts = []
        for page_num, page in enumerate(doc):
            text = page.get_text()
            if text.strip():
                text_parts.append(f"--- Page {page_num + 1} ---\n{text}")
        doc.close()
        return "\n\n".join(text_parts) if text_parts else "[No text extracted from PDF]"
    except Exception as e:
        return f"[Error extracting PDF text: {str(e)}]"


def read_document(file_path: str) -> Dict[str, Any]:
    """Read a document from file path and return structured document object."""
    path = Path(file_path).expanduser().resolve()

    if not path.exists():
        return {"error": f"File not found: {file_path}"}

    if not path.is_file():
        return {"error": f"Not a file: {file_path}"}

    # Determine file type
    suffix = path.suffix.lower()

    # For PDFs, read as base64 AND extract text
    if suffix == '.pdf':
        try:
            with open(path, 'rb') as f:
                content = base64.b64encode(f.read()).decode('utf-8')
            encoding = 'base64'
            # Also extract text for sample_text usage
            extracted_text = extract_pdf_text(path)
        except Exception as e:
            return {"error": f"Cannot read PDF: {str(e)}"}
    else:
        # For text files, read as text
        try:
            content = path.read_text(encoding='utf-8')
            encoding = 'text'
            extracted_text = content  # Same as content for text files
        except UnicodeDecodeError:
            return {"error": f"Cannot read file as text (try PDF format): {file_path}"}
        except Exception as e:
            return {"error": f"Cannot read file: {str(e)}"}

    return {
        "id": f"doc_{int(time.time())}_{hash(str(path)) % 10000}",
        "title": path.name,
        "content": content,
        "encoding": encoding,
        "extracted_text": extracted_text,  # Always have text available
        "path": str(path),
        "size": path.stat().st_size
    }


def send_notification(title: str, message: str, tags: str = "visualizer", sound: bool = True):
    """Send notification via ntfy.sh.

    Args:
        title: Notification title
        message: Notification body
        tags: Comma-separated tags
        sound: If True, plays sound (priority 4). If False, silent (priority 3).
    """
    try:
        requests.post(
            f"https://ntfy.sh/{NTFY_TOPIC}",
            json={
                "topic": NTFY_TOPIC,
                "title": title,
                "message": message,
                "tags": tags,
                "priority": 4 if sound else 3  # 4=high (sound), 3=default (silent)
            },
            timeout=5
        )
    except Exception as e:
        logger.warning(f"Failed to send notification: {e}")


def download_file_from_url(url: str, output_path: Path) -> bool:
    """Download a file from URL to the specified path."""
    try:
        response = requests.get(url, timeout=60, stream=True)
        response.raise_for_status()

        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        logger.info(f"Downloaded: {output_path}")
        return True
    except Exception as e:
        logger.error(f"Failed to download {url}: {e}")
        return False


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
    # Use extracted_text which has actual text content for PDFs
    sample_text = doc.get("extracted_text", doc["content"])[:2000]

    # Build llm_keys for request body
    llm_keys = get_llm_keys(anthropic_api_key, gemini_api_key)

    # Also build headers as backup (for curator endpoint)
    headers = {}
    if llm_keys.get('anthropic_api_key'):
        headers['X-Anthropic-Api-Key'] = llm_keys['anthropic_api_key']
    if llm_keys.get('gemini_api_key'):
        headers['X-Gemini-Api-Key'] = llm_keys['gemini_api_key']

    # Call curator endpoint with llm_keys in body
    result = api_request(
        VISUALIZER_API_URL,
        'POST',
        '/api/analyzer/curator/recommend',
        data={
            "sample_text": sample_text,
            "max_recommendations": max_recommendations,
            "llm_keys": llm_keys  # Include in body
        },
        headers=headers,  # Also in headers as backup
        timeout=120  # Curator may take longer
    )

    if "error" in result:
        return json.dumps({"error": result["error"], "details": result.get("details", "")})

    # Format response for readability
    output = {
        "document": doc["title"],
        "recommendations": result.get("primary_recommendations", []),
        "bundle_recommendations": result.get("bundle_recommendations", []),
        "pipeline_recommendations": result.get("pipeline_recommendations", []),
        "document_characteristics": result.get("document_characteristics", {}),
        "analysis_strategy": result.get("analysis_strategy", ""),
        "curator_note": "The AI has analyzed your document and suggests these engines, bundles, and pipelines for optimal insights."
    }

    num_engines = len(output['recommendations'])
    num_bundles = len(output['bundle_recommendations'])
    num_pipelines = len(output['pipeline_recommendations'])
    logger.info(f"Got {num_engines} engine, {num_bundles} bundle, {num_pipelines} pipeline recommendations")
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

    # Handle if result is a list directly
    engines_list = result if isinstance(result, list) else result.get('engines', [])

    # Filter by category if specified
    if category:
        engines = [e for e in engines_list if e.get('category', '').lower() == category.lower()]
    else:
        engines = engines_list

    # Format for readability - handle missing keys gracefully
    output = {
        "total_engines": len(engines),
        "engines": [
            {
                "key": e.get("key") or e.get("engine_key"),
                "name": e.get("name") or e.get("engine_name"),
                "description": (e.get("description", "") or "")[:100] + "..." if len(e.get("description", "") or "") > 100 else (e.get("description", "") or ""),
                "category": e.get("category")
            }
            for e in engines
        ]
    }

    return json.dumps(output, indent=2)


@mcp.tool()
def submit_analysis(
    document_path: Annotated[str, Field(description="Path to the document to analyze")],
    engine_keys: Annotated[List[str], Field(description="List of engine keys to use (e.g., ['anomaly_detector', 'argument_architecture'])")],
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

    # Build llm_keys for request body (this is what the visualizer expects!)
    llm_keys = get_llm_keys(anthropic_api_key, gemini_api_key)

    # Validate output mode
    if output_mode not in ['visual', 'textual']:
        return json.dumps({"error": "output_mode must be 'visual' or 'textual'"})

    # Determine the actual output mode string for the API
    # Valid modes: gemini_image, structured_text_report, executive_memo, mermaid, d3_interactive, etc.
    api_output_mode = "gemini_image" if output_mode == "visual" else "executive_memo"

    # Prepare document for submission (remove extra fields the API doesn't need)
    doc_for_api = {
        "id": doc["id"],
        "title": doc["title"],
        "content": doc["content"],
        "encoding": doc["encoding"]
    }

    job_ids = []
    errors = []

    # Submit each engine separately (more reliable than bundle for arbitrary engine lists)
    for engine_key in engine_keys:
        result = api_request(
            VISUALIZER_API_URL,
            'POST',
            '/api/analyzer/analyze',
            data={
                "documents": [doc_for_api],
                "engine": engine_key,
                "output_mode": api_output_mode,
                "collection_mode": "single",
                "llm_keys": llm_keys  # Pass API keys in body!
            },
            timeout=120
        )

        if "error" in result:
            errors.append(f"{engine_key}: {result['error']}")
        elif result.get("job_id"):
            job_ids.append({
                "engine": engine_key,
                "job_id": result["job_id"]
            })
        else:
            errors.append(f"{engine_key}: No job ID returned")

    if not job_ids and errors:
        return json.dumps({"error": "All submissions failed", "details": errors})

    output = {
        "jobs": job_ids,
        "status": "submitted",
        "document": doc["title"],
        "engines": engine_keys,
        "output_mode": output_mode,
        "errors": errors if errors else None,
        "message": f"Submitted {len(job_ids)} analysis job(s). Use check_job_status() with each job_id to monitor progress."
    }

    # Send notification (silent - no sound for submission)
    send_notification(
        "📊 Analysis Started",
        f"Analyzing {doc['title']} with {len(job_ids)} engine(s)",
        "visualizer,started",
        sound=False
    )

    if auto_monitor:
        output["message"] += " You'll be notified when jobs complete."

    logger.info(f"Jobs submitted: {job_ids}")
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
        send_notification(
            "✅ Job Complete",
            f"Analysis job {job_id[:8]}... is ready",
            "visualizer,completed"
        )
    elif status == "failed":
        output["error"] = result.get("error", "Job failed")
        send_notification(
            "❌ Job Failed",
            f"Analysis job {job_id[:8]}... failed",
            "visualizer,failed"
        )
    elif status in ["pending", "running", "extracting", "curating", "rendering"]:
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

    # Handle the CORRECT response structure: outputs.{engine_key}.{image_url|text|...}
    outputs = result.get("outputs", {})

    if isinstance(outputs, dict):
        for engine_key, engine_result in outputs.items():
            if not isinstance(engine_result, dict):
                continue

            # Check for image_url (visual mode - Gemini image)
            image_url = engine_result.get("image_url")
            if image_url:
                # Determine file extension from URL or default to .jpg
                if ".png" in image_url.lower():
                    ext = ".png"
                elif ".jpg" in image_url.lower() or ".jpeg" in image_url.lower():
                    ext = ".jpg"
                else:
                    ext = ".jpg"  # Default for S3 images

                file_path = job_dir / f"{engine_key}{ext}"
                if download_file_from_url(image_url, file_path):
                    downloaded_files.append(str(file_path))
                continue

            # Check for text content (textual mode)
            text_content = engine_result.get("text") or engine_result.get("content") or engine_result.get("output")
            if text_content and isinstance(text_content, str):
                file_path = job_dir / f"{engine_key}.md"
                file_path.write_text(text_content, encoding='utf-8')
                downloaded_files.append(str(file_path))
                logger.info(f"Saved text: {file_path}")
                continue

            # Check for S3 URL in output field (legacy format)
            output_data = engine_result.get("output")
            if isinstance(output_data, str):
                if output_data.startswith("s3://"):
                    # Convert S3 URL to HTTP URL
                    # s3://bucket-name/path -> https://bucket-name.s3.amazonaws.com/path
                    parts = output_data[5:].split("/", 1)
                    if len(parts) == 2:
                        bucket, key = parts
                        http_url = f"https://{bucket}.s3.amazonaws.com/{key}"

                        # Determine extension
                        ext = ".png" if any(x in key.lower() for x in ['.png', '.jpg', '.jpeg', 'image']) else ".md"
                        file_path = job_dir / f"{engine_key}{ext}"

                        if download_file_from_url(http_url, file_path):
                            downloaded_files.append(str(file_path))
                elif output_data.startswith("http"):
                    # Direct HTTP URL
                    ext = ".png" if any(x in output_data.lower() for x in ['.png', '.jpg', '.jpeg']) else ".md"
                    file_path = job_dir / f"{engine_key}{ext}"
                    if download_file_from_url(output_data, file_path):
                        downloaded_files.append(str(file_path))
                else:
                    # Inline text content
                    file_path = job_dir / f"{engine_key}.md"
                    file_path.write_text(output_data, encoding='utf-8')
                    downloaded_files.append(str(file_path))

    # Also check legacy "result" structure for backwards compatibility
    legacy_results = result.get("result", {})
    if isinstance(legacy_results, dict) and not outputs:
        for engine_key, engine_result in legacy_results.items():
            if isinstance(engine_result, dict) and "output" in engine_result:
                output_data = engine_result["output"]
                if isinstance(output_data, str):
                    if output_data.startswith("http"):
                        ext = ".png" if "image" in output_data.lower() else ".md"
                        file_path = job_dir / f"{engine_key}{ext}"
                        if download_file_from_url(output_data, file_path):
                            downloaded_files.append(str(file_path))
                    else:
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


# =============================================================================
# BATCH ANALYSIS TOOLS
# =============================================================================

@mcp.tool()
def scan_folder(
    folder_path: Annotated[str, Field(description="Path to folder containing documents to analyze")],
    file_types: Annotated[str, Field(description="Comma-separated file extensions to include (default: 'pdf,txt,md')")] = "pdf,txt,md"
) -> str:
    """
    Scan a folder and list all documents available for batch analysis.

    Use this to preview what files would be processed before submitting a batch.

    Returns: JSON with list of files found, sizes, and types.
    """
    logger.info(f"Scanning folder: {folder_path}")

    path = Path(folder_path).expanduser().resolve()

    if not path.exists():
        return json.dumps({"error": f"Folder not found: {folder_path}"})

    if not path.is_dir():
        return json.dumps({"error": f"Not a directory: {folder_path}"})

    # Parse file types
    extensions = {f".{ext.strip().lower().lstrip('.')}" for ext in file_types.split(",")}

    files = []
    total_size = 0

    for file_path in path.iterdir():
        if file_path.is_file() and file_path.suffix.lower() in extensions:
            size = file_path.stat().st_size
            files.append({
                "name": file_path.name,
                "path": str(file_path),
                "type": file_path.suffix.lower(),
                "size_bytes": size,
                "size_human": f"{size / 1024:.1f} KB" if size < 1024 * 1024 else f"{size / (1024 * 1024):.1f} MB"
            })
            total_size += size

    # Sort by name
    files.sort(key=lambda x: x["name"])

    output = {
        "folder": str(path),
        "file_types_searched": list(extensions),
        "files_found": len(files),
        "total_size_bytes": total_size,
        "total_size_human": f"{total_size / 1024:.1f} KB" if total_size < 1024 * 1024 else f"{total_size / (1024 * 1024):.1f} MB",
        "files": files
    }

    return json.dumps(output, indent=2)


@mcp.tool()
def submit_batch_analysis(
    folder_path: Annotated[str, Field(description="Path to folder containing documents")],
    engine_keys: Annotated[List[str], Field(description="List of engine keys to run on EACH document")],
    output_mode: Annotated[str, Field(description="Output mode: 'visual' for 4K images, 'textual' for reports")] = "visual",
    file_types: Annotated[str, Field(description="Comma-separated extensions to process (default: 'pdf,txt,md')")] = "pdf,txt,md",
    max_documents: Annotated[int, Field(description="Maximum number of documents to process (default: 20)")] = 20,
    anthropic_api_key: Annotated[Optional[str], Field(description="Anthropic API key")] = None,
    gemini_api_key: Annotated[Optional[str], Field(description="Gemini API key (required for visual mode)")] = None
) -> str:
    """
    Submit an entire folder of documents for batch analysis.

    Each document will be analyzed with ALL selected engines, generating N × M jobs
    (N documents × M engines).

    Returns: Grouped job IDs for tracking each document's analysis.
    """
    logger.info(f"Batch analysis: {folder_path} with {len(engine_keys)} engines")

    # First scan the folder
    scan_result = json.loads(scan_folder(folder_path, file_types))
    if "error" in scan_result:
        return json.dumps(scan_result)

    files = scan_result.get("files", [])
    if not files:
        return json.dumps({"error": f"No matching files found in {folder_path}"})

    # Limit number of documents
    if len(files) > max_documents:
        logger.warning(f"Limiting to {max_documents} documents (found {len(files)})")
        files = files[:max_documents]

    # Build llm_keys
    llm_keys = get_llm_keys(anthropic_api_key, gemini_api_key)

    # Determine API output mode
    api_output_mode = "gemini_image" if output_mode == "visual" else "executive_memo"

    batch_results = []
    total_jobs = 0
    total_errors = 0

    for file_info in files:
        file_path = file_info["path"]
        doc = read_document(file_path)

        if "error" in doc:
            batch_results.append({
                "file": file_info["name"],
                "status": "error",
                "error": doc["error"],
                "jobs": []
            })
            total_errors += 1
            continue

        doc_for_api = {
            "id": doc["id"],
            "title": doc["title"],
            "content": doc["content"],
            "encoding": doc["encoding"]
        }

        file_jobs = []
        file_errors = []

        for engine_key in engine_keys:
            result = api_request(
                VISUALIZER_API_URL,
                'POST',
                '/api/analyzer/analyze',
                data={
                    "documents": [doc_for_api],
                    "engine": engine_key,
                    "output_mode": api_output_mode,
                    "collection_mode": "single",
                    "llm_keys": llm_keys
                },
                timeout=120
            )

            if "error" in result:
                file_errors.append(f"{engine_key}: {result['error']}")
            elif result.get("job_id"):
                file_jobs.append({
                    "engine": engine_key,
                    "job_id": result["job_id"]
                })
                total_jobs += 1
            else:
                file_errors.append(f"{engine_key}: No job ID returned")

        batch_results.append({
            "file": file_info["name"],
            "status": "submitted" if file_jobs else "failed",
            "jobs": file_jobs,
            "errors": file_errors if file_errors else None
        })

    # Send notification (silent - no sound for submission)
    send_notification(
        "📊 Batch Analysis Started",
        f"Processing {len(files)} documents with {len(engine_keys)} engines ({total_jobs} total jobs)",
        "visualizer,batch,started",
        sound=False
    )

    output = {
        "batch_status": "submitted",
        "folder": folder_path,
        "documents_processed": len(files),
        "engines_per_document": len(engine_keys),
        "total_jobs": total_jobs,
        "total_errors": total_errors,
        "output_mode": output_mode,
        "documents": batch_results,
        "message": f"Submitted {total_jobs} job(s) for {len(files)} document(s). Use check_job_status() with individual job_ids to monitor."
    }

    logger.info(f"Batch submitted: {total_jobs} jobs for {len(files)} documents")
    return json.dumps(output, indent=2)


# =============================================================================
# PIPELINE ANALYSIS TOOLS
# =============================================================================

@mcp.tool()
def submit_pipeline_analysis(
    document_path: Annotated[str, Field(description="Path to the document to analyze")],
    pipeline_key: Annotated[str, Field(description="Pipeline key (e.g., 'argument_deep_dive', 'evidence_chain')")],
    output_mode: Annotated[str, Field(description="Output mode: 'visual' for 4K images, 'textual' for reports")] = "visual",
    anthropic_api_key: Annotated[Optional[str], Field(description="Anthropic API key")] = None,
    gemini_api_key: Annotated[Optional[str], Field(description="Gemini API key (required for visual mode)")] = None
) -> str:
    """
    Submit a document for pipeline analysis.

    Pipelines chain multiple engines sequentially, where each engine's output
    feeds into the next. This provides deeper, multi-stage analysis.

    Returns: Job ID for the pipeline execution.
    """
    logger.info(f"Pipeline analysis: {document_path} with pipeline '{pipeline_key}'")

    # Read document
    doc = read_document(document_path)
    if "error" in doc:
        return json.dumps({"error": doc["error"]})

    # Build llm_keys
    llm_keys = get_llm_keys(anthropic_api_key, gemini_api_key)

    # Determine API output mode
    api_output_mode = "gemini_image" if output_mode == "visual" else "executive_memo"

    # Prepare document
    doc_for_api = {
        "id": doc["id"],
        "title": doc["title"],
        "content": doc["content"],
        "encoding": doc["encoding"]
    }

    # Submit pipeline analysis (use /analyze/pipeline endpoint)
    result = api_request(
        VISUALIZER_API_URL,
        'POST',
        '/api/analyzer/analyze/pipeline',
        data={
            "documents": [doc_for_api],
            "pipeline": pipeline_key,
            "output_mode": api_output_mode,
            "include_intermediate_outputs": True,
            "llm_keys": llm_keys
        },
        timeout=120
    )

    if "error" in result:
        return json.dumps({"error": result["error"], "details": result.get("details", "")})

    job_id = result.get("job_id")
    if not job_id:
        return json.dumps({"error": "No job ID returned from pipeline submission"})

    output = {
        "job_id": job_id,
        "status": "submitted",
        "document": doc["title"],
        "pipeline": pipeline_key,
        "output_mode": output_mode,
        "message": f"Pipeline '{pipeline_key}' started. Use check_job_status('{job_id}') to monitor progress."
    }

    # Send notification (silent - no sound for submission)
    send_notification(
        "🔗 Pipeline Analysis Started",
        f"Running pipeline '{pipeline_key}' on {doc['title']}",
        "visualizer,pipeline,started",
        sound=False
    )

    logger.info(f"Pipeline submitted: {job_id}")
    return json.dumps(output, indent=2)


if __name__ == "__main__":
    # Run the MCP server
    mcp.run()
