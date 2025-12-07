#!/usr/bin/env python3
"""
Nano Banana 4K Visualizer

A production-ready web interface for Gemini's advanced image generation.

Features:
  - 4K image generation with reasoning display
  - Local file context support (images, text files)
  - Multi-turn conversation for iterative refinement
  - Automatic download to local machine
  - Google Search grounding option

Usage:
  1. pip install flask google-genai Pillow python-dotenv
  2. Create .env file with GEMINI_API_KEY=your_key
  3. python app.py
  4. Open http://localhost:5000
"""

import os
import io
import re
import base64
import datetime
import traceback
import uuid
import httpx
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any

from flask import Flask, request, jsonify, Response
from PIL import Image
from dotenv import load_dotenv

# Try to import PDF support
try:
    import fitz  # PyMuPDF
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False
    print("Warning: PyMuPDF not installed. PDF support disabled. Run: pip install pymupdf")

# Load environment variables from .env file
load_dotenv()

# Configuration
API_KEY = os.environ.get("GEMINI_API_KEY")
if not API_KEY:
    raise ValueError("GEMINI_API_KEY not found. Create a .env file with GEMINI_API_KEY=your_key")

# Model Definitions
MODELS = {
    "pro": {
        "id": "gemini-3-pro-image-preview",
        "name": "Nano Banana Pro",
        "description": "4K generation with advanced reasoning",
        "max_resolution": "4K",
        "supports_search": True
    },
    "flash": {
        "id": "gemini-2.5-flash-image",
        "name": "Nano Banana Flash",
        "description": "Fast generation, 1K resolution",
        "max_resolution": "1K",
        "supports_search": False
    }
}

# Download folder - uses system Downloads directory
DOWNLOAD_FOLDER = Path.home() / "Downloads"
DOWNLOAD_FOLDER.mkdir(exist_ok=True)

# Supported file types for context
IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.gif', '.webp', '.bmp', '.tiff'}
TEXT_EXTENSIONS = {'.txt', '.md', '.json', '.csv', '.xml', '.py', '.js', '.html', '.css', '.yaml', '.yml'}

# Analyzer API Configuration
ANALYZER_API_URL = os.environ.get("ANALYZER_API_URL", "http://localhost:8847")
ANALYZER_API_KEY = os.environ.get("ANALYZER_API_KEY", "dev-key-12345")

# Document extensions for analysis
DOCUMENT_EXTENSIONS = {'.pdf', '.md', '.txt'}

# Application Initialization
app = Flask(__name__)

# Auto-reload configuration
app.config['TEMPLATES_AUTO_RELOAD'] = True
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

# Initialize Gemini client
client = None
genai = None
types = None

def initialize_client():
    """Initialize the Gemini client with proper error handling."""
    global client, genai, types
    try:
        from google import genai as _genai
        from google.genai import types as _types
        genai = _genai
        types = _types
        client = genai.Client(api_key=API_KEY)
        return True
    except ImportError as e:
        print(f"Missing dependency: {e}")
        print("   Run: pip install google-genai")
        return False
    except Exception as e:
        print(f"Failed to initialize Gemini client: {e}")
        return False

# Chat session storage for multi-turn conversations
chat_sessions: Dict[str, Any] = {}


def get_file_type(path: Path) -> Optional[str]:
    """Determine file type from extension."""
    ext = path.suffix.lower()
    if ext in IMAGE_EXTENSIONS:
        return "image"
    if ext in TEXT_EXTENSIONS:
        return "text"
    return None


def load_context_file(path: Path) -> Tuple[Optional[Any], str, Optional[str]]:
    """
    Load a file for use as generation context.

    Returns:
        Tuple of (content, type, error_message)
    """
    if not path.exists():
        return None, "error", f"File not found: {path}"

    file_type = get_file_type(path)

    try:
        if file_type == "image":
            img = Image.open(path)
            # Ensure compatible color mode
            if img.mode not in ('RGB', 'RGBA', 'L'):
                img = img.convert('RGB')
            return img, "image", None

        elif file_type == "text":
            content = path.read_text(encoding='utf-8', errors='replace')
            # Limit text size to prevent token overflow
            if len(content) > 50000:
                content = content[:50000] + "\n[...truncated...]"
            formatted = f"[Content from {path.name}]:\n{content}"
            return formatted, "text", None

        else:
            return None, "error", f"Unsupported file type: {path.suffix}"

    except Exception as e:
        return None, "error", f"Error loading {path.name}: {str(e)}"


def image_to_base64(img: Image.Image, max_size: int = 2000) -> str:
    """
    Convert PIL Image to base64 string.
    Optionally resize for display efficiency.
    """
    # Resize if too large for display
    if max(img.size) > max_size:
        ratio = max_size / max(img.size)
        new_size = (int(img.width * ratio), int(img.height * ratio))
        img = img.resize(new_size, Image.Resampling.LANCZOS)

    buffer = io.BytesIO()
    # Use PNG for quality, JPEG for smaller files
    img_format = "PNG" if img.mode == "RGBA" else "JPEG"
    save_kwargs = {"format": img_format}
    if img_format == "JPEG":
        save_kwargs["quality"] = 92

    img.save(buffer, **save_kwargs)
    return base64.b64encode(buffer.getvalue()).decode('utf-8')


def save_generated_image(img: Image.Image, prompt_hint: str = "") -> Tuple[Path, str]:
    """
    Save generated image to downloads folder.

    Returns:
        Tuple of (full_path, filename)
    """
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # Sanitize prompt for filename
    safe_hint = re.sub(r'[\\/*?:"<>|]', "", prompt_hint)[:30].replace(" ", "_")
    safe_hint = safe_hint or "output"

    filename = f"nano_banana_{timestamp}_{safe_hint}.png"
    save_path = DOWNLOAD_FOLDER / filename

    img.save(save_path, format="PNG", optimize=True)
    return save_path, filename


def create_generation_config(
    model_key: str,
    aspect_ratio: str,
    resolution: str,
    use_search: bool
) -> Any:
    """Create the Gemini generation configuration."""

    config_params = {
        "response_modalities": ["TEXT", "IMAGE"],
        "temperature": 1.0,  # Maximum creativity
        "image_config": types.ImageConfig(
            aspect_ratio=aspect_ratio,
            image_size=resolution.upper()  # Must be uppercase: 1K, 2K, 4K
        )
    }

    # Add Google Search grounding if supported and requested
    model_info = MODELS.get(model_key, MODELS["pro"])
    if use_search and model_info.get("supports_search"):
        config_params["tools"] = [{"google_search": {}}]

    return types.GenerateContentConfig(**config_params)


def process_gemini_response(response) -> Dict[str, Any]:
    """
    Process Gemini response, extracting all components.

    Returns structured data containing:
      - text: Main response text
      - thinking: Model's reasoning process
      - images: List of base64 encoded images
      - saved_paths: List of local file paths
    """
    result = {
        "text": "",
        "thinking": "",
        "images": [],
        "saved_paths": [],
        "raw_images": []  # PIL images for saving
    }

    if not response or not response.parts:
        return result

    for part in response.parts:
        # Check if this is a thought/reasoning part
        is_thought = getattr(part, 'thought', False)

        # Extract text content
        if hasattr(part, 'text') and part.text:
            if is_thought:
                result["thinking"] += part.text + "\n"
            else:
                result["text"] += part.text

        # Extract image content (skip thought images)
        if hasattr(part, 'inline_data') and part.inline_data and not is_thought:
            try:
                # Get raw bytes from inline_data
                inline_data = part.inline_data
                if hasattr(inline_data, 'data'):
                    img_bytes = inline_data.data
                    # Handle if data is already bytes or base64 string
                    if isinstance(img_bytes, str):
                        img_bytes = base64.b64decode(img_bytes)
                    img = Image.open(io.BytesIO(img_bytes))
                else:
                    print("No data attribute in inline_data")
                    continue
            except Exception as e:
                print(f"Failed to decode image: {e}")
                continue

            result["raw_images"].append(img)
            result["images"].append(image_to_base64(img))

    return result


def format_error_message(error: Exception) -> str:
    """Convert exceptions to user-friendly error messages."""
    error_str = str(error).lower()

    if "permission_denied" in error_str or "api key" in error_str:
        return "API key is invalid or lacks required permissions. Check your GEMINI_API_KEY."
    elif "resource_exhausted" in error_str or "quota" in error_str:
        return "API quota exceeded. Please wait a moment and try again."
    elif "invalid_argument" in error_str:
        return "Invalid request parameters. Try adjusting your prompt or settings."
    elif "safety" in error_str or "blocked" in error_str:
        return "Content was blocked by safety filters. Please modify your prompt."
    elif "deadline" in error_str or "timeout" in error_str:
        return "Request timed out. 4K generation can take up to 60 seconds - please try again."
    elif "not found" in error_str:
        return "Model not available. It may be in limited preview access."
    else:
        return f"Generation failed: {str(error)}"


# API Endpoints

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy",
        "client_initialized": client is not None,
        "download_folder": str(DOWNLOAD_FOLDER)
    })


@app.route('/api/download/<filename>', methods=['GET'])
def download_file(filename):
    """Serve a saved image file for download at full resolution."""
    from flask import send_file

    # Security: only allow files from DOWNLOAD_FOLDER
    safe_filename = Path(filename).name  # Strip any path components
    file_path = DOWNLOAD_FOLDER / safe_filename

    if not file_path.exists():
        return jsonify({"error": "File not found"}), 404

    # Verify it's actually in download folder (prevent path traversal)
    try:
        file_path.resolve().relative_to(DOWNLOAD_FOLDER.resolve())
    except ValueError:
        return jsonify({"error": "Invalid file path"}), 403

    return send_file(
        file_path,
        mimetype='image/png',
        as_attachment=True,
        download_name=safe_filename
    )


@app.route('/api/models', methods=['GET'])
def list_models():
    """Return available models."""
    return jsonify({"models": MODELS})


@app.route('/api/validate-path', methods=['POST'])
def validate_path():
    """
    Validate a local file path and return file information.

    Request: {"path": "/path/to/file.png"}
    Response: {"valid": true, "type": "image", "name": "file.png", ...}
    """
    data = request.json or {}
    path_str = data.get('path', '').strip()

    if not path_str:
        return jsonify({"valid": False, "error": "Empty path provided"})

    # Expand user home directory (~)
    path = Path(path_str).expanduser().resolve()

    if not path.exists():
        return jsonify({"valid": False, "error": "File not found"})

    if not path.is_file():
        return jsonify({"valid": False, "error": "Path is not a file"})

    file_type = get_file_type(path)
    if not file_type:
        return jsonify({
            "valid": False,
            "error": f"Unsupported file type: {path.suffix}"
        })

    # Get file metadata
    stat = path.stat()
    size_bytes = stat.st_size

    # Human-readable size
    if size_bytes < 1024:
        size_str = f"{size_bytes} B"
    elif size_bytes < 1024 ** 2:
        size_str = f"{size_bytes / 1024:.1f} KB"
    else:
        size_str = f"{size_bytes / (1024 ** 2):.1f} MB"

    response_data = {
        "valid": True,
        "type": file_type,
        "name": path.name,
        "path": str(path),
        "size": size_str,
        "size_bytes": size_bytes
    }

    # Add thumbnail preview for images
    if file_type == "image":
        try:
            img = Image.open(path)
            img.thumbnail((80, 80))
            response_data["preview"] = image_to_base64(img, max_size=80)
            response_data["dimensions"] = f"{img.width}x{img.height}"
        except Exception:
            pass

    return jsonify(response_data)


@app.route('/api/session/new', methods=['POST'])
def create_session():
    """Create a new chat session for multi-turn conversations."""
    session_id = str(uuid.uuid4())
    return jsonify({"session_id": session_id})


@app.route('/api/session/clear', methods=['POST'])
def clear_session():
    """Clear an existing chat session."""
    data = request.json or {}
    session_id = data.get('session_id')

    if session_id and session_id in chat_sessions:
        del chat_sessions[session_id]

    return jsonify({"success": True})


@app.route('/api/generate', methods=['POST'])
def generate_image():
    """
    Generate an image using Gemini.

    Request body:
    {
        "prompt": "Description of desired image",
        "file_paths": ["/path/to/context/file.png"],
        "model": "pro",
        "aspect_ratio": "16:9",
        "resolution": "4K",
        "use_search": false,
        "session_id": "optional-uuid-for-multi-turn"
    }
    """
    # Ensure client is initialized
    if not client:
        if not initialize_client():
            return jsonify({
                "success": False,
                "error": "Failed to initialize Gemini client. Check your API key and dependencies."
            })

    # Parse request
    data = request.json or {}
    prompt = data.get('prompt', '').strip()
    file_paths = data.get('file_paths', [])
    model_key = data.get('model', 'pro')
    aspect_ratio = data.get('aspect_ratio', '16:9')
    resolution = data.get('resolution', '4K')
    use_search = data.get('use_search', False)
    session_id = data.get('session_id')

    # Validate prompt
    if not prompt:
        return jsonify({
            "success": False,
            "error": "Please provide a description of what you want to visualize."
        })

    # Validate model
    if model_key not in MODELS:
        model_key = "pro"

    model_info = MODELS[model_key]
    model_id = model_info["id"]

    # Enforce resolution limits
    max_res = model_info.get("max_resolution", "1K")
    res_order = ["1K", "2K", "4K"]
    if res_order.index(resolution) > res_order.index(max_res):
        resolution = max_res

    try:
        # Build content list
        contents = []
        loaded_files = []
        warnings = []

        # Load context files
        for path_str in file_paths:
            path_str = (path_str or "").strip()
            if not path_str:
                continue

            path = Path(path_str).expanduser().resolve()
            content, ftype, error = load_context_file(path)

            if error:
                warnings.append(error)
            elif content is not None:
                contents.append(content)
                loaded_files.append({
                    "type": ftype,
                    "name": path.name,
                    "status": "loaded"
                })

        # Add the prompt text
        contents.append(prompt)

        # Create generation configuration
        config = create_generation_config(
            model_key, aspect_ratio, resolution, use_search
        )

        # Execute generation
        if session_id and session_id in chat_sessions:
            # Continue existing chat session
            chat = chat_sessions[session_id]
            response = chat.send_message(contents, config=config)
        else:
            # New single-turn generation
            response = client.models.generate_content(
                model=model_id,
                contents=contents,
                config=config
            )

            # Create chat session for potential follow-ups
            if session_id:
                try:
                    chat_sessions[session_id] = client.chats.create(
                        model=model_id,
                        config=types.GenerateContentConfig(
                            response_modalities=['TEXT', 'IMAGE']
                        )
                    )
                except Exception:
                    pass  # Chat creation is optional

        # Process response
        result = process_gemini_response(response)

        # Save generated images and collect dimensions
        saved_paths = []
        image_dimensions = []
        for i, img in enumerate(result["raw_images"]):
            try:
                image_dimensions.append(f"{img.width}x{img.height}")
                path, filename = save_generated_image(img, prompt[:30])
                saved_paths.append(str(path))
            except Exception as e:
                warnings.append(f"Failed to save image {i+1}: {str(e)}")

        # Build success response
        return jsonify({
            "success": True,
            "text": result["text"].strip(),
            "thinking": result["thinking"].strip(),
            "images": result["images"],
            "saved_paths": saved_paths,
            "loaded_files": loaded_files,
            "warnings": warnings if warnings else None,
            "metadata": {
                "model": model_info["name"],
                "model_id": model_id,
                "resolution": resolution,
                "aspect_ratio": aspect_ratio,
                "search_enabled": use_search and model_info.get("supports_search", False),
                "image_dimensions": image_dimensions[0] if image_dimensions else None
            }
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({
            "success": False,
            "error": format_error_message(e)
        })


# ============================================================================
# ANALYZER INTEGRATION - Document Analysis API
# ============================================================================

def extract_pdf_text(file_path: Path) -> str:
    """Extract text content from a PDF file."""
    if not PDF_SUPPORT:
        return f"[PDF support not available. Install pymupdf: pip install pymupdf]"

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
        return f"[Error reading PDF: {str(e)}]"


def extract_document_content(file_path: Path) -> Tuple[str, str]:
    """
    Extract content from a document file.
    Returns: (content, error_message)
    """
    suffix = file_path.suffix.lower()

    try:
        if suffix == '.pdf':
            content = extract_pdf_text(file_path)
            return content, None
        elif suffix in {'.md', '.txt'}:
            content = file_path.read_text(encoding='utf-8', errors='replace')
            return content, None
        else:
            return None, f"Unsupported file type: {suffix}"
    except Exception as e:
        return None, f"Error reading {file_path.name}: {str(e)}"


@app.route('/api/analyzer/scan-folder', methods=['POST'])
def scan_folder():
    """
    Scan a folder for documents OR accept a single file.

    Request: {"path": "/path/to/folder"} or {"path": "/path/to/file.pdf"}
    Response: {"success": true, "files": [...], "count": N}
    """
    from urllib.parse import unquote

    data = request.json or {}
    input_path = data.get('path', '').strip()

    if not input_path:
        return jsonify({"success": False, "error": "No path provided"})

    # Clean up path - remove file:// prefix if present
    if input_path.startswith('file://'):
        input_path = input_path[7:]
    # Handle URL encoding (%20 -> space, etc.)
    input_path = unquote(input_path)

    path = Path(input_path).expanduser().resolve()

    if not path.exists():
        return jsonify({"success": False, "error": f"Path not found: {input_path}"})

    files = []

    if path.is_file():
        # Single file mode
        if path.suffix.lower() not in DOCUMENT_EXTENSIONS:
            return jsonify({"success": False, "error": f"Unsupported file type: {path.suffix}. Supported: PDF, MD, TXT"})

        stat = path.stat()
        size_kb = stat.st_size / 1024

        files.append({
            "name": path.name,
            "path": str(path),
            "type": path.suffix.lower()[1:],
            "size": f"{size_kb:.1f} KB" if size_kb < 1024 else f"{size_kb/1024:.1f} MB",
            "size_bytes": stat.st_size
        })

        return jsonify({
            "success": True,
            "files": files,
            "count": 1,
            "folder": str(path.parent),
            "pdf_support": PDF_SUPPORT,
            "mode": "single_file"
        })

    elif path.is_dir():
        # Folder mode - scan for documents
        for ext in DOCUMENT_EXTENSIONS:
            for file_path in path.glob(f"*{ext}"):
                if file_path.is_file():
                    stat = file_path.stat()
                    size_kb = stat.st_size / 1024

                    files.append({
                        "name": file_path.name,
                        "path": str(file_path),
                        "type": file_path.suffix.lower()[1:],
                        "size": f"{size_kb:.1f} KB" if size_kb < 1024 else f"{size_kb/1024:.1f} MB",
                        "size_bytes": stat.st_size
                    })

        files.sort(key=lambda x: x["name"].lower())

        return jsonify({
            "success": True,
            "files": files,
            "count": len(files),
            "folder": str(path),
            "pdf_support": PDF_SUPPORT,
            "mode": "folder"
        })

    else:
        return jsonify({"success": False, "error": f"Invalid path: {input_path}"})


@app.route('/api/analyzer/engines', methods=['GET'])
def list_analyzer_engines():
    """Fetch available engines from Analyzer API."""
    try:
        response = httpx.get(
            f"{ANALYZER_API_URL}/v1/engines",
            headers={"X-API-Key": ANALYZER_API_KEY},
            timeout=30.0,
        )
        response.raise_for_status()
        return jsonify(response.json())
    except httpx.HTTPError as e:
        return jsonify({"error": f"Failed to fetch engines: {str(e)}"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/analyzer/bundles', methods=['GET'])
def list_analyzer_bundles():
    """Fetch available bundles from Analyzer API."""
    try:
        response = httpx.get(
            f"{ANALYZER_API_URL}/v1/bundles",
            headers={"X-API-Key": ANALYZER_API_KEY},
            timeout=30.0,
        )
        response.raise_for_status()
        return jsonify(response.json())
    except httpx.HTTPError as e:
        return jsonify({"error": f"Failed to fetch bundles: {str(e)}"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/analyzer/output-modes', methods=['GET'])
def list_analyzer_output_modes():
    """Fetch available output modes from Analyzer API."""
    try:
        response = httpx.get(
            f"{ANALYZER_API_URL}/v1/output-modes",
            headers={"X-API-Key": ANALYZER_API_KEY},
            timeout=30.0,
        )
        response.raise_for_status()
        return jsonify(response.json())
    except httpx.HTTPError as e:
        return jsonify({"error": f"Failed to fetch output modes: {str(e)}"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/analyzer/analyze', methods=['POST'])
def submit_analysis():
    """
    Submit documents for analysis.

    Request: {
        "file_paths": ["/path/to/doc.pdf", ...],
        "engine": "thematic_synthesis",
        "output_mode": "structured_text_report",
        "collection_mode": "single" | "individual"
    }
    """
    data = request.json or {}
    file_paths = data.get('file_paths', [])
    engine = data.get('engine')
    output_mode = data.get('output_mode', 'structured_text_report')
    collection_mode = data.get('collection_mode', 'single')

    if not file_paths:
        return jsonify({"success": False, "error": "No files provided"})

    if not engine:
        return jsonify({"success": False, "error": "No engine selected"})

    # Extract content from all files
    documents = []
    errors = []

    for i, file_path in enumerate(file_paths):
        path = Path(file_path).expanduser().resolve()
        if not path.exists():
            errors.append(f"File not found: {file_path}")
            continue

        content, error = extract_document_content(path)
        if error:
            errors.append(error)
            continue

        documents.append({
            "id": f"doc_{i+1}",
            "title": path.stem,
            "content": content
        })

    if not documents:
        return jsonify({
            "success": False,
            "error": "No documents could be read",
            "details": errors
        })

    # For individual mode, we'll return multiple job IDs
    if collection_mode == "individual":
        jobs = []
        for doc in documents:
            try:
                response = httpx.post(
                    f"{ANALYZER_API_URL}/v1/analyze",
                    headers={"X-API-Key": ANALYZER_API_KEY},
                    json={
                        "documents": [doc],
                        "engine": engine,
                        "output_mode": output_mode
                    },
                    timeout=60.0,
                )
                response.raise_for_status()
                job_data = response.json()
                jobs.append({
                    "job_id": job_data.get("job_id"),
                    "title": doc["title"],
                    "status": "submitted"
                })
            except Exception as e:
                jobs.append({
                    "title": doc["title"],
                    "status": "failed",
                    "error": str(e)
                })

        return jsonify({
            "success": True,
            "mode": "individual",
            "jobs": jobs,
            "warnings": errors if errors else None
        })

    else:
        # Single collection mode - all documents together
        try:
            response = httpx.post(
                f"{ANALYZER_API_URL}/v1/analyze",
                headers={"X-API-Key": ANALYZER_API_KEY},
                json={
                    "documents": documents,
                    "engine": engine,
                    "output_mode": output_mode
                },
                timeout=60.0,
            )
            response.raise_for_status()
            job_data = response.json()

            return jsonify({
                "success": True,
                "mode": "collection",
                "job_id": job_data.get("job_id"),
                "document_count": len(documents),
                "warnings": errors if errors else None
            })

        except httpx.HTTPError as e:
            return jsonify({
                "success": False,
                "error": f"API error: {str(e)}"
            })


@app.route('/api/analyzer/analyze/bundle', methods=['POST'])
def submit_bundle_analysis():
    """Submit documents for bundle analysis (multiple engines)."""
    data = request.json or {}
    file_paths = data.get('file_paths', [])
    bundle = data.get('bundle')
    output_modes = data.get('output_modes', {})

    if not file_paths:
        return jsonify({"success": False, "error": "No files provided"})

    if not bundle:
        return jsonify({"success": False, "error": "No bundle selected"})

    # Extract content from all files
    documents = []
    errors = []

    for i, file_path in enumerate(file_paths):
        path = Path(file_path).expanduser().resolve()
        if not path.exists():
            errors.append(f"File not found: {file_path}")
            continue

        content, error = extract_document_content(path)
        if error:
            errors.append(error)
            continue

        documents.append({
            "id": f"doc_{i+1}",
            "title": path.stem,
            "content": content
        })

    if not documents:
        return jsonify({
            "success": False,
            "error": "No documents could be read",
            "details": errors
        })

    try:
        response = httpx.post(
            f"{ANALYZER_API_URL}/v1/analyze/bundle",
            headers={"X-API-Key": ANALYZER_API_KEY},
            json={
                "documents": documents,
                "bundle": bundle,
                "output_modes": output_modes
            },
            timeout=60.0,
        )
        response.raise_for_status()
        job_data = response.json()

        return jsonify({
            "success": True,
            "job_id": job_data.get("job_id"),
            "bundle": bundle,
            "document_count": len(documents),
            "warnings": errors if errors else None
        })

    except httpx.HTTPError as e:
        return jsonify({
            "success": False,
            "error": f"API error: {str(e)}"
        })


@app.route('/api/analyzer/jobs/<job_id>', methods=['GET'])
def get_analysis_job(job_id):
    """Get job status from Analyzer API."""
    try:
        response = httpx.get(
            f"{ANALYZER_API_URL}/v1/jobs/{job_id}",
            headers={"X-API-Key": ANALYZER_API_KEY},
            timeout=30.0,
        )
        response.raise_for_status()
        return jsonify(response.json())
    except httpx.HTTPError as e:
        return jsonify({"error": f"Failed to get job: {str(e)}"}), 500


@app.route('/api/analyzer/jobs/<job_id>/result', methods=['GET'])
def get_analysis_result(job_id):
    """Get completed job result from Analyzer API."""
    try:
        response = httpx.get(
            f"{ANALYZER_API_URL}/v1/jobs/{job_id}/result",
            headers={"X-API-Key": ANALYZER_API_KEY},
            timeout=30.0,
        )
        response.raise_for_status()
        return jsonify(response.json())
    except httpx.HTTPError as e:
        return jsonify({"error": f"Failed to get result: {str(e)}"}), 500


# Main Page

@app.route('/')
def index():
    """Serve the main application page."""
    return Response(HTML_PAGE, mimetype='text/html')


# Embedded HTML/CSS/JS

HTML_PAGE = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Nano Banana 4K Visualizer</title>
    <style>
        :root {
            --bg-dark: #0d0d14;
            --bg-card: #151521;
            --bg-input: #1c1c2e;
            --bg-hover: #252538;
            --accent: #fbbf24;
            --accent-glow: rgba(251, 191, 36, 0.15);
            --success: #34d399;
            --error: #f87171;
            --warning: #fbbf24;
            --purple: #a78bfa;
            --text: #f1f1f4;
            --text-dim: #71717a;
            --border: rgba(255,255,255,0.08);
            --radius: 10px;
        }

        * { margin: 0; padding: 0; box-sizing: border-box; }

        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: var(--bg-dark);
            color: var(--text);
            min-height: 100vh;
            line-height: 1.5;
        }

        .app {
            max-width: 1440px;
            margin: 0 auto;
            padding: 1.5rem;
        }

        /* Header */
        header {
            text-align: center;
            padding: 2rem 1rem;
            border-bottom: 1px solid var(--border);
            margin-bottom: 2rem;
        }

        header h1 {
            font-size: clamp(1.75rem, 5vw, 2.5rem);
            background: linear-gradient(135deg, var(--accent), #f97316);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            margin-bottom: 0.5rem;
        }

        header p { color: var(--text-dim); }

        .badge {
            display: inline-block;
            padding: 0.25rem 0.75rem;
            background: var(--accent-glow);
            border: 1px solid var(--accent);
            border-radius: 999px;
            font-size: 0.75rem;
            color: var(--accent);
            margin-top: 0.75rem;
        }

        /* Tab Navigation */
        .tab-nav {
            display: flex;
            gap: 0.5rem;
            margin-bottom: 2rem;
            border-bottom: 1px solid var(--border);
            padding-bottom: 1rem;
        }

        .tab-btn {
            padding: 0.75rem 1.5rem;
            background: transparent;
            border: 1px solid var(--border);
            border-radius: 8px 8px 0 0;
            color: var(--text-dim);
            font-size: 0.95rem;
            font-weight: 500;
            cursor: pointer;
            transition: all 0.2s;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }

        .tab-btn:hover { background: var(--bg-hover); color: var(--text); }

        .tab-btn.active {
            background: var(--bg-card);
            border-color: var(--accent);
            border-bottom-color: var(--bg-card);
            color: var(--accent);
        }

        .tab-content { display: none; }
        .tab-content.active { display: block; }

        /* Layout */
        .main {
            display: grid;
            grid-template-columns: minmax(300px, 1fr) minmax(400px, 1.4fr);
            gap: 2rem;
            align-items: start;
        }

        @media (max-width: 900px) {
            .main { grid-template-columns: 1fr; }
        }

        /* Cards */
        .card {
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: var(--radius);
            padding: 1.5rem;
        }

        .card-title {
            display: flex;
            align-items: center;
            gap: 0.5rem;
            font-size: 1rem;
            color: var(--accent);
            margin-bottom: 1.25rem;
            font-weight: 600;
        }

        /* Form controls */
        label {
            display: block;
            font-size: 0.75rem;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            color: var(--text-dim);
            margin-bottom: 0.5rem;
        }

        textarea, input[type="text"], select {
            width: 100%;
            background: var(--bg-input);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 0.75rem 1rem;
            color: var(--text);
            font-size: 0.9rem;
            font-family: inherit;
            transition: border-color 0.2s, box-shadow 0.2s;
        }

        textarea:focus, input:focus, select:focus {
            outline: none;
            border-color: var(--accent);
            box-shadow: 0 0 0 3px var(--accent-glow);
        }

        textarea {
            min-height: 140px;
            resize: vertical;
        }

        textarea::placeholder, input::placeholder {
            color: var(--text-dim);
        }

        select { cursor: pointer; }

        /* File inputs */
        .file-section { margin: 1.25rem 0; }

        .file-row {
            display: flex;
            gap: 0.5rem;
            margin-bottom: 0.5rem;
        }

        .file-row input {
            flex: 1;
            font-family: 'SF Mono', 'Monaco', 'Consolas', monospace;
            font-size: 0.85rem;
        }

        .file-list {
            display: flex;
            flex-direction: column;
            gap: 0.5rem;
            margin-top: 0.75rem;
        }

        .file-item {
            display: flex;
            align-items: center;
            gap: 0.75rem;
            padding: 0.625rem 0.875rem;
            background: var(--bg-hover);
            border-radius: 6px;
            font-size: 0.85rem;
            border-left: 3px solid var(--success);
        }

        .file-item.error { border-left-color: var(--error); }

        .file-item img.thumb {
            width: 32px;
            height: 32px;
            object-fit: cover;
            border-radius: 4px;
        }

        .file-item .info { flex: 1; min-width: 0; }
        .file-item .name { font-weight: 500; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
        .file-item .meta { font-size: 0.75rem; color: var(--text-dim); }
        .file-item .remove { color: var(--error); cursor: pointer; font-size: 1.1rem; padding: 0.25rem; }
        .file-item .remove:hover { opacity: 0.8; }

        /* Settings */
        .settings-row {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
            gap: 1rem;
            margin: 1.25rem 0;
        }

        /* Checkbox */
        .checkbox-wrap {
            display: flex;
            align-items: center;
            gap: 0.625rem;
            padding: 0.75rem;
            background: var(--bg-input);
            border-radius: 8px;
            cursor: pointer;
            margin: 1rem 0;
            font-size: 0.9rem;
        }

        .checkbox-wrap input {
            width: 16px;
            height: 16px;
            accent-color: var(--accent);
        }

        /* Buttons */
        .btn {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            gap: 0.5rem;
            padding: 0.625rem 1rem;
            border: none;
            border-radius: 8px;
            font-size: 0.875rem;
            font-weight: 500;
            cursor: pointer;
            transition: all 0.2s;
        }

        .btn-sm { padding: 0.5rem 0.75rem; font-size: 0.8rem; }

        .btn-outline {
            background: transparent;
            border: 1px solid var(--accent);
            color: var(--accent);
        }
        .btn-outline:hover { background: var(--accent-glow); }

        .btn-primary {
            width: 100%;
            padding: 0.875rem 1.5rem;
            background: linear-gradient(135deg, var(--accent), #f97316);
            color: var(--bg-dark);
            font-size: 0.95rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        .btn-primary:hover:not(:disabled) {
            transform: translateY(-1px);
            box-shadow: 0 4px 16px rgba(251, 191, 36, 0.3);
        }
        .btn-primary:disabled {
            opacity: 0.6;
            cursor: not-allowed;
            transform: none;
        }

        .btn-success { background: var(--success); color: var(--bg-dark); }
        .btn-ghost { background: var(--bg-hover); color: var(--text); }

        /* Status messages */
        .status {
            display: none;
            align-items: center;
            gap: 0.75rem;
            padding: 0.875rem 1rem;
            border-radius: 8px;
            margin-bottom: 1rem;
            font-size: 0.9rem;
        }

        .status.show { display: flex; }
        .status.loading { background: var(--accent-glow); border: 1px solid rgba(251,191,36,0.3); color: var(--accent); }
        .status.success { background: rgba(52,211,153,0.1); border: 1px solid rgba(52,211,153,0.3); color: var(--success); }
        .status.error { background: rgba(248,113,113,0.1); border: 1px solid rgba(248,113,113,0.3); color: var(--error); }

        .spinner {
            width: 16px;
            height: 16px;
            border: 2px solid currentColor;
            border-top-color: transparent;
            border-radius: 50%;
            animation: spin 0.7s linear infinite;
        }

        @keyframes spin { to { transform: rotate(360deg); } }

        /* Output area */
        .output { min-height: 400px; }

        .placeholder {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            padding: 3rem;
            text-align: center;
            color: var(--text-dim);
        }
        .placeholder .icon { font-size: 3.5rem; margin-bottom: 1rem; opacity: 0.5; }

        /* Thinking box */
        .thinking-box {
            background: rgba(167,139,250,0.08);
            border: 1px solid rgba(167,139,250,0.25);
            border-radius: 8px;
            margin-bottom: 1rem;
            display: none;
        }
        .thinking-box.show { display: block; }

        .thinking-box summary {
            padding: 0.75rem 1rem;
            cursor: pointer;
            color: var(--purple);
            font-weight: 500;
            font-size: 0.9rem;
        }

        .thinking-box .content {
            padding: 0.75rem 1rem;
            border-top: 1px solid rgba(167,139,250,0.15);
            font-size: 0.85rem;
            color: var(--text-dim);
            white-space: pre-wrap;
            max-height: 180px;
            overflow-y: auto;
        }

        /* Image display */
        .image-box {
            display: none;
            background: rgba(0,0,0,0.3);
            border-radius: var(--radius);
            padding: 1.25rem;
            text-align: center;
        }
        .image-box.show { display: block; }

        .image-box img {
            max-width: 100%;
            max-height: 500px;
            border-radius: 8px;
            box-shadow: 0 8px 32px rgba(0,0,0,0.4);
        }

        .image-actions {
            display: flex;
            gap: 0.625rem;
            justify-content: center;
            margin-top: 1rem;
            flex-wrap: wrap;
        }

        /* Save path */
        .save-info {
            display: none;
            align-items: center;
            gap: 0.5rem;
            padding: 0.75rem 1rem;
            background: rgba(52,211,153,0.08);
            border: 1px solid rgba(52,211,153,0.2);
            border-radius: 8px;
            margin-top: 1rem;
            font-size: 0.8rem;
        }
        .save-info.show { display: flex; }
        .save-info code {
            color: var(--success);
            word-break: break-all;
            font-family: 'SF Mono', monospace;
        }

        /* Text response */
        .text-output {
            display: none;
            background: var(--bg-input);
            border-radius: 8px;
            padding: 1rem;
            margin-top: 1rem;
            font-size: 0.9rem;
            white-space: pre-wrap;
            max-height: 200px;
            overflow-y: auto;
        }
        .text-output.show { display: block; }

        /* History */
        .history-card {
            display: none;
            margin-top: 1.5rem;
        }
        .history-card.show { display: block; }

        .history-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(100px, 1fr));
            gap: 0.625rem;
            margin-top: 1rem;
        }

        .history-item {
            aspect-ratio: 1;
            border-radius: 8px;
            overflow: hidden;
            cursor: pointer;
            border: 2px solid transparent;
            transition: all 0.2s;
        }
        .history-item:hover { border-color: var(--accent); transform: scale(1.03); }
        .history-item img { width: 100%; height: 100%; object-fit: cover; }

        /* Keyboard hint */
        .key-hint {
            text-align: center;
            font-size: 0.75rem;
            color: var(--text-dim);
            margin-top: 0.75rem;
        }
        .key-hint kbd {
            background: var(--bg-input);
            padding: 0.125rem 0.5rem;
            border-radius: 4px;
            font-family: inherit;
        }

        /* Scrollbar */
        ::-webkit-scrollbar { width: 6px; height: 6px; }
        ::-webkit-scrollbar-track { background: transparent; }
        ::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.15); border-radius: 3px; }
        ::-webkit-scrollbar-thumb:hover { background: rgba(255,255,255,0.25); }

        /* Animation */
        .fade-in { animation: fadeIn 0.25s ease-out; }
        @keyframes fadeIn { from { opacity: 0; transform: translateY(8px); } }

        /* ========== Document Analysis Styles ========== */

        .analysis-container {
            display: grid;
            grid-template-columns: 1fr 1.5fr;
            gap: 2rem;
            align-items: start;
        }

        @media (max-width: 1100px) {
            .analysis-container { grid-template-columns: 1fr; }
        }

        .folder-input-row {
            display: flex;
            gap: 0.5rem;
            margin-bottom: 1rem;
        }

        .folder-input-row input { flex: 1; font-family: 'SF Mono', monospace; }

        .doc-list {
            max-height: 300px;
            overflow-y: auto;
            border: 1px solid var(--border);
            border-radius: 8px;
            margin: 1rem 0;
        }

        .doc-item {
            display: flex;
            align-items: center;
            gap: 0.75rem;
            padding: 0.75rem 1rem;
            border-bottom: 1px solid var(--border);
            font-size: 0.85rem;
        }

        .doc-item:last-child { border-bottom: none; }

        .doc-item input[type="checkbox"] {
            width: 16px;
            height: 16px;
            accent-color: var(--accent);
        }

        .doc-item .icon {
            font-size: 1.25rem;
            opacity: 0.7;
        }

        .doc-item .info { flex: 1; }
        .doc-item .name { font-weight: 500; }
        .doc-item .meta { font-size: 0.75rem; color: var(--text-dim); }

        .mode-toggle {
            display: flex;
            gap: 0.5rem;
            margin: 1rem 0;
        }

        .mode-btn {
            flex: 1;
            padding: 0.875rem;
            background: var(--bg-input);
            border: 2px solid var(--border);
            border-radius: 8px;
            color: var(--text);
            cursor: pointer;
            text-align: center;
            transition: all 0.2s;
        }

        .mode-btn:hover { border-color: var(--accent); }

        .mode-btn.active {
            border-color: var(--accent);
            background: var(--accent-glow);
        }

        .mode-btn .title { font-weight: 600; font-size: 0.9rem; }
        .mode-btn .desc { font-size: 0.75rem; color: var(--text-dim); margin-top: 0.25rem; }

        .engine-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(180px, 1fr));
            gap: 0.5rem;
            margin: 1rem 0;
            max-height: 250px;
            overflow-y: auto;
            padding: 0.25rem;
        }

        .engine-card {
            padding: 0.75rem;
            background: var(--bg-input);
            border: 2px solid var(--border);
            border-radius: 8px;
            cursor: pointer;
            transition: all 0.2s;
        }

        .engine-card:hover { border-color: var(--text-dim); }

        .engine-card.selected {
            border-color: var(--accent);
            background: var(--accent-glow);
        }

        .engine-card .name { font-weight: 500; font-size: 0.85rem; }
        .engine-card .priority {
            display: inline-block;
            padding: 0.125rem 0.375rem;
            border-radius: 4px;
            font-size: 0.65rem;
            margin-top: 0.25rem;
        }

        .priority-1 { background: var(--success); color: var(--bg-dark); }
        .priority-2 { background: var(--warning); color: var(--bg-dark); }
        .priority-3 { background: var(--purple); color: white; }

        .bundle-card {
            padding: 1rem;
            background: var(--bg-input);
            border: 2px solid var(--border);
            border-radius: 8px;
            cursor: pointer;
            margin-bottom: 0.5rem;
            transition: all 0.2s;
        }

        .bundle-card:hover { border-color: var(--text-dim); }
        .bundle-card.selected { border-color: var(--accent); background: var(--accent-glow); }

        .bundle-card .name { font-weight: 600; }
        .bundle-card .engines { font-size: 0.75rem; color: var(--text-dim); margin-top: 0.25rem; }

        .output-select { margin: 1rem 0; }

        .analysis-progress {
            display: none;
            padding: 1.5rem;
            background: var(--bg-input);
            border-radius: 8px;
            margin: 1rem 0;
        }

        .analysis-progress.show { display: block; }

        .progress-bar-outer {
            height: 8px;
            background: var(--bg-hover);
            border-radius: 4px;
            overflow: hidden;
            margin: 1rem 0;
        }

        .progress-bar-inner {
            height: 100%;
            background: linear-gradient(90deg, var(--accent), #f97316);
            width: 0%;
            transition: width 0.3s;
        }

        .progress-stage {
            display: flex;
            gap: 0.5rem;
            flex-wrap: wrap;
            margin-top: 0.75rem;
        }

        .stage-badge {
            padding: 0.25rem 0.75rem;
            border-radius: 999px;
            font-size: 0.75rem;
            background: var(--bg-hover);
            color: var(--text-dim);
        }

        .stage-badge.active { background: var(--accent-glow); color: var(--accent); }
        .stage-badge.completed { background: rgba(52,211,153,0.2); color: var(--success); }

        .results-container {
            margin-top: 1.5rem;
        }

        .result-card {
            background: var(--bg-input);
            border-radius: 8px;
            margin-bottom: 1rem;
            overflow: hidden;
        }

        .result-card .header {
            padding: 1rem;
            background: var(--bg-hover);
            font-weight: 600;
            display: flex;
            align-items: center;
            justify-content: space-between;
        }

        .result-card .content {
            padding: 1rem;
            max-height: 70vh;
            overflow: auto;
        }

        .result-card pre {
            white-space: pre-wrap;
            font-size: 0.85rem;
            line-height: 1.6;
        }

        .result-card img {
            max-width: 100%;
            border-radius: 8px;
        }

        .result-card .markdown-content {
            font-size: 0.9rem;
            line-height: 1.7;
        }
        .result-card .markdown-content h1 { font-size: 1.5rem; margin: 1.5rem 0 1rem; color: var(--accent); }
        .result-card .markdown-content h2 { font-size: 1.25rem; margin: 1.25rem 0 0.75rem; color: var(--purple); border-bottom: 1px solid var(--bg-hover); padding-bottom: 0.5rem; }
        .result-card .markdown-content h3 { font-size: 1.1rem; margin: 1rem 0 0.5rem; color: var(--text); }
        .result-card .markdown-content p { margin: 0.75rem 0; }
        .result-card .markdown-content ul, .result-card .markdown-content ol { margin: 0.5rem 0; padding-left: 1.5rem; }
        .result-card .markdown-content li { margin: 0.25rem 0; }
        .result-card .markdown-content strong { color: var(--accent); }
        .result-card .markdown-content table { width: 100%; border-collapse: collapse; margin: 1rem 0; }
        .result-card .markdown-content th, .result-card .markdown-content td { padding: 0.5rem; border: 1px solid var(--bg-hover); text-align: left; }
        .result-card .markdown-content th { background: var(--bg-hover); }

        .result-actions {
            display: flex;
            gap: 0.5rem;
        }
        .result-actions button {
            padding: 0.4rem 0.8rem;
            font-size: 0.8rem;
            background: var(--bg-input);
            border: 1px solid var(--text-dim);
            border-radius: 4px;
            color: var(--text);
            cursor: pointer;
        }
        .result-actions button:hover {
            background: var(--accent);
            color: var(--bg-dark);
            border-color: var(--accent);
        }
        .result-meta {
            font-size: 0.75rem;
            color: var(--text-dim);
            margin-top: 0.5rem;
            padding-top: 0.5rem;
            border-top: 1px solid var(--bg-hover);
        }

        .analyzer-status {
            display: inline-flex;
            align-items: center;
            gap: 0.5rem;
            padding: 0.5rem 1rem;
            border-radius: 8px;
            font-size: 0.8rem;
            margin-bottom: 1rem;
        }

        .analyzer-status.connected { background: rgba(52,211,153,0.1); color: var(--success); }
        .analyzer-status.disconnected { background: rgba(248,113,113,0.1); color: var(--error); }
    </style>
</head>
<body>
    <div class="app">
        <header>
            <h1>Nano Banana Studio</h1>
            <p>AI-Powered Image Generation & Document Analysis</p>
        </header>

        <!-- Tab Navigation -->
        <div class="tab-nav">
            <button class="tab-btn active" onclick="switchTab('image-gen')">
                <span>Image Generation</span>
            </button>
            <button class="tab-btn" onclick="switchTab('doc-analysis')">
                <span>Document Analysis</span>
            </button>
        </div>

        <!-- Image Generation Tab -->
        <div id="image-gen" class="tab-content active">
        <div class="main">
            <!-- Input Panel -->
            <div class="input-panel">
                <div class="card">
                    <h2 class="card-title">Create Visualization</h2>

                    <!-- Session controls -->
                    <div style="display:flex; gap:0.5rem; margin-bottom:1rem; padding-bottom:1rem; border-bottom:1px solid var(--border);">
                        <button class="btn btn-sm btn-outline" onclick="newSession()">New Session</button>
                        <span id="session-indicator" class="badge" style="display:none;">Multi-turn active</span>
                    </div>

                    <!-- Prompt input -->
                    <div>
                        <label for="prompt">Describe your visualization</label>
                        <textarea id="prompt" placeholder="Describe what you want to create in detail...

Example prompts:
- A photorealistic 4K image of a futuristic city at golden hour
- An infographic explaining the water cycle for kids
- A Da Vinci style anatomical sketch of a butterfly"></textarea>
                    </div>

                    <!-- File context -->
                    <div class="file-section">
                        <label>Context Files (optional)</label>
                        <div class="file-row">
                            <input type="text" id="file-input" placeholder="/path/to/image.png">
                            <button class="btn btn-sm btn-outline" onclick="addFile()">+ Add</button>
                        </div>
                        <div id="file-list" class="file-list"></div>
                        <p style="font-size:0.7rem; color:var(--text-dim); margin-top:0.5rem;">
                            Supports images (PNG, JPG, WEBP) and text files (TXT, MD, JSON)
                        </p>
                    </div>

                    <!-- Settings -->
                    <div class="settings-row">
                        <div>
                            <label for="model">Model</label>
                            <select id="model">
                                <option value="pro" selected>Pro (4K + Reasoning)</option>
                                <option value="flash">Flash (Fast)</option>
                            </select>
                        </div>
                        <div>
                            <label for="aspect">Aspect Ratio</label>
                            <select id="aspect">
                                <option value="16:9" selected>16:9 Widescreen</option>
                                <option value="1:1">1:1 Square</option>
                                <option value="9:16">9:16 Portrait</option>
                                <option value="4:3">4:3 Standard</option>
                                <option value="3:2">3:2 Photo</option>
                                <option value="21:9">21:9 Ultrawide</option>
                            </select>
                        </div>
                        <div>
                            <label for="resolution">Resolution</label>
                            <select id="resolution">
                                <option value="4K" selected>4K Maximum</option>
                                <option value="2K">2K</option>
                                <option value="1K">1K</option>
                            </select>
                        </div>
                    </div>

                    <!-- Search toggle -->
                    <label class="checkbox-wrap">
                        <input type="checkbox" id="use-search">
                        <span>Enable Google Search grounding</span>
                    </label>

                    <!-- Generate button -->
                    <button id="gen-btn" class="btn btn-primary" onclick="generate()">
                        Generate Visualization
                    </button>
                    <p class="key-hint">Press <kbd>Ctrl+Enter</kbd> to generate</p>
                </div>
            </div>

            <!-- Output Panel -->
            <div class="output-panel">
                <div class="card output">
                    <h2 class="card-title">Generated Output</h2>

                    <!-- Status -->
                    <div id="status" class="status">
                        <div class="spinner"></div>
                        <span id="status-text">Generating...</span>
                    </div>

                    <!-- Thinking -->
                    <details id="thinking-box" class="thinking-box">
                        <summary>Model Reasoning</summary>
                        <div id="thinking-content" class="content"></div>
                    </details>

                    <!-- Image -->
                    <div id="image-box" class="image-box fade-in">
                        <img id="result-img" src="" alt="Generated image">
                        <div class="image-actions">
                            <button class="btn btn-success" onclick="downloadImg()">Download</button>
                            <button class="btn btn-ghost" onclick="openFull()">Full Size</button>
                        </div>
                    </div>

                    <!-- Save info -->
                    <div id="save-info" class="save-info">
                        Saved: <code id="save-path"></code> <span id="save-dims" style="color:var(--accent);margin-left:0.5rem;"></span>
                    </div>

                    <!-- Text output -->
                    <div id="text-output" class="text-output"></div>

                    <!-- Placeholder -->
                    <div id="placeholder" class="placeholder">
                        <div class="icon">Nano Banana</div>
                        <p>Your visualization will appear here</p>
                        <p style="font-size:0.85rem; margin-top:0.5rem;">Enter a prompt and click Generate</p>
                    </div>
                </div>

                <!-- History -->
                <div id="history-card" class="card history-card">
                    <h2 class="card-title">Recent</h2>
                    <div id="history-grid" class="history-grid"></div>
                </div>
            </div>
        </div>
        </div><!-- End Image Generation Tab -->

        <!-- Document Analysis Tab -->
        <div id="doc-analysis" class="tab-content">
            <div class="analysis-container">
                <!-- Left Panel: Document Selection -->
                <div class="card">
                    <h2 class="card-title">Document Selection</h2>

                    <div id="analyzer-status" class="analyzer-status disconnected">
                        <span class="status-dot"></span>
                        <span id="analyzer-status-text">Checking Analyzer API...</span>
                    </div>

                    <!-- Folder Input -->
                    <label>File or Folder Path</label>
                    <div class="folder-input-row">
                        <input type="text" id="folder-path" placeholder="~/Documents/articles">
                        <button class="btn btn-outline" onclick="scanFolder()">Scan</button>
                    </div>

                    <!-- Document List -->
                    <div id="doc-list-container" style="display:none;">
                        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:0.5rem;">
                            <label style="margin:0;">Documents Found</label>
                            <div>
                                <button class="btn btn-sm btn-ghost" onclick="selectAllDocs()">Select All</button>
                                <button class="btn btn-sm btn-ghost" onclick="deselectAllDocs()">Deselect All</button>
                            </div>
                        </div>
                        <div id="doc-list" class="doc-list"></div>
                        <p id="doc-count" style="font-size:0.8rem; color:var(--text-dim); margin-top:0.5rem;"></p>
                    </div>

                    <!-- Collection Mode -->
                    <label style="margin-top:1rem;">Analysis Mode</label>
                    <div class="mode-toggle">
                        <div class="mode-btn active" onclick="setCollectionMode('single')" id="mode-single">
                            <div class="title">Single Collection</div>
                            <div class="desc">Analyze all docs together as one topic</div>
                        </div>
                        <div class="mode-btn" onclick="setCollectionMode('individual')" id="mode-individual">
                            <div class="title">Individual Files</div>
                            <div class="desc">Each file is a separate topic</div>
                        </div>
                    </div>
                </div>

                <!-- Right Panel: Engine Selection & Results -->
                <div>
                    <div class="card">
                        <h2 class="card-title">Analysis Engine</h2>

                        <!-- Engine vs Bundle Toggle -->
                        <div class="mode-toggle">
                            <div class="mode-btn active" onclick="setEngineMode('engine')" id="engine-mode-single">
                                <div class="title">Single Engine</div>
                                <div class="desc">One analytical lens</div>
                            </div>
                            <div class="mode-btn" onclick="setEngineMode('bundle')" id="engine-mode-bundle">
                                <div class="title">Bundle</div>
                                <div class="desc">Multiple engines, shared extraction</div>
                            </div>
                        </div>

                        <!-- Single Engine Selection -->
                        <div id="engine-selection">
                            <label>Select Engine <span id="engine-count" style="color:var(--accent);"></span></label>
                            <div id="engine-grid" class="engine-grid"></div>
                        </div>

                        <!-- Bundle Selection -->
                        <div id="bundle-selection" style="display:none;">
                            <label>Select Bundle</label>
                            <div id="bundle-list"></div>
                        </div>

                        <!-- Output Mode -->
                        <div class="output-select">
                            <label>Output Format</label>
                            <select id="output-mode">
                                <option value="structured_text_report">Text Report</option>
                                <option value="gemini_image">Visual Diagram (Gemini)</option>
                                <option value="mermaid">Mermaid Diagram</option>
                                <option value="d3_interactive">Interactive D3</option>
                                <option value="comparative_matrix_table">Comparison Table</option>
                            </select>
                        </div>

                        <!-- Submit Button -->
                        <button id="analyze-btn" class="btn btn-primary" onclick="runAnalysis()" disabled>
                            Select Documents & Engine to Analyze
                        </button>
                    </div>

                    <!-- Progress -->
                    <div id="analysis-progress" class="analysis-progress">
                        <h3 style="margin-bottom:0.5rem;">Processing...</h3>
                        <div class="progress-bar-outer">
                            <div id="progress-bar" class="progress-bar-inner"></div>
                        </div>
                        <div id="progress-text" style="font-size:0.85rem; color:var(--text-dim);"></div>
                        <div class="progress-stage">
                            <span class="stage-badge" id="stage-extraction">Extraction</span>
                            <span class="stage-badge" id="stage-curation">Curation</span>
                            <span class="stage-badge" id="stage-concretization">Concretization</span>
                            <span class="stage-badge" id="stage-rendering">Rendering</span>
                        </div>
                    </div>

                    <!-- Results -->
                    <div id="results-container" class="results-container"></div>
                </div>
            </div>
        </div><!-- End Document Analysis Tab -->
    </div>

    <script>
        // State
        let files = [];
        let currentImg = null;
        let currentPath = null;
        let sessionId = null;
        let history = [];

        const $ = id => document.getElementById(id);

        // Initialization
        document.addEventListener('DOMContentLoaded', () => {
            $('prompt').focus();
            newSession();
        });

        document.addEventListener('keydown', e => {
            if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
                e.preventDefault();
                generate();
            }
        });

        // Session Management
        async function newSession() {
            try {
                const res = await fetch('/api/session/new', { method: 'POST' });
                const data = await res.json();
                sessionId = data.session_id;
                $('session-indicator').style.display = 'none';
            } catch (e) {
                console.error('Session creation failed:', e);
            }
        }

        // File Management
        async function addFile() {
            const input = $('file-input');
            const path = input.value.trim();

            if (!path || files.some(f => f.path === path)) return;

            try {
                const res = await fetch('/api/validate-path', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ path })
                });
                const data = await res.json();

                if (data.valid) {
                    files.push({ path, ...data });
                    renderFiles();
                    input.value = '';
                } else {
                    showStatus(data.error, 'error');
                    setTimeout(hideStatus, 3000);
                }
            } catch (e) {
                showStatus('Failed to validate path', 'error');
            }
        }

        function renderFiles() {
            const container = $('file-list');
            container.innerHTML = files.map((f, i) => `
                <div class="file-item fade-in">
                    ${f.preview ? `<img class="thumb" src="data:image/jpeg;base64,${f.preview}" alt="">` : ''}
                    <div class="info">
                        <div class="name">${f.name}</div>
                        <div class="meta">${f.type} - ${f.size}${f.dimensions ? ` - ${f.dimensions}` : ''}</div>
                    </div>
                    <span class="remove" onclick="removeFile(${i})">x</span>
                </div>
            `).join('');
        }

        function removeFile(i) {
            files.splice(i, 1);
            renderFiles();
        }

        // Status Display
        function showStatus(msg, type = 'loading') {
            const el = $('status');
            const spinner = el.querySelector('.spinner');
            el.className = `status show ${type}`;
            $('status-text').textContent = msg;
            spinner.style.display = type === 'loading' ? 'block' : 'none';
        }

        function hideStatus() {
            $('status').classList.remove('show');
        }

        // Generation
        async function generate() {
            const prompt = $('prompt').value.trim();

            if (!prompt) {
                showStatus('Please enter a prompt describing what to visualize', 'error');
                setTimeout(hideStatus, 3000);
                return;
            }

            const btn = $('gen-btn');
            btn.disabled = true;
            btn.textContent = 'Generating...';

            // Reset UI
            $('placeholder').style.display = 'none';
            $('image-box').classList.remove('show');
            $('save-info').classList.remove('show');
            $('text-output').classList.remove('show');
            $('thinking-box').classList.remove('show');

            const resolution = $('resolution').value;
            showStatus(`Generating ${resolution} image... This may take up to 30 seconds`, 'loading');

            try {
                const res = await fetch('/api/generate', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        prompt,
                        file_paths: files.map(f => f.path),
                        model: $('model').value,
                        aspect_ratio: $('aspect').value,
                        resolution: $('resolution').value,
                        use_search: $('use-search').checked,
                        session_id: sessionId
                    })
                });

                const data = await res.json();

                if (data.success) {
                    const meta = data.metadata || {};
                    showStatus(`Generated with ${meta.model || 'Nano Banana'} at ${meta.resolution || resolution}!`, 'success');
                    setTimeout(hideStatus, 4000);

                    // Show multi-turn indicator
                    $('session-indicator').style.display = 'inline-block';

                    // Thinking
                    if (data.thinking) {
                        $('thinking-box').classList.add('show');
                        $('thinking-content').textContent = data.thinking;
                    }

                    // Image
                    if (data.images && data.images.length > 0) {
                        currentImg = data.images[0];
                        $('result-img').src = `data:image/png;base64,${currentImg}`;
                        $('image-box').classList.add('show');
                        addToHistory(currentImg, prompt);
                    }

                    // Save path and dimensions
                    if (data.saved_paths && data.saved_paths.length > 0) {
                        currentPath = data.saved_paths[0];
                        $('save-path').textContent = currentPath;
                        const dims = data.metadata?.image_dimensions;
                        $('save-dims').textContent = dims ? `[${dims}]` : '';
                        $('save-info').classList.add('show');
                    }

                    // Text
                    if (data.text) {
                        $('text-output').textContent = data.text;
                        $('text-output').classList.add('show');
                    }
                } else {
                    showStatus(data.error || 'Generation failed', 'error');
                    $('placeholder').style.display = 'flex';
                }
            } catch (e) {
                showStatus(`Error: ${e.message}`, 'error');
                $('placeholder').style.display = 'flex';
            } finally {
                btn.disabled = false;
                btn.textContent = 'Generate Visualization';
            }
        }

        // Image Actions
        function downloadImg() {
            // Use full-size file from server if available
            if (currentPath) {
                const filename = currentPath.split('/').pop();
                const a = document.createElement('a');
                a.href = `/api/download/${encodeURIComponent(filename)}`;
                a.download = filename;
                a.click();
                return;
            }
            // Fallback to base64 display image
            if (!currentImg) return;
            const a = document.createElement('a');
            a.href = `data:image/png;base64,${currentImg}`;
            a.download = `nano_banana_${Date.now()}.png`;
            a.click();
        }

        function openFull() {
            // Use full-size file from server if available
            if (currentPath) {
                const filename = currentPath.split('/').pop();
                const w = window.open();
                w.document.write(`<html><head><title>Nano Banana Output - Full Size</title></head>
                    <body style="margin:0;background:#0d0d14;display:flex;justify-content:center;align-items:center;min-height:100vh;">
                    <img src="/api/download/${encodeURIComponent(filename)}" style="max-width:100%;max-height:100vh;"></body></html>`);
                return;
            }
            // Fallback to base64 display image
            if (!currentImg) return;
            const w = window.open();
            w.document.write(`<html><head><title>Nano Banana Output</title></head>
                <body style="margin:0;background:#0d0d14;display:flex;justify-content:center;align-items:center;min-height:100vh;">
                <img src="data:image/png;base64,${currentImg}" style="max-width:100%;max-height:100vh;"></body></html>`);
        }

        // History
        function addToHistory(img, prompt) {
            history.unshift({ img, prompt, time: Date.now() });
            if (history.length > 6) history = history.slice(0, 6);
            renderHistory();
        }

        function renderHistory() {
            if (!history.length) {
                $('history-card').classList.remove('show');
                return;
            }

            $('history-card').classList.add('show');
            $('history-grid').innerHTML = history.map((h, i) => `
                <div class="history-item" onclick="loadHistory(${i})" title="${h.prompt.slice(0, 80)}">
                    <img src="data:image/png;base64,${h.img}" alt="">
                </div>
            `).join('');
        }

        function loadHistory(i) {
            if (!history[i]) return;
            currentImg = history[i].img;
            $('result-img').src = `data:image/png;base64,${currentImg}`;
            $('image-box').classList.add('show');
            $('placeholder').style.display = 'none';
        }

        // ============================================================
        // DOCUMENT ANALYSIS FUNCTIONALITY
        // ============================================================

        // Analysis State
        let scannedDocs = [];
        let selectedDocs = new Set();
        let engines = [];
        let bundles = [];
        let selectedEngine = null;
        let selectedBundle = null;
        let collectionMode = 'single';
        let engineMode = 'engine';
        let currentJobId = null;

        // Tab Switching
        function switchTab(tabId) {
            document.querySelectorAll('.tab-content').forEach(el => el.classList.remove('active'));
            document.querySelectorAll('.tab-btn').forEach(el => el.classList.remove('active'));
            document.getElementById(tabId).classList.add('active');
            event.target.closest('.tab-btn').classList.add('active');

            // Load analysis data when switching to that tab
            if (tabId === 'doc-analysis' && engines.length === 0) {
                loadAnalyzerData();
            }
        }

        // Check Analyzer API Status
        async function checkAnalyzerStatus() {
            const statusEl = $('analyzer-status');
            const textEl = $('analyzer-status-text');

            try {
                const res = await fetch('/api/analyzer/engines');
                if (res.ok) {
                    statusEl.className = 'analyzer-status connected';
                    textEl.textContent = 'Analyzer API Connected';
                    return true;
                }
            } catch (e) {}

            statusEl.className = 'analyzer-status disconnected';
            textEl.textContent = 'Analyzer API Not Available - Start the Analyzer service';
            return false;
        }

        // Load Engines and Bundles
        async function loadAnalyzerData() {
            const connected = await checkAnalyzerStatus();
            if (!connected) return;

            try {
                // Load engines
                const enginesRes = await fetch('/api/analyzer/engines');
                if (enginesRes.ok) {
                    engines = await enginesRes.json();
                    renderEngines();
                }

                // Load bundles
                const bundlesRes = await fetch('/api/analyzer/bundles');
                if (bundlesRes.ok) {
                    bundles = await bundlesRes.json();
                    renderBundles();
                }
            } catch (e) {
                console.error('Failed to load analyzer data:', e);
            }
        }

        // Render Engine Cards
        function renderEngines() {
            const grid = $('engine-grid');
            $('engine-count').textContent = `(${engines.length} available)`;

            grid.innerHTML = engines.map(e => `
                <div class="engine-card ${selectedEngine === e.engine_key ? 'selected' : ''}"
                     onclick="selectEngine('${e.engine_key}')"
                     title="${e.description || ''}">
                    <div class="name">${e.name || e.engine_key}</div>
                    <span class="priority priority-${e.priority || 2}">P${e.priority || 2}</span>
                </div>
            `).join('');
        }

        // Render Bundle Cards
        function renderBundles() {
            const list = $('bundle-list');

            list.innerHTML = bundles.map(b => `
                <div class="bundle-card ${selectedBundle === b.bundle_key ? 'selected' : ''}"
                     onclick="selectBundle('${b.bundle_key}')">
                    <div class="name">${b.name || b.bundle_key}</div>
                    <div class="engines">${(b.member_engines || []).length} engines: ${(b.member_engines || []).slice(0, 3).join(', ')}${(b.member_engines || []).length > 3 ? '...' : ''}</div>
                </div>
            `).join('');
        }

        // Select Engine
        function selectEngine(key) {
            selectedEngine = key;
            selectedBundle = null;
            renderEngines();
            updateAnalyzeButton();
        }

        // Select Bundle
        function selectBundle(key) {
            selectedBundle = key;
            selectedEngine = null;
            renderBundles();
            updateAnalyzeButton();
        }

        // Scan Folder
        async function scanFolder() {
            const folderPath = $('folder-path').value.trim();
            if (!folderPath) {
                alert('Please enter a folder path');
                return;
            }

            try {
                const res = await fetch('/api/analyzer/scan-folder', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({ path: folderPath })
                });

                const data = await res.json();

                if (data.success) {
                    scannedDocs = data.files;
                    selectedDocs = new Set(scannedDocs.map(d => d.path));
                    renderDocList();
                    $('doc-list-container').style.display = 'block';

                    if (!data.pdf_support && scannedDocs.some(d => d.type === 'pdf')) {
                        alert('Note: PDF support requires pymupdf. Run: pip install pymupdf');
                    }
                } else {
                    alert(data.error || 'Failed to scan folder');
                }
            } catch (e) {
                alert('Error scanning folder: ' + e.message);
            }
        }

        // Render Document List
        function renderDocList() {
            const list = $('doc-list');
            const icons = { pdf: '📄', md: '📝', txt: '📃' };

            list.innerHTML = scannedDocs.map(d => `
                <div class="doc-item">
                    <input type="checkbox" ${selectedDocs.has(d.path) ? 'checked' : ''}
                           onchange="toggleDoc('${d.path}')">
                    <span class="icon">${icons[d.type] || '📄'}</span>
                    <div class="info">
                        <div class="name">${d.name}</div>
                        <div class="meta">${d.type.toUpperCase()} - ${d.size}</div>
                    </div>
                </div>
            `).join('');

            $('doc-count').textContent = `${selectedDocs.size} of ${scannedDocs.length} documents selected`;
            updateAnalyzeButton();
        }

        // Toggle Document Selection
        function toggleDoc(path) {
            if (selectedDocs.has(path)) {
                selectedDocs.delete(path);
            } else {
                selectedDocs.add(path);
            }
            renderDocList();
        }

        // Select/Deselect All
        function selectAllDocs() {
            selectedDocs = new Set(scannedDocs.map(d => d.path));
            renderDocList();
        }

        function deselectAllDocs() {
            selectedDocs.clear();
            renderDocList();
        }

        // Set Collection Mode
        function setCollectionMode(mode) {
            collectionMode = mode;
            $('mode-single').classList.toggle('active', mode === 'single');
            $('mode-individual').classList.toggle('active', mode === 'individual');
        }

        // Set Engine Mode
        function setEngineMode(mode) {
            engineMode = mode;
            $('engine-mode-single').classList.toggle('active', mode === 'engine');
            $('engine-mode-bundle').classList.toggle('active', mode === 'bundle');
            $('engine-selection').style.display = mode === 'engine' ? 'block' : 'none';
            $('bundle-selection').style.display = mode === 'bundle' ? 'block' : 'none';
        }

        // Update Analyze Button
        function updateAnalyzeButton() {
            const btn = $('analyze-btn');
            const hasSelection = engineMode === 'engine' ? selectedEngine : selectedBundle;
            const hasDocs = selectedDocs.size > 0;

            btn.disabled = !hasSelection || !hasDocs;

            if (!hasDocs) {
                btn.textContent = 'Select Documents First';
            } else if (!hasSelection) {
                btn.textContent = `Select ${engineMode === 'engine' ? 'Engine' : 'Bundle'} to Analyze`;
            } else {
                btn.textContent = `Analyze ${selectedDocs.size} Document${selectedDocs.size > 1 ? 's' : ''}`;
            }
        }

        // Run Analysis
        async function runAnalysis() {
            const filePaths = Array.from(selectedDocs);
            const outputMode = $('output-mode').value;

            $('analyze-btn').disabled = true;
            $('analyze-btn').textContent = 'Submitting...';
            $('analysis-progress').classList.add('show');
            $('results-container').innerHTML = '';

            resetStages();

            try {
                let response;
                if (engineMode === 'engine') {
                    response = await fetch('/api/analyzer/analyze', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({
                            file_paths: filePaths,
                            engine: selectedEngine,
                            output_mode: outputMode,
                            collection_mode: collectionMode
                        })
                    });
                } else {
                    // Bundle analysis
                    const outputModes = {};
                    const bundle = bundles.find(b => b.bundle_key === selectedBundle);
                    if (bundle) {
                        bundle.member_engines.forEach(e => {
                            outputModes[e] = outputMode;
                        });
                    }

                    response = await fetch('/api/analyzer/analyze/bundle', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({
                            file_paths: filePaths,
                            bundle: selectedBundle,
                            output_modes: outputModes
                        })
                    });
                }

                const data = await response.json();

                if (data.success) {
                    if (data.mode === 'individual') {
                        // Multiple jobs - track all
                        pollMultipleJobs(data.jobs);
                    } else {
                        // Single job
                        currentJobId = data.job_id;
                        pollJobStatus(data.job_id);
                    }
                } else {
                    showAnalysisError(data.error || 'Failed to submit analysis');
                }
            } catch (e) {
                showAnalysisError('Error: ' + e.message);
            }
        }

        // Poll Job Status
        async function pollJobStatus(jobId) {
            try {
                const res = await fetch(`/api/analyzer/jobs/${jobId}`);
                const job = await res.json();

                updateProgress(job);

                if (job.status === 'completed') {
                    await fetchAndDisplayResult(jobId);
                } else if (job.status === 'failed') {
                    showAnalysisError(job.error_message || 'Analysis failed');
                } else {
                    setTimeout(() => pollJobStatus(jobId), 2000);
                }
            } catch (e) {
                showAnalysisError('Error polling status: ' + e.message);
            }
        }

        // Poll Multiple Jobs
        async function pollMultipleJobs(jobs) {
            const pending = jobs.filter(j => j.status === 'submitted');
            let allDone = true;

            for (const job of pending) {
                if (!job.job_id) continue;

                try {
                    const res = await fetch(`/api/analyzer/jobs/${job.job_id}`);
                    const status = await res.json();

                    if (status.status === 'completed') {
                        job.status = 'completed';
                        const resultRes = await fetch(`/api/analyzer/jobs/${job.job_id}/result`);
                        const result = await resultRes.json();
                        displayResult(result, job.title);
                    } else if (status.status === 'failed') {
                        job.status = 'failed';
                        displayError(job.title, status.error_message);
                    } else {
                        allDone = false;
                    }
                } catch (e) {
                    job.status = 'failed';
                }
            }

            const completed = jobs.filter(j => j.status === 'completed' || j.status === 'failed').length;
            updateProgressMulti(completed, jobs.length);

            if (!allDone) {
                setTimeout(() => pollMultipleJobs(jobs), 2000);
            } else {
                finishAnalysis();
            }
        }

        // Update Progress Display
        function updateProgress(job) {
            const percent = job.progress_percent || 0;
            $('progress-bar').style.width = `${percent}%`;
            $('progress-text').textContent = `${job.current_stage || job.status} (${percent}%)`;

            // Update stage badges
            const stages = ['extraction', 'curation', 'concretization', 'rendering'];
            const currentIdx = stages.indexOf(job.current_stage?.toLowerCase());

            stages.forEach((stage, i) => {
                const el = $(`stage-${stage}`);
                if (!el) return;

                if (i < currentIdx) {
                    el.className = 'stage-badge completed';
                } else if (i === currentIdx) {
                    el.className = 'stage-badge active';
                } else {
                    el.className = 'stage-badge';
                }
            });
        }

        function updateProgressMulti(completed, total) {
            const percent = Math.round((completed / total) * 100);
            $('progress-bar').style.width = `${percent}%`;
            $('progress-text').textContent = `${completed} of ${total} documents processed`;
        }

        function resetStages() {
            ['extraction', 'curation', 'concretization', 'rendering'].forEach(s => {
                const el = $(`stage-${s}`);
                if (el) el.className = 'stage-badge';
            });
        }

        // Fetch and Display Result
        async function fetchAndDisplayResult(jobId) {
            try {
                const res = await fetch(`/api/analyzer/jobs/${jobId}/result`);
                const result = await res.json();
                displayResult(result);
                finishAnalysis();
            } catch (e) {
                showAnalysisError('Error fetching result: ' + e.message);
            }
        }

        // Escape HTML for safe display
        function escapeHtml(text) {
            var div = document.createElement('div');
            div.textContent = text;
            return div.innerHTML;
        }

        // Display Result
        function displayResult(result, title = null) {
            const container = $('results-container');
            const metadata = result.metadata || {};

            // Display outputs
            const outputs = result.outputs || {};
            for (const [key, output] of Object.entries(outputs)) {
                const card = document.createElement('div');
                card.className = 'result-card fade-in';
                const cardId = 'result-' + Date.now() + '-' + Math.random().toString(36).substr(2, 9);

                let contentHtml = '';
                let rawContent = '';

                if (output.image_url) {
                    contentHtml = `<img src="${output.image_url}" alt="${key}">`;
                } else if (output.html_content) {
                    contentHtml = output.html_content;
                    rawContent = output.html_content;
                } else if (output.content) {
                    contentHtml = '<pre>' + escapeHtml(output.content) + '<'+'/pre>';
                    rawContent = output.content;
                } else if (output.data) {
                    rawContent = JSON.stringify(output.data, null, 2);
                    contentHtml = '<pre>' + rawContent + '<\/pre>';
                }

                // Build metadata string
                let metaHtml = '';
                if (metadata.total_ms || metadata.cost_usd) {
                    const time = metadata.total_ms ? (metadata.total_ms / 1000).toFixed(1) + 's' : '';
                    const cost = metadata.cost_usd ? '$' + metadata.cost_usd.toFixed(4) : '';
                    metaHtml = '<div class="result-meta">⏱️ ' + time + ' | 💰 ' + cost + '<\/div>';
                }

                card.innerHTML =
                    '<div class="header">' +
                        '<span>' + (title ? title + ' - ' : '') + key.replace(/_/g, ' ') + '<\/span>' +
                        '<div class="result-actions">' +
                            '<button onclick="downloadResult(\'' + cardId + '\', \'' + key + '\')">📥 Download<\/button>' +
                            '<span class="badge">' + (output.mode || output.renderer_type || '') + '<\/span>' +
                        '<\/div>' +
                    '<\/div>' +
                    '<div class="content" id="' + cardId + '">' + contentHtml + metaHtml + '<\/div>';

                // Store raw content for download
                card.dataset.rawContent = rawContent;
                card.dataset.filename = key;

                container.appendChild(card);
            }

            // If no outputs but has canonical data, show that
            if (Object.keys(outputs).length === 0 && result.canonical_data) {
                const card = document.createElement('div');
                card.className = 'result-card fade-in';
                const rawJson = JSON.stringify(result.canonical_data, null, 2);
                card.innerHTML =
                    '<div class="header">' +
                        '<span>Canonical Data<\/span>' +
                        '<div class="result-actions">' +
                            '<button onclick="downloadJson(this, \'canonical_data.json\')">📥 Download JSON<\/button>' +
                        '<\/div>' +
                    '<\/div>' +
                    '<div class="content"><pre>' + rawJson + '<\/pre><\/div>';
                card.dataset.rawContent = rawJson;
                container.appendChild(card);
            }
        }

        // Download result as file
        function downloadResult(contentId, filename) {
            const card = document.getElementById(contentId)?.closest('.result-card');
            if (!card) return;

            const content = card.dataset.rawContent || card.querySelector('.content')?.innerText || '';
            const ext = content.startsWith('{') || content.startsWith('[') ? '.json' : '.md';

            const blob = new Blob([content], { type: 'text/plain' });
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = filename.replace(/_/g, '-') + ext;
            a.click();
            URL.revokeObjectURL(url);
        }

        function downloadJson(btn, filename) {
            const card = btn.closest('.result-card');
            const content = card.dataset.rawContent || '';
            const blob = new Blob([content], { type: 'application/json' });
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = filename;
            a.click();
            URL.revokeObjectURL(url);
        }

        function displayError(title, message) {
            const container = $('results-container');
            const card = document.createElement('div');
            card.className = 'result-card fade-in';
            card.innerHTML =
                '<div class="header" style="background:rgba(248,113,113,0.2);">' + title + '<\/div>' +
                '<div class="content" style="color:var(--error);">' + (message || 'Analysis failed') + '<\/div>';
            container.appendChild(card);
        }

        function showAnalysisError(message) {
            $('progress-text').textContent = message;
            $('progress-text').style.color = 'var(--error)';
            finishAnalysis();
        }

        function finishAnalysis() {
            $('analyze-btn').disabled = false;
            updateAnalyzeButton();

            // Mark all stages complete
            ['extraction', 'curation', 'concretization', 'rendering'].forEach(s => {
                const el = $(`stage-${s}`);
                if (el && !$('progress-text').style.color.includes('error')) {
                    el.className = 'stage-badge completed';
                }
            });
        }

        function escapeHtml(text) {
            const div = document.createElement('div');
            div.textContent = text;
            return div.innerHTML;
        }
    </script>
</body>
</html>'''


# Application Entry Point

if __name__ == '__main__':
    # Print startup banner
    print()
    print("=" * 65)
    print("  NANO BANANA 4K VISUALIZER")
    print("=" * 65)
    print()
    print(f"  Download folder  : {DOWNLOAD_FOLDER}")
    print(f"  API Key          : {'Environment variable' if os.environ.get('GEMINI_API_KEY') else 'Embedded'}")
    print(f"  Available models : {', '.join(m['name'] for m in MODELS.values())}")
    print()
    print("=" * 65)
    print("  Starting server at: http://localhost:5010")
    print("  Press Ctrl+C to stop")
    print("=" * 65)
    print()

    # Initialize client before starting server
    if initialize_client():
        print("  Gemini client initialized successfully")
    else:
        print("  Client initialization deferred to first request")

    print()

    # Collect files to watch for auto-reload
    extra_files = []
    templates_dir = Path(__file__).parent / 'templates'
    if templates_dir.exists():
        for template_file in templates_dir.rglob('*.html'):
            extra_files.append(str(template_file))

    static_dir = Path(__file__).parent / 'static'
    if static_dir.exists():
        for static_file in static_dir.rglob('*'):
            if static_file.is_file():
                extra_files.append(str(static_file))

    if extra_files:
        print(f"  Watching {len(extra_files)} template and static files for changes")

    # Run the Flask server with auto-reload enabled
    app.run(
        host='0.0.0.0',
        port=5847,
        debug=True,
        use_reloader=True,
        extra_files=extra_files if extra_files else None,
        threaded=True
    )
