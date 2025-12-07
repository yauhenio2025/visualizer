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

    # Text-based formats that can be read directly
    TEXT_FORMATS = {'.md', '.txt', '.json', '.xml', '.html', '.htm', '.csv',
                    '.py', '.js', '.ts', '.yaml', '.yml', '.rst', '.tex', '.log'}

    try:
        if suffix == '.pdf':
            content = extract_pdf_text(file_path)
            return content, None
        elif suffix in TEXT_FORMATS:
            content = file_path.read_text(encoding='utf-8', errors='replace')
            return content, None
        else:
            return None, f"Unsupported file type: {suffix}. Supported: PDF, TXT, MD, JSON, XML, HTML, CSV, and more text formats"
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


def extract_pdf_from_base64(base64_content: str) -> str:
    """Extract text from base64-encoded PDF."""
    import base64
    import io

    if not PDF_SUPPORT:
        return "[PDF support not available. Install pymupdf: pip install pymupdf]"

    try:
        pdf_bytes = base64.b64decode(base64_content)
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        text_parts = []
        for page_num, page in enumerate(doc):
            text = page.get_text()
            if text.strip():
                text_parts.append(f"--- Page {page_num + 1} ---\n{text}")
        doc.close()
        return "\n\n".join(text_parts) if text_parts else "[No text extracted from PDF]"
    except Exception as e:
        return f"[Error reading PDF: {str(e)}]"


@app.route('/api/analyzer/analyze', methods=['POST'])
def submit_analysis():
    """
    Submit documents for analysis.

    Request (file paths):
    {
        "file_paths": ["/path/to/doc.pdf", ...],
        "engine": "thematic_synthesis",
        "output_mode": "structured_text_report",
        "collection_mode": "single" | "individual"
    }

    Request (inline documents - for browser uploads):
    {
        "documents": [{"id": "doc_1", "title": "...", "content": "...", "encoding": "text|base64"}],
        "engine": "thematic_synthesis",
        "output_mode": "executive_memo"
    }
    """
    data = request.json or {}
    file_paths = data.get('file_paths', [])
    inline_documents = data.get('documents', [])
    engine = data.get('engine')
    output_mode = data.get('output_mode', 'structured_text_report')
    collection_mode = data.get('collection_mode', 'single')

    if not file_paths and not inline_documents:
        return jsonify({"success": False, "error": "No files provided"})

    if not engine:
        return jsonify({"success": False, "error": "No engine selected"})

    documents = []
    errors = []

    # Handle inline documents (from browser upload)
    if inline_documents:
        for i, doc in enumerate(inline_documents):
            doc_id = doc.get('id', f'doc_{i+1}')
            title = doc.get('title', f'Document {i+1}')
            content = doc.get('content', '')
            encoding = doc.get('encoding', 'text')

            if encoding == 'base64':
                # PDF - decode and extract text
                content = extract_pdf_from_base64(content)

            if content and not content.startswith('[Error'):
                documents.append({
                    "id": doc_id,
                    "title": title,
                    "content": content
                })
            else:
                errors.append(f"Failed to extract content from {title}")

    # Handle file paths (from server folder scan)
    else:
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

    # Debug: log document info
    for doc in documents:
        print(f"[DEBUG] Document: {doc.get('title', 'unknown')}, content length: {len(doc.get('content', ''))}")

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
                timeout=300.0,  # 5 minutes for large document sets
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

        except httpx.HTTPStatusError as e:
            # Get the response body for more details
            error_detail = ""
            try:
                error_detail = e.response.text
            except:
                pass
            print(f"[DEBUG] API Error: {e}, Response: {error_detail}")
            return jsonify({
                "success": False,
                "error": f"API error: {str(e)}",
                "detail": error_detail
            })
        except httpx.HTTPError as e:
            print(f"[DEBUG] HTTP Error: {e}")
            return jsonify({
                "success": False,
                "error": f"API error: {str(e)}"
            })


@app.route('/api/analyzer/analyze/bundle', methods=['POST'])
def submit_bundle_analysis():
    """Submit documents for bundle analysis (multiple engines)."""
    data = request.json or {}
    file_paths = data.get('file_paths', [])
    inline_documents = data.get('documents', [])
    bundle = data.get('bundle')
    output_modes = data.get('output_modes', {})

    if not file_paths and not inline_documents:
        return jsonify({"success": False, "error": "No files provided"})

    if not bundle:
        return jsonify({"success": False, "error": "No bundle selected"})

    documents = []
    errors = []

    # Handle inline documents (from browser upload)
    if inline_documents:
        for i, doc in enumerate(inline_documents):
            doc_id = doc.get('id', f'doc_{i+1}')
            title = doc.get('title', f'Document {i+1}')
            content = doc.get('content', '')
            encoding = doc.get('encoding', 'text')

            if encoding == 'base64':
                content = extract_pdf_from_base64(content)

            if content and not content.startswith('[Error'):
                documents.append({
                    "id": doc_id,
                    "title": title,
                    "content": content
                })
            else:
                errors.append(f"Failed to extract content from {title}")

    # Handle file paths (from server folder scan)
    else:
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
    <title>The Visualizer</title>
    <link href="https://fonts.googleapis.com/css2?family=Libre+Baskerville:ital,wght@0,400;0,700;1,400&family=Inter:wght@400;500;600&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-page: #faf9f6;
            --bg-card: #ffffff;
            --bg-input: #f5f4f0;
            --bg-hover: #eeeee8;
            --accent: #1a1a1a;
            --accent-muted: #4a4a4a;
            --success: #2d7d46;
            --error: #b33a3a;
            --warning: #c67e00;
            --text: #1a1a1a;
            --text-secondary: #666666;
            --text-muted: #999999;
            --border: #e0ddd5;
            --border-dark: #c5c2ba;
            --radius: 4px;
            --shadow: 0 1px 3px rgba(0,0,0,0.08);
            --shadow-lg: 0 4px 12px rgba(0,0,0,0.1);
        }

        * { margin: 0; padding: 0; box-sizing: border-box; }

        body {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            background: var(--bg-page);
            color: var(--text);
            min-height: 100vh;
            line-height: 1.6;
            font-size: 15px;
        }

        .app {
            max-width: 1200px;
            margin: 0 auto;
            padding: 0 2rem 3rem;
        }

        /* Header - NYT/New Yorker Style */
        header {
            text-align: center;
            padding: 2.5rem 1rem 2rem;
            border-bottom: 3px double var(--border-dark);
            margin-bottom: 2rem;
        }

        header h1 {
            font-family: 'Libre Baskerville', Georgia, serif;
            font-size: clamp(2rem, 5vw, 3rem);
            font-weight: 400;
            letter-spacing: -0.02em;
            color: var(--text);
            margin-bottom: 0.5rem;
        }

        header .tagline {
            font-size: 0.85rem;
            text-transform: uppercase;
            letter-spacing: 0.15em;
            color: var(--text-muted);
        }

        /* Navigation */
        .nav-bar {
            display: flex;
            justify-content: center;
            gap: 0;
            margin-bottom: 2.5rem;
            border-bottom: 1px solid var(--border);
        }

        .nav-btn {
            padding: 1rem 2.5rem;
            background: transparent;
            border: none;
            border-bottom: 2px solid transparent;
            color: var(--text-secondary);
            font-family: 'Inter', sans-serif;
            font-size: 0.8rem;
            font-weight: 500;
            text-transform: uppercase;
            letter-spacing: 0.1em;
            cursor: pointer;
            transition: all 0.2s;
            margin-bottom: -1px;
        }

        .nav-btn:hover { color: var(--text); }
        .nav-btn.active { color: var(--text); border-bottom-color: var(--text); }

        .view-content { display: none; }
        .view-content.active { display: block; }

        /* Main Layout */
        .main-layout {
            display: grid;
            grid-template-columns: 340px 1fr;
            gap: 2.5rem;
            align-items: start;
        }

        @media (max-width: 1100px) {
            .main-layout { grid-template-columns: 1fr; }
        }

        /* Cards */
        .card {
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: var(--radius);
            padding: 1.75rem;
            box-shadow: var(--shadow);
        }

        .card + .card { margin-top: 1.5rem; }

        .card-header {
            font-family: 'Libre Baskerville', Georgia, serif;
            font-size: 1.1rem;
            font-weight: 400;
            color: var(--text);
            margin-bottom: 1.5rem;
            padding-bottom: 0.75rem;
            border-bottom: 1px solid var(--border);
        }

        .section-label {
            display: block;
            font-size: 0.7rem;
            text-transform: uppercase;
            letter-spacing: 0.12em;
            color: var(--text-muted);
            margin-bottom: 0.5rem;
            font-weight: 500;
        }

        /* Form controls */
        input[type="text"], select {
            width: 100%;
            background: var(--bg-input);
            border: 1px solid var(--border);
            border-radius: var(--radius);
            padding: 0.75rem 1rem;
            color: var(--text);
            font-size: 0.9rem;
            font-family: inherit;
            transition: border-color 0.2s, box-shadow 0.2s;
        }

        input[type="text"]:focus, select:focus {
            outline: none;
            border-color: var(--accent);
            box-shadow: 0 0 0 2px rgba(26,26,26,0.1);
        }

        input::placeholder { color: var(--text-muted); }

        select {
            cursor: pointer;
            appearance: none;
            background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='12' height='12' viewBox='0 0 12 12'%3E%3Cpath fill='%23666' d='M6 8L1 3h10z'/%3E%3C/svg%3E");
            background-repeat: no-repeat;
            background-position: right 1rem center;
            padding-right: 2.5rem;
        }

        /* Upload zone */
        .upload-zone {
            border: 2px dashed var(--border);
            border-radius: var(--radius);
            padding: 2rem;
            text-align: center;
            background: var(--bg-input);
            cursor: pointer;
            transition: all 0.2s;
        }

        .upload-zone:hover { border-color: var(--accent-muted); background: var(--bg-hover); }
        .upload-zone.dragover { border-color: var(--accent); background: rgba(26,26,26,0.05); }
        .upload-zone-icon { font-size: 2rem; margin-bottom: 0.5rem; opacity: 0.5; }
        .upload-zone-text { font-size: 0.9rem; color: var(--text-secondary); }
        .upload-zone-hint { font-size: 0.75rem; color: var(--text-muted); margin-top: 0.5rem; }

        /* Hidden file inputs */
        .hidden-input { display: none; }

        /* Path input row */
        .path-input-row {
            display: flex;
            gap: 0.5rem;
            margin-top: 0.75rem;
        }

        .path-input-row input {
            flex: 1;
            font-family: 'SF Mono', Monaco, Consolas, monospace;
            font-size: 0.85rem;
        }

        /* Document list */
        .doc-list {
            max-height: 300px;
            overflow-y: auto;
            border: 1px solid var(--border);
            border-radius: var(--radius);
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
        .doc-item input[type="checkbox"] { width: 16px; height: 16px; accent-color: var(--accent); cursor: pointer; }
        .doc-item .icon { font-size: 1.25rem; opacity: 0.6; }
        .doc-item .info { flex: 1; min-width: 0; }
        .doc-item .name { font-weight: 500; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
        .doc-item .meta { font-size: 0.75rem; color: var(--text-muted); }

        .doc-count {
            font-size: 0.8rem;
            color: var(--text-muted);
            padding: 0.5rem 0;
        }

        .doc-actions {
            display: flex;
            gap: 0.5rem;
            margin-bottom: 0.5rem;
        }

        /* Mode toggle */
        .mode-toggle {
            display: flex;
            gap: 0.5rem;
            margin: 1rem 0;
        }

        .mode-btn {
            flex: 1;
            padding: 0.875rem;
            background: var(--bg-input);
            border: 1px solid var(--border);
            border-radius: var(--radius);
            color: var(--text);
            cursor: pointer;
            text-align: center;
            transition: all 0.15s;
        }

        .mode-btn:hover { border-color: var(--accent-muted); }
        .mode-btn.active { border-color: var(--accent); background: var(--bg-card); box-shadow: var(--shadow); }
        .mode-btn .title { font-weight: 600; font-size: 0.85rem; }
        .mode-btn .desc { font-size: 0.7rem; color: var(--text-muted); margin-top: 0.25rem; }

        /* Engine grid - expansive, inviting layout */
        .engine-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
            gap: 1rem;
            margin: 1.5rem 0;
            padding: 0;
        }

        .engine-card {
            padding: 1.25rem;
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: var(--radius);
            cursor: pointer;
            transition: all 0.15s;
            display: flex;
            flex-direction: column;
            gap: 0.5rem;
        }

        .engine-card:hover {
            border-color: var(--accent-muted);
            box-shadow: var(--shadow);
            transform: translateY(-1px);
        }

        .engine-card.selected {
            border-color: var(--accent);
            background: var(--bg-card);
            box-shadow: var(--shadow-lg);
            border-width: 2px;
        }

        .engine-card .name {
            font-family: 'Libre Baskerville', Georgia, serif;
            font-weight: 400;
            font-size: 1rem;
            color: var(--text);
            line-height: 1.3;
        }

        .engine-card .desc {
            font-size: 0.8rem;
            color: var(--text-muted);
            line-height: 1.4;
            flex: 1;
        }

        .engine-card .priority {
            display: inline-block;
            padding: 0.2rem 0.5rem;
            border-radius: 3px;
            font-size: 0.65rem;
            font-weight: 500;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            align-self: flex-start;
        }

        .priority-1 { background: rgba(45,125,70,0.12); color: var(--success); }
        .priority-2 { background: rgba(198,126,0,0.12); color: var(--warning); }
        .priority-3 { background: rgba(102,102,102,0.12); color: var(--text-secondary); }

        /* Bundles */
        .bundle-card {
            padding: 1rem;
            background: var(--bg-input);
            border: 1px solid var(--border);
            border-radius: var(--radius);
            cursor: pointer;
            margin-bottom: 0.5rem;
            transition: all 0.15s;
        }

        .bundle-card:hover { border-color: var(--accent-muted); }
        .bundle-card.selected { border-color: var(--accent); background: var(--bg-card); }
        .bundle-card .name { font-weight: 600; font-size: 0.9rem; }
        .bundle-card .engines { font-size: 0.75rem; color: var(--text-muted); margin-top: 0.25rem; }

        /* Buttons */
        .btn {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            gap: 0.5rem;
            padding: 0.65rem 1.25rem;
            border: 1px solid var(--border);
            border-radius: var(--radius);
            font-size: 0.85rem;
            font-weight: 500;
            cursor: pointer;
            transition: all 0.15s;
            background: var(--bg-card);
            color: var(--text);
        }

        .btn:hover { background: var(--bg-hover); border-color: var(--border-dark); }
        .btn-sm { padding: 0.45rem 0.75rem; font-size: 0.8rem; }
        .btn-outline { background: transparent; border-color: var(--accent); color: var(--accent); }
        .btn-outline:hover { background: var(--accent); color: var(--bg-card); }

        .btn-primary {
            width: 100%;
            padding: 0.9rem 1.5rem;
            background: var(--accent);
            border-color: var(--accent);
            color: var(--bg-card);
            font-size: 0.85rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.08em;
        }

        .btn-primary:hover:not(:disabled) { background: var(--accent-muted); border-color: var(--accent-muted); }
        .btn-primary:disabled { opacity: 0.5; cursor: not-allowed; background: var(--text-muted); border-color: var(--text-muted); }
        .btn-ghost { background: transparent; border-color: transparent; color: var(--text-secondary); }
        .btn-ghost:hover { background: var(--bg-hover); color: var(--text); }

        /* API Status */
        .api-status {
            display: inline-flex;
            align-items: center;
            gap: 0.5rem;
            padding: 0.5rem 0.75rem;
            border-radius: var(--radius);
            font-size: 0.75rem;
            font-weight: 500;
            margin-bottom: 1.25rem;
        }

        .api-status.connected { background: rgba(45,125,70,0.1); color: var(--success); }
        .api-status.disconnected { background: rgba(179,58,58,0.1); color: var(--error); }
        .status-dot { width: 6px; height: 6px; border-radius: 50%; background: currentColor; }

        /* Progress section */
        .progress-section {
            display: none;
            padding: 1.5rem;
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: var(--radius);
            margin-top: 1.5rem;
            box-shadow: var(--shadow);
        }

        .progress-section.show { display: block; }

        .progress-header {
            font-family: 'Libre Baskerville', Georgia, serif;
            font-size: 1rem;
            margin-bottom: 1rem;
        }

        .progress-bar-outer {
            height: 6px;
            background: var(--bg-input);
            border-radius: 3px;
            overflow: hidden;
            margin: 1rem 0;
        }

        .progress-bar-inner {
            height: 100%;
            background: var(--accent);
            width: 0%;
            transition: width 0.3s;
        }

        .progress-text {
            font-size: 0.85rem;
            color: var(--text-secondary);
        }

        .progress-stages {
            display: flex;
            gap: 0.5rem;
            flex-wrap: wrap;
            margin-top: 1rem;
        }

        .stage-badge {
            padding: 0.25rem 0.75rem;
            border-radius: 999px;
            font-size: 0.7rem;
            font-weight: 500;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            background: var(--bg-input);
            color: var(--text-muted);
        }

        .stage-badge.active { background: rgba(26,26,26,0.1); color: var(--accent); }
        .stage-badge.completed { background: rgba(45,125,70,0.15); color: var(--success); }

        /* Results Gallery */
        .results-gallery {
            margin-top: 2rem;
        }

        .results-gallery-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 1rem;
            padding-bottom: 0.75rem;
            border-bottom: 1px solid var(--border);
        }

        .results-gallery-header h3 {
            font-family: 'Libre Baskerville', Georgia, serif;
            font-size: 1.1rem;
            font-weight: 400;
        }

        .results-count {
            font-size: 0.8rem;
            color: var(--text-muted);
        }

        .results-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
            gap: 1.25rem;
        }

        .gallery-card {
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: var(--radius);
            overflow: hidden;
            transition: transform 0.15s, box-shadow 0.15s;
            cursor: pointer;
        }

        .gallery-card:hover {
            transform: translateY(-2px);
            box-shadow: var(--shadow-lg);
        }

        .gallery-card-preview {
            height: 180px;
            background: var(--bg-input);
            display: flex;
            align-items: center;
            justify-content: center;
            overflow: hidden;
        }

        .gallery-card-preview img { width: 100%; height: 100%; object-fit: cover; }
        .gallery-card-preview .text-preview {
            padding: 1rem;
            font-size: 0.75rem;
            line-height: 1.5;
            color: var(--text-secondary);
            overflow: hidden;
            max-height: 100%;
            font-family: 'SF Mono', Monaco, monospace;
        }
        .gallery-card-preview .icon-preview { font-size: 2.5rem; opacity: 0.4; }

        .gallery-card-info { padding: 1rem; }
        .gallery-card-title { font-weight: 500; font-size: 0.9rem; margin-bottom: 0.5rem; }
        .gallery-card-meta {
            display: flex;
            gap: 0.75rem;
            font-size: 0.75rem;
            color: var(--text-muted);
        }

        .gallery-card-actions {
            display: flex;
            gap: 0.5rem;
            padding: 0 1rem 1rem;
        }

        .gallery-card-actions button {
            flex: 1;
            padding: 0.5rem;
            font-size: 0.8rem;
            background: var(--bg-input);
            border: 1px solid var(--border);
            border-radius: var(--radius);
            color: var(--text);
            cursor: pointer;
            transition: all 0.15s;
        }

        .gallery-card-actions button:hover {
            background: var(--accent);
            border-color: var(--accent);
            color: var(--bg-card);
        }

        /* Modal */
        .result-modal {
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: rgba(0,0,0,0.8);
            z-index: 1000;
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 2rem;
        }

        .result-modal-content {
            background: var(--bg-card);
            border-radius: var(--radius);
            max-width: 90vw;
            max-height: 90vh;
            overflow: auto;
            position: relative;
        }

        .result-modal-header {
            position: sticky;
            top: 0;
            background: var(--bg-input);
            padding: 1rem 1.5rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
            z-index: 1;
            border-bottom: 1px solid var(--border);
        }

        .result-modal-header h3 {
            font-family: 'Libre Baskerville', Georgia, serif;
            font-size: 1rem;
            font-weight: 400;
        }

        .result-modal-close {
            background: none;
            border: none;
            font-size: 1.5rem;
            cursor: pointer;
            color: var(--text-muted);
            padding: 0.5rem;
        }

        .result-modal-close:hover { color: var(--text); }

        .result-modal-body {
            padding: 1.5rem;
        }

        .result-modal-body img { max-width: 100%; border-radius: var(--radius); }
        .result-modal-body pre {
            white-space: pre-wrap;
            font-size: 0.85rem;
            line-height: 1.6;
            background: var(--bg-input);
            padding: 1rem;
            border-radius: var(--radius);
            overflow: auto;
            max-height: 70vh;
        }

        .result-modal-actions {
            padding: 1rem 1.5rem;
            border-top: 1px solid var(--border);
            display: flex;
            gap: 0.5rem;
            justify-content: flex-end;
        }

        /* Library View */
        .library-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(250px, 1fr));
            gap: 1.5rem;
        }

        .library-empty {
            text-align: center;
            padding: 4rem 2rem;
            color: var(--text-muted);
        }

        .library-empty-icon {
            font-size: 3rem;
            margin-bottom: 1rem;
            opacity: 0.4;
        }

        .library-empty-text {
            font-size: 1rem;
        }

        .library-empty-hint {
            font-size: 0.85rem;
            margin-top: 0.5rem;
        }

        /* Output select */
        .output-select { margin: 1rem 0 1.5rem; }

        /* Spinner */
        .spinner {
            width: 16px;
            height: 16px;
            border: 2px solid var(--border);
            border-top-color: var(--accent);
            border-radius: 50%;
            animation: spin 0.7s linear infinite;
        }

        @keyframes spin { to { transform: rotate(360deg); } }

        /* Scrollbar */
        ::-webkit-scrollbar { width: 8px; height: 8px; }
        ::-webkit-scrollbar-track { background: var(--bg-input); }
        ::-webkit-scrollbar-thumb { background: var(--border-dark); border-radius: 4px; }
        ::-webkit-scrollbar-thumb:hover { background: var(--text-muted); }

        /* Animations */
        .fade-in { animation: fadeIn 0.2s ease-out; }
        @keyframes fadeIn { from { opacity: 0; transform: translateY(4px); } }
    </style>
</head>
<body>
    <div class="app">
        <header>
            <h1>The Visualizer</h1>
            <p class="tagline">Document Intelligence & Visual Analysis</p>
        </header>

        <!-- Navigation -->
        <nav class="nav-bar">
            <button class="nav-btn active" onclick="switchView('analyze', event)">Analyze</button>
            <button class="nav-btn" onclick="switchView('library', event)">Library</button>
        </nav>

        <!-- Analyze View -->
        <div id="analyze-view" class="view-content active">
            <div class="main-layout">
                <!-- Left Panel: Document Selection -->
                <div class="left-panel">
                    <div class="card">
                        <h2 class="card-header">Documents</h2>

                        <div id="api-status" class="api-status disconnected">
                            <span class="status-dot"></span>
                            <span id="api-status-text">Checking connection...</span>
                        </div>

                        <!-- Upload Zone -->
                        <div class="upload-zone" id="upload-zone" onclick="triggerFileUpload()">
                            <div class="upload-zone-icon">&#128194;</div>
                            <div class="upload-zone-text">Drop files here or click to browse</div>
                            <div class="upload-zone-hint">Supports PDF, TXT, MD files</div>
                        </div>
                        <input type="file" id="file-upload" class="hidden-input" multiple accept=".pdf,.txt,.md,.json,.xml" onchange="handleFileUpload(event)">
                        <input type="file" id="folder-upload" class="hidden-input" webkitdirectory directory onchange="handleFolderUpload(event)">

                        <!-- Or use path -->
                        <span class="section-label" style="margin-top: 1rem;">Or enter path</span>
                        <div class="path-input-row">
                            <input type="text" id="folder-path" placeholder="~/Documents/articles">
                            <button class="btn btn-sm" onclick="scanFolder()">Scan</button>
                        </div>

                        <!-- Document List -->
                        <div id="doc-list-container" style="display:none; margin-top: 1.25rem;">
                            <div class="doc-actions">
                                <button class="btn btn-sm btn-ghost" onclick="selectAllDocs()">Select All</button>
                                <button class="btn btn-sm btn-ghost" onclick="deselectAllDocs()">Deselect All</button>
                            </div>
                            <div id="doc-list" class="doc-list"></div>
                            <p id="doc-count" class="doc-count"></p>
                        </div>

                        <!-- Analysis Mode -->
                        <span class="section-label" style="margin-top: 1.5rem;">Analysis Mode</span>
                        <div class="mode-toggle">
                            <div class="mode-btn active" onclick="setCollectionMode('single')" id="mode-single">
                                <div class="title">Single Collection</div>
                                <div class="desc">Analyze together</div>
                            </div>
                            <div class="mode-btn" onclick="setCollectionMode('individual')" id="mode-individual">
                                <div class="title">Individual Files</div>
                                <div class="desc">Analyze separately</div>
                            </div>
                        </div>
                    </div>
                </div>

                <!-- Right Panel: Engine Selection & Results -->
                <div class="right-panel">
                    <div class="card">
                        <h2 class="card-header">Analysis Engine</h2>

                        <!-- Engine vs Bundle Toggle -->
                        <div class="mode-toggle">
                            <div class="mode-btn active" onclick="setEngineMode('engine')" id="engine-mode-single">
                                <div class="title">Single Engine</div>
                                <div class="desc">One analytical lens</div>
                            </div>
                            <div class="mode-btn" onclick="setEngineMode('bundle')" id="engine-mode-bundle">
                                <div class="title">Bundle</div>
                                <div class="desc">Multiple engines</div>
                            </div>
                        </div>

                        <!-- Single Engine Selection -->
                        <div id="engine-selection">
                            <span class="section-label">Select Engine <span id="engine-count" style="color:var(--text-secondary);"></span></span>
                            <div id="engine-grid" class="engine-grid"></div>
                        </div>

                        <!-- Bundle Selection -->
                        <div id="bundle-selection" style="display:none;">
                            <span class="section-label">Select Bundle</span>
                            <div id="bundle-list"></div>
                        </div>

                        <!-- Output Mode -->
                        <div class="output-select">
                            <span class="section-label">Output Format</span>
                            <select id="output-mode">
                                <option value="structured_text_report">Text Report</option>
                                <option value="executive_memo">Executive Memo</option>
                                <option value="research_report">Research Report</option>
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

                    <!-- Progress Section -->
                    <div id="progress-section" class="progress-section">
                        <div class="progress-header">Processing</div>
                        <div class="progress-bar-outer">
                            <div id="progress-bar" class="progress-bar-inner"></div>
                        </div>
                        <div id="progress-text" class="progress-text"></div>
                        <div class="progress-stages">
                            <span class="stage-badge" id="stage-extraction">Extraction</span>
                            <span class="stage-badge" id="stage-curation">Curation</span>
                            <span class="stage-badge" id="stage-concretization">Concretization</span>
                            <span class="stage-badge" id="stage-rendering">Rendering</span>
                        </div>
                    </div>

                    <!-- Results Gallery -->
                    <div id="results-gallery" class="results-gallery" style="display:none;">
                        <div class="results-gallery-header">
                            <h3>Analysis Results</h3>
                            <span id="results-count" class="results-count"></span>
                        </div>
                        <div id="results-grid" class="results-grid"></div>
                    </div>
                </div>
            </div>
        </div>

        <!-- Library View -->
        <div id="library-view" class="view-content">
            <div class="library-empty" id="library-empty">
                <div class="library-empty-icon">&#128218;</div>
                <div class="library-empty-text">Your library is empty</div>
                <div class="library-empty-hint">Analyzed documents and generated visualizations will appear here</div>
            </div>
            <div id="library-grid" class="library-grid"></div>
        </div>
    </div>

    <script>
        // State
        let scannedDocs = [];
        let selectedDocs = new Set();
        let engines = [];
        let bundles = [];
        let selectedEngine = null;
        let selectedBundle = null;
        let collectionMode = 'single';
        let engineMode = 'engine';
        let currentJobId = null;
        let allResults = [];
        let libraryItems = [];

        function $(id) { return document.getElementById(id); }

        // Initialize
        document.addEventListener('DOMContentLoaded', function() {
            loadAnalyzerData();
            loadLibrary();
            setupDragDrop();
        });

        // View switching
        function switchView(viewId, evt) {
            document.querySelectorAll('.view-content').forEach(function(el) { el.classList.remove('active'); });
            document.querySelectorAll('.nav-btn').forEach(function(el) { el.classList.remove('active'); });
            document.getElementById(viewId + '-view').classList.add('active');
            if (evt && evt.target) evt.target.classList.add('active');
        }

        // File Upload
        function triggerFileUpload() {
            $('file-upload').click();
        }

        function handleFileUpload(event) {
            var files = Array.from(event.target.files);
            if (files.length === 0) return;

            scannedDocs = files.map(function(f) {
                return {
                    name: f.name,
                    path: f.name,
                    type: f.name.split('.').pop().toLowerCase(),
                    size: formatSize(f.size),
                    file: f
                };
            });
            selectedDocs = new Set(scannedDocs.map(function(d) { return d.path; }));
            renderDocList();
            $('doc-list-container').style.display = 'block';
        }

        function handleFolderUpload(event) {
            var files = Array.from(event.target.files).filter(function(f) {
                var ext = f.name.split('.').pop().toLowerCase();
                return ['pdf', 'txt', 'md', 'json', 'xml'].includes(ext);
            });
            if (files.length === 0) return;

            scannedDocs = files.map(function(f) {
                return {
                    name: f.name,
                    path: f.webkitRelativePath || f.name,
                    type: f.name.split('.').pop().toLowerCase(),
                    size: formatSize(f.size),
                    file: f
                };
            });
            selectedDocs = new Set(scannedDocs.map(function(d) { return d.path; }));
            renderDocList();
            $('doc-list-container').style.display = 'block';
        }

        function formatSize(bytes) {
            if (bytes < 1024) return bytes + ' B';
            if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
            return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
        }

        // Drag and drop
        function setupDragDrop() {
            var zone = $('upload-zone');

            zone.addEventListener('dragover', function(e) {
                e.preventDefault();
                zone.classList.add('dragover');
            });

            zone.addEventListener('dragleave', function() {
                zone.classList.remove('dragover');
            });

            zone.addEventListener('drop', function(e) {
                e.preventDefault();
                zone.classList.remove('dragover');
                var files = Array.from(e.dataTransfer.files);
                if (files.length > 0) {
                    $('file-upload').files = e.dataTransfer.files;
                    handleFileUpload({ target: { files: files } });
                }
            });
        }

        // Check API Status
        async function checkAnalyzerStatus() {
            const statusEl = $('api-status');
            const textEl = $('api-status-text');

            try {
                const res = await fetch('/api/analyzer/engines');
                if (res.ok) {
                    statusEl.className = 'api-status connected';
                    textEl.textContent = 'API Connected';
                    return true;
                }
            } catch (e) {}

            statusEl.className = 'api-status disconnected';
            textEl.textContent = 'API Not Available';
            return false;
        }

        // Load Engines and Bundles
        async function loadAnalyzerData() {
            const connected = await checkAnalyzerStatus();
            if (!connected) return;

            try {
                const enginesRes = await fetch('/api/analyzer/engines');
                if (enginesRes.ok) {
                    engines = await enginesRes.json();
                    renderEngines();
                }

                const bundlesRes = await fetch('/api/analyzer/bundles');
                if (bundlesRes.ok) {
                    bundles = await bundlesRes.json();
                    renderBundles();
                }
            } catch (e) {
                console.error('Failed to load analyzer data:', e);
            }
        }

        // Format engine name - convert snake_case to Title Case
        function formatEngineName(key) {
            return key.split('_').map(function(word) {
                return word.charAt(0).toUpperCase() + word.slice(1);
            }).join(' ');
        }

        // Truncate description to ~80 chars
        function truncateDesc(desc, maxLen) {
            if (!desc) return '';
            if (desc.length <= maxLen) return desc;
            return desc.substring(0, maxLen).replace(/\\s+\\S*$/, '') + '...';
        }

        // Render Engines
        function renderEngines() {
            var grid = $('engine-grid');
            $('engine-count').textContent = '(' + engines.length + ' available)';

            grid.innerHTML = engines.map(function(e) {
                var displayName = e.name || formatEngineName(e.engine_key);
                var shortDesc = truncateDesc(e.description || '', 100);
                return '<div class="engine-card ' + (selectedEngine === e.engine_key ? 'selected' : '') + '" ' +
                'onclick="selectEngine(\\'' + e.engine_key + '\\')">' +
                '<div class="name">' + displayName + '</div>' +
                '<div class="desc">' + shortDesc + '</div>' +
                '</div>';
            }).join('');
        }

        // Render Bundles
        function renderBundles() {
            var list = $('bundle-list');

            list.innerHTML = bundles.map(function(b) {
                var displayName = b.name || formatEngineName(b.bundle_key);
                var engineList = (b.member_engines || []).map(formatEngineName).slice(0, 3).join(', ');
                var moreCount = (b.member_engines || []).length > 3 ? ' +' + ((b.member_engines || []).length - 3) + ' more' : '';
                return '<div class="bundle-card ' + (selectedBundle === b.bundle_key ? 'selected' : '') + '" ' +
                'onclick="selectBundle(\\'' + b.bundle_key + '\\')">' +
                '<div class="name">' + displayName + '</div>' +
                '<div class="engines">' + (b.member_engines || []).length + ' engines: ' + engineList + moreCount + '</div>' +
                '</div>';
            }).join('');
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
                    selectedDocs = new Set(scannedDocs.map(function(d) { return d.path; }));
                    renderDocList();
                    $('doc-list-container').style.display = 'block';
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
            const icons = { pdf: '&#128196;', md: '&#128221;', txt: '&#128195;', json: '&#128196;', xml: '&#128196;' };

            list.innerHTML = scannedDocs.map(function(d) {
                var escapedPath = d.path.replace(/\\\\/g, '\\\\\\\\').replace(/'/g, "\\\\'");
                return '<div class="doc-item">' +
                '<input type="checkbox" ' + (selectedDocs.has(d.path) ? 'checked' : '') + ' onchange="toggleDoc(\\'' + escapedPath + '\\')">' +
                '<span class="icon">' + (icons[d.type] || '&#128196;') + '</span>' +
                '<div class="info">' +
                '<div class="name">' + d.name + '</div>' +
                '<div class="meta">' + d.type.toUpperCase() + ' - ' + d.size + '</div>' +
                '</div>' +
                '</div>';
            }).join('');

            $('doc-count').textContent = selectedDocs.size + ' of ' + scannedDocs.length + ' documents selected';
            updateAnalyzeButton();
        }

        // Toggle Document
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
            selectedDocs = new Set(scannedDocs.map(function(d) { return d.path; }));
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
                btn.textContent = 'Select ' + (engineMode === 'engine' ? 'Engine' : 'Bundle') + ' to Analyze';
            } else {
                btn.textContent = 'Analyze ' + selectedDocs.size + ' Document' + (selectedDocs.size > 1 ? 's' : '');
            }
        }

        // Read file as text or base64 for PDFs
        async function readFileContent(file) {
            return new Promise((resolve, reject) => {
                const reader = new FileReader();
                const isPdf = file.name.toLowerCase().endsWith('.pdf');

                reader.onload = function(e) {
                    if (isPdf) {
                        // For PDFs, send as base64
                        const base64 = e.target.result.split(',')[1];
                        resolve({ content: base64, encoding: 'base64' });
                    } else {
                        resolve({ content: e.target.result, encoding: 'text' });
                    }
                };
                reader.onerror = function() { reject(new Error('Failed to read file')); };

                if (isPdf) {
                    reader.readAsDataURL(file);
                } else {
                    reader.readAsText(file);
                }
            });
        }

        // Get documents for analysis - either paths or inline content
        async function getDocumentsForAnalysis() {
            const selectedPaths = Array.from(selectedDocs);
            const selectedDocObjects = scannedDocs.filter(d => selectedPaths.includes(d.path));

            // Check if these are browser-uploaded files (have file object) or server-scanned (have full paths)
            const hasBrowserFiles = selectedDocObjects.some(d => d.file && d.file instanceof File);

            if (!hasBrowserFiles) {
                // Server-scanned files - just return paths
                return { type: 'paths', file_paths: selectedPaths };
            }

            // Browser-uploaded files - read contents
            $('analyze-btn').textContent = 'Reading files...';
            const documents = [];

            for (let i = 0; i < selectedDocObjects.length; i++) {
                const doc = selectedDocObjects[i];
                if (doc.file && doc.file instanceof File) {
                    try {
                        const { content, encoding } = await readFileContent(doc.file);
                        documents.push({
                            id: 'doc_' + (i + 1),
                            title: doc.name,  // Keep full filename with extension for citations
                            content: content,
                            encoding: encoding
                        });
                    } catch (e) {
                        console.error('Failed to read file:', doc.name, e);
                    }
                }
            }

            return { type: 'inline', documents: documents };
        }

        // Run Analysis
        async function runAnalysis() {
            const outputMode = $('output-mode').value;

            $('analyze-btn').disabled = true;
            $('analyze-btn').textContent = 'Preparing...';
            $('progress-section').classList.add('show');
            $('results-grid').innerHTML = '';
            $('results-gallery').style.display = 'none';
            allResults = [];

            resetStages();

            try {
                // Get documents (either paths or inline content)
                const docData = await getDocumentsForAnalysis();

                // Track document count for progress display
                currentDocCount = docData.type === 'paths'
                    ? docData.file_paths.length
                    : docData.documents.length;

                $('analyze-btn').textContent = 'Submitting...';

                let response;
                if (engineMode === 'engine') {
                    const payload = {
                        engine: selectedEngine,
                        output_mode: outputMode,
                        collection_mode: collectionMode
                    };

                    if (docData.type === 'paths') {
                        payload.file_paths = docData.file_paths;
                    } else {
                        payload.documents = docData.documents;
                    }

                    response = await fetch('/api/analyzer/analyze', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify(payload)
                    });
                } else {
                    var outputModes = {};
                    var bundle = bundles.find(function(b) { return b.bundle_key === selectedBundle; });
                    if (bundle) {
                        bundle.member_engines.forEach(function(e) { outputModes[e] = outputMode; });
                    }

                    const payload = {
                        bundle: selectedBundle,
                        output_modes: outputModes
                    };

                    if (docData.type === 'paths') {
                        payload.file_paths = docData.file_paths;
                    } else {
                        payload.documents = docData.documents;
                    }

                    response = await fetch('/api/analyzer/analyze/bundle', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify(payload)
                    });
                }

                const data = await response.json();
                console.log('Analysis response:', data);

                if (data.success) {
                    if (data.mode === 'individual') {
                        pollMultipleJobs(data.jobs);
                    } else {
                        currentJobId = data.job_id;
                        pollJobStatus(data.job_id);
                    }
                } else {
                    console.log('Analysis failed:', data.error, data);
                    var errorMsg = data.error || 'Failed to submit analysis';
                    if (data.detail) {
                        console.log('Error detail:', data.detail);
                        errorMsg += ' - ' + data.detail.substring(0, 200);
                    }
                    showAnalysisError(errorMsg);
                }
            } catch (e) {
                showAnalysisError('Error: ' + e.message);
            }
        }

        // Track document count for progress display
        let currentDocCount = 0;

        // Poll Job Status
        async function pollJobStatus(jobId) {
            try {
                const res = await fetch('/api/analyzer/jobs/' + jobId);
                const job = await res.json();

                // Add doc count to job for display
                job.doc_count = currentDocCount;
                updateProgress(job);

                if (job.status === 'completed') {
                    await fetchAndDisplayResult(jobId);
                } else if (job.status === 'failed') {
                    showAnalysisError(job.error_message || 'Analysis failed');
                } else {
                    setTimeout(function() { pollJobStatus(jobId); }, 2000);
                }
            } catch (e) {
                showAnalysisError('Error polling status: ' + e.message);
            }
        }

        // Poll Multiple Jobs
        async function pollMultipleJobs(jobs) {
            const pending = jobs.filter(function(j) { return j.status === 'submitted'; });
            let allDone = true;

            for (const job of pending) {
                if (!job.job_id) continue;

                try {
                    const res = await fetch('/api/analyzer/jobs/' + job.job_id);
                    const status = await res.json();

                    if (status.status === 'completed') {
                        job.status = 'completed';
                        const resultRes = await fetch('/api/analyzer/jobs/' + job.job_id + '/result');
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

            const completed = jobs.filter(function(j) { return j.status === 'completed' || j.status === 'failed'; }).length;
            updateProgressMulti(completed, jobs.length);

            if (!allDone) {
                setTimeout(function() { pollMultipleJobs(jobs); }, 2000);
            } else {
                finishAnalysis();
            }
        }

        // Update Progress
        function updateProgress(job) {
            const percent = job.progress_percent || 0;
            $('progress-bar').style.width = percent + '%';

            // Build status text
            let statusText = '';
            const stage = job.current_stage ? job.current_stage.toLowerCase() : '';
            const status = job.status || 'pending';
            const docCount = currentDocCount > 0 ? currentDocCount : '';
            const docSuffix = docCount ? ' (' + docCount + ' doc' + (docCount > 1 ? 's' : '') + ')' : '';

            if (status === 'pending' || status === 'queued') {
                statusText = 'Queued...' + docSuffix;
            } else if (stage === 'extraction' || status === 'extracting') {
                statusText = 'Extracting from ' + (docCount || '') + ' document' + (docCount > 1 ? 's' : '') + '... (' + percent + '%)';
            } else if (stage === 'curation' || status === 'curating') {
                statusText = 'Curating analysis... (' + percent + '%)';
            } else if (stage === 'concretization' || status === 'concretizing') {
                statusText = 'Refining labels... (' + percent + '%)';
            } else if (stage === 'rendering' || status === 'rendering') {
                statusText = 'Generating output... (' + percent + '%)';
            } else if (status === 'completed') {
                statusText = 'Complete!' + docSuffix;
            } else if (status === 'failed') {
                statusText = 'Failed';
            } else if (stage || status) {
                statusText = (stage || status) + ' (' + percent + '%)';
            } else {
                statusText = 'Processing...' + docSuffix + ' (' + percent + '%)';
            }

            $('progress-text').textContent = statusText;

            // Update stage badges
            const stages = ['extraction', 'curation', 'concretization', 'rendering'];
            const stagesCompleted = job.stages_completed || [];
            const currentStage = stage || '';

            stages.forEach(function(stageName, i) {
                const el = $('stage-' + stageName);
                if (!el) return;

                if (stagesCompleted.includes(stageName)) {
                    el.className = 'stage-badge completed';
                } else if (currentStage === stageName) {
                    el.className = 'stage-badge active';
                } else {
                    el.className = 'stage-badge';
                }
            });
        }

        function updateProgressMulti(completed, total) {
            const percent = Math.round((completed / total) * 100);
            $('progress-bar').style.width = percent + '%';
            $('progress-text').textContent = completed + ' of ' + total + ' documents processed';
        }

        function resetStages() {
            ['extraction', 'curation', 'concretization', 'rendering'].forEach(function(s) {
                const el = $('stage-' + s);
                if (el) el.className = 'stage-badge';
            });
        }

        // Fetch and Display Result
        async function fetchAndDisplayResult(jobId) {
            try {
                const res = await fetch('/api/analyzer/jobs/' + jobId + '/result');
                const result = await res.json();
                displayResult(result);
                finishAnalysis();
            } catch (e) {
                showAnalysisError('Error fetching result: ' + e.message);
            }
        }

        // Display Result
        function displayResult(result, title) {
            var gallery = $('results-gallery');
            var grid = $('results-grid');
            var countEl = $('results-count');

            gallery.style.display = 'block';

            var outputs = result.outputs || {};
            var metadata = result.metadata || {};
            var count = 0;

            for (var key in outputs) {
                var output = outputs[key];
                count++;

                var resultData = {
                    key: key,
                    title: (title ? title + ' - ' : '') + key.replace(/_/g, ' '),
                    output: output,
                    metadata: metadata,
                    isImage: !!output.image_url,
                    imageUrl: output.image_url || null,
                    content: output.content || '',
                    data: output.data || null
                };
                allResults.push(resultData);
                addToLibrary(resultData);

                var card = createGalleryCard(resultData, allResults.length - 1);
                grid.appendChild(card);
            }

            if (count === 0 && result.canonical_data) {
                var canonicalData = {
                    key: 'canonical_data',
                    title: 'Canonical Data',
                    output: { data: result.canonical_data },
                    metadata: metadata,
                    isImage: false,
                    imageUrl: null,
                    content: '',
                    data: result.canonical_data
                };
                allResults.push(canonicalData);
                addToLibrary(canonicalData);
                var card = createGalleryCard(canonicalData, allResults.length - 1);
                grid.appendChild(card);
                count++;
            }

            countEl.textContent = count + ' result' + (count !== 1 ? 's' : '');
        }

        function createGalleryCard(data, index) {
            var card = document.createElement('div');
            card.className = 'gallery-card fade-in';

            var preview = document.createElement('div');
            preview.className = 'gallery-card-preview';

            if (data.isImage && data.imageUrl) {
                var img = document.createElement('img');
                if (data.imageUrl.startsWith('/static/')) {
                    img.src = 'http://localhost:8847' + data.imageUrl;
                } else {
                    img.src = data.imageUrl;
                }
                img.alt = data.title;
                img.onerror = function() {
                    this.style.display = 'none';
                    var icon = document.createElement('div');
                    icon.className = 'icon-preview';
                    icon.innerHTML = '&#128444;';
                    preview.appendChild(icon);
                };
                preview.appendChild(img);
            } else if (data.content) {
                var textPre = document.createElement('div');
                textPre.className = 'text-preview';
                textPre.textContent = data.content.substring(0, 500) + (data.content.length > 500 ? '...' : '');
                preview.appendChild(textPre);
            } else if (data.data) {
                var jsonPre = document.createElement('div');
                jsonPre.className = 'text-preview';
                var jsonStr = JSON.stringify(data.data, null, 2);
                jsonPre.textContent = jsonStr.substring(0, 500) + (jsonStr.length > 500 ? '...' : '');
                preview.appendChild(jsonPre);
            } else {
                var icon = document.createElement('div');
                icon.className = 'icon-preview';
                icon.innerHTML = data.isImage ? '&#128444;' : '&#128196;';
                preview.appendChild(icon);
            }

            var info = document.createElement('div');
            info.className = 'gallery-card-info';

            var titleEl = document.createElement('div');
            titleEl.className = 'gallery-card-title';
            titleEl.textContent = data.title;
            info.appendChild(titleEl);

            var meta = document.createElement('div');
            meta.className = 'gallery-card-meta';
            var type = data.isImage ? 'Image' : 'Text';
            var renderer = data.output.renderer_type || data.output.mode || '';
            meta.innerHTML = '<span>' + type + '</span><span>' + renderer + '</span>';
            if (data.metadata.cost_usd) {
                meta.innerHTML += '<span>$' + data.metadata.cost_usd.toFixed(3) + '</span>';
            }
            info.appendChild(meta);

            var actions = document.createElement('div');
            actions.className = 'gallery-card-actions';

            var viewBtn = document.createElement('button');
            viewBtn.textContent = 'View';
            viewBtn.onclick = function(e) {
                e.stopPropagation();
                openResultModal(index);
            };

            var dlBtn = document.createElement('button');
            dlBtn.textContent = 'Download';
            dlBtn.onclick = function(e) {
                e.stopPropagation();
                downloadGalleryResult(index);
            };

            actions.appendChild(viewBtn);
            actions.appendChild(dlBtn);

            card.appendChild(preview);
            card.appendChild(info);
            card.appendChild(actions);

            card.onclick = function() { openResultModal(index); };

            return card;
        }

        function openResultModal(index) {
            var data = allResults[index];
            if (!data) return;

            var modal = document.createElement('div');
            modal.className = 'result-modal';
            modal.onclick = function(e) {
                if (e.target === modal) closeResultModal();
            };

            var content = document.createElement('div');
            content.className = 'result-modal-content';

            var header = document.createElement('div');
            header.className = 'result-modal-header';
            var h3 = document.createElement('h3');
            h3.textContent = data.title;
            var closeBtn = document.createElement('button');
            closeBtn.className = 'result-modal-close';
            closeBtn.innerHTML = '&times;';
            closeBtn.onclick = closeResultModal;
            header.appendChild(h3);
            header.appendChild(closeBtn);

            var body = document.createElement('div');
            body.className = 'result-modal-body';

            if (data.isImage && data.imageUrl) {
                var img = document.createElement('img');
                if (data.imageUrl.startsWith('/static/')) {
                    img.src = 'http://localhost:8847' + data.imageUrl;
                } else {
                    img.src = data.imageUrl;
                }
                img.alt = data.title;
                body.appendChild(img);
            } else if (data.content) {
                var pre = document.createElement('pre');
                pre.textContent = data.content;
                body.appendChild(pre);
            } else if (data.data) {
                var pre = document.createElement('pre');
                pre.textContent = JSON.stringify(data.data, null, 2);
                body.appendChild(pre);
            }

            var actions = document.createElement('div');
            actions.className = 'result-modal-actions';

            var dlBtn = document.createElement('button');
            dlBtn.className = 'btn btn-primary';
            dlBtn.style.width = 'auto';
            dlBtn.textContent = 'Download';
            dlBtn.onclick = function() { downloadGalleryResult(index); };
            actions.appendChild(dlBtn);

            content.appendChild(header);
            content.appendChild(body);
            content.appendChild(actions);
            modal.appendChild(content);

            document.body.appendChild(modal);
            document.addEventListener('keydown', handleModalEscape);
        }

        function handleModalEscape(e) {
            if (e.key === 'Escape') closeResultModal();
        }

        function closeResultModal() {
            var modal = document.querySelector('.result-modal');
            if (modal) modal.remove();
            document.removeEventListener('keydown', handleModalEscape);
        }

        function downloadGalleryResult(index) {
            var data = allResults[index];
            if (!data) return;

            if (data.isImage && data.imageUrl) {
                var url = data.imageUrl;
                if (url.startsWith('/static/')) {
                    url = 'http://localhost:8847' + url;
                }
                window.open(url, '_blank');
                return;
            }

            var content, filename, mimeType;
            if (data.content) {
                content = data.content;
                filename = data.key.replace(/_/g, '-') + '.md';
                mimeType = 'text/markdown';
            } else if (data.data) {
                content = JSON.stringify(data.data, null, 2);
                filename = data.key.replace(/_/g, '-') + '.json';
                mimeType = 'application/json';
            } else {
                return;
            }

            var blob = new Blob([content], { type: mimeType });
            var url = URL.createObjectURL(blob);
            var a = document.createElement('a');
            a.href = url;
            a.download = filename;
            a.click();
            URL.revokeObjectURL(url);
        }

        function displayError(title, message) {
            var gallery = $('results-gallery');
            var grid = $('results-grid');

            gallery.style.display = 'block';
            $('results-count').textContent = 'Error';

            var card = document.createElement('div');
            card.className = 'gallery-card';
            card.style.borderColor = 'var(--error)';

            var preview = document.createElement('div');
            preview.className = 'gallery-card-preview';
            preview.style.background = 'rgba(179,58,58,0.1)';
            var icon = document.createElement('div');
            icon.className = 'icon-preview';
            icon.innerHTML = '&#10060;';
            preview.appendChild(icon);

            var info = document.createElement('div');
            info.className = 'gallery-card-info';
            var titleEl = document.createElement('div');
            titleEl.className = 'gallery-card-title';
            titleEl.style.color = 'var(--error)';
            titleEl.textContent = title;
            info.appendChild(titleEl);

            var msgEl = document.createElement('div');
            msgEl.className = 'gallery-card-meta';
            msgEl.textContent = message || 'Analysis failed';
            info.appendChild(msgEl);

            card.appendChild(preview);
            card.appendChild(info);
            grid.appendChild(card);
        }

        function showAnalysisError(message) {
            $('progress-text').textContent = message;
            $('progress-text').style.color = 'var(--error)';
            finishAnalysis();
        }

        function finishAnalysis() {
            $('analyze-btn').disabled = false;
            updateAnalyzeButton();

            // Hide progress section after completion
            setTimeout(function() {
                $('progress-section').classList.remove('show');
            }, 1500);

            // Mark all stages complete
            ['extraction', 'curation', 'concretization', 'rendering'].forEach(function(s) {
                var el = $('stage-' + s);
                if (el && !$('progress-text').style.color.includes('error')) {
                    el.className = 'stage-badge completed';
                }
            });
        }

        // Library
        function loadLibrary() {
            var saved = localStorage.getItem('visualizer_library');
            if (saved) {
                try {
                    libraryItems = JSON.parse(saved);
                    renderLibrary();
                } catch (e) {
                    libraryItems = [];
                }
            }
        }

        function addToLibrary(item) {
            item.addedAt = new Date().toISOString();
            libraryItems.unshift(item);
            if (libraryItems.length > 100) libraryItems = libraryItems.slice(0, 100);
            localStorage.setItem('visualizer_library', JSON.stringify(libraryItems));
            renderLibrary();
        }

        function renderLibrary() {
            var grid = $('library-grid');
            var empty = $('library-empty');

            if (libraryItems.length === 0) {
                empty.style.display = 'block';
                grid.innerHTML = '';
                return;
            }

            empty.style.display = 'none';
            grid.innerHTML = '';

            libraryItems.forEach(function(item, index) {
                var card = createLibraryCard(item, index);
                grid.appendChild(card);
            });
        }

        function createLibraryCard(data, index) {
            var card = document.createElement('div');
            card.className = 'gallery-card';

            var preview = document.createElement('div');
            preview.className = 'gallery-card-preview';

            if (data.isImage && data.imageUrl) {
                var img = document.createElement('img');
                if (data.imageUrl.startsWith('/static/')) {
                    img.src = 'http://localhost:8847' + data.imageUrl;
                } else {
                    img.src = data.imageUrl;
                }
                img.alt = data.title;
                preview.appendChild(img);
            } else if (data.content) {
                var textPre = document.createElement('div');
                textPre.className = 'text-preview';
                textPre.textContent = data.content.substring(0, 300) + '...';
                preview.appendChild(textPre);
            } else {
                var icon = document.createElement('div');
                icon.className = 'icon-preview';
                icon.innerHTML = '&#128196;';
                preview.appendChild(icon);
            }

            var info = document.createElement('div');
            info.className = 'gallery-card-info';

            var titleEl = document.createElement('div');
            titleEl.className = 'gallery-card-title';
            titleEl.textContent = data.title;
            info.appendChild(titleEl);

            var meta = document.createElement('div');
            meta.className = 'gallery-card-meta';
            meta.innerHTML = '<span>' + (data.isImage ? 'Image' : 'Text') + '</span>';
            if (data.addedAt) {
                meta.innerHTML += '<span>' + new Date(data.addedAt).toLocaleDateString() + '</span>';
            }
            info.appendChild(meta);

            card.appendChild(preview);
            card.appendChild(info);

            card.onclick = function() {
                allResults = [data];
                openResultModal(0);
            };

            return card;
        }
    </script>
</body>
</html>
'''


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
