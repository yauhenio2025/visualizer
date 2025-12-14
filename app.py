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

# Configuration - API keys are optional (users can provide their own via UI)
API_KEY = os.environ.get("GEMINI_API_KEY", "")
if not API_KEY:
    print("Note: GEMINI_API_KEY not set. Users must provide their own via the KEYS button.")

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

# Web-Saver Export API Configuration
WEBSAVER_API_URL = os.environ.get("WEBSAVER_API_URL", "http://localhost:5001/api/export")
WEBSAVER_API_KEY = os.environ.get("WEBSAVER_API_KEY", "ws-export-key-12345")

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


# =============================================================================
# Web-Saver Integration Endpoints
# =============================================================================

def get_websaver_headers() -> Dict[str, str]:
    """Build headers for Web-Saver Export API requests."""
    return {"X-API-Key": WEBSAVER_API_KEY}


@app.route('/api/websaver/health', methods=['GET'])
def websaver_health():
    """Check if Web-Saver is reachable."""
    try:
        response = httpx.get(
            f"{WEBSAVER_API_URL}/health",
            timeout=5.0,
        )
        if response.status_code == 200:
            data = response.json()
            return jsonify({
                "success": True,
                "available": True,
                "url": WEBSAVER_API_URL,
                "article_count": data.get("article_count", 0),
                "collection_count": data.get("collection_count", 0),
            })
        else:
            return jsonify({"success": True, "available": False, "url": WEBSAVER_API_URL})
    except Exception as e:
        return jsonify({"success": True, "available": False, "url": WEBSAVER_API_URL, "error": str(e)})


@app.route('/api/websaver/collections', methods=['GET'])
def websaver_list_collections():
    """List collections from Web-Saver."""
    try:
        min_articles = request.args.get('min_articles', 1, type=int)
        search = request.args.get('search', '')

        params = {"min_articles": min_articles}
        if search:
            params["search"] = search

        response = httpx.get(
            f"{WEBSAVER_API_URL}/collections",
            headers=get_websaver_headers(),
            params=params,
            timeout=30.0,
        )
        response.raise_for_status()
        data = response.json()

        return jsonify({
            "success": True,
            "collections": data.get("collections", []),
            "total": data.get("total", 0),
        })
    except httpx.HTTPStatusError as e:
        return jsonify({"success": False, "error": f"Web-Saver API error: {e.response.status_code}"})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})


@app.route('/api/websaver/collections/<int:collection_id>', methods=['GET'])
def websaver_get_collection(collection_id: int):
    """
    Fetch collection with articles from Web-Saver.

    Returns documents in format ready for Analyzer.
    """
    try:
        response = httpx.get(
            f"{WEBSAVER_API_URL}/collections/{collection_id}",
            headers=get_websaver_headers(),
            timeout=60.0,
        )
        response.raise_for_status()
        data = response.json()

        # Transform to format compatible with scan-folder response
        files = []
        documents = []

        for article in data.get("articles", []):
            # Create document entry
            documents.append({
                "id": article.get("id"),
                "title": article.get("title", "Untitled"),
                "content": article.get("content", ""),
                "source_name": article.get("source_name"),
                "url": article.get("url"),
                "date_published": article.get("date_published"),
                "word_count": article.get("word_count", 0),
            })

            # Create file-like entry for UI display
            word_count = article.get("word_count", 0)
            files.append({
                "name": article.get("title", "Untitled")[:60] + ("..." if len(article.get("title", "")) > 60 else ""),
                "path": f"websaver://article/{article.get('id')}",
                "type": "web-saver",
                "size": f"{word_count:,} words",
                "size_bytes": word_count * 5,  # Approximate bytes
                "source": article.get("source_name", "Unknown"),
                "article_id": article.get("id"),
            })

        return jsonify({
            "success": True,
            "collection": {
                "id": data.get("id"),
                "name": data.get("name"),
                "description": data.get("description"),
            },
            "files": files,
            "documents": documents,  # Ready for Analyzer API
            "count": len(files),
            "mode": "web-saver",
        })
    except httpx.HTTPStatusError as e:
        return jsonify({"success": False, "error": f"Web-Saver API error: {e.response.status_code}"})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})


def get_analyzer_headers(llm_keys: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    """
    Build headers for analyzer API requests.
    Includes the API key and any forwarded LLM keys from the frontend.

    Args:
        llm_keys: Optional dict with 'anthropic_api_key' and/or 'gemini_api_key' from request body
    """
    headers = {"X-API-Key": ANALYZER_API_KEY}

    # First check llm_keys from request body (preferred - from JSON payload)
    if llm_keys:
        if llm_keys.get('anthropic_api_key'):
            headers['X-Anthropic-Api-Key'] = llm_keys['anthropic_api_key']
        if llm_keys.get('gemini_api_key'):
            headers['X-Gemini-Api-Key'] = llm_keys['gemini_api_key']
    else:
        # Fallback: check request headers (legacy support)
        anthropic_key = request.headers.get('X-Anthropic-Api-Key')
        gemini_key = request.headers.get('X-Gemini-Api-Key')

        if anthropic_key:
            headers['X-Anthropic-Api-Key'] = anthropic_key
        if gemini_key:
            headers['X-Gemini-Api-Key'] = gemini_key

    return headers


@app.route('/api/analyzer/engines', methods=['GET'])
def list_analyzer_engines():
    """Fetch available engines from Analyzer API."""
    try:
        response = httpx.get(
            f"{ANALYZER_API_URL}/v1/engines",
            headers=get_analyzer_headers(),
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
            headers=get_analyzer_headers(),
            timeout=30.0,
        )
        response.raise_for_status()
        return jsonify(response.json())
    except httpx.HTTPError as e:
        return jsonify({"error": f"Failed to fetch bundles: {str(e)}"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/analyzer/pipelines', methods=['GET'])
def list_analyzer_pipelines():
    """Fetch available meta-engine pipelines from Analyzer API."""
    try:
        response = httpx.get(
            f"{ANALYZER_API_URL}/v1/pipelines",
            headers=get_analyzer_headers(),
            timeout=30.0,
        )
        response.raise_for_status()
        return jsonify(response.json())
    except httpx.HTTPError as e:
        return jsonify({"error": f"Failed to fetch pipelines: {str(e)}"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/analyzer/pipeline-tiers', methods=['GET'])
def list_analyzer_pipeline_tiers():
    """Fetch pipeline tier groupings from Analyzer API."""
    try:
        response = httpx.get(
            f"{ANALYZER_API_URL}/v1/pipeline-tiers",
            headers=get_analyzer_headers(),
            timeout=30.0,
        )
        response.raise_for_status()
        return jsonify(response.json())
    except httpx.HTTPError as e:
        return jsonify({"error": f"Failed to fetch pipeline tiers: {str(e)}"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/analyzer/output-modes', methods=['GET'])
def list_analyzer_output_modes():
    """Fetch available output modes from Analyzer API."""
    try:
        response = httpx.get(
            f"{ANALYZER_API_URL}/v1/output-modes",
            headers=get_analyzer_headers(),
            timeout=30.0,
        )
        response.raise_for_status()
        return jsonify(response.json())
    except httpx.HTTPError as e:
        return jsonify({"error": f"Failed to fetch output modes: {str(e)}"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/analyzer/categories', methods=['GET'])
def list_analyzer_categories():
    """Fetch engine categories from Analyzer API."""
    try:
        response = httpx.get(
            f"{ANALYZER_API_URL}/v1/categories",
            headers=get_analyzer_headers(),
            timeout=30.0,
        )
        response.raise_for_status()
        return jsonify(response.json())
    except httpx.HTTPError as e:
        return jsonify({"error": f"Failed to fetch categories: {str(e)}"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/analyzer/curator/recommend', methods=['POST'])
def curator_recommend():
    """Get engine recommendations from the Curator AI."""
    try:
        data = request.get_json()
        response = httpx.post(
            f"{ANALYZER_API_URL}/v1/curator/recommend",
            headers=get_analyzer_headers(),
            json=data,
            timeout=120.0,  # Longer timeout for AI processing
        )
        response.raise_for_status()
        return jsonify(response.json())
    except httpx.HTTPError as e:
        return jsonify({"error": f"Curator request failed: {str(e)}"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/analyzer/curator/quick-suggest', methods=['GET'])
def curator_quick_suggest():
    """Get quick heuristic-based engine suggestions."""
    try:
        text = request.args.get('text', '')
        response = httpx.get(
            f"{ANALYZER_API_URL}/v1/curator/quick-suggest",
            headers=get_analyzer_headers(),
            params={'text': text},
            timeout=10.0,
        )
        response.raise_for_status()
        return jsonify(response.json())
    except httpx.HTTPError as e:
        return jsonify({"error": f"Quick suggest failed: {str(e)}"}), 500
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
    llm_keys = data.get('llm_keys')  # Forward user-provided API keys

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
                payload = {
                    "documents": [doc],
                    "engine": engine,
                    "output_mode": output_mode
                }
                response = httpx.post(
                    f"{ANALYZER_API_URL}/v1/analyze",
                    headers=get_analyzer_headers(llm_keys),
                    json=payload,
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
            payload = {
                "documents": documents,
                "engine": engine,
                "output_mode": output_mode
            }
            response = httpx.post(
                f"{ANALYZER_API_URL}/v1/analyze",
                headers=get_analyzer_headers(llm_keys),
                json=payload,
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
    llm_keys = data.get('llm_keys')  # Forward user-provided API keys

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
            headers=get_analyzer_headers(llm_keys),
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


@app.route('/api/analyzer/analyze/pipeline', methods=['POST'])
def submit_pipeline_analysis():
    """Submit documents for pipeline analysis (chained engines)."""
    data = request.json or {}
    file_paths = data.get('file_paths', [])
    inline_documents = data.get('documents', [])
    pipeline = data.get('pipeline')
    output_mode = data.get('output_mode', 'executive_memo')
    include_intermediate = data.get('include_intermediate_outputs', True)
    llm_keys = data.get('llm_keys')

    if not file_paths and not inline_documents:
        return jsonify({"success": False, "error": "No files provided"})

    if not pipeline:
        return jsonify({"success": False, "error": "No pipeline selected"})

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
        for i, path in enumerate(file_paths):
            content = extract_text_from_file(path)
            filename = os.path.basename(path)

            if content and not content.startswith('[Error'):
                documents.append({
                    "id": f"doc_{i+1}",
                    "title": filename,
                    "content": content
                })
            else:
                errors.append(f"Failed to read {filename}: {content}")

    if not documents:
        return jsonify({
            "success": False,
            "error": "No valid documents could be processed",
            "details": errors
        })

    # Build headers for analyzer API
    headers = get_analyzer_headers()

    # Forward user-provided LLM API keys if present
    if llm_keys:
        if llm_keys.get('anthropic_api_key'):
            headers['X-Anthropic-Api-Key'] = llm_keys['anthropic_api_key']
        if llm_keys.get('gemini_api_key'):
            headers['X-Gemini-Api-Key'] = llm_keys['gemini_api_key']

    try:
        response = httpx.post(
            f"{ANALYZER_API_URL}/v1/analyze/pipeline",
            json={
                "documents": documents,
                "pipeline": pipeline,
                "output_mode": output_mode,
                "include_intermediate_outputs": include_intermediate,
            },
            headers=headers,
            timeout=60.0,
        )
        response.raise_for_status()
        job_data = response.json()

        return jsonify({
            "success": True,
            "job_id": job_data.get("job_id"),
            "pipeline": pipeline,
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
            headers=get_analyzer_headers(),
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
            headers=get_analyzer_headers(),
            timeout=120.0,  # Increased for large image data URLs
        )
        response.raise_for_status()
        result = response.json()
        print(f"Got result for job {job_id}, outputs: {list(result.get('outputs', {}).keys())}")
        # Log size of any image URLs
        for key, output in result.get('outputs', {}).items():
            if output.get('image_url'):
                print(f"  {key}: image_url length = {len(output['image_url'])}")
        return jsonify(result)
    except httpx.HTTPError as e:
        print(f"HTTPError fetching result for {job_id}: {e}")
        return jsonify({"error": f"Failed to get result: {str(e)}"}), 500
    except Exception as e:
        print(f"Error fetching result for {job_id}: {e}")
        return jsonify({"error": f"Failed to get result: {str(e)}"}), 500


@app.route('/api/analyzer/jobs/<job_id>', methods=['DELETE'])
def delete_analysis_job(job_id):
    """Delete a job from the Analyzer API."""
    try:
        response = httpx.delete(
            f"{ANALYZER_API_URL}/v1/jobs/{job_id}",
            headers=get_analyzer_headers(),
            timeout=30.0,
        )
        response.raise_for_status()
        print(f"Deleted job {job_id} from server")
        return jsonify({"success": True, "job_id": job_id})
    except httpx.HTTPError as e:
        print(f"HTTPError deleting job {job_id}: {e}")
        return jsonify({"error": f"Failed to delete job: {str(e)}"}), 500


@app.route('/api/analyzer/jobs', methods=['GET'])
def list_analysis_jobs():
    """List recent jobs from Analyzer API."""
    try:
        response = httpx.get(
            f"{ANALYZER_API_URL}/v1/jobs",
            headers=get_analyzer_headers(),
            params={"status": "completed", "limit": 20},
            timeout=30.0,
        )
        response.raise_for_status()
        return jsonify(response.json())
    except httpx.HTTPError as e:
        return jsonify({"error": f"Failed to list jobs: {str(e)}"}), 500


@app.route('/api/admin/cleanup-orphaned-jobs', methods=['POST'])
def cleanup_orphaned_jobs():
    """Cleanup orphaned Procrastinate jobs stuck in 'doing' status."""
    try:
        response = httpx.post(
            f"{ANALYZER_API_URL}/v1/admin/cleanup-orphaned-jobs",
            headers=get_analyzer_headers(),
            timeout=30.0,
        )
        response.raise_for_status()
        return jsonify(response.json())
    except httpx.HTTPError as e:
        return jsonify({"error": f"Failed to cleanup: {str(e)}"}), 500


@app.route('/api/admin/debug-status', methods=['GET'])
def get_debug_status():
    """Get detailed debug status from analyzer for troubleshooting."""
    job_id = request.args.get('job_id')
    try:
        params = {}
        if job_id:
            params['job_id'] = job_id
        response = httpx.get(
            f"{ANALYZER_API_URL}/v1/admin/debug-status",
            headers=get_analyzer_headers(),
            params=params,
            timeout=30.0,
        )
        response.raise_for_status()
        return jsonify(response.json())
    except httpx.HTTPError as e:
        return jsonify({"error": f"Failed to get debug status: {str(e)}"}), 500


@app.route('/api/admin/requeue-pending-jobs', methods=['POST'])
def requeue_pending_jobs():
    """Re-queue pending jobs that aren't in the Procrastinate queue."""
    try:
        response = httpx.post(
            f"{ANALYZER_API_URL}/v1/admin/requeue-pending-jobs",
            headers=get_analyzer_headers(),
            timeout=30.0,
        )
        response.raise_for_status()
        return jsonify(response.json())
    except httpx.HTTPError as e:
        return jsonify({"error": f"Failed to requeue: {str(e)}"}), 500


@app.route('/api/admin/force-requeue/<job_id>', methods=['POST'])
def force_requeue_job(job_id):
    """Force-requeue a specific job regardless of its current status."""
    try:
        response = httpx.post(
            f"{ANALYZER_API_URL}/v1/admin/force-requeue/{job_id}",
            headers=get_analyzer_headers(),
            timeout=30.0,
        )
        response.raise_for_status()
        return jsonify(response.json())
    except httpx.HTTPError as e:
        return jsonify({"error": f"Failed to force-requeue: {str(e)}"}), 500


@app.route('/api/admin/cancel-all', methods=['POST'])
def cancel_all_jobs():
    """Nuclear option: Cancel ALL active jobs."""
    try:
        response = httpx.post(
            f"{ANALYZER_API_URL}/v1/admin/cancel-all",
            headers=get_analyzer_headers(),
            timeout=30.0,
        )
        response.raise_for_status()
        return jsonify(response.json())
    except httpx.HTTPError as e:
        return jsonify({"error": f"Failed to cancel all: {str(e)}"}), 500


@app.route('/api/analyzer/jobs/<job_id>/resume', methods=['POST'])
def resume_job(job_id):
    """Resume a failed job from its last completed stage."""
    try:
        response = httpx.post(
            f"{ANALYZER_API_URL}/v1/jobs/{job_id}/resume",
            headers=get_analyzer_headers(),
            timeout=30.0,
        )
        response.raise_for_status()
        return jsonify(response.json())
    except httpx.HTTPError as e:
        return jsonify({"error": f"Failed to resume job: {str(e)}"}), 500


# Main Page

@app.route('/')
def index():
    """Serve the main application page."""
    return Response(HTML_PAGE, mimetype='text/html')


@app.route('/job/<job_id>')
def job_page(job_id):
    """Serve a job-specific page with stable URL."""
    # The HTML page will detect the job_id from URL and auto-load it
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

        /* KEYS Button */
        .btn-keys {
            display: flex;
            align-items: center;
            gap: 0.5rem;
            padding: 0.6rem 1.2rem;
            background: var(--bg-input);
            border: 1px solid var(--border);
            border-radius: var(--radius);
            color: var(--text-secondary);
            font-size: 0.75rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.1em;
            cursor: pointer;
            transition: all 0.2s ease;
        }

        .btn-keys:hover {
            background: var(--bg-card);
            border-color: var(--accent);
            color: var(--accent);
        }

        .btn-keys.has-keys {
            border-color: var(--success);
            color: var(--success);
        }

        .keys-icon {
            font-size: 1rem;
        }

        /* Keys Modal */
        .keys-modal-overlay {
            display: none;
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: rgba(0, 0, 0, 0.7);
            z-index: 1000;
            justify-content: center;
            align-items: center;
        }

        .keys-modal-overlay.active {
            display: flex;
        }

        .keys-modal {
            background: var(--bg-card);
            border-radius: var(--radius-lg);
            width: 90%;
            max-width: 500px;
            border: 1px solid var(--border);
            box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.5);
        }

        .keys-modal-header {
            padding: 1.5rem;
            border-bottom: 1px solid var(--border);
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        .keys-modal-header h3 {
            font-family: 'Libre Baskerville', Georgia, serif;
            font-size: 1.25rem;
            font-weight: 400;
            margin: 0;
        }

        .keys-modal-body {
            padding: 1.5rem;
        }

        .key-field {
            margin-bottom: 1.25rem;
        }

        .key-field label {
            display: block;
            font-size: 0.75rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: var(--text-secondary);
            margin-bottom: 0.5rem;
        }

        .key-field input {
            width: 100%;
            padding: 0.75rem;
            background: var(--bg-input);
            border: 1px solid var(--border);
            border-radius: var(--radius);
            color: var(--text);
            font-family: 'SF Mono', 'Consolas', monospace;
            font-size: 0.85rem;
        }

        .key-field input:focus {
            outline: none;
            border-color: var(--accent);
        }

        .key-field .hint {
            font-size: 0.7rem;
            color: var(--text-muted);
            margin-top: 0.4rem;
        }

        .keys-modal-footer {
            padding: 1rem 1.5rem;
            border-top: 1px solid var(--border);
            display: flex;
            gap: 0.75rem;
            justify-content: flex-end;
        }

        .keys-status {
            padding: 0.75rem 1rem;
            background: var(--bg-input);
            border-radius: var(--radius);
            font-size: 0.8rem;
            margin-bottom: 1rem;
        }

        .keys-status.configured {
            border-left: 3px solid var(--success);
            color: var(--success);
        }

        .keys-status.missing {
            border-left: 3px solid var(--warning);
            color: var(--warning);
        }

        /* Profile Management */
        .profile-section {
            margin-bottom: 1.25rem;
            padding-bottom: 1.25rem;
            border-bottom: 1px solid var(--border);
        }

        .profile-section label {
            display: block;
            font-size: 0.75rem;
            font-weight: 500;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: var(--text-muted);
            margin-bottom: 0.5rem;
        }

        .profile-selector {
            display: flex;
            gap: 0.5rem;
            align-items: center;
        }

        .profile-selector select {
            flex: 1;
            padding: 0.6rem 0.8rem;
            background: var(--bg-input);
            border: 1px solid var(--border);
            border-radius: var(--radius);
            color: var(--text-primary);
            font-size: 0.9rem;
            cursor: pointer;
        }

        .profile-selector select:focus {
            outline: none;
            border-color: var(--accent);
        }

        .btn-icon {
            padding: 0.6rem;
            background: var(--bg-input);
            border: 1px solid var(--border);
            border-radius: var(--radius);
            color: var(--text-secondary);
            cursor: pointer;
            transition: all 0.2s;
            display: flex;
            align-items: center;
            justify-content: center;
        }

        .btn-icon:hover {
            background: var(--bg-hover);
            color: var(--text-primary);
        }

        .btn-icon.danger:hover {
            background: rgba(239, 68, 68, 0.1);
            color: var(--error);
            border-color: var(--error);
        }

        .profile-actions {
            display: flex;
            gap: 0.5rem;
            margin-top: 0.75rem;
        }

        .profile-actions input {
            flex: 1;
            padding: 0.5rem 0.75rem;
            background: var(--bg-input);
            border: 1px solid var(--border);
            border-radius: var(--radius);
            color: var(--text-primary);
            font-size: 0.85rem;
        }

        .profile-actions input::placeholder {
            color: var(--text-muted);
        }

        .profile-actions .btn {
            padding: 0.5rem 1rem;
            font-size: 0.8rem;
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

        /* Pipeline Cards (Meta-Engines) */
        .pipeline-list {
            display: flex;
            flex-direction: column;
            gap: 0.75rem;
        }

        .pipeline-card {
            padding: 1rem;
            background: var(--bg-input);
            border: 1px solid var(--border);
            border-radius: var(--radius);
            cursor: pointer;
            transition: all 0.15s;
        }

        .pipeline-card:hover { border-color: var(--accent-muted); box-shadow: var(--shadow-sm); }
        .pipeline-card.selected { border-color: var(--accent); background: var(--bg-card); border-width: 2px; }
        .pipeline-card .name { font-weight: 600; font-size: 0.9rem; display: flex; align-items: center; gap: 0.5rem; }
        .pipeline-card .desc { font-size: 0.8rem; color: var(--text-secondary); margin-top: 0.35rem; line-height: 1.4; }
        .pipeline-card .synergy { font-size: 0.75rem; color: var(--accent); margin-top: 0.5rem; font-style: italic; }
        .pipeline-card .stages { margin-top: 0.75rem; display: flex; align-items: center; gap: 0.25rem; flex-wrap: wrap; }
        .pipeline-card .stage-chip {
            padding: 0.2rem 0.5rem;
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 4px;
            font-size: 0.7rem;
            font-family: var(--mono);
        }
        .pipeline-card .stage-arrow { color: var(--text-muted); font-size: 0.8rem; }
        .pipeline-card .tier-badge {
            font-size: 0.65rem;
            padding: 0.15rem 0.4rem;
            background: var(--accent);
            color: white;
            border-radius: 3px;
            font-weight: 500;
        }

        /* Tier Filter Tabs */
        .tier-tabs {
            display: flex;
            flex-wrap: wrap;
            gap: 0.5rem;
            margin: 0.75rem 0;
            padding-bottom: 0.75rem;
            border-bottom: 1px solid var(--border);
        }

        .tier-tab {
            padding: 0.4rem 0.75rem;
            background: var(--bg-input);
            border: 1px solid var(--border);
            border-radius: 6px;
            cursor: pointer;
            font-size: 0.75rem;
            transition: all 0.15s;
        }

        .tier-tab:hover { border-color: var(--accent-muted); }
        .tier-tab.active { background: var(--accent); color: white; border-color: var(--accent); }
        .tier-tab .count { opacity: 0.7; margin-left: 0.25rem; }

        /* Category Tabs */
        .category-tabs {
            display: flex;
            flex-wrap: wrap;
            gap: 0.5rem;
            margin: 1rem 0;
            padding: 0.5rem 0;
            border-bottom: 1px solid var(--border);
        }

        .category-tab {
            padding: 0.4rem 0.8rem;
            background: var(--bg-input);
            border: 1px solid var(--border);
            border-radius: 20px;
            cursor: pointer;
            font-size: 0.75rem;
            display: flex;
            align-items: center;
            gap: 0.3rem;
            transition: all 0.15s;
            font-family: var(--font-sans);
        }

        .category-tab:hover {
            border-color: var(--accent-muted);
            background: var(--bg-card);
        }

        .category-tab.active {
            background: var(--accent);
            color: white;
            border-color: var(--accent);
        }

        .category-tab .cat-icon { font-size: 1em; }
        .category-tab .cat-count { opacity: 0.7; }

        /* Curator Section */
        .curator-section {
            background: linear-gradient(135deg, rgba(45,125,70,0.05) 0%, rgba(45,125,70,0.02) 100%);
            border: 1px solid rgba(45,125,70,0.2);
            border-radius: var(--radius);
            padding: 1rem;
            margin-bottom: 1rem;
        }

        .curator-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        .curator-result {
            margin-top: 0.75rem;
            font-size: 0.85rem;
            color: var(--text-muted);
        }

        .curator-result.loading {
            color: var(--accent);
            font-style: italic;
        }

        .curator-result .doc-type {
            font-weight: 600;
            color: var(--text);
        }

        .curator-result .strategy {
            margin-top: 0.5rem;
            padding: 0.5rem;
            background: var(--bg-card);
            border-radius: 4px;
            font-size: 0.8rem;
        }

        /* Recommended Engine Badge */
        .engine-card.recommended {
            border-color: var(--success);
            background: linear-gradient(135deg, rgba(45,125,70,0.05) 0%, transparent 100%);
        }

        .rec-badge {
            font-size: 0.65rem;
            font-weight: 600;
            color: var(--success);
            background: rgba(45,125,70,0.1);
            padding: 0.2rem 0.5rem;
            border-radius: 3px;
            margin-bottom: 0.5rem;
            align-self: flex-start;
        }

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

        .progress-details {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 0.5rem;
        }

        .progress-counter {
            font-size: 0.8rem;
            font-weight: 600;
            color: var(--accent);
        }

        .progress-doc-name {
            font-size: 0.8rem;
            color: var(--text-secondary);
            max-width: 60%;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
            text-align: right;
        }

        .progress-stages {
            display: flex;
            gap: 0.5rem;
            flex-wrap: wrap;
            margin-top: 1rem;
        }

        .progress-warnings {
            margin-top: 0.75rem;
            padding: 0.5rem 0.75rem;
            background: var(--warning-bg, #fff8e6);
            border-left: 3px solid var(--warning, #f59e0b);
            font-size: 0.75rem;
            color: var(--text-secondary);
            border-radius: 0 4px 4px 0;
        }
        .progress-warnings .warning-count {
            font-weight: 600;
            color: var(--warning, #f59e0b);
        }
        .progress-warnings .warning-list {
            margin-top: 0.25rem;
            font-family: var(--mono-font);
            font-size: 0.7rem;
            max-height: 100px;
            overflow-y: auto;
        }
        .progress-warnings .warning-item {
            margin: 2px 0;
            opacity: 0.85;
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

        /* Job URL Section */
        .job-url-section {
            margin-top: 1rem;
            padding: 0.75rem 1rem;
            background: var(--bg-input);
            border-radius: var(--radius);
            display: flex;
            align-items: center;
            gap: 0.5rem;
            font-size: 0.85rem;
        }

        .job-url-label {
            color: var(--text-muted);
            font-weight: 500;
        }

        .job-url-link {
            color: var(--accent);
            text-decoration: none;
            font-family: monospace;
            font-size: 0.8rem;
            word-break: break-all;
        }

        .job-url-link:hover {
            text-decoration: underline;
        }

        .btn-small {
            padding: 0.25rem 0.5rem;
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: var(--radius);
            cursor: pointer;
            font-size: 0.75rem;
        }

        .btn-small:hover {
            background: var(--bg-hover);
        }

        /* Debug Panel */
        .debug-toggle {
            position: fixed;
            bottom: 1rem;
            right: 1rem;
            z-index: 1000;
            display: flex;
            align-items: center;
            gap: 0.5rem;
            padding: 0.5rem 1rem;
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: var(--radius);
            box-shadow: var(--shadow);
            cursor: pointer;
            font-size: 0.8rem;
        }

        .debug-toggle:hover {
            background: var(--bg-hover);
        }

        .debug-toggle.active {
            background: var(--accent);
            color: white;
            border-color: var(--accent);
        }

        .debug-panel {
            display: none;
            position: fixed;
            bottom: 4rem;
            right: 1rem;
            width: 450px;
            max-height: 60vh;
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: var(--radius);
            box-shadow: 0 4px 20px rgba(0,0,0,0.15);
            z-index: 999;
            overflow: hidden;
        }

        .debug-panel.show {
            display: flex;
            flex-direction: column;
        }

        .debug-panel-header {
            padding: 0.75rem 1rem;
            background: var(--accent);
            color: white;
            font-weight: 600;
            font-size: 0.85rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        .debug-panel-actions {
            display: flex;
            gap: 0.5rem;
        }

        .debug-panel-actions button {
            padding: 0.25rem 0.5rem;
            background: rgba(255,255,255,0.2);
            border: none;
            border-radius: 3px;
            color: white;
            cursor: pointer;
            font-size: 0.75rem;
        }

        .debug-panel-actions button:hover {
            background: rgba(255,255,255,0.3);
        }

        .debug-panel-content {
            flex: 1;
            overflow-y: auto;
            padding: 1rem;
            font-family: monospace;
            font-size: 0.75rem;
            line-height: 1.5;
        }

        .debug-section {
            margin-bottom: 1rem;
        }

        .debug-section-title {
            font-weight: 600;
            color: var(--accent);
            margin-bottom: 0.5rem;
            font-size: 0.8rem;
        }

        .debug-item {
            display: flex;
            justify-content: space-between;
            padding: 0.25rem 0;
            border-bottom: 1px solid var(--border-light);
        }

        .debug-item:last-child {
            border-bottom: none;
        }

        .debug-key {
            color: var(--text-muted);
        }

        .debug-value {
            color: var(--text-primary);
            font-weight: 500;
        }

        .debug-value.success { color: var(--success); }
        .debug-value.warning { color: #b8860b; }
        .debug-value.error { color: var(--error); }

        .debug-log {
            background: var(--bg-input);
            padding: 0.5rem;
            border-radius: 3px;
            max-height: 150px;
            overflow-y: auto;
            white-space: pre-wrap;
            word-break: break-all;
        }

        .debug-status-indicator {
            display: inline-block;
            width: 8px;
            height: 8px;
            border-radius: 50%;
            margin-right: 0.5rem;
        }

        .debug-status-indicator.green { background: var(--success); }
        .debug-status-indicator.yellow { background: #b8860b; }
        .debug-status-indicator.red { background: var(--error); }

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
            line-height: 1.6;
            color: var(--text-secondary);
            overflow: hidden;
            max-height: 100%;
            font-family: 'Inter', -apple-system, sans-serif;
            white-space: pre-wrap;
            word-wrap: break-word;
        }
        .gallery-card-preview .text-preview.json-preview {
            font-family: 'SF Mono', Monaco, monospace;
            font-size: 0.7rem;
            line-height: 1.5;
            color: var(--text-muted);
        }
        .gallery-card-preview .html-preview {
            width: 100%;
            height: 100%;
            overflow: hidden;
            transform: scale(0.4);
            transform-origin: top left;
            width: 250%;
            height: 250%;
            padding: 0.5rem;
            pointer-events: none;
        }
        .gallery-card-preview .html-preview table {
            font-size: 0.65rem;
            width: 100%;
            border-collapse: collapse;
        }
        .gallery-card-preview .html-preview th,
        .gallery-card-preview .html-preview td {
            border: 1px solid var(--border);
            padding: 0.25rem 0.4rem;
            text-align: left;
        }
        .gallery-card-preview .html-preview th {
            background: var(--bg-input);
            font-weight: 600;
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

        .gallery-card-actions button.btn-delete {
            background: transparent;
            border-color: var(--error);
            color: var(--error);
        }

        .gallery-card-actions button.btn-delete:hover {
            background: var(--error);
            border-color: var(--error);
            color: white;
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
            display: flex;
            flex-direction: column;
            gap: 1rem;
            max-width: 900px;
            margin: 0 auto;
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

        /* Job Group in Library - Clean Design */
        .job-group {
            border: 1px solid var(--border);
            border-radius: 8px;
            overflow: hidden;
            background: var(--bg-card);
        }
        .job-group-header {
            display: flex;
            align-items: center;
            padding: 0.75rem 1rem;
            background: var(--bg-secondary);
            border-bottom: 1px solid var(--border);
            cursor: pointer;
            gap: 0.75rem;
        }
        .job-group-header:hover {
            background: var(--bg-tertiary);
        }
        .job-group-toggle {
            font-size: 0.6rem;
            color: var(--text-muted);
            transition: transform 0.2s;
            flex-shrink: 0;
        }
        .job-group.collapsed .job-group-toggle {
            transform: rotate(-90deg);
        }
        .job-group-title {
            font-weight: 500;
            font-size: 0.9rem;
            flex: 1;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
            min-width: 0;
        }
        .job-group-count {
            background: var(--accent);
            color: white;
            padding: 0.15rem 0.5rem;
            border-radius: 10px;
            font-size: 0.7rem;
            font-weight: 500;
            flex-shrink: 0;
        }
        .job-group-date {
            font-size: 0.75rem;
            color: var(--text-muted);
            flex-shrink: 0;
        }
        .job-group-delete {
            width: 28px;
            height: 28px;
            border-radius: 4px;
            border: none;
            background: transparent;
            color: var(--text-muted);
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1rem;
            flex-shrink: 0;
            transition: all 0.15s;
        }
        .job-group-delete:hover {
            background: #fee2e2;
            color: #dc2626;
        }
        .job-group-items {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(160px, 1fr));
            gap: 0.75rem;
            padding: 0.75rem;
        }
        .job-group.collapsed .job-group-items {
            display: none;
        }
        .job-group.collapsed .job-group-toggle {
            transform: rotate(-90deg);
        }
        /* Compact cards inside job groups */
        .job-group-items .gallery-card {
            border-radius: 6px;
        }
        .job-group-items .gallery-card-preview {
            height: 100px;
        }
        .job-group-items .gallery-card-info {
            padding: 0.5rem 0.75rem;
        }
        .job-group-items .gallery-card-title {
            font-size: 0.8rem;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }
        .job-group-items .gallery-card-meta {
            font-size: 0.7rem;
        }
        .job-group-items .gallery-card-actions {
            padding: 0 0.75rem 0.5rem;
        }
        .job-group-items .gallery-card-actions button {
            padding: 0.35rem;
            font-size: 0.7rem;
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

        /* Rendered Markdown Styles */
        .markdown-body {
            font-family: 'Libre Baskerville', Georgia, serif;
            font-size: 1rem;
            line-height: 1.8;
            color: var(--text);
            max-width: 800px;
            margin: 0 auto;
        }

        .markdown-body h1 {
            font-size: 1.8rem;
            font-weight: 700;
            margin: 0 0 1.5rem 0;
            padding-bottom: 0.75rem;
            border-bottom: 2px solid var(--border);
            line-height: 1.3;
        }

        .markdown-body h2 {
            font-size: 1.35rem;
            font-weight: 700;
            margin: 2rem 0 1rem 0;
            color: var(--accent);
            line-height: 1.4;
        }

        .markdown-body h3 {
            font-size: 1.1rem;
            font-weight: 700;
            margin: 1.5rem 0 0.75rem 0;
            color: var(--accent-muted);
        }

        .markdown-body h4 {
            font-size: 1rem;
            font-weight: 600;
            margin: 1.25rem 0 0.5rem 0;
            color: var(--accent-muted);
        }

        .markdown-body p {
            margin: 0 0 1.25rem 0;
            text-align: justify;
            hyphens: auto;
        }

        .markdown-body blockquote {
            border-left: 3px solid var(--accent);
            margin: 1.5rem 0;
            padding: 0.75rem 1.25rem;
            background: var(--bg-input);
            font-style: italic;
            color: var(--accent-muted);
        }

        .markdown-body blockquote p:last-child { margin-bottom: 0; }

        .markdown-body ul, .markdown-body ol {
            margin: 1rem 0 1.25rem 1.5rem;
            padding: 0;
        }

        .markdown-body li {
            margin-bottom: 0.5rem;
            line-height: 1.7;
        }

        .markdown-body code {
            font-family: 'SF Mono', 'Consolas', monospace;
            font-size: 0.875em;
            background: var(--bg-input);
            padding: 0.15em 0.4em;
            border-radius: 3px;
            color: var(--accent-muted);
        }

        .markdown-body pre {
            background: #1a1a1a;
            color: #e0e0e0;
            padding: 1rem 1.25rem;
            border-radius: var(--radius);
            overflow-x: auto;
            margin: 1.25rem 0;
            font-size: 0.85rem;
            line-height: 1.5;
        }

        .markdown-body pre code {
            background: none;
            padding: 0;
            color: inherit;
        }

        .markdown-body hr {
            border: none;
            border-top: 1px solid var(--border);
            margin: 2rem 0;
        }

        .markdown-body a {
            color: var(--accent);
            text-decoration: underline;
            text-decoration-color: var(--border-dark);
            text-underline-offset: 2px;
        }

        .markdown-body a:hover {
            text-decoration-color: var(--accent);
        }

        .markdown-body strong { font-weight: 700; }
        .markdown-body em { font-style: italic; }

        /* Footnote styling */
        .markdown-body sup {
            font-size: 0.75em;
            line-height: 0;
            position: relative;
            vertical-align: baseline;
            top: -0.5em;
            color: var(--accent-muted);
        }

        /* Table styling */
        .markdown-body table {
            width: 100%;
            border-collapse: collapse;
            margin: 1.5rem 0;
            font-size: 0.9rem;
        }

        .markdown-body th, .markdown-body td {
            border: 1px solid var(--border);
            padding: 0.6rem 0.8rem;
            text-align: left;
        }

        .markdown-body th {
            background: var(--bg-input);
            font-weight: 600;
        }

        .markdown-body tr:nth-child(even) {
            background: var(--bg-input);
        }

        /* Sources/References section */
        .markdown-body h2:last-of-type + p,
        .markdown-body h2:last-of-type ~ p {
            font-family: 'Inter', sans-serif;
            font-size: 0.85rem;
            line-height: 1.8;
        }

        /* Each footnote on its own line */
        .markdown-body h2:last-of-type + p sup,
        .markdown-body h2:last-of-type ~ p sup {
            display: block;
            margin-top: 0.5rem;
        }
        .markdown-body h2:last-of-type + p sup:first-child,
        .markdown-body h2:last-of-type ~ p sup:first-child {
            margin-top: 0;
        }

        /* Modal adjustments for markdown */
        .result-modal-body.markdown-view {
            padding: 2rem 3rem;
            background: var(--bg-card);
        }

        .result-modal-content.wide {
            max-width: 900px;
            width: 90vw;
        }

        /* Copy button success state */
        .btn-copied {
            background: var(--success) !important;
            color: white !important;
        }

        /* ========================================
           Web-Saver Integration Styles
           ======================================== */

        .btn-websaver {
            background: linear-gradient(135deg, #4a90a4 0%, #357a8a 100%);
            color: white;
            border: none;
            padding: 0.5rem 1rem;
            border-radius: 6px;
            cursor: pointer;
            font-size: 0.85rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
            transition: all 0.2s ease;
        }

        .btn-websaver:hover {
            background: linear-gradient(135deg, #357a8a 0%, #2a6570 100%);
            transform: translateY(-1px);
        }

        .btn-websaver:disabled {
            opacity: 0.5;
            cursor: not-allowed;
            transform: none;
        }

        .websaver-status {
            font-size: 0.75rem;
            color: var(--text-muted);
            margin-left: 0.5rem;
        }

        .websaver-status.connected {
            color: var(--success);
        }

        .websaver-status.error {
            color: var(--danger);
        }

        /* Web-Saver Modal */
        .ws-modal {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0, 0, 0, 0.7);
            display: flex;
            align-items: center;
            justify-content: center;
            z-index: 2000;
            backdrop-filter: blur(4px);
        }

        .ws-modal-content {
            background: var(--bg-card);
            border-radius: 12px;
            width: 90%;
            max-width: 600px;
            max-height: 80vh;
            display: flex;
            flex-direction: column;
            box-shadow: 0 20px 60px rgba(0, 0, 0, 0.4);
            border: 1px solid var(--border-color);
        }

        .ws-modal-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 1rem 1.5rem;
            border-bottom: 1px solid var(--border-color);
        }

        .ws-modal-header h3 {
            margin: 0;
            font-size: 1.1rem;
            color: var(--text-primary);
        }

        .ws-modal-close {
            background: none;
            border: none;
            font-size: 1.5rem;
            cursor: pointer;
            color: var(--text-muted);
            padding: 0;
            line-height: 1;
        }

        .ws-modal-close:hover {
            color: var(--text-primary);
        }

        .ws-modal-body {
            padding: 1rem 1.5rem;
            overflow-y: auto;
            flex: 1;
        }

        .ws-search-row {
            display: flex;
            gap: 0.5rem;
            margin-bottom: 1rem;
        }

        .ws-search-row input {
            flex: 1;
            padding: 0.5rem 1rem;
            background: var(--bg-input);
            border: 1px solid var(--border-color);
            border-radius: 6px;
            color: var(--text-primary);
            font-size: 0.9rem;
        }

        .ws-search-row input:focus {
            outline: none;
            border-color: var(--accent);
        }

        .ws-collections-list {
            display: flex;
            flex-direction: column;
            gap: 0.5rem;
            max-height: 400px;
            overflow-y: auto;
        }

        .ws-collection-item {
            padding: 0.75rem 1rem;
            background: var(--bg-hover);
            border: 1px solid transparent;
            border-radius: 8px;
            cursor: pointer;
            transition: all 0.2s ease;
        }

        .ws-collection-item:hover {
            background: var(--bg-input);
            border-color: var(--border-color);
        }

        .ws-collection-item.selected {
            background: rgba(74, 144, 164, 0.15);
            border-color: #4a90a4;
        }

        .ws-collection-name {
            font-weight: 500;
            color: var(--text-primary);
            margin-bottom: 0.25rem;
        }

        .ws-collection-meta {
            font-size: 0.8rem;
            color: var(--text-muted);
            display: flex;
            gap: 1rem;
        }

        .ws-loading {
            text-align: center;
            padding: 2rem;
            color: var(--text-muted);
        }

        .ws-error {
            text-align: center;
            padding: 2rem;
            color: var(--danger);
        }

        .ws-empty {
            text-align: center;
            padding: 2rem;
            color: var(--text-muted);
        }

        .ws-modal-footer {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 1rem 1.5rem;
            border-top: 1px solid var(--border-color);
            gap: 0.5rem;
        }

        .ws-modal-footer span {
            flex: 1;
            font-size: 0.85rem;
            color: var(--text-muted);
        }
    </style>
    <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
</head>
<body>
    <div class="app">
        <header>
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <h1>The Visualizer</h1>
                    <p class="tagline">Document Intelligence & Visual Analysis</p>
                </div>
                <button class="btn btn-keys" onclick="openKeysModal()">
                    <span class="keys-icon">&#128273;</span> KEYS
                </button>
            </div>
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

                        <!-- Import from Web-Saver -->
                        <span class="section-label" style="margin-top: 1rem;">Or import from Web-Saver</span>
                        <div class="path-input-row">
                            <button class="btn btn-sm btn-websaver" onclick="openWebSaverModal()" id="websaver-import-btn">
                                📚 Import Collection
                            </button>
                            <span id="websaver-status" class="websaver-status"></span>
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

                        <!-- Engine vs Bundle vs Pipeline Toggle -->
                        <div class="mode-toggle">
                            <div class="mode-btn active" onclick="setEngineMode('engine')" id="engine-mode-single">
                                <div class="title">Single Engine</div>
                                <div class="desc">One analytical lens</div>
                            </div>
                            <div class="mode-btn" onclick="setEngineMode('bundle')" id="engine-mode-bundle">
                                <div class="title">Bundle</div>
                                <div class="desc">Multiple engines</div>
                            </div>
                            <div class="mode-btn" onclick="setEngineMode('pipeline')" id="engine-mode-pipeline">
                                <div class="title">Pipeline</div>
                                <div class="desc">Chained engines</div>
                            </div>
                        </div>

                        <!-- Smart Curator -->
                        <div id="curator-section" class="curator-section">
                            <div class="curator-header">
                                <span class="section-label">Smart Curator</span>
                                <button class="btn btn-sm" onclick="getCuratorRecommendations()" id="curator-btn" disabled title="Upload documents first">
                                    Get AI Recommendations
                                </button>
                            </div>
                            <div id="curator-result" class="curator-result"></div>
                        </div>

                        <!-- Category Filter Tabs -->
                        <div id="category-tabs" class="category-tabs"></div>

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

                        <!-- Pipeline Selection (Meta-Engines) -->
                        <div id="pipeline-selection" style="display:none;">
                            <div id="pipeline-tier-tabs" class="tier-tabs"></div>
                            <span class="section-label">Select Pipeline <span id="pipeline-count" style="color:var(--text-secondary);"></span></span>
                            <div id="pipeline-list" class="pipeline-list"></div>
                        </div>

                        <!-- Output Mode -->
                        <div class="output-select">
                            <span class="section-label">Output Format <span id="output-mode-count" style="color: #666; font-size: 11px;"></span></span>
                            <select id="output-mode">
                                <option value="">Loading output modes...</option>
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
                        <div class="progress-details">
                            <div id="progress-counter" class="progress-counter"></div>
                            <div id="progress-doc-name" class="progress-doc-name"></div>
                        </div>
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
                        <div id="progress-warnings" class="progress-warnings" style="display:none;"></div>
                        <div id="job-url-section" class="job-url-section" style="display:none;">
                            <span class="job-url-label">Job URL:</span>
                            <a id="job-url-link" href="#" class="job-url-link" target="_blank"></a>
                            <button class="btn-small" onclick="copyJobUrl()" title="Copy URL">📋</button>
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
            <div style="margin-bottom: 1rem; text-align: right;">
                <button class="btn btn-sm" onclick="loadRecentJobs()">Load Recent Jobs</button>
            </div>
            <div class="library-empty" id="library-empty">
                <div class="library-empty-icon">&#128218;</div>
                <div class="library-empty-text">Your library is empty</div>
                <div class="library-empty-hint">Analyzed documents and generated visualizations will appear here.<br>Click "Load Recent Jobs" above to fetch from server.</div>
            </div>
            <div id="library-grid" class="library-grid"></div>
        </div>
    </div>

    <!-- KEYS Modal -->
    <div id="keys-modal-overlay" class="keys-modal-overlay" onclick="closeKeysModal(event)">
        <div class="keys-modal" onclick="event.stopPropagation()">
            <div class="keys-modal-header">
                <h3>API Keys Configuration</h3>
                <button class="btn btn-ghost" onclick="closeKeysModal()">&times;</button>
            </div>
            <div class="keys-modal-body">
                <div id="keys-status" class="keys-status missing">
                    Keys are stored locally in your browser. They are never sent to our servers.
                </div>

                <!-- Profile Section -->
                <div class="profile-section">
                    <label>Key Profile</label>
                    <div class="profile-selector">
                        <select id="profile-select" onchange="onProfileSelect()">
                            <option value="">-- Select Profile --</option>
                        </select>
                        <button class="btn-icon danger" onclick="deleteCurrentProfile()" title="Delete Profile" id="delete-profile-btn" style="display: none;">
                            &#128465;
                        </button>
                    </div>
                    <div class="profile-actions">
                        <input type="text" id="new-profile-name" placeholder="New profile name...">
                        <button class="btn btn-sm" onclick="saveAsNewProfile()">Save as Profile</button>
                    </div>
                </div>

                <div class="key-field">
                    <label for="anthropic-key">Anthropic API Key</label>
                    <input type="password" id="anthropic-key" placeholder="sk-ant-..." autocomplete="off">
                    <div class="hint">Required for document analysis (Claude models)</div>
                </div>
                <div class="key-field">
                    <label for="gemini-key">Google Gemini API Key</label>
                    <input type="password" id="gemini-key" placeholder="AIzaSy..." autocomplete="off">
                    <div class="hint">Required for visual diagram generation</div>
                </div>
            </div>
            <div class="keys-modal-footer">
                <button class="btn btn-ghost" onclick="clearKeys()">Clear All</button>
                <button class="btn btn-primary" onclick="saveKeys()">Update Active Keys</button>
            </div>
        </div>
    </div>

    <script>
        // State
        let scannedDocs = [];
        let selectedDocs = new Set();
        let engines = [];
        let bundles = [];
        let pipelines = [];  // Meta-engines
        let pipelineTiers = [];  // Tier groupings
        let categories = [];
        let outputModes = [];
        let selectedEngine = null;
        let selectedBundle = null;
        let selectedPipeline = null;  // Meta-engine selection
        let selectedTier = null;  // Filter pipelines by tier
        let selectedCategory = null;  // Filter engines by category
        let curatorRecommendations = null;  // AI recommendations
        let collectionMode = 'single';
        let engineMode = 'engine';  // 'engine', 'bundle', or 'pipeline'
        let currentJobId = null;
        let allResults = [];
        let libraryItems = [];

        // Progress tracking
        let currentDocCount = 0;
        let currentDocNames = [];  // Array of document names being processed
        let currentDocIndex = 0;   // Index of document currently being processed

        function $(id) { return document.getElementById(id); }

        // ==================== API KEYS MANAGEMENT ====================
        const KEYS_STORAGE_KEY = 'visualizer_api_keys';
        const PROFILES_STORAGE_KEY = 'visualizer_key_profiles';
        const ACTIVE_PROFILE_KEY = 'visualizer_active_profile';

        function getStoredKeys() {
            try {
                const stored = localStorage.getItem(KEYS_STORAGE_KEY);
                return stored ? JSON.parse(stored) : { anthropic: '', gemini: '' };
            } catch (e) {
                return { anthropic: '', gemini: '' };
            }
        }

        function saveStoredKeys(keys) {
            localStorage.setItem(KEYS_STORAGE_KEY, JSON.stringify(keys));
        }

        function getProfiles() {
            try {
                const stored = localStorage.getItem(PROFILES_STORAGE_KEY);
                return stored ? JSON.parse(stored) : [];
            } catch (e) {
                return [];
            }
        }

        function saveProfiles(profiles) {
            localStorage.setItem(PROFILES_STORAGE_KEY, JSON.stringify(profiles));
        }

        function getActiveProfileName() {
            return localStorage.getItem(ACTIVE_PROFILE_KEY) || '';
        }

        function setActiveProfileName(name) {
            if (name) {
                localStorage.setItem(ACTIVE_PROFILE_KEY, name);
            } else {
                localStorage.removeItem(ACTIVE_PROFILE_KEY);
            }
        }

        function renderProfileSelector() {
            const select = $('profile-select');
            const profiles = getProfiles();
            const activeProfile = getActiveProfileName();

            console.log('Rendering profiles:', profiles.length, 'profiles, active:', activeProfile);

            select.innerHTML = '<option value="">-- Select Profile --</option>';
            profiles.forEach(function(profile) {
                const opt = document.createElement('option');
                opt.value = profile.name;
                opt.textContent = profile.name;
                if (profile.name === activeProfile) {
                    opt.selected = true;
                }
                select.appendChild(opt);
            });

            // Show/hide delete button based on selection
            const deleteBtn = $('delete-profile-btn');
            deleteBtn.style.display = activeProfile ? 'flex' : 'none';
        }

        function onProfileSelect() {
            const select = $('profile-select');
            const profileName = select.value;
            const profiles = getProfiles();

            if (profileName) {
                const profile = profiles.find(function(p) { return p.name === profileName; });
                if (profile) {
                    $('anthropic-key').value = profile.anthropic || '';
                    $('gemini-key').value = profile.gemini || '';
                    setActiveProfileName(profileName);

                    // Also save as current active keys
                    saveStoredKeys({ anthropic: profile.anthropic, gemini: profile.gemini });
                    updateKeysButtonState();
                    updateKeysStatus();
                }
            } else {
                setActiveProfileName('');
            }

            // Show/hide delete button
            const deleteBtn = $('delete-profile-btn');
            deleteBtn.style.display = profileName ? 'flex' : 'none';
        }

        function saveAsNewProfile() {
            const nameInput = $('new-profile-name');
            const name = nameInput.value.trim();

            if (!name) {
                alert('Please enter a profile name');
                return;
            }

            const profiles = getProfiles();
            const existingIndex = profiles.findIndex(function(p) { return p.name === name; });

            const profile = {
                name: name,
                anthropic: $('anthropic-key').value.trim(),
                gemini: $('gemini-key').value.trim()
            };

            if (existingIndex >= 0) {
                if (!confirm('Profile "' + name + '" already exists. Overwrite?')) {
                    return;
                }
                profiles[existingIndex] = profile;
            } else {
                profiles.push(profile);
            }

            saveProfiles(profiles);
            setActiveProfileName(name);

            // Also save as current active keys
            saveStoredKeys({ anthropic: profile.anthropic, gemini: profile.gemini });

            nameInput.value = '';
            renderProfileSelector();

            // Select the newly saved profile in dropdown
            $('profile-select').value = name;
            $('delete-profile-btn').style.display = 'flex';

            updateKeysButtonState();
            updateKeysStatus();

            // Visual feedback
            alert('Profile "' + name + '" saved!');
            console.log('Saved profile:', name);
        }

        function deleteCurrentProfile() {
            const activeProfile = getActiveProfileName();
            if (!activeProfile) return;

            if (!confirm('Delete profile "' + activeProfile + '"?')) {
                return;
            }

            const profiles = getProfiles();
            const newProfiles = profiles.filter(function(p) { return p.name !== activeProfile; });
            saveProfiles(newProfiles);
            setActiveProfileName('');

            $('profile-select').value = '';
            renderProfileSelector();

            console.log('Deleted profile:', activeProfile);
        }

        function openKeysModal() {
            const keys = getStoredKeys();
            $('anthropic-key').value = keys.anthropic || '';
            $('gemini-key').value = keys.gemini || '';
            $('new-profile-name').value = '';
            renderProfileSelector();
            updateKeysStatus();
            $('keys-modal-overlay').classList.add('active');
        }

        function closeKeysModal(event) {
            if (event && event.target !== event.currentTarget) return;
            $('keys-modal-overlay').classList.remove('active');
        }

        function saveKeys() {
            const keys = {
                anthropic: $('anthropic-key').value.trim(),
                gemini: $('gemini-key').value.trim()
            };
            saveStoredKeys(keys);

            // If a profile is selected, also update it
            const activeProfile = getActiveProfileName();
            if (activeProfile) {
                const profiles = getProfiles();
                const profileIndex = profiles.findIndex(function(p) { return p.name === activeProfile; });
                if (profileIndex >= 0) {
                    profiles[profileIndex].anthropic = keys.anthropic;
                    profiles[profileIndex].gemini = keys.gemini;
                    saveProfiles(profiles);
                }
            }

            updateKeysButtonState();
            updateKeysStatus();
            closeKeysModal();
        }

        function clearKeys() {
            if (confirm('Remove all stored API keys? (Profiles will be preserved)')) {
                saveStoredKeys({ anthropic: '', gemini: '' });
                setActiveProfileName('');
                $('anthropic-key').value = '';
                $('gemini-key').value = '';
                $('profile-select').value = '';
                renderProfileSelector();
                updateKeysButtonState();
                updateKeysStatus();
            }
        }

        function updateKeysButtonState() {
            const keys = getStoredKeys();
            const hasKeys = keys.anthropic || keys.gemini;
            const btn = document.querySelector('.btn-keys');
            if (btn) {
                btn.classList.toggle('has-keys', hasKeys);
            }
        }

        function updateKeysStatus() {
            const keys = getStoredKeys();
            const statusEl = $('keys-status');
            if (!statusEl) return;

            const hasAnthropic = !!keys.anthropic;
            const hasGemini = !!keys.gemini;
            const activeProfile = getActiveProfileName();

            let profileInfo = activeProfile ? ' [Profile: ' + activeProfile + ']' : '';

            if (hasAnthropic && hasGemini) {
                statusEl.className = 'keys-status configured';
                statusEl.textContent = 'Both API keys configured. Ready for analysis.' + profileInfo;
            } else if (hasAnthropic || hasGemini) {
                statusEl.className = 'keys-status configured';
                const missing = !hasAnthropic ? 'Anthropic' : 'Gemini';
                statusEl.textContent = 'Partial configuration. ' + missing + ' key not set.' + profileInfo;
            } else {
                statusEl.className = 'keys-status missing';
                statusEl.textContent = 'No API keys configured. Enter your keys to enable analysis.';
            }
        }

        function getApiHeaders() {
            const keys = getStoredKeys();
            const headers = { 'Content-Type': 'application/json' };
            if (keys.anthropic) {
                headers['X-Anthropic-Api-Key'] = keys.anthropic;
            }
            if (keys.gemini) {
                headers['X-Gemini-Api-Key'] = keys.gemini;
            }
            return headers;
        }
        // ==================== END KEYS MANAGEMENT ====================

        // Check if we're on a job URL
        var isJobUrl = window.location.pathname.match(/^\\/job\\/([a-f0-9-]+)$/i);

        // Initialize
        document.addEventListener('DOMContentLoaded', function() {
            setupDragDrop();
            updateKeysButtonState();

            if (isJobUrl) {
                // Loading job from URL - hide panels immediately before any async loading
                var leftPanel = document.querySelector('.left-panel');
                var engineSelector = document.querySelector('.engine-selector');
                var curatorSection = document.getElementById('curator-section');
                if (leftPanel) leftPanel.style.display = 'none';
                if (engineSelector) engineSelector.style.display = 'none';
                if (curatorSection) curatorSection.style.display = 'none';

                // Then load job
                var jobId = isJobUrl[1];
                console.log('Loading job from URL:', jobId);
                loadJobFromUrl(jobId);
            } else {
                // Normal page load
                loadAnalyzerData();
            }
            loadLibrary();
        });

        // Load a job directly from URL
        async function loadJobFromUrl(jobId) {
            try {
                // Switch to analyze view
                switchView('analyze');

                // Show progress section
                var progressSection = document.getElementById('progress-section');
                if (progressSection) progressSection.classList.add('show');

                // First check job status
                const statusRes = await fetch('/api/analyzer/jobs/' + jobId, { headers: getApiHeaders() });
                if (!statusRes.ok) {
                    showAnalysisError('Job not found: ' + jobId);
                    return;
                }

                const job = await statusRes.json();
                currentJobId = jobId;

                // Update URL to show job URL section
                updateJobUrl(jobId);

                if (job.status === 'completed') {
                    // Job already complete - fetch and display result
                    await fetchAndDisplayResult(jobId);
                } else if (job.status === 'failed') {
                    showAnalysisError(job.error_message || 'Job failed');
                } else {
                    // Job still running - start polling
                    updateProgress(job);
                    pollJobStatus(jobId);
                }
            } catch (e) {
                showAnalysisError('Error loading job: ' + e.message);
            }
        }

        // Update browser URL when job starts (without page reload)
        function updateJobUrl(jobId) {
            var newUrl = '/job/' + jobId;
            window.history.pushState({ jobId: jobId }, '', newUrl);

            // Show job URL section with link
            var fullUrl = window.location.origin + newUrl;
            var urlSection = document.getElementById('job-url-section');
            var urlLink = document.getElementById('job-url-link');
            if (urlSection && urlLink) {
                urlLink.href = fullUrl;
                urlLink.textContent = fullUrl;
                urlSection.style.display = 'flex';
            }
        }

        // Copy job URL to clipboard
        function copyJobUrl() {
            var urlLink = document.getElementById('job-url-link');
            if (urlLink) {
                navigator.clipboard.writeText(urlLink.href).then(function() {
                    // Brief visual feedback
                    var btn = event.target;
                    var originalText = btn.textContent;
                    btn.textContent = '✓';
                    setTimeout(function() { btn.textContent = originalText; }, 1000);
                });
            }
        }

        // Hide job URL section when starting new analysis
        function hideJobUrl() {
            var urlSection = document.getElementById('job-url-section');
            if (urlSection) urlSection.style.display = 'none';
        }

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

        // Load Engines, Bundles, and Categories
        async function loadAnalyzerData() {
            const connected = await checkAnalyzerStatus();
            if (!connected) return;

            try {
                // Load categories first for filtering
                const categoriesRes = await fetch('/api/analyzer/categories');
                if (categoriesRes.ok) {
                    categories = await categoriesRes.json();
                    renderCategoryTabs();
                }

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

                // Load pipelines (meta-engines)
                const pipelinesRes = await fetch('/api/analyzer/pipelines');
                if (pipelinesRes.ok) {
                    pipelines = await pipelinesRes.json();
                    renderPipelineTierTabs();
                    renderPipelines();
                }

                // Load pipeline tiers for display info
                const tiersRes = await fetch('/api/analyzer/pipeline-tiers');
                if (tiersRes.ok) {
                    pipelineTiers = await tiersRes.json();
                    renderPipelineTierTabs();  // Re-render with full tier info
                }

                const outputModesRes = await fetch('/api/analyzer/output-modes');
                if (outputModesRes.ok) {
                    outputModes = await outputModesRes.json();
                    renderOutputModes();
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

        // Category icons mapping
        var categoryIcons = {
            'argument': '&#128218;',    // Books
            'concepts': '&#128161;',    // Lightbulb
            'temporal': '&#128336;',    // Clock
            'power': '&#9889;',         // Lightning bolt
            'evidence': '&#128200;',    // Chart
            'rhetoric': '&#128172;',    // Speech bubble
            'epistemology': '&#129504;', // Brain
            'scholarly': '&#127891;',   // Graduation cap
            'market': '&#128176;'       // Money bag
        };

        // Render category filter tabs
        function renderCategoryTabs() {
            var container = $('category-tabs');
            if (!container) return;

            var html = '<button class="category-tab ' + (!selectedCategory ? 'active' : '') + '" ' +
                       'onclick="filterByCategory(null)">All Engines</button>';

            categories.forEach(function(cat) {
                var icon = categoryIcons[cat.category_key] || '';
                var active = selectedCategory === cat.category_key ? 'active' : '';
                html += '<button class="category-tab ' + active + '" ' +
                        'onclick="filterByCategory(\\'' + cat.category_key + '\\')" ' +
                        'title="' + (cat.description || '') + '">' +
                        '<span class="cat-icon">' + icon + '</span>' +
                        '<span class="cat-name">' + cat.name + '</span>' +
                        '<span class="cat-count">(' + cat.engine_count + ')</span>' +
                        '</button>';
            });

            container.innerHTML = html;
        }

        // Filter engines by category
        function filterByCategory(categoryKey) {
            selectedCategory = categoryKey;
            selectedEngine = null;  // Reset selection when filtering
            curatorRecommendations = null;  // Clear recommendations
            renderCategoryTabs();
            renderEngines();
            updateAnalyzeButton();
        }

        // Render Engines (with category filtering and recommendations)
        function renderEngines() {
            var grid = $('engine-grid');

            // Filter engines by selected category
            var filteredEngines = engines;
            if (selectedCategory) {
                filteredEngines = engines.filter(function(e) {
                    return e.category === selectedCategory;
                });
            }

            // If we have curator recommendations, highlight them
            var recommendedKeys = new Set();
            if (curatorRecommendations && curatorRecommendations.primary_recommendations) {
                curatorRecommendations.primary_recommendations.forEach(function(r) {
                    recommendedKeys.add(r.engine_key);
                });
            }

            $('engine-count').textContent = '(' + filteredEngines.length + (selectedCategory ? ' in category' : ' available') + ')';

            // Sort: recommended engines first, then alphabetically
            filteredEngines = filteredEngines.slice().sort(function(a, b) {
                var aRec = recommendedKeys.has(a.engine_key) ? 0 : 1;
                var bRec = recommendedKeys.has(b.engine_key) ? 0 : 1;
                if (aRec !== bRec) return aRec - bRec;
                return a.engine_name.localeCompare(b.engine_name);
            });

            grid.innerHTML = filteredEngines.map(function(e) {
                var displayName = e.engine_name || formatEngineName(e.engine_key);
                var shortDesc = truncateDesc(e.description || '', 100);
                var isRecommended = recommendedKeys.has(e.engine_key);
                var recInfo = '';
                if (isRecommended) {
                    var rec = curatorRecommendations.primary_recommendations.find(function(r) { return r.engine_key === e.engine_key; });
                    recInfo = rec ? '<div class="rec-badge" title="' + (rec.rationale || '') + '">AI Recommended (' + Math.round(rec.confidence * 100) + '%)</div>' : '';
                }
                var catIcon = categoryIcons[e.category] || '';
                return '<div class="engine-card ' + (selectedEngine === e.engine_key ? 'selected' : '') + ' ' + (isRecommended ? 'recommended' : '') + '" ' +
                'onclick="selectEngine(\\'' + e.engine_key + '\\')">' +
                recInfo +
                '<div class="name">' + catIcon + ' ' + displayName + '</div>' +
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

        // Tier names for display
        var tierNames = {
            1: 'Foundation \\u2192 Enhancement',
            2: 'Extraction \\u2192 Relational',
            3: 'Critique \\u2192 Generation',
            4: 'Multi-Stage Deep'
        };

        // Render pipeline tier filter tabs
        function renderPipelineTierTabs() {
            var tabs = $('pipeline-tier-tabs');
            if (!tabs) return;

            // Count pipelines per tier
            var tierCounts = {};
            pipelines.forEach(function(p) {
                tierCounts[p.tier] = (tierCounts[p.tier] || 0) + 1;
            });

            var html = '<div class="tier-tab ' + (selectedTier === null ? 'active' : '') + '" onclick="filterByTier(null)">' +
                'All <span class="count">(' + pipelines.length + ')</span></div>';

            for (var tier = 1; tier <= 4; tier++) {
                if (tierCounts[tier]) {
                    html += '<div class="tier-tab ' + (selectedTier === tier ? 'active' : '') + '" onclick="filterByTier(' + tier + ')">' +
                        tierNames[tier] + ' <span class="count">(' + tierCounts[tier] + ')</span></div>';
                }
            }

            tabs.innerHTML = html;
        }

        // Filter pipelines by tier
        function filterByTier(tier) {
            selectedTier = tier;
            selectedPipeline = null;
            renderPipelineTierTabs();
            renderPipelines();
            updateAnalyzeButton();
        }

        // Render pipeline cards
        function renderPipelines() {
            var list = $('pipeline-list');
            var countSpan = $('pipeline-count');
            if (!list) return;

            // Filter by tier if selected
            var filtered = selectedTier !== null
                ? pipelines.filter(function(p) { return p.tier === selectedTier; })
                : pipelines;

            if (countSpan) {
                countSpan.textContent = '(' + filtered.length + ' available)';
            }

            list.innerHTML = filtered.map(function(p) {
                // Build stage chips with arrows
                var stagesHtml = (p.engine_sequence || []).map(function(engineKey, idx) {
                    var arrow = idx < p.engine_sequence.length - 1
                        ? '<span class="stage-arrow">\\u2192</span>'
                        : '';
                    return '<span class="stage-chip">' + formatEngineName(engineKey) + '</span>' + arrow;
                }).join('');

                return '<div class="pipeline-card ' + (selectedPipeline === p.pipeline_key ? 'selected' : '') + '" ' +
                    'onclick="selectPipeline(\\'' + p.pipeline_key + '\\')">' +
                    '<div class="name">' +
                        '<span class="tier-badge">Tier ' + p.tier + '</span> ' +
                        p.pipeline_name +
                    '</div>' +
                    '<div class="desc">' + (p.description || '') + '</div>' +
                    (p.synergy_rationale ? '<div class="synergy">"' + p.synergy_rationale + '"</div>' : '') +
                    '<div class="stages">' + stagesHtml + '</div>' +
                '</div>';
            }).join('');
        }

        // Select a pipeline
        function selectPipeline(pipelineKey) {
            selectedPipeline = pipelineKey;
            selectedEngine = null;
            selectedBundle = null;
            renderPipelines();
            renderOutputModes();
            updateAnalyzeButton();
        }

        // Get sample text from documents for curator (async - reads file contents)
        async function getSampleTextForCurator() {
            var sampleParts = [];
            var maxDocs = 3;  // Sample from up to 3 docs
            var maxCharsPerDoc = 2000;

            var selectedDocObjects = scannedDocs.filter(function(doc) {
                return selectedDocs.has(doc.path);
            }).slice(0, maxDocs);

            for (var i = 0; i < selectedDocObjects.length; i++) {
                var doc = selectedDocObjects[i];
                var content = '';

                // If doc has a File object, read it
                if (doc.file && doc.file instanceof File) {
                    try {
                        var fileData = await readFileContent(doc.file);
                        if (fileData.encoding === 'base64') {
                            // For PDFs, we can't easily extract text in browser
                            // Just note it's a PDF and let the backend handle it
                            content = '[PDF: ' + doc.name + '] - Content will be extracted by analyzer';
                        } else {
                            content = fileData.content || '';
                        }
                    } catch (e) {
                        content = '[Error reading: ' + doc.name + ']';
                    }
                } else if (doc.content) {
                    // Server-scanned doc with content
                    content = doc.content;
                } else {
                    // Server-scanned doc without content - just use metadata
                    content = '[Document: ' + doc.name + ' at ' + doc.path + ']';
                }

                if (content.length > maxCharsPerDoc) {
                    content = content.substring(0, maxCharsPerDoc) + '...';
                }
                if (content) {
                    sampleParts.push(content);
                }
            }

            return sampleParts.join('\\n\\n---\\n\\n');
        }

        // Update curator button state based on document selection
        function updateCuratorButton() {
            var btn = $('curator-btn');
            if (!btn) return;
            var hasSelectedDocs = selectedDocs.size > 0;
            btn.disabled = !hasSelectedDocs;
            btn.title = hasSelectedDocs ? 'Get AI-powered engine recommendations' : 'Upload documents first';
        }

        // Get curator recommendations
        async function getCuratorRecommendations() {
            var resultDiv = $('curator-result');
            var btn = $('curator-btn');

            // Show loading state while reading files
            resultDiv.innerHTML = '<span class="loading">Reading documents...</span>';
            btn.disabled = true;

            var sampleText = await getSampleTextForCurator();
            if (!sampleText || sampleText.length < 100) {
                resultDiv.innerHTML = '<span style="color:var(--warning);">Need more document content for analysis.</span>';
                btn.disabled = false;
                return;
            }

            // Show loading state for AI analysis
            resultDiv.innerHTML = '<span class="loading">Analyzing documents to suggest best engines...</span>';

            try {
                var response = await fetch('/api/analyzer/curator/recommend', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        sample_text: sampleText,
                        max_recommendations: 6
                    })
                });

                if (!response.ok) {
                    throw new Error('Curator request failed');
                }

                curatorRecommendations = await response.json();

                // Display results
                var chars = curatorRecommendations.document_characteristics || {};
                var html = '<div class="doc-type">Document type: ' + (chars.type || 'Unknown') + '</div>';
                html += '<div>Style: ' + (chars.style || 'N/A') + ' | Focus: ' + (chars.focus || 'N/A') + '</div>';

                if (curatorRecommendations.primary_recommendations && curatorRecommendations.primary_recommendations.length > 0) {
                    html += '<div style="margin-top:0.5rem;"><strong>Top recommendations:</strong> ';
                    html += curatorRecommendations.primary_recommendations.slice(0, 4).map(function(r) {
                        return r.engine_name + ' (' + Math.round(r.confidence * 100) + '%)';
                    }).join(', ');
                    html += '</div>';
                }

                if (curatorRecommendations.analysis_strategy) {
                    html += '<div class="strategy">' + curatorRecommendations.analysis_strategy + '</div>';
                }

                resultDiv.innerHTML = html;

                // Re-render engines to show recommendations
                renderEngines();

            } catch (e) {
                console.error('Curator error:', e);
                resultDiv.innerHTML = '<span style="color:var(--error);">Recommendation failed. Try selecting specific category manually.</span>';
            }

            btn.disabled = false;
        }

        // Check if output mode is compatible with selected engine
        function isOutputModeCompatible(mode) {
            // If no engine selected, all modes are available
            if (!selectedEngine) return true;
            // If compatible_engines is empty, mode works with all engines
            if (!mode.compatible_engines || mode.compatible_engines.length === 0) return true;
            // Check if selected engine is in compatible list
            return mode.compatible_engines.includes(selectedEngine);
        }

        // Render Output Modes dropdown
        function renderOutputModes() {
            var select = $('output-mode');
            var countSpan = $('output-mode-count');

            if (!outputModes || outputModes.length === 0) {
                select.innerHTML = '<option value="">No output modes available</option>';
                return;
            }

            // Group by renderer type for better organization
            var groups = {
                'visual': [],
                'document': [],
                'structured': [],
                'other': []
            };

            var compatibleCount = 0;

            outputModes.forEach(function(mode) {
                // Add compatibility info to mode
                mode._isCompatible = isOutputModeCompatible(mode);
                if (mode._isCompatible) compatibleCount++;

                var type = mode.renderer_type || 'other';
                // Visual outputs: gemini images, mermaid diagrams, d3 interactive
                if (type === 'gemini_image' || type === 'mermaid' || type === 'd3_interactive') {
                    groups.visual.push(mode);
                // Document outputs: text-based reports and memos
                } else if (type === 'text' || type === 'document') {
                    groups.document.push(mode);
                // Structured outputs: tables, matrices
                } else if (type === 'table' || type === 'structured') {
                    groups.structured.push(mode);
                } else {
                    groups.other.push(mode);
                }
            });

            var html = '';

            // Helper to render option with compatibility check
            function renderOption(mode) {
                var label = mode.mode_name || formatEngineName(mode.mode_key);
                var requiresVisual = mode.requires_visual_llm ? ' (requires API key)' : '';
                var disabled = !mode._isCompatible ? ' disabled' : '';
                var incompatNote = !mode._isCompatible ? ' [incompatible]' : '';
                return '<option value="' + mode.mode_key + '"' + disabled + ' title="' + (mode.description || '') + '">' + label + requiresVisual + incompatNote + '</option>';
            }

            // Visual outputs first (most common)
            if (groups.visual.length > 0) {
                html += '<optgroup label="Visual Outputs">';
                groups.visual.forEach(function(mode) {
                    html += renderOption(mode);
                });
                html += '</optgroup>';
            }

            // Document outputs
            if (groups.document.length > 0) {
                html += '<optgroup label="Document Outputs">';
                groups.document.forEach(function(mode) {
                    html += renderOption(mode);
                });
                html += '</optgroup>';
            }

            // Structured outputs
            if (groups.structured.length > 0) {
                html += '<optgroup label="Structured Outputs">';
                groups.structured.forEach(function(mode) {
                    html += renderOption(mode);
                });
                html += '</optgroup>';
            }

            // Other outputs
            if (groups.other.length > 0) {
                html += '<optgroup label="Other">';
                groups.other.forEach(function(mode) {
                    html += renderOption(mode);
                });
                html += '</optgroup>';
            }

            select.innerHTML = html;

            // Show count - highlight if some are incompatible
            if (selectedEngine && compatibleCount < outputModes.length) {
                countSpan.textContent = '(' + compatibleCount + '/' + outputModes.length + ' compatible)';
            } else {
                countSpan.textContent = '(' + outputModes.length + ' available)';
            }

            // Auto-select first compatible option if current selection is incompatible
            var currentValue = select.value;
            var currentMode = outputModes.find(function(m) { return m.mode_key === currentValue; });
            if (currentMode && !currentMode._isCompatible) {
                // Find first compatible mode
                var firstCompatible = outputModes.find(function(m) { return m._isCompatible; });
                if (firstCompatible) {
                    select.value = firstCompatible.mode_key;
                }
            }
        }

        // Select Engine
        function selectEngine(key) {
            selectedEngine = key;
            selectedBundle = null;
            renderEngines();
            renderOutputModes();  // Update output modes based on engine compatibility
            updateAnalyzeButton();
        }

        // Select Bundle
        function selectBundle(key) {
            selectedBundle = key;
            selectedEngine = null;
            renderBundles();
            renderOutputModes();  // Reset output modes (all available for bundles)
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
            $('engine-mode-pipeline').classList.toggle('active', mode === 'pipeline');
            $('engine-selection').style.display = mode === 'engine' ? 'block' : 'none';
            $('bundle-selection').style.display = mode === 'bundle' ? 'block' : 'none';
            $('pipeline-selection').style.display = mode === 'pipeline' ? 'block' : 'none';
            // Show/hide category tabs (only for single engine mode)
            $('category-tabs').style.display = mode === 'engine' ? 'flex' : 'none';
            // Show/hide curator (only for single engine mode)
            $('curator-section').style.display = mode === 'engine' ? 'block' : 'none';
            updateAnalyzeButton();
        }

        // Update Analyze Button
        function updateAnalyzeButton() {
            const btn = $('analyze-btn');
            let hasSelection;
            if (engineMode === 'engine') {
                hasSelection = selectedEngine;
            } else if (engineMode === 'bundle') {
                hasSelection = selectedBundle;
            } else {
                hasSelection = selectedPipeline;
            }
            const hasDocs = selectedDocs.size > 0;

            btn.disabled = !hasSelection || !hasDocs;

            if (!hasDocs) {
                btn.textContent = 'Select Documents First';
            } else if (!hasSelection) {
                var modeLabel = engineMode === 'engine' ? 'Engine' : (engineMode === 'bundle' ? 'Bundle' : 'Pipeline');
                btn.textContent = 'Select ' + modeLabel + ' to Analyze';
            } else {
                btn.textContent = 'Analyze ' + selectedDocs.size + ' Document' + (selectedDocs.size > 1 ? 's' : '');
            }

            // Also update curator button state
            updateCuratorButton();
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
            // Check if these are Web-Saver imported docs (have content directly)
            const hasInlineContent = selectedDocObjects.some(d => d.content && !d.file);

            if (!hasBrowserFiles && !hasInlineContent) {
                // Server-scanned files - just return paths
                return { type: 'paths', file_paths: selectedPaths };
            }

            // Browser-uploaded files or Web-Saver imports - read/use contents
            $('analyze-btn').textContent = 'Reading files...';
            const documents = [];

            for (let i = 0; i < selectedDocObjects.length; i++) {
                const doc = selectedDocObjects[i];
                if (doc.file && doc.file instanceof File) {
                    // Browser-uploaded file - read from File object
                    try {
                        const { content, encoding } = await readFileContent(doc.file);
                        documents.push({
                            id: 'doc_' + (i + 1),
                            title: doc.name,
                            content: content,
                            encoding: encoding
                        });
                    } catch (e) {
                        console.error('Failed to read file:', doc.name, e);
                    }
                } else if (doc.content) {
                    // Web-Saver imported doc - content already available
                    documents.push({
                        id: 'doc_' + (i + 1),
                        title: doc.name,
                        content: doc.content,
                        encoding: 'text'
                    });
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
            resetProgressDetails();
            hideJobUrl();  // Hide previous job URL

            try {
                // Get documents (either paths or inline content)
                const docData = await getDocumentsForAnalysis();

                // Track document count and names for progress display
                if (docData.type === 'paths') {
                    currentDocCount = docData.file_paths.length;
                    // Extract filenames from paths
                    currentDocNames = docData.file_paths.map(p => p.split('/').pop());
                } else {
                    currentDocCount = docData.documents.length;
                    currentDocNames = docData.documents.map(d => d.title);
                }
                currentDocIndex = 0;

                // Show initial document info
                updateProgressDetails(0, currentDocCount, currentDocNames[0] || '');

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
                        headers: getApiHeaders(),
                        body: JSON.stringify(payload)
                    });
                } else if (engineMode === 'bundle') {
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
                        headers: getApiHeaders(),
                        body: JSON.stringify(payload)
                    });
                } else {
                    // Pipeline mode
                    const payload = {
                        pipeline: selectedPipeline,
                        output_mode: outputMode,
                        include_intermediate_outputs: true
                    };

                    if (docData.type === 'paths') {
                        payload.file_paths = docData.file_paths;
                    } else {
                        payload.documents = docData.documents;
                    }

                    response = await fetch('/api/analyzer/analyze/pipeline', {
                        method: 'POST',
                        headers: getApiHeaders(),
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
                        // Update browser URL to stable job URL
                        updateJobUrl(data.job_id);
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

        // Poll Job Status
        let pollCount = 0;
        async function pollJobStatus(jobId) {
            try {
                const res = await fetch('/api/analyzer/jobs/' + jobId, { headers: getApiHeaders() });
                const job = await res.json();

                // Add doc count to job for display
                job.doc_count = currentDocCount;
                pollCount++;

                // Warn if stuck in queued for too long (>10 polls = 20+ seconds)
                if ((job.status === 'pending' || job.status === 'queued') && pollCount > 10) {
                    job.stuckWarning = true;
                }

                updateProgress(job);

                if (job.status === 'completed') {
                    pollCount = 0;
                    await fetchAndDisplayResult(jobId);
                } else if (job.status === 'failed') {
                    pollCount = 0;
                    showAnalysisError(job.error_message || 'Analysis failed');
                } else {
                    setTimeout(function() { pollJobStatus(jobId); }, 2000);
                }
            } catch (e) {
                pollCount = 0;
                showAnalysisError('Error polling status: ' + e.message);
            }
        }

        // Poll Multiple Jobs
        async function pollMultipleJobs(jobs) {
            const pending = jobs.filter(function(j) { return j.status === 'submitted'; });
            let allDone = true;
            let currentProcessingJob = null;

            for (const job of pending) {
                if (!job.job_id) continue;

                try {
                    const res = await fetch('/api/analyzer/jobs/' + job.job_id, { headers: getApiHeaders() });
                    const status = await res.json();

                    if (status.status === 'completed') {
                        job.status = 'completed';
                        const resultRes = await fetch('/api/analyzer/jobs/' + job.job_id + '/result', { headers: getApiHeaders() });
                        const result = await resultRes.json();
                        displayResult(result, job.title);
                    } else if (status.status === 'failed') {
                        job.status = 'failed';
                        displayError(job.title, status.error_message);
                    } else {
                        allDone = false;
                        // Track which job is currently being processed
                        if (!currentProcessingJob) {
                            currentProcessingJob = job;
                        }
                    }
                } catch (e) {
                    job.status = 'failed';
                }
            }

            const completed = jobs.filter(function(j) { return j.status === 'completed' || j.status === 'failed'; }).length;
            updateProgressMulti(completed, jobs.length, currentProcessingJob);

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

            // Debug: log job status
            console.log('Job status update:', {
                status: job.status,
                stage: job.current_stage,
                percent: percent,
                stages_completed: job.stages_completed,
                pipeline_stages_completed: job.pipeline_stages_completed,
                total_pipeline_stages: job.total_pipeline_stages
            });

            // Build status text - normalize to lowercase for comparison
            let statusText = '';
            const stage = job.current_stage ? String(job.current_stage).toLowerCase() : '';
            const status = job.status ? String(job.status).toLowerCase() : 'pending';
            const docCount = currentDocCount > 0 ? currentDocCount : '';

            // Check if this is a pipeline job
            const isPipeline = engineMode === 'pipeline' || job.total_pipeline_stages > 0;
            const pipelineStagesCompleted = job.pipeline_stages_completed || 0;
            const totalPipelineStages = job.total_pipeline_stages || 0;

            // Update progress counter (document count)
            if (docCount) {
                $('progress-counter').textContent = docCount + ' document' + (docCount > 1 ? 's' : '');
            }

            // Show document names on the right (or pipeline info for pipeline mode)
            if (isPipeline && selectedPipeline) {
                var pipeline = pipelines.find(function(p) { return p.pipeline_key === selectedPipeline; });
                if (pipeline) {
                    $('progress-doc-name').textContent = pipeline.pipeline_name;
                }
            } else if (currentDocNames.length > 0) {
                if (currentDocNames.length === 1) {
                    $('progress-doc-name').textContent = currentDocNames[0];
                } else if (currentDocNames.length <= 3) {
                    $('progress-doc-name').textContent = currentDocNames.join(', ');
                } else {
                    $('progress-doc-name').textContent = currentDocNames.slice(0, 2).join(', ') + ' +' + (currentDocNames.length - 2) + ' more';
                }
            }

            // Reset text color
            $('progress-text').style.color = '';

            // Match status or stage (API may return either) and set status text
            if (status === 'pending' || status === 'queued') {
                if (job.stuckWarning) {
                    statusText = 'Still queued - is the analyzer worker running?';
                    $('progress-text').style.color = 'var(--error)';
                } else {
                    statusText = 'Queued... waiting to start';
                }
            } else if (isPipeline && totalPipelineStages > 0) {
                // Pipeline-specific progress display
                var pipelineStageNum = pipelineStagesCompleted + 1;
                if (stage.includes('extract') || status.includes('extract')) {
                    statusText = 'Pipeline ' + pipelineStageNum + '/' + totalPipelineStages + ': Extracting (' + percent + '%)';
                } else if (stage.includes('curat') || status.includes('curat')) {
                    statusText = 'Pipeline ' + pipelineStageNum + '/' + totalPipelineStages + ': Curating (' + percent + '%)';
                } else if (stage.includes('concret') || status.includes('concret')) {
                    statusText = 'Finalizing: Refining labels (' + percent + '%)';
                } else if (stage.includes('render') || status.includes('render')) {
                    statusText = 'Finalizing: Generating outputs (' + percent + '%)';
                } else {
                    statusText = 'Pipeline stage ' + pipelineStageNum + '/' + totalPipelineStages + ' (' + percent + '%)';
                }
            } else if (stage.includes('extract') || status.includes('extract')) {
                statusText = 'Stage 1/4: Extracting content (' + percent + '%)';
            } else if (stage.includes('curat') || status.includes('curat')) {
                statusText = 'Stage 2/4: Curating insights (' + percent + '%)';
            } else if (stage.includes('concret') || status.includes('concret')) {
                statusText = 'Stage 3/4: Refining labels (' + percent + '%)';
            } else if (stage.includes('render') || status.includes('render')) {
                statusText = 'Stage 4/4: Generating output (' + percent + '%)';
            } else if (status === 'completed') {
                statusText = 'Complete!';
            } else if (status === 'failed') {
                statusText = 'Failed';
            } else if (stage || status) {
                // Capitalize first letter for display
                const displayStage = stage || status;
                statusText = displayStage.charAt(0).toUpperCase() + displayStage.slice(1) + '... (' + percent + '%)';
            } else {
                statusText = 'Processing... (' + percent + '%)';
            }

            $('progress-text').textContent = statusText;

            // Update stage badges
            const stages = ['extraction', 'curation', 'concretization', 'rendering'];
            // Normalize stages_completed to lowercase strings
            const stagesCompleted = (job.stages_completed || []).map(s => String(s).toLowerCase());
            const currentStage = stage || '';

            stages.forEach(function(stageName, i) {
                const el = $('stage-' + stageName);
                if (!el) return;

                // Check if completed (includes partial match for flexibility)
                const isCompleted = stagesCompleted.some(s => s.includes(stageName.substring(0, 5)));
                // Check if current (includes partial match)
                const isCurrent = currentStage.includes(stageName.substring(0, 5));

                if (isCompleted) {
                    el.className = 'stage-badge completed';
                } else if (isCurrent) {
                    el.className = 'stage-badge active';
                } else {
                    el.className = 'stage-badge';
                }
            });

            // Display processing warnings (validation fallbacks, JSON repairs, etc.)
            const warningsEl = $('progress-warnings');
            if (warningsEl && job.warnings && job.warnings.length > 0) {
                const count = job.warnings.length;
                let html = '<span class="warning-count">⚠️ ' + count + ' warning' + (count > 1 ? 's' : '') + ' during processing</span>';
                html += '<div class="warning-list">';
                // Show up to 10 warnings
                job.warnings.slice(0, 10).forEach(function(w) {
                    const typeLabel = (w.type || 'warning').replace(/_/g, ' ');
                    const context = w.context ? ': ' + w.context : '';
                    html += '<div class="warning-item">' + typeLabel + context + '</div>';
                });
                if (count > 10) {
                    html += '<div class="warning-item">... and ' + (count - 10) + ' more</div>';
                }
                html += '</div>';
                warningsEl.innerHTML = html;
                warningsEl.style.display = 'block';
            } else if (warningsEl) {
                warningsEl.style.display = 'none';
            }
        }

        function updateProgressMulti(completed, total, currentJob) {
            const percent = Math.round((completed / total) * 100);
            $('progress-bar').style.width = percent + '%';
            $('progress-text').textContent = completed + ' of ' + total + ' documents processed';

            // Update document details
            $('progress-counter').textContent = 'Document ' + (completed + 1) + ' of ' + total;
            if (currentJob && currentJob.title) {
                $('progress-doc-name').textContent = currentJob.title;
            }
        }

        function resetStages() {
            ['extraction', 'curation', 'concretization', 'rendering'].forEach(function(s) {
                const el = $('stage-' + s);
                if (el) el.className = 'stage-badge';
            });
            // Reset warnings display
            const warningsEl = $('progress-warnings');
            if (warningsEl) {
                warningsEl.style.display = 'none';
                warningsEl.innerHTML = '';
            }
        }

        function resetProgressDetails() {
            $('progress-counter').textContent = '';
            $('progress-doc-name').textContent = '';
            currentDocNames = [];
            currentDocIndex = 0;
        }

        function updateProgressDetails(current, total, docName) {
            if (total > 0) {
                // For single collection mode, show overall count
                // For individual mode, show current/total
                if (collectionMode === 'single' && current === 0) {
                    $('progress-counter').textContent = total + ' document' + (total > 1 ? 's' : '');
                } else if (total > 1) {
                    $('progress-counter').textContent = 'Document ' + (current + 1) + ' of ' + total;
                } else {
                    $('progress-counter').textContent = '1 document';
                }
            }
            if (docName) {
                $('progress-doc-name').textContent = docName;
            }
        }

        // Fetch and Display Result
        async function fetchAndDisplayResult(jobId) {
            try {
                const res = await fetch('/api/analyzer/jobs/' + jobId + '/result', { headers: getApiHeaders() });
                const result = await res.json();
                displayResult(result);
                finishAnalysis();
            } catch (e) {
                showAnalysisError('Error fetching result: ' + e.message);
            }
        }

        // Display Result
        function displayResult(result, title) {
            console.log('displayResult called with:', result);

            var gallery = $('results-gallery');
            var grid = $('results-grid');
            var countEl = $('results-count');

            gallery.style.display = 'block';

            var outputs = result.outputs || {};
            var metadata = result.metadata || {};
            var count = 0;

            console.log('Outputs:', Object.keys(outputs));

            for (var key in outputs) {
                var output = outputs[key];
                count++;

                console.log('Processing output:', key, 'has image_url:', !!output.image_url, 'url length:', output.image_url ? output.image_url.length : 0);

                var resultData = {
                    key: key,
                    title: (title ? title + ' - ' : '') + key.replace(/_/g, ' '),
                    job_id: currentJobId,  // Include job_id for library grouping
                    output: output,
                    metadata: metadata,
                    isImage: !!output.image_url,
                    imageUrl: output.image_url || null,
                    content: output.content || output.html_content || '',
                    data: output.data || null,
                    isInteractive: !!output.html_content
                };
                allResults.push(resultData);

                // Try to add to library but don't let failures block display
                try {
                    addToLibrary(resultData);
                } catch (e) {
                    console.warn('Failed to add to library:', e);
                }

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
                console.log('Creating image element, URL length:', data.imageUrl.length);
                var img = document.createElement('img');
                if (data.imageUrl.startsWith('/static/')) {
                    img.src = 'http://localhost:8847' + data.imageUrl;
                } else {
                    img.src = data.imageUrl;
                }
                img.alt = data.title;
                img.onload = function() {
                    console.log('Image loaded successfully:', data.title);
                };
                img.onerror = function() {
                    console.error('Image failed to load:', data.title);
                    this.style.display = 'none';
                    var icon = document.createElement('div');
                    icon.className = 'icon-preview';
                    icon.innerHTML = '&#128444;';
                    preview.appendChild(icon);
                };
                preview.appendChild(img);
            } else if (data.isInteractive && data.content) {
                // Interactive content (D3, Mermaid) - show icon preview
                var icon = document.createElement('div');
                icon.className = 'icon-preview';
                icon.innerHTML = '&#128202;';  // Chart emoji
                icon.title = 'Interactive visualization - click to view';
                preview.appendChild(icon);
                var interLabel = document.createElement('div');
                interLabel.style.cssText = 'font-size: 0.7rem; color: var(--accent); margin-top: 0.5rem;';
                interLabel.textContent = 'Interactive';
                preview.appendChild(interLabel);
            } else if (data.content && isHtmlContent(data.content)) {
                // HTML content - render scaled preview
                var htmlPre = document.createElement('div');
                htmlPre.className = 'html-preview';
                htmlPre.innerHTML = data.content;
                preview.appendChild(htmlPre);
            } else if (data.content) {
                var textPre = document.createElement('div');
                textPre.className = 'text-preview';
                var cleanText = cleanTextForPreview(data.content);
                textPre.textContent = cleanText.substring(0, 400) + (cleanText.length > 400 ? '...' : '');
                preview.appendChild(textPre);
            } else if (data.data) {
                var jsonPre = document.createElement('div');
                jsonPre.className = 'text-preview json-preview';
                jsonPre.textContent = formatJsonPreview(data.data);
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

        function isMarkdown(text) {
            // Check if text looks like markdown
            if (!text) return false;
            // Has headers, or starts with # title, or has markdown links/emphasis
            return /^#+ /m.test(text) || /\\[.+\\]\\(.+\\)/.test(text) || /\\*\\*.+\\*\\*/.test(text);
        }

        function isHtmlContent(text) {
            // Check if text looks like HTML (tables, divs, etc)
            if (!text) return false;
            return /<(table|div|html|body|p|h[1-6]|ul|ol|span)[^>]*>/i.test(text);
        }

        function cleanTextForPreview(text) {
            if (!text) return '';
            // Simple cleanup - remove # headers and ** bold
            return text
                .replace(/^#+ /gm, '')
                .replace(/[*][*]/g, '')
                .replace(/^---+$/gm, '')
                .trim();
        }

        function formatJsonPreview(data) {
            // Create a cleaner summary of JSON data
            if (!data) return '';
            if (Array.isArray(data)) {
                var count = data.length;
                var sample = data.slice(0, 2).map(function(item) {
                    if (typeof item === 'object' && item !== null) {
                        var keys = Object.keys(item).slice(0, 3);
                        return keys.join(', ');
                    }
                    return String(item).substring(0, 30);
                });
                return count + ' items: ' + sample.join(' | ') + (count > 2 ? '...' : '');
            }
            if (typeof data === 'object') {
                var keys = Object.keys(data).slice(0, 5);
                return keys.map(function(k) {
                    var v = data[k];
                    if (typeof v === 'string') return k + ': ' + v.substring(0, 40);
                    if (typeof v === 'number') return k + ': ' + v;
                    if (Array.isArray(v)) return k + ': [' + v.length + ' items]';
                    return k + ': {...}';
                }).join('\\n');
            }
            return String(data);
        }

        function formatSourcesFootnotes(container) {
            // Find the Sources heading and format footnotes in the following paragraph
            var headings = container.querySelectorAll('h2');
            headings.forEach(function(h2) {
                if (h2.textContent.toLowerCase().includes('source')) {
                    // Get the next sibling paragraph(s)
                    var sibling = h2.nextElementSibling;
                    while (sibling && sibling.tagName === 'P') {
                        // Split on footnote markers (¹²³⁴⁵⁶⁷⁸⁹⁰) and reformat
                        var html = sibling.innerHTML;
                        // Add line break before each footnote number (except first)
                        html = html.replace(/([^\\s])\\s*([¹²³⁴⁵⁶⁷⁸⁹⁰]+)\\s+/g, '$1<br><br>$2 ');
                        // Remove leading br if present
                        html = html.replace(/^<br><br>/, '');
                        sibling.innerHTML = html;
                        sibling = sibling.nextElementSibling;
                    }
                }
            });
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

            var isMarkdownContent = false;

            if (data.isImage && data.imageUrl) {
                var img = document.createElement('img');
                if (data.imageUrl.startsWith('/static/')) {
                    img.src = 'http://localhost:8847' + data.imageUrl;
                } else {
                    img.src = data.imageUrl;
                }
                img.alt = data.title;
                body.appendChild(img);
            } else if (data.isInteractive && data.content) {
                // Interactive content (D3, Mermaid) - use iframe for script execution
                content.classList.add('wide');
                content.classList.add('interactive-view');
                var iframe = document.createElement('iframe');
                iframe.className = 'interactive-iframe';
                iframe.style.cssText = 'width: 100%; height: 70vh; border: none; background: white; border-radius: 8px;';
                iframe.srcdoc = data.content;
                body.appendChild(iframe);
            } else if (data.content && isHtmlContent(data.content)) {
                // Render as HTML (tables, etc)
                isMarkdownContent = true;
                content.classList.add('wide');
                body.classList.add('markdown-view');
                var htmlContainer = document.createElement('div');
                htmlContainer.className = 'markdown-body';
                htmlContainer.innerHTML = data.content;
                body.appendChild(htmlContainer);
            } else if (data.content && isMarkdown(data.content)) {
                // Render as formatted markdown
                isMarkdownContent = true;
                content.classList.add('wide');
                body.classList.add('markdown-view');
                var mdContainer = document.createElement('div');
                mdContainer.className = 'markdown-body';
                mdContainer.innerHTML = marked.parse(data.content);

                // Format footnotes in Sources section - each on its own line
                formatSourcesFootnotes(mdContainer);

                body.appendChild(mdContainer);
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

            // Copy button for text content
            if (data.content || data.data) {
                var copyBtn = document.createElement('button');
                copyBtn.className = 'btn';
                copyBtn.style.width = 'auto';
                copyBtn.textContent = 'Copy';
                copyBtn.onclick = function() {
                    var textToCopy = data.content || JSON.stringify(data.data, null, 2);
                    navigator.clipboard.writeText(textToCopy).then(function() {
                        copyBtn.textContent = 'Copied!';
                        copyBtn.classList.add('btn-copied');
                        setTimeout(function() {
                            copyBtn.textContent = 'Copy';
                            copyBtn.classList.remove('btn-copied');
                        }, 2000);
                    });
                };
                actions.appendChild(copyBtn);
            }

            // Open in New Tab button for interactive content
            if (data.isInteractive && data.content) {
                var newTabBtn = document.createElement('button');
                newTabBtn.className = 'btn';
                newTabBtn.style.width = 'auto';
                newTabBtn.style.background = '#4fc3f7';
                newTabBtn.style.color = '#1a1a2e';
                newTabBtn.textContent = '↗ Open Full Screen';
                newTabBtn.onclick = function() {
                    var newWindow = window.open('', '_blank');
                    newWindow.document.write(data.content);
                    newWindow.document.close();
                };
                actions.appendChild(newTabBtn);
            }

            var dlBtn = document.createElement('button');
            dlBtn.className = 'btn btn-primary';
            dlBtn.style.width = 'auto';
            dlBtn.textContent = 'DOWNLOAD';
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
                // Handle data URLs (base64 embedded images)
                if (url.startsWith('data:')) {
                    var a = document.createElement('a');
                    a.href = url;
                    a.download = data.key.replace(/_/g, '-') + '.png';
                    document.body.appendChild(a);
                    a.click();
                    document.body.removeChild(a);
                    return;
                }
                window.open(url, '_blank');
                return;
            }

            var content, filename, mimeType;
            if (data.isInteractive && data.content) {
                // Interactive HTML content
                content = data.content;
                filename = data.key.replace(/_/g, '-') + '.html';
                mimeType = 'text/html';
            } else if (data.content) {
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

        // Cache for recent jobs - avoid re-fetching too often
        let recentJobsCache = { jobs: [], lastFetch: 0 };
        const CACHE_TTL_MS = 60000; // 1 minute cache

        async function loadRecentJobs(forceRefresh = false) {
            console.log('Loading recent jobs from server...');

            // Check cache first (unless force refresh)
            const now = Date.now();
            if (!forceRefresh && recentJobsCache.jobs.length > 0 && (now - recentJobsCache.lastFetch) < CACHE_TTL_MS) {
                console.log('Using cached jobs (' + recentJobsCache.jobs.length + ' items, ' + Math.round((now - recentJobsCache.lastFetch)/1000) + 's old)');
                // Just merge cached items into library
                recentJobsCache.jobs.forEach(function(item) {
                    var exists = libraryItems.some(function(it) {
                        return it.job_id === item.job_id && it.key === item.key;
                    });
                    if (!exists) libraryItems.push(item);
                });
                renderLibrary();
                return;
            }

            // Show loading state
            const btn = document.querySelector('button[onclick*="loadRecentJobs"]');
            const originalText = btn ? btn.textContent : '';
            if (btn) {
                btn.textContent = 'Loading...';
                btn.disabled = true;
            }

            try {
                const res = await fetch('/api/analyzer/jobs');
                const data = await res.json();
                console.log('Found jobs:', data.jobs?.length || 0);

                if (!data.jobs || data.jobs.length === 0) {
                    alert('No completed jobs found on server');
                    return;
                }

                // Clear cache for fresh fetch
                recentJobsCache.jobs = [];

                // Fetch results for each completed job
                let loaded = 0;
                for (const job of data.jobs.slice(0, 10)) {
                    if (btn) btn.textContent = 'Loading ' + (++loaded) + '/10...';
                    console.log('Fetching result for job:', job.job_id);
                    try {
                        const resultRes = await fetch('/api/analyzer/jobs/' + job.job_id + '/result');
                        const result = await resultRes.json();

                        if (result.outputs) {
                            for (var key in result.outputs) {
                                var output = result.outputs[key];
                                var resultData = {
                                    key: key,
                                    title: key.replace(/_/g, ' '),
                                    job_id: job.job_id,
                                    output: output,
                                    metadata: result.metadata || {},
                                    isImage: !!output.image_url,
                                    imageUrl: output.image_url || null,
                                    content: output.content || output.html_content || '',
                                    data: output.data || null,
                                    addedAt: job.completed_at || new Date().toISOString(),
                                    isInteractive: !!output.html_content
                                };

                                // Add to cache
                                recentJobsCache.jobs.push(resultData);

                                // Check if already in library
                                var exists = libraryItems.some(function(it) {
                                    return it.job_id === resultData.job_id && it.key === key;
                                });

                                if (!exists) {
                                    libraryItems.push(resultData);
                                }
                            }
                        }
                    } catch (e) {
                        console.warn('Failed to fetch result for job', job.job_id, e);
                    }
                }

                // Update cache timestamp
                recentJobsCache.lastFetch = Date.now();

                renderLibrary();
                console.log('Library now has', libraryItems.length, 'items (cached ' + recentJobsCache.jobs.length + ')');
            } catch (e) {
                console.error('Failed to load recent jobs:', e);
                alert('Failed to load jobs: ' + e.message);
            } finally {
                // Restore button
                if (btn) {
                    btn.textContent = originalText;
                    btn.disabled = false;
                }
            }
        }

        function addToLibrary(item) {
            item.addedAt = new Date().toISOString();

            // Add original item to in-memory array for display
            libraryItems.unshift(item);
            if (libraryItems.length > 100) libraryItems = libraryItems.slice(0, 100);

            // Clone for localStorage - don't store huge data URLs
            var storageItems = libraryItems.map(function(it) {
                if (it.imageUrl && it.imageUrl.startsWith('data:') && it.imageUrl.length > 100000) {
                    var clone = Object.assign({}, it);
                    clone.imageUrl = null;
                    clone.imageTooLarge = true;
                    return clone;
                }
                return it;
            });

            try {
                localStorage.setItem('visualizer_library', JSON.stringify(storageItems));
            } catch (e) {
                console.warn('Failed to save to library (storage full?):', e);
            }
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

            // Group items by job_id
            var groups = {};
            var ungrouped = [];

            libraryItems.forEach(function(item, index) {
                item._libraryIndex = index;  // Store original index for deletion
                if (item.job_id) {
                    if (!groups[item.job_id]) {
                        groups[item.job_id] = {
                            items: [],
                            addedAt: item.addedAt,
                            metadata: item.metadata || {}
                        };
                    }
                    groups[item.job_id].items.push(item);
                    // Use earliest addedAt for the group
                    if (item.addedAt && item.addedAt < groups[item.job_id].addedAt) {
                        groups[item.job_id].addedAt = item.addedAt;
                    }
                } else {
                    ungrouped.push(item);
                }
            });

            // Sort groups by date (newest first)
            var sortedJobIds = Object.keys(groups).sort(function(a, b) {
                return new Date(groups[b].addedAt) - new Date(groups[a].addedAt);
            });

            // Render grouped items
            sortedJobIds.forEach(function(jobId) {
                var group = groups[jobId];
                var groupEl = createJobGroup(jobId, group);
                grid.appendChild(groupEl);
            });

            // Render ungrouped items (legacy items without job_id)
            if (ungrouped.length > 0) {
                var ungroupedSection = document.createElement('div');
                ungroupedSection.className = 'job-group';

                var header = document.createElement('div');
                header.className = 'job-group-header';
                header.innerHTML =
                    '<span class="job-group-toggle">▼</span>' +
                    '<span class="job-group-title">Ungrouped Items</span>' +
                    '<span class="job-group-count">' + ungrouped.length + ' items</span>' +
                    '<button class="job-group-delete" onclick="event.stopPropagation(); clearUngrouped()" title="Clear all ungrouped">🗑</button>';
                header.onclick = function(e) {
                    if (e.target.tagName !== 'BUTTON') {
                        ungroupedSection.classList.toggle('collapsed');
                    }
                };

                var itemsContainer = document.createElement('div');
                itemsContainer.className = 'job-group-items';

                ungrouped.forEach(function(item) {
                    var card = createLibraryCard(item, item._libraryIndex);
                    itemsContainer.appendChild(card);
                });

                ungroupedSection.appendChild(header);
                ungroupedSection.appendChild(itemsContainer);
                grid.appendChild(ungroupedSection);
            }
        }

        function createJobGroup(jobId, group) {
            var groupEl = document.createElement('div');
            groupEl.className = 'job-group';
            groupEl.dataset.jobId = jobId;

            // Deduplicate items by key (keep the latest one)
            var seen = {};
            var uniqueItems = [];
            group.items.forEach(function(item) {
                var key = item.key || item.title;
                if (!seen[key]) {
                    seen[key] = true;
                    uniqueItems.push(item);
                }
            });

            // Get pipeline/engine info from first item
            var firstItem = uniqueItems[0];
            var pipelineName = '';
            if (firstItem.metadata && firstItem.metadata.pipeline) {
                pipelineName = firstItem.metadata.pipeline.replace(/_/g, ' ');
            } else if (firstItem.metadata && firstItem.metadata.engine) {
                pipelineName = firstItem.metadata.engine.replace(/_/g, ' ');
            } else {
                // Try to infer from unique output keys
                var keys = uniqueItems.map(function(i) { return i.key || i.title; });
                pipelineName = keys.slice(0, 3).join(' + ').replace(/_/g, ' ');
                if (keys.length > 3) pipelineName += ' +' + (keys.length - 3) + ' more';
            }

            var dateStr = group.addedAt ? new Date(group.addedAt).toLocaleDateString() : '';

            var header = document.createElement('div');
            header.className = 'job-group-header';
            header.innerHTML =
                '<span class="job-group-toggle">▼</span>' +
                '<span class="job-group-title" title="' + pipelineName.replace(/"/g, '&quot;') + '">' + pipelineName + '</span>' +
                '<span class="job-group-count">' + uniqueItems.length + ' outputs</span>' +
                '<span class="job-group-date">' + dateStr + '</span>' +
                '<button class="job-group-delete" onclick="event.stopPropagation(); deleteJob(\'' + jobId + '\')" title="Delete job">🗑</button>';

            header.onclick = function(e) {
                if (e.target.tagName !== 'BUTTON') {
                    groupEl.classList.toggle('collapsed');
                }
            };

            var itemsContainer = document.createElement('div');
            itemsContainer.className = 'job-group-items';

            uniqueItems.forEach(function(item) {
                var card = createLibraryCard(item, item._libraryIndex);
                itemsContainer.appendChild(card);
            });

            groupEl.appendChild(header);
            groupEl.appendChild(itemsContainer);

            return groupEl;
        }

        // Delete entire job from library
        function deleteJob(jobId) {
            if (!confirm('Delete all outputs from this job?')) return;

            // Remove all items with this job_id
            libraryItems = libraryItems.filter(function(item) {
                return item.job_id !== jobId;
            });

            // Update localStorage
            try {
                localStorage.setItem('vizLibrary', JSON.stringify(libraryItems));
            } catch (e) {
                console.error('Failed to save library:', e);
            }

            renderLibrary();
        }

        // Clear all ungrouped items
        function clearUngrouped() {
            if (!confirm('Delete all ungrouped items?')) return;

            libraryItems = libraryItems.filter(function(item) {
                return item.job_id;  // Keep only items WITH job_id
            });

            try {
                localStorage.setItem('vizLibrary', JSON.stringify(libraryItems));
            } catch (e) {
                console.error('Failed to save library:', e);
            }

            renderLibrary();
        }

        function viewFullJob(jobId) {
            // Find all items for this job and display them in the modal
            var jobItems = libraryItems.filter(function(item) {
                return item.job_id === jobId;
            });

            if (jobItems.length === 0) {
                alert('No items found for job ' + jobId);
                return;
            }

            // Set up allResults for the modal navigation
            allResults = jobItems;
            openResultModal(0);
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
            } else if (data.isImage && data.imageTooLarge) {
                // Large image that couldn't be stored - show placeholder
                var icon = document.createElement('div');
                icon.className = 'icon-preview';
                icon.innerHTML = '&#128444;';
                icon.title = 'Image too large to store in library';
                preview.appendChild(icon);
                var label = document.createElement('div');
                label.style.cssText = 'font-size: 0.7rem; color: var(--text-muted); margin-top: 0.5rem;';
                label.textContent = '(Image not cached)';
                preview.appendChild(label);
            } else if (data.isInteractive && data.content) {
                // Interactive content (D3, Mermaid) - show icon preview
                var icon = document.createElement('div');
                icon.className = 'icon-preview';
                icon.innerHTML = '&#128202;';  // Chart emoji
                icon.title = 'Interactive visualization - click to view';
                preview.appendChild(icon);
                var interLabel = document.createElement('div');
                interLabel.style.cssText = 'font-size: 0.7rem; color: var(--accent); margin-top: 0.5rem;';
                interLabel.textContent = 'Interactive';
                preview.appendChild(interLabel);
            } else if (data.content && isHtmlContent(data.content)) {
                // HTML content - render scaled preview
                var htmlPre = document.createElement('div');
                htmlPre.className = 'html-preview';
                htmlPre.innerHTML = data.content;
                preview.appendChild(htmlPre);
            } else if (data.content) {
                var textPre = document.createElement('div');
                textPre.className = 'text-preview';
                var cleanText = cleanTextForPreview(data.content);
                textPre.textContent = cleanText.substring(0, 300) + (cleanText.length > 300 ? '...' : '');
                preview.appendChild(textPre);
            } else if (data.data) {
                var jsonPre = document.createElement('div');
                jsonPre.className = 'text-preview json-preview';
                jsonPre.textContent = formatJsonPreview(data.data);
                preview.appendChild(jsonPre);
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
            var typeLabel = data.isImage ? 'Image' : (data.isInteractive ? 'Interactive' : 'Text');
            meta.innerHTML = '<span>' + typeLabel + '</span>';
            if (data.addedAt) {
                meta.innerHTML += '<span>' + new Date(data.addedAt).toLocaleDateString() + '</span>';
            }
            info.appendChild(meta);

            // Add action buttons
            var actions = document.createElement('div');
            actions.className = 'gallery-card-actions';

            var viewBtn = document.createElement('button');
            viewBtn.textContent = 'View';
            viewBtn.onclick = function(e) {
                e.stopPropagation();
                allResults = [data];
                openResultModal(0);
            };

            var deleteBtn = document.createElement('button');
            deleteBtn.textContent = 'Delete';
            deleteBtn.className = 'btn-delete';
            deleteBtn.onclick = function(e) {
                e.stopPropagation();
                deleteFromLibrary(index);
            };

            actions.appendChild(viewBtn);
            actions.appendChild(deleteBtn);

            card.appendChild(preview);
            card.appendChild(info);
            card.appendChild(actions);

            card.onclick = function() {
                allResults = [data];
                openResultModal(0);
            };

            return card;
        }

        async function deleteFromLibrary(index) {
            if (confirm('Delete this item from library?')) {
                var item = libraryItems[index];

                // If item has a job_id, also delete from server
                if (item && item.job_id) {
                    try {
                        const res = await fetch('/api/analyzer/jobs/' + item.job_id, {
                            method: 'DELETE'
                        });
                        if (res.ok) {
                            console.log('Deleted job from server:', item.job_id);
                        } else {
                            console.warn('Failed to delete from server:', await res.text());
                        }
                    } catch (e) {
                        console.warn('Error deleting from server:', e);
                    }
                }

                // Remove from local library
                libraryItems.splice(index, 1);
                localStorage.setItem('visualizer_library', JSON.stringify(libraryItems));
                renderLibrary();
            }
        }

        // ===== DEBUG PANEL =====
        var debugEnabled = false;
        var debugInterval = null;

        function toggleDebugPanel() {
            debugEnabled = !debugEnabled;
            var toggle = document.getElementById('debug-toggle');
            var panel = document.getElementById('debug-panel');

            if (debugEnabled) {
                toggle.classList.add('active');
                toggle.innerHTML = '🔧 Debug ON';
                panel.classList.add('show');
                fetchDebugStatus();
                debugInterval = setInterval(fetchDebugStatus, 3000);
            } else {
                toggle.classList.remove('active');
                toggle.innerHTML = '🔧 Debug';
                panel.classList.remove('show');
                if (debugInterval) {
                    clearInterval(debugInterval);
                    debugInterval = null;
                }
            }
        }

        function fetchDebugStatus() {
            var jobId = currentJobId || null;
            var url = '/api/admin/debug-status';
            if (jobId) {
                url += '?job_id=' + jobId;
            }

            fetch(url)
                .then(function(response) { return response.json(); })
                .then(function(data) {
                    renderDebugPanel(data);
                })
                .catch(function(error) {
                    console.error('Debug fetch error:', error);
                    document.getElementById('debug-content').innerHTML =
                        '<div class="debug-section"><div class="debug-value error">Error fetching debug status: ' + error.message + '</div></div>';
                });
        }

        function renderDebugPanel(data) {
            var content = document.getElementById('debug-content');
            var html = '';

            // Queue Status
            html += '<div class="debug-section">';
            html += '<div class="debug-section-title">📊 Job Queue</div>';
            if (data.queue) {
                var queueKeys = ['todo', 'doing', 'succeeded', 'failed'];
                queueKeys.forEach(function(key) {
                    var count = data.queue[key] || 0;
                    var valueClass = '';
                    if (key === 'doing' && count > 0) valueClass = 'warning';
                    if (key === 'failed' && count > 0) valueClass = 'error';
                    if (key === 'succeeded') valueClass = 'success';
                    html += '<div class="debug-item"><span class="debug-key">' + key + '</span><span class="debug-value ' + valueClass + '">' + count + '</span></div>';
                });
            }
            html += '</div>';

            // Orphaned Jobs Warning
            if (data.orphaned_jobs && data.orphaned_jobs > 0) {
                html += '<div class="debug-section">';
                html += '<div class="debug-section-title">⚠️ Orphaned Jobs</div>';
                html += '<div class="debug-item"><span class="debug-key">Stuck in "doing"</span><span class="debug-value error">' + data.orphaned_jobs + '</span></div>';
                html += '<button onclick="cleanupOrphanedJobs()" style="margin-top:0.5rem;padding:0.25rem 0.5rem;background:var(--error);color:white;border:none;border-radius:3px;cursor:pointer;font-size:0.75rem;">Clean Up Orphaned Jobs</button>';
                html += '</div>';
            }

            // Stuck Pending Jobs
            if (data.stuck_pending_jobs && data.stuck_pending_jobs > 0) {
                html += '<div class="debug-section">';
                html += '<div class="debug-section-title">⏳ Stuck Pending</div>';
                html += '<div class="debug-item"><span class="debug-key">Not queued</span><span class="debug-value warning">' + data.stuck_pending_jobs + '</span></div>';
                html += '<button onclick="requeuePendingJobs()" style="margin-top:0.5rem;padding:0.25rem 0.5rem;background:var(--warning);color:black;border:none;border-radius:3px;cursor:pointer;font-size:0.75rem;">Requeue Pending Jobs</button>';
                html += '</div>';
            }

            // Current Job Details
            if (data.job) {
                html += '<div class="debug-section">';
                html += '<div class="debug-section-title">🎯 Current Job</div>';
                html += '<div class="debug-item"><span class="debug-key">ID</span><span class="debug-value">' + (data.job.id || 'N/A').substring(0,8) + '...</span></div>';
                html += '<div class="debug-item"><span class="debug-key">Status</span><span class="debug-value">' + (data.job.status || 'N/A') + '</span></div>';
                html += '<div class="debug-item"><span class="debug-key">Stage</span><span class="debug-value">' + (data.job.current_stage || 'N/A') + '</span></div>';
                html += '<div class="debug-item"><span class="debug-key">Progress</span><span class="debug-value">' + (data.job.progress_percent || 0) + '%</span></div>';
                if (data.job.stages_completed !== null && data.job.total_stages) {
                    html += '<div class="debug-item"><span class="debug-key">Pipeline</span><span class="debug-value">' + data.job.stages_completed + '/' + data.job.total_stages + '</span></div>';
                }
                if (data.job.error_message) {
                    html += '<div class="debug-item"><span class="debug-key">Error</span><span class="debug-value error">' + data.job.error_message.substring(0,100) + '</span></div>';
                }
                // Resume button for failed jobs
                if (data.job.status === 'failed' && data.job.id) {
                    html += '<button onclick="resumeJob(\\'' + data.job.id + '\\')" style="margin-top:0.5rem;padding:0.25rem 0.5rem;background:#4CAF50;color:white;border:none;border-radius:3px;cursor:pointer;font-size:0.75rem;">▶ Resume This Job</button>';
                }
                html += '</div>';
            }

            // Recent Procrastinate Jobs
            if (data.recent_procrastinate_jobs && data.recent_procrastinate_jobs.length > 0) {
                html += '<div class="debug-section">';
                html += '<div class="debug-section-title">📋 Recent Queue Jobs</div>';
                data.recent_procrastinate_jobs.slice(0, 5).forEach(function(job) {
                    var statusClass = '';
                    if (job.status === 'succeeded') statusClass = 'success';
                    if (job.status === 'failed') statusClass = 'error';
                    if (job.status === 'doing') statusClass = 'warning';
                    html += '<div class="debug-item">';
                    html += '<span class="debug-key">#' + job.procrastinate_id + ' ' + job.task.replace('process_', '').replace('_job', '') + '</span>';
                    html += '<span class="debug-value ' + statusClass + '">' + job.status + '</span>';
                    html += '</div>';
                });
                html += '</div>';
            }

            // Analysis Jobs 24h Summary
            if (data.analysis_jobs_24h) {
                html += '<div class="debug-section">';
                html += '<div class="debug-section-title">📈 Last 24h</div>';
                Object.keys(data.analysis_jobs_24h).forEach(function(status) {
                    var valueClass = '';
                    if (status === 'completed') valueClass = 'success';
                    if (status === 'failed') valueClass = 'error';
                    html += '<div class="debug-item"><span class="debug-key">' + status + '</span><span class="debug-value ' + valueClass + '">' + data.analysis_jobs_24h[status] + '</span></div>';
                });
                html += '</div>';
            }

            // Quick Actions
            html += '<div class="debug-section">';
            html += '<div class="debug-section-title">🔧 Quick Actions</div>';
            html += '<a href="https://dashboard.render.com/worker/srv-d4qqhs2li9vc73a2nqeg" target="_blank" style="color:#4CAF50;font-size:0.75rem;text-decoration:none;">📊 Worker Dashboard (Render)</a>';
            html += '</div>';

            // Timestamp
            html += '<div class="debug-section" style="opacity:0.5;font-size:0.7rem;">';
            html += 'Last updated: ' + new Date().toLocaleTimeString();
            html += '</div>';

            content.innerHTML = html;
        }

        function cleanupOrphanedJobs() {
            fetch('/api/admin/cleanup-orphaned-jobs', { method: 'POST' })
                .then(function(response) { return response.json(); })
                .then(function(data) {
                    alert(data.message || 'Cleanup complete');
                    fetchDebugStatus();
                })
                .catch(function(error) {
                    alert('Cleanup failed: ' + error.message);
                });
        }

        function requeuePendingJobs() {
            fetch('/api/admin/requeue-pending-jobs', { method: 'POST' })
                .then(function(response) { return response.json(); })
                .then(function(data) {
                    alert(data.message || 'Requeue complete: ' + (data.requeued_count || 0) + ' jobs requeued');
                    fetchDebugStatus();
                })
                .catch(function(error) {
                    alert('Requeue failed: ' + error.message);
                });
        }

        function resumeJob(jobId) {
            if (!confirm('Resume job ' + jobId.substring(0, 8) + '... from last completed stage?')) {
                return;
            }
            fetch('/api/analyzer/jobs/' + jobId + '/resume', { method: 'POST' })
                .then(function(response) { return response.json(); })
                .then(function(data) {
                    if (data.error) {
                        alert('Resume failed: ' + data.error);
                    } else {
                        alert(data.message || 'Job resumed from ' + (data.resuming_from || 'start'));
                        fetchDebugStatus();
                        // Also trigger job progress polling if we're on that job
                        if (currentJobId === jobId) {
                            pollJobProgress();
                        }
                    }
                })
                .catch(function(error) {
                    alert('Resume failed: ' + error.message);
                });
        }

        function refreshDebug() {
            fetchDebugStatus();
        }

        // ========================================
        // Utility Functions
        // ========================================

        function escapeHtml(text) {
            const div = document.createElement('div');
            div.textContent = text || '';
            return div.innerHTML;
        }

        // ========================================
        // Web-Saver Integration
        // ========================================

        let wsCollections = [];
        let wsSelectedCollection = null;

        // Check Web-Saver availability on page load
        function checkWebSaverAvailability() {
            fetch('/api/websaver/health')
                .then(response => response.json())
                .then(data => {
                    const statusEl = document.getElementById('websaver-status');
                    const btnEl = document.getElementById('websaver-import-btn');
                    if (data.success && data.available) {
                        statusEl.textContent = '✓ Connected';
                        statusEl.className = 'websaver-status connected';
                        btnEl.disabled = false;
                    } else {
                        statusEl.textContent = '✗ Offline';
                        statusEl.className = 'websaver-status error';
                        btnEl.disabled = true;
                    }
                })
                .catch(() => {
                    const statusEl = document.getElementById('websaver-status');
                    const btnEl = document.getElementById('websaver-import-btn');
                    statusEl.textContent = '✗ Unavailable';
                    statusEl.className = 'websaver-status error';
                    btnEl.disabled = true;
                });
        }

        function openWebSaverModal() {
            document.getElementById('websaver-modal').style.display = 'flex';
            refreshWebSaverCollections();
        }

        function closeWebSaverModal() {
            document.getElementById('websaver-modal').style.display = 'none';
            wsSelectedCollection = null;
            document.getElementById('ws-selected-info').textContent = '';
            document.getElementById('ws-import-btn').disabled = true;
        }

        function refreshWebSaverCollections() {
            const listEl = document.getElementById('ws-collections-list');
            listEl.innerHTML = '<div class="ws-loading">Loading collections...</div>';

            fetch('/api/websaver/collections?min_articles=1&limit=100')
                .then(response => response.json())
                .then(data => {
                    if (!data.success) {
                        listEl.innerHTML = '<div class="ws-error">Failed to load collections: ' + (data.error || 'Unknown error') + '</div>';
                        return;
                    }

                    wsCollections = data.collections || [];

                    if (wsCollections.length === 0) {
                        listEl.innerHTML = '<div class="ws-empty">No collections with articles found.</div>';
                        return;
                    }

                    renderCollectionsList(wsCollections);
                })
                .catch(error => {
                    listEl.innerHTML = '<div class="ws-error">Connection error: ' + error.message + '</div>';
                });
        }

        function renderCollectionsList(collections) {
            const listEl = document.getElementById('ws-collections-list');
            listEl.innerHTML = collections.map(col => `
                <div class="ws-collection-item ${wsSelectedCollection && wsSelectedCollection.id === col.id ? 'selected' : ''}"
                     onclick="selectCollection(${col.id})" data-id="${col.id}">
                    <div class="ws-collection-name">${escapeHtml(col.name)}</div>
                    <div class="ws-collection-meta">
                        <span>📄 ${col.article_count} articles</span>
                        <span>📅 ${formatDate(col.updated_at)}</span>
                    </div>
                </div>
            `).join('');
        }

        function filterWebSaverCollections() {
            const search = document.getElementById('ws-search').value.toLowerCase();
            const filtered = wsCollections.filter(col =>
                col.name.toLowerCase().includes(search) ||
                (col.description && col.description.toLowerCase().includes(search))
            );
            renderCollectionsList(filtered);
        }

        function selectCollection(collectionId) {
            wsSelectedCollection = wsCollections.find(c => c.id === collectionId);

            // Update selection UI
            document.querySelectorAll('.ws-collection-item').forEach(el => {
                el.classList.remove('selected');
                if (parseInt(el.dataset.id) === collectionId) {
                    el.classList.add('selected');
                }
            });

            // Update footer
            if (wsSelectedCollection) {
                document.getElementById('ws-selected-info').textContent =
                    `Selected: ${wsSelectedCollection.name} (${wsSelectedCollection.article_count} articles)`;
                document.getElementById('ws-import-btn').disabled = false;
            } else {
                document.getElementById('ws-selected-info').textContent = '';
                document.getElementById('ws-import-btn').disabled = true;
            }
        }

        function importSelectedCollection() {
            if (!wsSelectedCollection) return;

            const importBtn = document.getElementById('ws-import-btn');
            importBtn.disabled = true;
            importBtn.textContent = 'Importing...';

            fetch('/api/websaver/collections/' + wsSelectedCollection.id)
                .then(response => response.json())
                .then(data => {
                    if (!data.success) {
                        alert('Failed to import: ' + (data.error || 'Unknown error'));
                        importBtn.disabled = false;
                        importBtn.textContent = 'Import';
                        return;
                    }

                    // Add documents to the analysis panel
                    const documents = data.documents || [];
                    if (documents.length === 0) {
                        alert('Collection has no documents with content.');
                        importBtn.disabled = false;
                        importBtn.textContent = 'Import';
                        return;
                    }

                    // Clear existing docs and add imported documents
                    scannedDocs = [];  // Clear existing

                    documents.forEach((doc, i) => {
                        // Add to scannedDocs array (with content for inline analysis)
                        const sizeBytes = doc.word_count * 5;
                        const sizeStr = sizeBytes > 1024 ? (sizeBytes / 1024).toFixed(1) + ' KB' : sizeBytes + ' B';
                        scannedDocs.push({
                            path: doc.url || `websaver_doc_${doc.id}`,
                            name: doc.title || 'Untitled',
                            type: 'article',  // Type for display
                            content: doc.content,  // Content stored for inline analysis
                            size: sizeStr,
                            wordCount: doc.word_count,
                            source: doc.source_name || ''
                        });
                    });

                    // Use existing selection system - select all imported docs
                    selectAllDocs();

                    // Show doc list container
                    document.getElementById('doc-list-container').style.display = 'block';

                    // Save name before closing modal (which clears wsSelectedCollection)
                    const collectionName = wsSelectedCollection?.name || 'Collection';

                    // Close modal
                    closeWebSaverModal();

                    // Show success message
                    showToast(`Imported ${documents.length} documents from "${collectionName}"`);
                })
                .catch(error => {
                    alert('Import failed: ' + error.message);
                })
                .finally(() => {
                    importBtn.disabled = false;
                    importBtn.textContent = 'Import';
                });
        }

        function formatDate(isoString) {
            if (!isoString) return '';
            const date = new Date(isoString);
            return date.toLocaleDateString(undefined, { month: 'short', day: 'numeric', year: 'numeric' });
        }

        function showToast(message) {
            const toast = document.createElement('div');
            toast.className = 'toast';
            toast.textContent = message;
            toast.style.cssText = `
                position: fixed; bottom: 2rem; right: 2rem;
                background: var(--success); color: white;
                padding: 1rem 1.5rem; border-radius: 8px;
                z-index: 3000; animation: fadeIn 0.3s ease;
            `;
            document.body.appendChild(toast);
            setTimeout(() => toast.remove(), 3000);
        }

        // Handle URL parameters for direct linking
        function handleUrlParameters() {
            const params = new URLSearchParams(window.location.search);
            const source = params.get('source');
            const collectionId = params.get('collection');

            if (source === 'web-saver' && collectionId) {
                // Auto-import from Web-Saver collection
                setTimeout(() => {
                    importFromUrlParam(collectionId);
                }, 500); // Small delay to ensure page is loaded
            }
        }

        function importFromUrlParam(collectionId) {
            showToast('Loading collection from Web-Saver...');

            fetch('/api/websaver/collections/' + collectionId)
                .then(response => response.json())
                .then(data => {
                    if (!data.success) {
                        showToast('Failed to load collection: ' + (data.error || 'Not found'));
                        return;
                    }

                    const documents = data.documents || [];
                    if (documents.length === 0) {
                        showToast('Collection has no documents.');
                        return;
                    }

                    // Clear and add documents
                    scannedDocs = [];  // Clear existing

                    documents.forEach((doc, i) => {
                        const sizeBytes = doc.word_count * 5;
                        const sizeStr = sizeBytes > 1024 ? (sizeBytes / 1024).toFixed(1) + ' KB' : sizeBytes + ' B';
                        scannedDocs.push({
                            path: doc.url || `websaver_doc_${doc.id}`,
                            name: doc.title || 'Untitled',
                            type: 'article',
                            content: doc.content,
                            size: sizeStr,
                            wordCount: doc.word_count,
                            source: doc.source_name || ''
                        });
                    });

                    // Use existing selection system
                    selectAllDocs();
                    document.getElementById('doc-list-container').style.display = 'block';
                    showToast(`Imported ${documents.length} documents from "${data.collection?.name || 'Collection'}"`);

                    // Clean up URL
                    window.history.replaceState({}, document.title, window.location.pathname);
                })
                .catch(error => {
                    showToast('Failed to import: ' + error.message);
                });
        }

        // Initialize on page load
        document.addEventListener('DOMContentLoaded', function() {
            checkWebSaverAvailability();
            handleUrlParameters();
        });
    </script>

    <!-- Debug Toggle and Panel -->
    <div id="debug-toggle" class="debug-toggle" onclick="toggleDebugPanel()">🔧 Debug</div>
    <div id="debug-panel" class="debug-panel">
        <div class="debug-panel-header">
            <span>System Status</span>
            <div class="debug-panel-actions">
                <button onclick="refreshDebug()">↻ Refresh</button>
                <button onclick="toggleDebugPanel()">✕</button>
            </div>
        </div>
        <div id="debug-content" class="debug-panel-content">
            <div class="debug-section">Loading...</div>
        </div>
    </div>

    <!-- Web-Saver Collection Picker Modal -->
    <div id="websaver-modal" class="ws-modal" style="display:none;">
        <div class="ws-modal-content">
            <div class="ws-modal-header">
                <h3>📚 Import from Web-Saver</h3>
                <button class="ws-modal-close" onclick="closeWebSaverModal()">✕</button>
            </div>
            <div class="ws-modal-body">
                <div class="ws-search-row">
                    <input type="text" id="ws-search" placeholder="Search collections..." oninput="filterWebSaverCollections()">
                    <button class="btn btn-sm" onclick="refreshWebSaverCollections()">↻</button>
                </div>
                <div id="ws-collections-list" class="ws-collections-list">
                    <div class="ws-loading">Loading collections...</div>
                </div>
            </div>
            <div class="ws-modal-footer">
                <span id="ws-selected-info"></span>
                <button class="btn btn-sm" onclick="closeWebSaverModal()">Cancel</button>
                <button class="btn btn-sm btn-primary" onclick="importSelectedCollection()" id="ws-import-btn" disabled>Import</button>
            </div>
        </div>
    </div>
</body>
</html>
'''


# Application Entry Point

if __name__ == '__main__':
    # Production-ready configuration from environment
    PORT = int(os.environ.get('PORT', 5847))
    DEBUG = os.environ.get('DEBUG', 'true').lower() in ('true', '1', 'yes')
    ENVIRONMENT = os.environ.get('ENVIRONMENT', 'development')

    # Print startup banner
    print()
    print("=" * 65)
    print("  THE VISUALIZER")
    print("=" * 65)
    print()
    print(f"  Environment      : {ENVIRONMENT}")
    print(f"  Download folder  : {DOWNLOAD_FOLDER}")
    print(f"  Analyzer URL     : {ANALYZER_API_URL}")
    print(f"  Gemini API Key   : {'Set' if os.environ.get('GEMINI_API_KEY') else 'Not set (users provide their own)'}")
    print()
    print("=" * 65)
    print(f"  Starting server at: http://localhost:{PORT}")
    print(f"  Debug mode       : {'ON' if DEBUG else 'OFF'}")
    print("  Press Ctrl+C to stop")
    print("=" * 65)
    print()

    # Initialize client before starting server (optional in production)
    if os.environ.get('GEMINI_API_KEY'):
        if initialize_client():
            print("  Gemini client initialized successfully")
        else:
            print("  Client initialization deferred to first request")
    else:
        print("  No server-side Gemini key - users will provide their own")

    print()

    # Development mode: watch files for auto-reload
    extra_files = []
    if DEBUG:
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

    # Run the Flask server
    app.run(
        host='0.0.0.0',
        port=PORT,
        debug=DEBUG,
        use_reloader=DEBUG,
        extra_files=extra_files if extra_files else None,
        threaded=True
    )
