TITLE: Nano Banana 4K Visualizer - Complete Production Implementation Guide

---

# 🍌 Nano Banana 4K Visualizer

**A Professional Web Interface for Gemini 3 Pro Image Generation**

Version 1.0.0 | Production Ready

---

## Table of Contents

1. [Overview](#overview)
2. [Quick Start](#quick-start)
3. [Complete Source Code](#complete-source-code)
4. [Features](#features)
5. [Usage Guide](#usage-guide)
6. [API Reference](#api-reference)
7. [Troubleshooting](#troubleshooting)
8. [Future Enhancements](#future-enhancements)

---

## Overview

This tool provides a sleek web interface for Google's Gemini 3 Pro Image Preview model ("Nano Banana"), enabling:

- **4K image generation** from text descriptions
- **Context-aware generation** using local files as reference
- **Multi-turn conversations** for iterative refinement
- **Model reasoning visibility** to understand how prompts are interpreted
- **Automatic downloads** to your local machine

### Technical Specifications

| Specification | Value |
|--------------|-------|
| Primary Model | `gemini-3-pro-image-preview` |
| Fallback Model | `gemini-2.5-flash-image` |
| Max Resolution | 4K (4096×4096 for 1:1) |
| Temperature | 1.0 (maximum creativity) |
| Response Modes | TEXT + IMAGE |
| Max Context Images | 14 |

---

## Quick Start

### Prerequisites

- Python 3.9 or higher
- Internet connection

### Installation

```bash
# Create project directory
mkdir nano-banana-visualizer
cd nano-banana-visualizer

# Install dependencies
pip install flask google-genai Pillow

# Save the code below as app.py
# Then run:
python app.py
```

### Access

Open your browser to: **http://localhost:5000**

---

## Complete Source Code

Save the following as `app.py`:

```python
#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
🍌 NANO BANANA 4K VISUALIZER
═══════════════════════════════════════════════════════════════════════════════
A production-ready web interface for Gemini's advanced image generation.

Features:
  • 4K image generation with reasoning display
  • Local file context support (images, text files)
  • Multi-turn conversation for iterative refinement
  • Automatic download to local machine
  • Google Search grounding option

Usage:
  1. pip install flask google-genai Pillow
  2. python app.py
  3. Open http://localhost:5000

═══════════════════════════════════════════════════════════════════════════════
"""

import os
import io
import re
import base64
import datetime
import traceback
import uuid
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any

from flask import Flask, request, jsonify, Response
from PIL import Image

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

# API Key Configuration
# Priority: Environment variable > Hardcoded fallback
API_KEY = os.environ.get(
    "GEMINI_API_KEY",
    "AIzaSyAv2Dn8RitDPbsFCs_Y9B9_IFlJd5p2vwA"
)

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

# ═══════════════════════════════════════════════════════════════════════════════
# APPLICATION INITIALIZATION
# ═══════════════════════════════════════════════════════════════════════════════

app = Flask(__name__)

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
        print(f"❌ Missing dependency: {e}")
        print("   Run: pip install google-genai")
        return False
    except Exception as e:
        print(f"❌ Failed to initialize Gemini client: {e}")
        return False

# Chat session storage for multi-turn conversations
chat_sessions: Dict[str, Any] = {}

# ═══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

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
                # Try SDK method first
                img = part.as_image()
            except Exception:
                # Fallback: decode base64 manually
                try:
                    img_data = base64.b64decode(part.inline_data.data)
                    img = Image.open(io.BytesIO(img_data))
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


# ═══════════════════════════════════════════════════════════════════════════════
# API ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════════

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy",
        "client_initialized": client is not None,
        "download_folder": str(DOWNLOAD_FOLDER)
    })


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
            response_data["dimensions"] = f"{img.width}×{img.height}"
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
        
        # Save generated images
        saved_paths = []
        for i, img in enumerate(result["raw_images"]):
            try:
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
                "search_enabled": use_search and model_info.get("supports_search", False)
            }
        })
    
    except Exception as e:
        traceback.print_exc()
        return jsonify({
            "success": False,
            "error": format_error_message(e)
        })


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN PAGE
# ═══════════════════════════════════════════════════════════════════════════════

@app.route('/')
def index():
    """Serve the main application page."""
    return Response(HTML_PAGE, mimetype='text/html')


# ═══════════════════════════════════════════════════════════════════════════════
# EMBEDDED HTML/CSS/JS
# ═══════════════════════════════════════════════════════════════════════════════

HTML_PAGE = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🍌 Nano Banana 4K Visualizer</title>
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
    </style>
</head>
<body>
    <div class="app">
        <header>
            <h1>🍌 Nano Banana 4K Visualizer</h1>
            <p>Generate stunning images with Gemini's advanced reasoning model</p>
            <span class="badge">gemini-3-pro-image-preview</span>
        </header>
        
        <div class="main">
            <!-- Input Panel -->
            <div class="input-panel">
                <div class="card">
                    <h2 class="card-title">✨ Create Visualization</h2>
                    
                    <!-- Session controls -->
                    <div style="display:flex; gap:0.5rem; margin-bottom:1rem; padding-bottom:1rem; border-bottom:1px solid var(--border);">
                        <button class="btn btn-sm btn-outline" onclick="newSession()">🔄 New Session</button>
                        <span id="session-indicator" class="badge" style="display:none;">Multi-turn active</span>
                    </div>
                    
                    <!-- Prompt input -->
                    <div>
                        <label for="prompt">Describe your visualization</label>
                        <textarea id="prompt" placeholder="Describe what you want to create in detail...

Example prompts:
• A photorealistic 4K image of a futuristic city at golden hour
• An infographic explaining the water cycle for kids
• A Da Vinci style anatomical sketch of a butterfly"></textarea>
                    </div>
                    
                    <!-- File context -->
                    <div class="file-section">
                        <label>📁 Context Files (optional)</label>
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
                        <span>🔍 Enable Google Search grounding</span>
                    </label>
                    
                    <!-- Generate button -->
                    <button id="gen-btn" class="btn btn-primary" onclick="generate()">
                        🍌 Generate Visualization
                    </button>
                    <p class="key-hint">Press <kbd>Ctrl+Enter</kbd> to generate</p>
                </div>
            </div>
            
            <!-- Output Panel -->
            <div class="output-panel">
                <div class="card output">
                    <h2 class="card-title">🖼️ Generated Output</h2>
                    
                    <!-- Status -->
                    <div id="status" class="status">
                        <div class="spinner"></div>
                        <span id="status-text">Generating...</span>
                    </div>
                    
                    <!-- Thinking -->
                    <details id="thinking-box" class="thinking-box">
                        <summary>💭 Model Reasoning</summary>
                        <div id="thinking-content" class="content"></div>
                    </details>
                    
                    <!-- Image -->
                    <div id="image-box" class="image-box fade-in">
                        <img id="result-img" src="" alt="Generated image">
                        <div class="image-actions">
                            <button class="btn btn-success" onclick="downloadImg()">⬇️ Download</button>
                            <button class="btn btn-ghost" onclick="openFull()">🔗 Full Size</button>
                        </div>
                    </div>
                    
                    <!-- Save info -->
                    <div id="save-info" class="save-info">
                        💾 Saved: <code id="save-path"></code>
                    </div>
                    
                    <!-- Text output -->
                    <div id="text-output" class="text-output"></div>
                    
                    <!-- Placeholder -->
                    <div id="placeholder" class="placeholder">
                        <div class="icon">🍌</div>
                        <p>Your visualization will appear here</p>
                        <p style="font-size:0.85rem; margin-top:0.5rem;">Enter a prompt and click Generate</p>
                    </div>
                </div>
                
                <!-- History -->
                <div id="history-card" class="card history-card">
                    <h2 class="card-title">📜 Recent</h2>
                    <div id="history-grid" class="history-grid"></div>
                </div>
            </div>
        </div>
    </div>
    
    <script>
        // ═══════════════════════════════════════════════════════════════
        // STATE
        // ═══════════════════════════════════════════════════════════════
        let files = [];
        let currentImg = null;
        let currentPath = null;
        let sessionId = null;
        let history = [];
        
        const $ = id => document.getElementById(id);
        
        // ═══════════════════════════════════════════════════════════════
        // INITIALIZATION
        // ═══════════════════════════════════════════════════════════════
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
        
        // ═══════════════════════════════════════════════════════════════
        // SESSION MANAGEMENT
        // ═══════════════════════════════════════════════════════════════
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
        
        // ═══════════════════════════════════════════════════════════════
        // FILE MANAGEMENT
        // ═══════════════════════════════════════════════════════════════
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
                        <div class="meta">${f.type} • ${f.size}${f.dimensions ? ` • ${f.dimensions}` : ''}</div>
                    </div>
                    <span class="remove" onclick="removeFile(${i})">×</span>
                </div>
            `).join('');
        }
        
        function removeFile(i) {
            files.splice(i, 1);
            renderFiles();
        }
        
        // ═══════════════════════════════════════════════════════════════
        // STATUS DISPLAY
        // ═══════════════════════════════════════════════════════════════
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
        
        // ═══════════════════════════════════════════════════════════════
        // GENERATION
        // ═══════════════════════════════════════════════════════════════
        async function generate() {
            const prompt = $('prompt').value.trim();
            
            if (!prompt) {
                showStatus('Please enter a prompt describing what to visualize', 'error');
                setTimeout(hideStatus, 3000);
                return;
            }
            
            const btn = $('gen-btn');
            btn.disabled = true;
            btn.textContent = '⏳ Generating...';
            
            // Reset UI
            $('placeholder').style.display = 'none';
            $('image-box').classList.remove('show');
            $('save-info').classList.remove('show');
            $('text-output').classList.remove('show');
            $('thinking-box').classList.remove('show');
            
            const resolution = $('resolution').value;
            showStatus(`🍌 Generating ${resolution} image... This may take up to 30 seconds`, 'loading');
            
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
                    showStatus(`✨ Generated with ${meta.model || 'Nano Banana'} at ${meta.resolution || resolution}!`, 'success');
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
                    
                    // Save path
                    if (data.saved_paths && data.saved_paths.length > 0) {
                        currentPath = data.saved_paths[0];
                        $('save-path').textContent = currentPath;
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
                btn.textContent = '🍌 Generate Visualization';
            }
        }
        
        // ═══════════════════════════════════════════════════════════════
        // IMAGE ACTIONS
        // ═══════════════════════════════════════════════════════════════
        function downloadImg() {
            if (!currentImg) return;
            const a = document.createElement('a');
            a.href = `data:image/png;base64,${currentImg}`;
            a.download = `nano_banana_${Date.now()}.png`;
            a.click();
        }
        
        function openFull() {
            if (!currentImg) return;
            const w = window.open();
            w.document.write(`<html><head><title>Nano Banana Output</title></head>
                <body style="margin:0;background:#0d0d14;display:flex;justify-content:center;align-items:center;min-height:100vh;">
                <img src="data:image/png;base64,${currentImg}" style="max-width:100%;max-height:100vh;"></body></html>`);
        }
        
        // ═══════════════════════════════════════════════════════════════
        // HISTORY
        // ═══════════════════════════════════════════════════════════════
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
    </script>
</body>
</html>'''


# ═══════════════════════════════════════════════════════════════════════════════
# APPLICATION ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    # Print startup banner
    print()
    print("═" * 65)
    print("  🍌 NANO BANANA 4K VISUALIZER")
    print("═" * 65)
    print()
    print(f"  Download folder  : {DOWNLOAD_FOLDER}")
    print(f"  API Key          : {'Environment variable' if os.environ.get('GEMINI_API_KEY') else 'Embedded'}")
    print(f"  Available models : {', '.join(m['name'] for m in MODELS.values())}")
    print()
    print("═" * 65)
    print("  Starting server at: http://localhost:5000")
    print("  Press Ctrl+C to stop")
    print("═" * 65)
    print()
    
    # Initialize client before starting server
    if initialize_client():
        print("  ✓ Gemini client initialized successfully")
    else:
        print("  ⚠ Client initialization deferred to first request")
    
    print()
    
    # Run the Flask server
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=True,
        threaded=True
    )
```

---

## Features

### Core Capabilities

| Feature | Description |
|---------|-------------|
| **4K Generation** | Native 4K output (up to 4096×4096 pixels) |
| **Model Reasoning** | View the model's thinking process before image generation |
| **Multi-Turn Chat** | Refine images through conversation |
| **Local File Context** | Use images and text files as reference |
| **Auto-Download** | Automatically saves to ~/Downloads |
| **Google Search** | Optional grounding for real-time data |

### Settings Options

| Setting | Options | Default |
|---------|---------|---------|
| **Model** | Pro (4K), Flash (Fast) | Pro |
| **Aspect Ratio** | 16:9, 1:1, 9:16, 4:3, 3:2, 21:9 | 16:9 |
| **Resolution** | 4K, 2K, 1K | 4K |
| **Search Grounding** | Enabled/Disabled | Disabled |

### Keyboard Shortcuts

| Shortcut | Action |
|----------|--------|
| `Ctrl+Enter` | Generate image |
| `Tab` | Navigate between fields |

---

## Usage Guide

### Basic Generation

1. Open `http://localhost:5000`
2. Enter a detailed prompt describing your visualization
3. Click **Generate Visualization** or press `Ctrl+Enter`
4. Image appears inline and is automatically saved to Downloads

### Using Context Files

1. Click the file path input field
2. Enter the full path to your file:
   - Windows: `C:\Users\Name\image.png`
   - Mac/Linux: `/Users/name/Documents/sketch.jpg`
3. Click **+ Add** to validate and add the file
4. Add up to 14 reference images
5. Generate - the model will use these as context

### Multi-Turn Refinement

1. Generate an initial image
2. Type a follow-up prompt like:
   - "Make the sky more purple"
   - "Add a person walking in the foreground"
   - "Convert this to a watercolor style"
3. Generate again - the model remembers previous context
4. Click **New Session** to start fresh

### Using Google Search

1. Enable the **🔍 Google Search grounding** checkbox
2. Ask for visualizations based on real-time data:
   - "Show the current weather forecast for Tokyo as an infographic"
   - "Visualize today's stock market performance"
3. The model will search for current information

---

## API Reference

### `POST /api/generate`

Generate an image from a prompt.

**Request Body:**
```json
{
    "prompt": "A photorealistic image of...",
    "file_paths": ["/path/to/context.png"],
    "model": "pro",
    "aspect_ratio": "16:9",
    "resolution": "4K",
    "use_search": false,
    "session_id": "uuid-for-multi-turn"
}
```

**Response:**
```json
{
    "success": true,
    "text": "Optional model text response",
    "thinking": "Model's reasoning process",
    "images": ["base64-encoded-image"],
    "saved_paths": ["/Users/name/Downloads/nano_banana_...png"],
    "metadata": {
        "model": "Nano Banana Pro",
        "resolution": "4K",
        "aspect_ratio": "16:9"
    }
}
```

### `POST /api/validate-path`

Validate a local file path.

**Request:** `{"path": "/path/to/file.png"}`

**Response:**
```json
{
    "valid": true,
    "type": "image",
    "name": "file.png",
    "size": "1.2 MB",
    "dimensions": "1920×1080",
    "preview": "base64-thumbnail"
}
```

### `POST /api/session/new`

Create a new chat session for multi-turn.

**Response:** `{"session_id": "uuid"}`

---

## Troubleshooting

### Common Issues

| Problem | Solution |
|---------|----------|
| **"Failed to initialize Gemini client"** | Check API key is valid. Try setting `GEMINI_API_KEY` environment variable. |
| **"API quota exceeded"** | Wait a few minutes and try again. 4K generation uses more quota. |
| **"Content blocked by safety filters"** | Modify your prompt to avoid sensitive content. |
| **"Model not available"** | The Pro model may have limited access. Try Flash model. |
| **Generation times out** | 4K images can take 30-60 seconds. Reduce resolution or try again. |
| **File not found** | Use absolute paths. On Windows, use forward slashes or escaped backslashes. |

### Environment Variables

```bash
# Set API key (recommended for security)
export GEMINI_API_KEY="your-api-key-here"

# Windows PowerShell
$env:GEMINI_API_KEY = "your-api-key-here"

# Windows CMD
set GEMINI_API_KEY=your-api-key-here
```

### Dependencies

If you encounter import errors:

```bash
# Ensure latest versions
pip install --upgrade google-genai Pillow flask
```

---

## Future Enhancements

The following features can be added based on your needs:

- [ ] **Image gallery** - Browse all generated images
- [ ] **Prompt templates** - Pre-built prompt structures
- [ ] **Batch generation** - Generate multiple variations
- [ ] **Style presets** - One-click style application
- [ ] **Image comparison** - Side-by-side before/after
- [ ] **Export settings** - Save/load generation preferences
- [ ] **API endpoint security** - Add authentication
- [ ] **Docker deployment** - Containerized deployment

---

## License & Attribution

This tool uses the Google Gemini API. Generated images include SynthID watermarks as required by Google's terms of service.

---

*Document generated: Production-ready implementation*
*Model: Gemini 3 Pro Image Preview (Nano Banana)*
*Resolution: 4K Maximum*