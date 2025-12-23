"""
Analyzer module for document analysis and textual output generation.

This module provides:
- Differentiated textual output templates (8 types)
- Textual output renderer using Claude API
- Engine-to-output affinity mapping
- Complementarity analysis for visual-text pairing
"""

from .renderer import (
    TextualOutputRenderer,
    TextualOutput,
    ComplementarityAnalysis,
    render_textual_output,
    get_output_recommendations,
)

from .prompts.textual_output_templates import (
    TextualOutputType,
    OUTPUT_TYPE_METADATA,
    TEXTUAL_OUTPUT_TEMPLATES,
    ENGINE_OUTPUT_AFFINITY,
    get_template,
    get_output_metadata,
    get_recommended_outputs,
    format_template,
)

__all__ = [
    # Renderer
    "TextualOutputRenderer",
    "TextualOutput",
    "ComplementarityAnalysis",
    "render_textual_output",
    "get_output_recommendations",
    # Templates
    "TextualOutputType",
    "OUTPUT_TYPE_METADATA",
    "TEXTUAL_OUTPUT_TEMPLATES",
    "ENGINE_OUTPUT_AFFINITY",
    "get_template",
    "get_output_metadata",
    "get_recommended_outputs",
    "format_template",
]
