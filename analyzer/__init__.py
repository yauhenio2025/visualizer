"""
Analyzer module for document analysis and textual output generation.

This module provides:
- Differentiated textual output templates (8 types)
- Textual output renderer using Claude API
- Engine-to-output affinity mapping
- Complementarity analysis for visual-text pairing
- Output Curator agent (Opus 4.5 with extended thinking)
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

from .output_curator import (
    OutputCurator,
    CuratorOutput,
    FormatRecommendation,
    OutputCategory,
    curate_output,
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
    # Output Curator
    "OutputCurator",
    "CuratorOutput",
    "FormatRecommendation",
    "OutputCategory",
    "curate_output",
]
