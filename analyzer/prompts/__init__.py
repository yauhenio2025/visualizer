"""
Prompt templates for textual output generation.
"""

from .textual_output_templates import (
    TextualOutputType,
    OUTPUT_TYPE_METADATA,
    TEXTUAL_OUTPUT_TEMPLATES,
    ENGINE_OUTPUT_AFFINITY,
    OutputTypeMetadata,
    get_template,
    get_output_metadata,
    get_recommended_outputs,
    format_template,
)

__all__ = [
    "TextualOutputType",
    "OUTPUT_TYPE_METADATA",
    "TEXTUAL_OUTPUT_TEMPLATES",
    "ENGINE_OUTPUT_AFFINITY",
    "OutputTypeMetadata",
    "get_template",
    "get_output_metadata",
    "get_recommended_outputs",
    "format_template",
]
