"""
Textual Output Renderer

Generates differentiated textual outputs using Claude API
based on the 8 output type templates.

This module:
1. Takes analysis data from engine extraction
2. Applies the appropriate template for the output type
3. Sends to Claude for generation
4. Returns the formatted text with metadata
"""

import os
import json
import logging
from typing import Any, Optional
from dataclasses import dataclass, asdict

import anthropic

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

logger = logging.getLogger(__name__)


@dataclass
class TextualOutput:
    """Result of textual output generation."""
    output_type: str
    content: str
    title: str
    metadata: dict
    word_count: int
    generation_model: str
    complementarity_note: Optional[str] = None


@dataclass
class ComplementarityAnalysis:
    """Analysis of what visual output shows vs. what text should add."""
    visual_shows: list[str]
    text_should_add: list[str]
    avoid_duplicating: list[str]
    focus_areas: list[str]


class TextualOutputRenderer:
    """
    Renders differentiated textual outputs using Claude API.

    Usage:
        renderer = TextualOutputRenderer(api_key="...")
        result = renderer.render(
            output_type="deep_dive",
            analysis_data={"findings": [...], "evidence": [...]},
            visual_summary="Network graph showing 15 actors with 3 clusters...",
            topic="Stakeholder Analysis of AI Policy"
        )
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "claude-sonnet-4-5-20250929",
        max_tokens: int = 8192,
    ):
        """
        Initialize the renderer.

        Args:
            api_key: Anthropic API key (defaults to ANTHROPIC_API_KEY env var)
            model: Claude model to use
            max_tokens: Maximum tokens for generation
        """
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("Anthropic API key required")

        self.client = anthropic.Anthropic(api_key=self.api_key)
        self.model = model
        self.max_tokens = max_tokens

    def analyze_complementarity(
        self,
        visual_summary: str,
        output_type: str,
    ) -> ComplementarityAnalysis:
        """
        Analyze what the visual shows to ensure text complements rather than duplicates.

        Args:
            visual_summary: Description of what the visual output shows
            output_type: The textual output type being generated

        Returns:
            ComplementarityAnalysis with guidance for text generation
        """
        if not visual_summary or visual_summary == "No visual output available for this analysis.":
            return ComplementarityAnalysis(
                visual_shows=[],
                text_should_add=["Full analysis (no visual to complement)"],
                avoid_duplicating=[],
                focus_areas=["Complete coverage of findings"]
            )

        metadata = get_output_metadata(output_type)

        # Use Claude to analyze what the visual shows
        prompt = f"""Analyze this visual output summary to determine what the accompanying {metadata.name} text should focus on.

VISUAL OUTPUT SUMMARY:
{visual_summary}

OUTPUT TYPE: {metadata.name}
PURPOSE: {metadata.description}
CORE QUESTION: {metadata.core_question}

Respond in JSON format:
{{
    "visual_shows": ["list of what the visual already communicates well"],
    "text_should_add": ["list of what the text should explain that visual cannot"],
    "avoid_duplicating": ["specific things text should NOT repeat from visual"],
    "focus_areas": ["top 3-5 areas the text should focus on"]
}}

Remember: Visual shows WHAT (structure, relationships, patterns). Text explains WHY, SO WHAT, and NOW WHAT."""

        try:
            response = self.client.messages.create(
                model="claude-haiku-4-5-20251001",  # Use Haiku for fast analysis
                max_tokens=1024,
                messages=[{"role": "user", "content": prompt}]
            )

            result_text = response.content[0].text
            # Extract JSON from response
            if "```json" in result_text:
                result_text = result_text.split("```json")[1].split("```")[0]
            elif "```" in result_text:
                result_text = result_text.split("```")[1].split("```")[0]

            result = json.loads(result_text.strip())
            return ComplementarityAnalysis(**result)

        except Exception as e:
            logger.warning(f"Complementarity analysis failed: {e}")
            return ComplementarityAnalysis(
                visual_shows=["Visual content (analysis failed)"],
                text_should_add=["Explanation and interpretation"],
                avoid_duplicating=["Direct description of visual elements"],
                focus_areas=["Why it matters", "Implications", "Recommendations"]
            )

    def render(
        self,
        output_type: str,
        analysis_data: dict[str, Any],
        visual_summary: Optional[str] = None,
        topic: Optional[str] = None,
        run_complementarity_check: bool = True,
    ) -> TextualOutput:
        """
        Render a textual output from analysis data.

        Args:
            output_type: One of the 8 output types (e.g., "deep_dive", "snapshot")
            analysis_data: Structured data from engine extraction
            visual_summary: Optional description of accompanying visual output
            topic: Optional topic/title for the analysis
            run_complementarity_check: Whether to analyze visual first

        Returns:
            TextualOutput with generated content and metadata
        """
        # Validate output type
        if output_type not in TEXTUAL_OUTPUT_TEMPLATES:
            raise ValueError(f"Unknown output type: {output_type}. Valid types: {list(TEXTUAL_OUTPUT_TEMPLATES.keys())}")

        metadata = get_output_metadata(output_type)

        # Run complementarity check if visual available
        complementarity_note = None
        if run_complementarity_check and visual_summary:
            comp_analysis = self.analyze_complementarity(visual_summary, output_type)
            # Add to analysis data for template
            analysis_data["_complementarity"] = asdict(comp_analysis)
            complementarity_note = f"Text focuses on: {', '.join(comp_analysis.focus_areas[:3])}"

        # Format the template with data
        formatted_prompt = format_template(
            output_type=output_type,
            analysis_data=analysis_data,
            visual_summary=visual_summary
        )

        # Add topic context if provided
        if topic:
            formatted_prompt = f"TOPIC: {topic}\n\n{formatted_prompt}"

        # Determine max tokens based on output type
        type_max_tokens = {
            "snapshot": 1024,
            "deep_dive": 8192,
            "evidence_pack": 16384,
            "signal_report": 2048,
            "status_brief": 3072,
            "stakeholder_profile": 4096,
            "gap_analysis": 4096,
            "options_brief": 3072,
        }
        max_tokens = type_max_tokens.get(output_type, self.max_tokens)

        # Generate with Claude
        logger.info(f"Generating {output_type} with {self.model} (max_tokens={max_tokens})")

        response = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": formatted_prompt}]
        )

        content = response.content[0].text
        word_count = len(content.split())

        # Generate title
        title = topic or f"{metadata.name}: Analysis Results"

        return TextualOutput(
            output_type=output_type,
            content=content,
            title=title,
            metadata={
                "output_type_name": metadata.name,
                "output_type_icon": metadata.icon,
                "audience": metadata.audience,
                "reading_time": metadata.reading_time,
                "core_question": metadata.core_question,
            },
            word_count=word_count,
            generation_model=self.model,
            complementarity_note=complementarity_note,
        )

    def render_multiple(
        self,
        output_types: list[str],
        analysis_data: dict[str, Any],
        visual_summary: Optional[str] = None,
        topic: Optional[str] = None,
    ) -> dict[str, TextualOutput]:
        """
        Render multiple output types from the same analysis data.

        Args:
            output_types: List of output types to generate
            analysis_data: Structured data from engine extraction
            visual_summary: Optional description of accompanying visual output
            topic: Optional topic/title for the analysis

        Returns:
            Dict mapping output_type -> TextualOutput
        """
        results = {}

        # Run complementarity check once
        comp_analysis = None
        if visual_summary:
            comp_analysis = self.analyze_complementarity(visual_summary, output_types[0])
            analysis_data["_complementarity"] = asdict(comp_analysis)

        for output_type in output_types:
            try:
                result = self.render(
                    output_type=output_type,
                    analysis_data=analysis_data,
                    visual_summary=visual_summary,
                    topic=topic,
                    run_complementarity_check=False,  # Already done
                )
                results[output_type] = result
            except Exception as e:
                logger.error(f"Failed to render {output_type}: {e}")
                results[output_type] = TextualOutput(
                    output_type=output_type,
                    content=f"Error generating {output_type}: {str(e)}",
                    title=f"Error: {output_type}",
                    metadata={},
                    word_count=0,
                    generation_model=self.model,
                )

        return results

    def get_recommended_outputs_for_engine(self, engine_key: str) -> list[dict]:
        """
        Get recommended output types for an engine with full metadata.

        Args:
            engine_key: The engine key (e.g., "stakeholder_power_interest")

        Returns:
            List of output type metadata dicts, sorted by affinity
        """
        recommended = get_recommended_outputs(engine_key)
        result = []

        for output_type in recommended:
            try:
                metadata = get_output_metadata(output_type)
                affinity = ENGINE_OUTPUT_AFFINITY.get(engine_key, {}).get(output_type, 0)
                result.append({
                    "output_type": output_type,
                    "name": metadata.name,
                    "icon": metadata.icon,
                    "description": metadata.description,
                    "affinity": affinity,
                    "affinity_label": "Ideal" if affinity == 3 else "Good" if affinity == 2 else "Possible",
                })
            except ValueError:
                continue

        return result


# Convenience function for quick rendering
def render_textual_output(
    output_type: str,
    analysis_data: dict[str, Any],
    visual_summary: Optional[str] = None,
    topic: Optional[str] = None,
    api_key: Optional[str] = None,
) -> TextualOutput:
    """
    Convenience function to render a single textual output.

    Args:
        output_type: One of the 8 output types
        analysis_data: Structured data from engine extraction
        visual_summary: Optional description of accompanying visual
        topic: Optional topic/title
        api_key: Optional Anthropic API key

    Returns:
        TextualOutput with generated content
    """
    renderer = TextualOutputRenderer(api_key=api_key)
    return renderer.render(
        output_type=output_type,
        analysis_data=analysis_data,
        visual_summary=visual_summary,
        topic=topic,
    )


# Export convenience function for getting recommendations
def get_output_recommendations(engine_key: str) -> list[dict]:
    """
    Get recommended output types for an engine.

    Args:
        engine_key: The engine key

    Returns:
        List of recommended output types with metadata
    """
    renderer = TextualOutputRenderer.__new__(TextualOutputRenderer)
    # Bypass __init__ since we don't need API key for recommendations
    return [
        {
            "output_type": ot,
            "name": get_output_metadata(ot).name,
            "icon": get_output_metadata(ot).icon,
            "description": get_output_metadata(ot).description,
            "affinity": ENGINE_OUTPUT_AFFINITY.get(engine_key, {}).get(ot, 0),
        }
        for ot in get_recommended_outputs(engine_key)
    ]
