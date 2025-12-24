"""
Output Curator Agent

An LLM-powered agent (Claude Opus 4.5 with extended thinking) that analyzes
extracted data from engines and recommends optimal output formats.

The curator:
1. Detects data structure from engine output
2. Recommends visual, textual, and structured formats
3. Generates optimized Gemini prompts for visual outputs
4. Explains its reasoning

Uses streaming for extended thinking support.
"""

import os
import json
import logging
from typing import Optional, Any
from dataclasses import dataclass, asdict
from enum import Enum

import anthropic

logger = logging.getLogger(__name__)


class OutputCategory(Enum):
    VISUAL = "visual"
    TEXTUAL = "textual"
    STRUCTURED = "structured"
    DIAGRAM = "diagram"


@dataclass
class FormatRecommendation:
    """A single format recommendation with rationale."""
    format_key: str
    category: str
    name: str
    confidence: float  # 0.0 - 1.0
    rationale: str
    gemini_prompt: Optional[str] = None  # Only for visual formats
    data_mapping: Optional[dict] = None  # How data maps to visual elements


@dataclass
class CuratorOutput:
    """Complete output from the curator."""
    data_structure_analysis: str
    primary_recommendation: FormatRecommendation
    secondary_recommendations: list[FormatRecommendation]
    audience_considerations: str
    thinking_summary: str
    raw_thinking: Optional[str] = None


# Knowledge base summaries for the curator
VISUAL_FORMAT_KNOWLEDGE = """
VISUAL FORMAT CATEGORIES:

RELATIONAL (for nodes[], edges[] data):
- network_graph: Actor relationships, citations, concepts
- chord_diagram: Bilateral flows, mutual dependencies
- hierarchical_tree: Taxonomies, org charts
- radial_tree: Concept maps, influence radii

FLOW (for flows[], sources[], targets[], values[] data):
- sankey: Resource flows, conversions, budget allocation
- alluvial: Evolution over time, categorical shifts
- flowchart: Processes, decision trees
- value_stream: Process efficiency, bottlenecks

TEMPORAL (for events[], dates[] data):
- timeline: Event sequences, history
- gantt: Overlapping durations, project schedules
- parallel_timelines: Simultaneous developments
- cycle_diagram: Recurring patterns, boom-bust

COMPARATIVE (for items[], scores[] data):
- matrix_heatmap: Values across two dimensions
- quadrant_chart: Power-interest, impact-effort (2x2)
- radar_chart: Multi-dimensional comparison
- bar_chart: Ranked comparisons

PART-OF-WHOLE (for categories[], sizes[] data):
- treemap: Hierarchical composition by size
- sunburst: Hierarchical composition radially
- stacked_bar: Composition across categories
- waterfall: Cumulative sequential changes

EVIDENCE/ANALYTICAL:
- ach_matrix: Hypotheses vs evidence consistency
- confidence_thermometer: Confidence levels for findings
- indicator_dashboard: Warning indicator status
- gap_analysis_visual: Current vs desired state

ARGUMENTATIVE:
- argument_tree: Logical structure (claims, premises)
- scenario_cone: Multiple possible futures
- dialectical_map: Thesis-antithesis-synthesis
"""

TEXTUAL_FORMAT_KNOWLEDGE = """
TEXTUAL OUTPUT FORMATS:

1. snapshot: ~400 words, "What do I need to know NOW?"
   - For: Any data, executive audience
   - Structure: Bottom line, key finding, implications, confidence

2. deep_dive: 2000-5000 words, "What's the full picture?"
   - For: Complex multi-faceted analysis
   - Structure: Key judgments, competing hypotheses, detailed analysis, outlook

3. evidence_pack: Variable length, "What's the evidence chain?"
   - For: Source documentation, verification
   - Structure: Evidence index, detailed items, contradictions, gaps

4. signal_report: ~800 words, "What signals indicate change?"
   - For: Detection/warning data
   - Structure: Signal summary, indicators, threshold analysis, watch list

5. status_brief: ~1200 words, "What changed and where are we?"
   - For: Temporal/evolution data
   - Structure: Situation summary, developments, change from previous, outlook

6. stakeholder_profile: ~1500 words/actor, "Who is this and how will they act?"
   - For: Actor analysis
   - Structure: Summary, interests, red lines, decision patterns, predictions

7. gap_analysis: ~1500 words, "Where are the weaknesses?"
   - For: Argument/logic data
   - Structure: Vulnerability inventory, detailed analysis, mitigations

8. options_brief: ~1200 words, "What should I choose?"
   - For: Comparison/decision data
   - Structure: Decision required, options with tradeoffs, recommendation
"""

STRUCTURED_FORMAT_KNOWLEDGE = """
STRUCTURED OUTPUT FORMATS:

- smart_table: Dynamic columns optimized for data structure
- matrix_table: Rows × Columns for comparisons
- evidence_table: Claim | Source | Reliability Rating
- timeline_table: Date | Event | Significance
- text_qna: Question-Answer pairs for key findings
- ranked_list: Ordered items with scores
- hierarchy_list: Nested bullet points
"""


class OutputCurator:
    """
    LLM-powered agent that recommends output formats based on extracted data.

    Uses Claude Opus 4.5 with extended thinking for deep analysis.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        thinking_budget: int = 16000,  # Generous thinking budget
    ):
        """
        Initialize the Output Curator.

        Args:
            api_key: Anthropic API key (defaults to ANTHROPIC_API_KEY env var)
            thinking_budget: Token budget for extended thinking (default 16000)
        """
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("Anthropic API key required")

        self.client = anthropic.Anthropic(api_key=self.api_key)
        self.model = "claude-opus-4-5-20251101"
        self.thinking_budget = thinking_budget

    def curate(
        self,
        engine_key: str,
        extracted_data: dict[str, Any],
        audience: str = "analyst",
        context: Optional[str] = None,
    ) -> CuratorOutput:
        """
        Analyze extracted data and recommend output formats.

        Args:
            engine_key: The engine that produced the data
            extracted_data: The structured data from the engine
            audience: Target audience (analyst, executive, researcher, etc.)
            context: Optional additional context about the analysis

        Returns:
            CuratorOutput with recommendations and rationale
        """
        logger.info(f"Curating output for engine: {engine_key}, audience: {audience}")

        # Build the prompt
        prompt = self._build_prompt(engine_key, extracted_data, audience, context)

        # Call Opus 4.5 with extended thinking (streaming required)
        thinking_content = ""
        response_content = ""

        with self.client.messages.stream(
            model=self.model,
            max_tokens=16000,
            thinking={
                "type": "enabled",
                "budget_tokens": self.thinking_budget
            },
            messages=[{"role": "user", "content": prompt}]
        ) as stream:
            for event in stream:
                if event.type == "content_block_delta":
                    if hasattr(event.delta, "thinking"):
                        thinking_content += event.delta.thinking
                    elif hasattr(event.delta, "text"):
                        response_content += event.delta.text

        logger.info(f"Curator thinking: {len(thinking_content)} chars")
        logger.info(f"Curator response: {len(response_content)} chars")

        # Parse the response
        return self._parse_response(response_content, thinking_content)

    def _build_prompt(
        self,
        engine_key: str,
        extracted_data: dict[str, Any],
        audience: str,
        context: Optional[str],
    ) -> str:
        """Build the prompt for the curator."""

        # Format extracted data for the prompt
        data_str = json.dumps(extracted_data, indent=2, default=str)
        if len(data_str) > 10000:
            # Truncate but show structure
            data_str = data_str[:10000] + "\n... [truncated, total keys: " + str(len(extracted_data)) + "]"

        prompt = f"""You are the Output Curator, an expert in information visualization and document intelligence.

Your task: Analyze the extracted data from an analysis engine and recommend the optimal output formats.

## KNOWLEDGE BASE

{VISUAL_FORMAT_KNOWLEDGE}

{TEXTUAL_FORMAT_KNOWLEDGE}

{STRUCTURED_FORMAT_KNOWLEDGE}

## INPUT

Engine: {engine_key}
Audience: {audience}
{f"Context: {context}" if context else ""}

Extracted Data:
```json
{data_str}
```

## YOUR TASK

1. ANALYZE the data structure:
   - What are the key data types present? (nodes/edges, flows, events, scores, etc.)
   - What relationships exist in the data?
   - What is the cardinality (how many items of each type)?
   - What dimensions can be visualized?

2. RECOMMEND output formats:
   - PRIMARY: The single best format for this data + audience
   - SECONDARY: 2-3 alternative formats that would also work well
   - For each, explain WHY it fits this data structure

3. For VISUAL formats, generate a specific Gemini prompt:
   - Be explicit about visualization type
   - Map data fields to visual properties (size, color, position)
   - Include styling guidance
   - Request appropriate labels and legends

4. Consider the AUDIENCE:
   - Executive: Prefers high-level, actionable (quadrant, snapshot)
   - Analyst: Prefers detailed, exploratory (network, deep_dive)
   - Researcher: Prefers evidence-focused (evidence_pack, citations)

## OUTPUT FORMAT

Respond in this exact JSON structure:

```json
{{
  "data_structure_analysis": "Description of the data structure, key types, relationships, cardinality",
  "primary_recommendation": {{
    "format_key": "e.g., sankey or deep_dive",
    "category": "visual|textual|structured|diagram",
    "name": "Human-readable name",
    "confidence": 0.95,
    "rationale": "Why this format is ideal for this data and audience",
    "gemini_prompt": "Full prompt for Gemini if visual format, null otherwise",
    "data_mapping": {{"size": "field_name", "color": "field_name", "etc": "..."}}
  }},
  "secondary_recommendations": [
    {{
      "format_key": "...",
      "category": "...",
      "name": "...",
      "confidence": 0.8,
      "rationale": "...",
      "gemini_prompt": "...",
      "data_mapping": {{}}
    }}
  ],
  "audience_considerations": "How the audience preference influenced recommendations",
  "thinking_summary": "Brief summary of your analytical process"
}}
```

Think deeply about the data structure and what visualization would best reveal its insights.
"""
        return prompt

    def _parse_response(
        self,
        response_content: str,
        thinking_content: str,
    ) -> CuratorOutput:
        """Parse the curator's response into structured output."""

        # Extract JSON from response
        try:
            # Find JSON block
            if "```json" in response_content:
                json_str = response_content.split("```json")[1].split("```")[0]
            elif "```" in response_content:
                json_str = response_content.split("```")[1].split("```")[0]
            else:
                json_str = response_content

            data = json.loads(json_str.strip())

            # Parse primary recommendation
            primary = data.get("primary_recommendation", {})
            primary_rec = FormatRecommendation(
                format_key=primary.get("format_key", "deep_dive"),
                category=primary.get("category", "textual"),
                name=primary.get("name", "Deep Dive"),
                confidence=primary.get("confidence", 0.8),
                rationale=primary.get("rationale", ""),
                gemini_prompt=primary.get("gemini_prompt"),
                data_mapping=primary.get("data_mapping"),
            )

            # Parse secondary recommendations
            secondary_recs = []
            for sec in data.get("secondary_recommendations", []):
                secondary_recs.append(FormatRecommendation(
                    format_key=sec.get("format_key", ""),
                    category=sec.get("category", ""),
                    name=sec.get("name", ""),
                    confidence=sec.get("confidence", 0.5),
                    rationale=sec.get("rationale", ""),
                    gemini_prompt=sec.get("gemini_prompt"),
                    data_mapping=sec.get("data_mapping"),
                ))

            return CuratorOutput(
                data_structure_analysis=data.get("data_structure_analysis", ""),
                primary_recommendation=primary_rec,
                secondary_recommendations=secondary_recs,
                audience_considerations=data.get("audience_considerations", ""),
                thinking_summary=data.get("thinking_summary", ""),
                raw_thinking=thinking_content if thinking_content else None,
            )

        except Exception as e:
            logger.error(f"Failed to parse curator response: {e}")
            logger.error(f"Response was: {response_content[:500]}...")

            # Return a default recommendation
            return CuratorOutput(
                data_structure_analysis="Failed to analyze - using defaults",
                primary_recommendation=FormatRecommendation(
                    format_key="deep_dive",
                    category="textual",
                    name="Deep Dive",
                    confidence=0.5,
                    rationale="Default recommendation due to parsing error",
                ),
                secondary_recommendations=[
                    FormatRecommendation(
                        format_key="smart_table",
                        category="structured",
                        name="Smart Table",
                        confidence=0.5,
                        rationale="Default alternative",
                    )
                ],
                audience_considerations="Unable to analyze",
                thinking_summary=f"Error: {str(e)}",
                raw_thinking=thinking_content,
            )

    def curate_multiple(
        self,
        engine_outputs: dict[str, dict[str, Any]],
        audience: str = "analyst",
        context: Optional[str] = None,
    ) -> dict[str, CuratorOutput]:
        """
        Curate output formats for multiple engine results.

        Args:
            engine_outputs: Dict mapping engine_key -> extracted_data
            audience: Target audience
            context: Optional context

        Returns:
            Dict mapping engine_key -> CuratorOutput
        """
        results = {}
        for engine_key, extracted_data in engine_outputs.items():
            try:
                results[engine_key] = self.curate(
                    engine_key=engine_key,
                    extracted_data=extracted_data,
                    audience=audience,
                    context=context,
                )
            except Exception as e:
                logger.error(f"Failed to curate for {engine_key}: {e}")
                results[engine_key] = CuratorOutput(
                    data_structure_analysis=f"Error: {str(e)}",
                    primary_recommendation=FormatRecommendation(
                        format_key="deep_dive",
                        category="textual",
                        name="Deep Dive",
                        confidence=0.3,
                        rationale="Default due to error",
                    ),
                    secondary_recommendations=[],
                    audience_considerations="",
                    thinking_summary=f"Error occurred: {str(e)}",
                )
        return results

    def generate_gemini_prompt(
        self,
        format_key: str,
        extracted_data: dict[str, Any],
        style: str = "professional",
    ) -> str:
        """
        Generate an optimized Gemini prompt for a specific visual format.

        This is a fallback/utility method when you already know the format.
        """
        # Simplified prompt generation without full curator analysis
        data_str = json.dumps(extracted_data, indent=2, default=str)[:5000]

        prompt = f"""You are generating a Gemini prompt for a {format_key} visualization.

Data to visualize:
{data_str}

Generate a detailed prompt that:
1. Specifies the exact visualization type
2. Maps data fields to visual properties
3. Includes styling for {style} presentation
4. Requests clear labels and legend

Output only the Gemini prompt, nothing else."""

        # Use a faster model for this utility function
        response = self.client.messages.create(
            model="claude-sonnet-4-5-20250929",
            max_tokens=2000,
            messages=[{"role": "user", "content": prompt}]
        )

        return response.content[0].text


# Convenience function
def curate_output(
    engine_key: str,
    extracted_data: dict[str, Any],
    audience: str = "analyst",
    context: Optional[str] = None,
    api_key: Optional[str] = None,
    thinking_budget: int = 16000,
) -> CuratorOutput:
    """
    Convenience function to curate output formats for engine results.

    Args:
        engine_key: The engine that produced the data
        extracted_data: The structured data from the engine
        audience: Target audience
        context: Optional context
        api_key: Anthropic API key
        thinking_budget: Token budget for extended thinking

    Returns:
        CuratorOutput with recommendations
    """
    curator = OutputCurator(api_key=api_key, thinking_budget=thinking_budget)
    return curator.curate(
        engine_key=engine_key,
        extracted_data=extracted_data,
        audience=audience,
        context=context,
    )


# Export for module
__all__ = [
    "OutputCurator",
    "CuratorOutput",
    "FormatRecommendation",
    "OutputCategory",
    "curate_output",
]
