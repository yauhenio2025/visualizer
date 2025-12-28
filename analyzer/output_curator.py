"""
Output Curator Agent

An LLM-powered agent (Claude Opus 4.5 with extended thinking) that analyzes
extracted data from engines and recommends optimal output formats.

The curator:
1. Detects data structure from engine output
2. Recommends visual, textual, and structured formats
3. Generates optimized Gemini prompts for visual outputs
4. Applies style curation from 5 major dataviz schools
5. Explains its reasoning

Uses streaming for extended thinking support.

Style Schools (integrated via style_curator.py):
- Tufte: Maximum data-ink ratio, no chartjunk
- NYT/Cox: Explanatory, annotated, reader-friendly
- FT/Burn-Murdoch: Restrained elegance, high-signal
- Lupi/Data Humanism: Ornate but rigorous, emotional
- Stefaner/Truth+Beauty: Complex, cultural, artistic
"""

import os
import json
import logging
from typing import Optional, Any
from dataclasses import dataclass, asdict, field
from enum import Enum

import anthropic

# Import style curator for integration
from .style_curator import (
    StyleCurator,
    StyleRecommendation,
    StyleSchool,
    get_quick_style,
    merge_gemini_prompt_with_style,
    STYLE_GUIDES,
)

# Import display utilities for data sanitization
from .display_utils import (
    sanitize_for_display,
    format_label,
    get_display_instructions,
)

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
    # Style integration fields
    style_school: Optional[str] = None  # e.g., "tufte", "nyt_cox", etc.
    style_name: Optional[str] = None  # Human-readable style name
    styled_gemini_prompt: Optional[str] = None  # Gemini prompt with style applied


@dataclass
class CuratorOutput:
    """Complete output from the curator."""
    data_structure_analysis: str
    primary_recommendation: FormatRecommendation
    secondary_recommendations: list[FormatRecommendation]
    audience_considerations: str
    thinking_summary: str
    raw_thinking: Optional[str] = None
    # Style curation fields
    style_rationale: Optional[str] = None
    style_thinking: Optional[str] = None


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

    Now includes style curation from 5 major dataviz schools:
    - Tufte: Maximum data-ink ratio, no chartjunk
    - NYT/Cox: Explanatory, annotated, reader-friendly
    - FT/Burn-Murdoch: Restrained elegance, high-signal
    - Lupi/Data Humanism: Ornate but rigorous, emotional
    - Stefaner/Truth+Beauty: Complex, cultural, artistic
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        thinking_budget: int = 16000,  # Generous thinking budget
        enable_style_curation: bool = True,  # Enable style selection
        use_quick_style: bool = False,  # Use fast affinity-based style (no LLM)
    ):
        """
        Initialize the Output Curator.

        Args:
            api_key: Anthropic API key (defaults to ANTHROPIC_API_KEY env var)
            thinking_budget: Token budget for extended thinking (default 16000)
            enable_style_curation: Whether to apply dataviz style curation
            use_quick_style: Use fast affinity-based style selection (no LLM call)
        """
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("Anthropic API key required")

        self.client = anthropic.Anthropic(api_key=self.api_key)
        self.model = "claude-opus-4-5-20251101"
        self.thinking_budget = thinking_budget
        self.enable_style_curation = enable_style_curation
        self.use_quick_style = use_quick_style

        # Initialize style curator if enabled (and not using quick mode)
        self.style_curator = None
        if enable_style_curation and not use_quick_style:
            self.style_curator = StyleCurator(
                api_key=self.api_key,
                thinking_budget=10000,  # Smaller budget for style
            )

    def curate(
        self,
        engine_key: str,
        extracted_data: dict[str, Any],
        audience: str = "analyst",
        context: Optional[str] = None,
        compatible_formats: Optional[list[str]] = None,
    ) -> CuratorOutput:
        """
        Analyze extracted data and recommend output formats.

        Args:
            engine_key: The engine that produced the data
            extracted_data: The structured data from the engine
            audience: Target audience (analyst, executive, researcher, etc.)
            context: Optional additional context about the analysis
            compatible_formats: Optional list of format_keys to constrain recommendations to.
                               First format is the primary recommendation if provided.

        Returns:
            CuratorOutput with recommendations and rationale
        """
        logger.info(f"Curating output for engine: {engine_key}, audience: {audience}")
        if compatible_formats:
            logger.info(f"Constraining to compatible formats: {compatible_formats}")

        # Build the prompt
        prompt = self._build_prompt(engine_key, extracted_data, audience, context, compatible_formats)

        # Call Opus 4.5 with extended thinking (streaming required)
        thinking_content = ""
        response_content = ""

        # max_tokens must be greater than thinking.budget_tokens
        max_tokens = self.thinking_budget + 8000  # thinking budget + room for response

        with self.client.messages.stream(
            model=self.model,
            max_tokens=max_tokens,
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
        result = self._parse_response(response_content, thinking_content)

        # Apply style curation to visual recommendations
        if self.enable_style_curation:
            result = self._apply_style_curation(result, engine_key, audience, extracted_data, context)

        return result

    def _apply_style_curation(
        self,
        result: "CuratorOutput",
        engine_key: str,
        audience: str,
        extracted_data: dict[str, Any],
        context: Optional[str],
    ) -> "CuratorOutput":
        """
        Apply style curation to visual recommendations.

        For each visual format recommendation, selects the optimal style
        and merges style instructions into the Gemini prompt.
        """
        style_thinking = None

        def apply_style_to_recommendation(rec: FormatRecommendation) -> FormatRecommendation:
            """Apply style to a single recommendation."""
            if rec.category != "visual" or not rec.gemini_prompt:
                return rec

            # Get style recommendation
            if self.use_quick_style:
                style_rec = get_quick_style(engine_key, rec.format_key, audience)
                logger.info(f"Quick style for {rec.format_key}: {style_rec.school.value}")
            elif self.style_curator:
                try:
                    style_output = self.style_curator.curate_style(
                        engine_key=engine_key,
                        format_key=rec.format_key,
                        extracted_data=extracted_data,
                        audience=audience,
                        document_context=context,
                    )
                    style_rec = style_output.primary_style
                    logger.info(f"LLM style for {rec.format_key}: {style_rec.school.value}")
                except Exception as e:
                    logger.warning(f"Style curation failed, using quick style: {e}")
                    style_rec = get_quick_style(engine_key, rec.format_key, audience)
            else:
                style_rec = get_quick_style(engine_key, rec.format_key, audience)

            # Merge style into Gemini prompt
            styled_prompt = merge_gemini_prompt_with_style(
                rec.gemini_prompt,
                style_rec.gemini_style_instructions,
            )

            # Update recommendation with style info
            rec.style_school = style_rec.school.value
            rec.style_name = style_rec.style_guide.name
            rec.styled_gemini_prompt = styled_prompt

            return rec

        # Apply style to primary recommendation
        result.primary_recommendation = apply_style_to_recommendation(result.primary_recommendation)

        # Apply style to secondary recommendations
        result.secondary_recommendations = [
            apply_style_to_recommendation(rec)
            for rec in result.secondary_recommendations
        ]

        # Add style rationale
        if result.primary_recommendation.style_school:
            result.style_rationale = f"Applied {result.primary_recommendation.style_name} style based on engine type, format, and audience"

        return result

    def _build_prompt(
        self,
        engine_key: str,
        extracted_data: dict[str, Any],
        audience: str,
        context: Optional[str],
        compatible_formats: Optional[list[str]] = None,
    ) -> str:
        """Build the prompt for the curator."""

        # Sanitize extracted data for display:
        # - Convert snake_case keys to Title Case
        # - Remove internal score fields that shouldn't appear on visualizations
        sanitized_data = sanitize_for_display(
            extracted_data,
            format_keys=True,
            convert_scores=True,
            hide_score_fields=False,  # Keep fields but transform values
        )

        # Format sanitized data for the prompt
        data_str = json.dumps(sanitized_data, indent=2, default=str)
        if len(data_str) > 10000:
            # Truncate but show structure
            data_str = data_str[:10000] + "\n... [truncated, total keys: " + str(len(sanitized_data)) + "]"

        # Build compatible formats constraint section
        format_constraint = ""
        if compatible_formats:
            format_list = ", ".join(compatible_formats)
            format_constraint = f"""
## COMPATIBLE FORMATS CONSTRAINT

IMPORTANT: For visual formats, you MUST choose ONLY from these compatible formats:
{format_list}

These formats have been pre-selected as appropriate for the "{engine_key}" engine's output structure.
The first format ({compatible_formats[0]}) is the primary recommendation based on archetype analysis.

Do NOT recommend visual formats outside this list.
"""

        prompt = f"""You are the Output Curator, an expert in information visualization and document intelligence.

Your task: Analyze the extracted data from an analysis engine and recommend the optimal output formats.

## KNOWLEDGE BASE

{VISUAL_FORMAT_KNOWLEDGE}

{TEXTUAL_FORMAT_KNOWLEDGE}

{STRUCTURED_FORMAT_KNOWLEDGE}
{format_constraint}
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
   - PRIMARY: The single best format for this data + audience{" (from compatible formats list)" if compatible_formats else ""}
   - SECONDARY: 2-3 alternative formats that would also work well{" (from compatible formats list for visual)" if compatible_formats else ""}
   - For each, explain WHY it fits this data structure

3. For VISUAL formats, generate a specific Gemini prompt:
   - Be explicit about visualization type
   - Map data fields to visual properties (size, color, position)
   - Include styling guidance
   - Request appropriate labels and legends
   - IMPORTANT: Instruct Gemini to NEVER display raw numeric scores (0.85, 0.75) on the visualization
   - IMPORTANT: Instruct Gemini to convert any snake_case identifiers to Title Case with spaces

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

    def curate_batch_by_engines(
        self,
        engine_keys: list[str],
        audience: str = "analyst",
        context: Optional[str] = None,
        sample_data: Optional[dict[str, Any]] = None,
        compatible_formats_map: Optional[dict[str, list[str]]] = None,
    ) -> dict[str, FormatRecommendation]:
        """
        Recommend visualization formats for multiple engines in ONE API call.

        Unlike curate() which needs extracted data, this method recommends formats
        based on engine knowledge - what kind of output each engine typically produces.

        Args:
            engine_keys: List of engine keys to get recommendations for
            audience: Target audience (analyst, executive, researcher)
            context: Optional context about the analysis
            sample_data: Optional sample data to help inform recommendations
            compatible_formats_map: Dict mapping engine_key -> list of compatible format_keys.
                                   If provided, recommendations will be constrained to these formats.

        Returns:
            Dict mapping engine_key -> FormatRecommendation (primary only)
        """
        if not engine_keys:
            return {}

        logger.info(f"Batch curating formats for {len(engine_keys)} engines: {engine_keys}")
        if compatible_formats_map:
            logger.info(f"Constraining to compatible formats per engine")

        # Build the batch prompt
        prompt = self._build_batch_prompt(engine_keys, audience, context, sample_data, compatible_formats_map)

        # Call Opus 4.5 with extended thinking (streaming required)
        thinking_content = ""
        response_content = ""

        max_tokens = self.thinking_budget + 8000

        with self.client.messages.stream(
            model=self.model,
            max_tokens=max_tokens,
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

        logger.info(f"Batch curator thinking: {len(thinking_content)} chars")
        logger.info(f"Batch curator response: {len(response_content)} chars")

        # Parse the batch response
        return self._parse_batch_response(response_content, engine_keys)

    def _build_batch_prompt(
        self,
        engine_keys: list[str],
        audience: str,
        context: Optional[str],
        sample_data: Optional[dict[str, Any]],
        compatible_formats_map: Optional[dict[str, list[str]]] = None,
    ) -> str:
        """Build prompt for batch format recommendation."""

        # Build engine list with compatible formats constraints
        engines_with_constraints = []
        for key in engine_keys:
            if compatible_formats_map and key in compatible_formats_map:
                formats = compatible_formats_map[key]
                engines_with_constraints.append(
                    f"- {key} → MUST choose from: [{', '.join(formats)}]"
                )
            else:
                engines_with_constraints.append(f"- {key}")
        engines_list = "\n".join(engines_with_constraints)

        sample_str = ""
        if sample_data:
            # Sanitize sample data to show clean format
            sanitized_sample = sanitize_for_display(sample_data, format_keys=True, convert_scores=True)
            sample_str = f"\nSample data from documents:\n```json\n{json.dumps(sanitized_sample, indent=2, default=str)[:3000]}\n```\n"

        # Add constraint section if we have compatible formats
        format_constraint = ""
        if compatible_formats_map:
            format_constraint = """
## COMPATIBLE FORMATS CONSTRAINT

IMPORTANT: For each engine, you MUST choose ONLY from the compatible formats listed next to it.
These have been pre-selected based on each engine's archetype and typical output structure.
Do NOT recommend formats outside the allowed list for each engine.
"""

        prompt = f"""You are the Output Curator, an expert in information visualization.

Your task: Recommend the BEST visualization format for EACH of these analysis engines.

## KNOWLEDGE BASE

{VISUAL_FORMAT_KNOWLEDGE}
{format_constraint}
## ENGINES TO RECOMMEND FOR

{engines_list}

## TARGET AUDIENCE: {audience}
{f"Context: {context}" if context else ""}
{sample_str}

## ENGINE OUTPUT PATTERNS

Based on your knowledge, these engines typically produce:

- stakeholder_power_interest → 2D positioning data (power vs interest scores per actor)
- resource_flow_asymmetry → flow data (source → target with values)
- event_timeline_causal → temporal data (events with dates and causal links)
- argument_architecture → logical structure (claims, premises, evidence)
- thematic_synthesis → categorical data (themes with supporting quotes)
- opportunity_vulnerability_matrix → comparative data (options × criteria scores)
- competitive_landscape → positioning data (players × dimensions)
- concept_evolution → temporal data (concept changes over time)
- dialectical_structure → thesis-antithesis-synthesis patterns
- citation_network → relational data (nodes and citation edges)
- intellectual_genealogy → hierarchical/temporal lineage
- metaphor_analogy_network → relational data (source-target mappings)

## YOUR TASK

For EACH engine listed above, recommend the SINGLE BEST visualization format{" from its allowed list" if compatible_formats_map else ""}.

Consider:
1. What data structure does this engine typically output?
2. What visualization best reveals insights from that structure?
3. How does the audience preference affect the choice?

## OUTPUT FORMAT

Respond with a JSON object mapping each engine_key to its recommended format:

```json
{{
  "engine_key_1": {{
    "format_key": "quadrant_chart",
    "name": "Power-Interest Quadrant",
    "confidence": 0.9,
    "rationale": "Stakeholder data naturally maps to 2D positioning..."
  }},
  "engine_key_2": {{
    "format_key": "sankey",
    "name": "Resource Flow Diagram",
    "confidence": 0.85,
    "rationale": "Flow data best shown as Sankey..."
  }}
}}
```

Recommend formats for ALL {len(engine_keys)} engines: {', '.join(engine_keys)}
"""
        return prompt

    def _parse_batch_response(
        self,
        response_content: str,
        engine_keys: list[str],
    ) -> dict[str, FormatRecommendation]:
        """Parse batch curator response into per-engine recommendations."""

        results = {}

        try:
            # Extract JSON from response
            if "```json" in response_content:
                json_str = response_content.split("```json")[1].split("```")[0]
            elif "```" in response_content:
                json_str = response_content.split("```")[1].split("```")[0]
            else:
                json_str = response_content

            data = json.loads(json_str.strip())

            for engine_key in engine_keys:
                if engine_key in data:
                    rec_data = data[engine_key]
                    results[engine_key] = FormatRecommendation(
                        format_key=rec_data.get("format_key", "network_graph"),
                        category="visual",  # Batch mode focuses on visual formats
                        name=rec_data.get("name", rec_data.get("format_key", "").replace("_", " ").title()),
                        confidence=rec_data.get("confidence", 0.7),
                        rationale=rec_data.get("rationale", ""),
                        gemini_prompt=rec_data.get("gemini_prompt"),
                        data_mapping=rec_data.get("data_mapping"),
                    )
                else:
                    # Engine not in response - use default
                    results[engine_key] = FormatRecommendation(
                        format_key="network_graph",
                        category="visual",
                        name="Network Graph",
                        confidence=0.5,
                        rationale="Default recommendation - engine not specifically analyzed",
                    )

        except Exception as e:
            logger.error(f"Failed to parse batch response: {e}")
            logger.error(f"Response was: {response_content[:500]}...")

            # Return defaults for all engines
            for engine_key in engine_keys:
                results[engine_key] = FormatRecommendation(
                    format_key="network_graph",
                    category="visual",
                    name="Network Graph",
                    confidence=0.3,
                    rationale=f"Default due to parsing error: {str(e)}",
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
        # Sanitize data before including in prompt
        sanitized_data = sanitize_for_display(
            extracted_data,
            format_keys=True,
            convert_scores=True,
            hide_score_fields=False,
        )
        data_str = json.dumps(sanitized_data, indent=2, default=str)[:5000]

        # Get display instructions
        display_instructions = get_display_instructions()

        prompt = f"""You are generating a Gemini prompt for a {format_key} visualization.

Data to visualize:
{data_str}

Generate a detailed prompt that:
1. Specifies the exact visualization type
2. Maps data fields to visual properties
3. Includes styling for {style} presentation
4. Requests clear labels and legend
5. NEVER displays raw numeric scores (0.85, 0.75) on the visualization
6. Converts all snake_case identifiers to Title Case with spaces

Include these display formatting rules in your generated prompt:
{display_instructions}

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
    compatible_formats: Optional[list[str]] = None,
    enable_style_curation: bool = True,
    use_quick_style: bool = False,
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
        compatible_formats: Optional list of format_keys to constrain recommendations to
        enable_style_curation: Whether to apply dataviz style curation (default True)
        use_quick_style: Use fast affinity-based style without LLM call (default False)

    Returns:
        CuratorOutput with recommendations (including style-enhanced Gemini prompts)
    """
    curator = OutputCurator(
        api_key=api_key,
        thinking_budget=thinking_budget,
        enable_style_curation=enable_style_curation,
        use_quick_style=use_quick_style,
    )
    return curator.curate(
        engine_key=engine_key,
        extracted_data=extracted_data,
        audience=audience,
        context=context,
        compatible_formats=compatible_formats,
    )


# Export for module
__all__ = [
    "OutputCurator",
    "CuratorOutput",
    "FormatRecommendation",
    "OutputCategory",
    "curate_output",
    # Style curation exports (re-exported from style_curator)
    "StyleCurator",
    "StyleRecommendation",
    "StyleSchool",
    "STYLE_GUIDES",
    "get_quick_style",
    "merge_gemini_prompt_with_style",
]
