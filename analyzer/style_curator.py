"""
Style Curator Agent

An LLM-powered agent (Claude Opus 4.5 with extended thinking) that selects
the optimal visualization style for each engine+format combination.

The curator draws on a knowledge base of 5 major dataviz schools/practitioners:
1. Tufte/Classic - Maximum data-ink ratio, no chartjunk
2. NYT/Cox - Explanatory, annotated, reader-friendly
3. FT/Burn-Murdoch - Restrained elegance, high-signal
4. Lupi/Data Humanism - Ornate but rigorous, emotional
5. Stefaner/Truth+Beauty - Complex, cultural, artistic precision

Each style school has distinct:
- Color palettes
- Typography guidance
- Layout principles
- Annotation philosophy
- Gemini prompt modifiers

The curator uses extended thinking to match content to the most appropriate style.
"""

import os
import json
import logging
from typing import Optional, Any
from dataclasses import dataclass, asdict, field
from enum import Enum

import anthropic

from .display_utils import get_display_instructions

logger = logging.getLogger(__name__)


class StyleSchool(Enum):
    TUFTE = "tufte"
    NYT_COX = "nyt_cox"
    FT_BURN_MURDOCH = "ft_burn_murdoch"
    LUPI_DATA_HUMANISM = "lupi_data_humanism"
    STEFANER_TRUTH_BEAUTY = "stefaner_truth_beauty"


@dataclass
class StyleGuide:
    """Complete style specification for visualization."""
    school: StyleSchool
    name: str
    philosophy: str
    color_palette: dict[str, str]
    typography: dict[str, str]
    layout_principles: list[str]
    annotation_style: str
    gemini_modifiers: str  # Specific instructions to append to Gemini prompts
    best_for: list[str]  # What this style excels at
    avoid_for: list[str]  # When to avoid this style


@dataclass
class StyleRecommendation:
    """A style recommendation with rationale."""
    school: StyleSchool
    style_guide: StyleGuide
    confidence: float  # 0.0 - 1.0
    rationale: str
    gemini_style_instructions: str  # Ready to append to Gemini prompt


@dataclass
class StyleCuratorOutput:
    """Complete output from the style curator."""
    primary_style: StyleRecommendation
    secondary_style: Optional[StyleRecommendation]
    style_rationale: str
    thinking_summary: str
    raw_thinking: Optional[str] = None


# ============================================================================
# STYLE GUIDE KNOWLEDGE BASE
# ============================================================================

TUFTE_STYLE = StyleGuide(
    school=StyleSchool.TUFTE,
    name="Tufte / Classic Statistical Graphics",
    philosophy="""
    Edward Tufte's principles from 'The Visual Display of Quantitative Information':
    - Maximize the data-ink ratio: Every drop of ink should convey data
    - Eliminate chartjunk: No decorative grids, unnecessary frames, or ornament
    - Small multiples: Repeat small, consistent charts to enable comparison
    - Sparklines: Dense, word-sized graphics embedded in text
    - Show the data: Let the numbers speak, minimize interpretation
    - Micro/macro readings: Works at glance AND on close inspection
    - Graphical integrity: No lie factors, no misleading perspectives
    """,
    color_palette={
        "primary": "#1a1a1a",  # Near-black for data
        "secondary": "#666666",  # Medium gray for secondary elements
        "tertiary": "#999999",  # Light gray for grids (if any)
        "accent": "#b8860b",  # Dark goldenrod for selective emphasis
        "background": "#ffffff",  # Pure white
        "highlight": "#cc0000",  # Red only for critical emphasis
        "text": "#333333",  # Dark gray text
        "muted": "#cccccc",  # Very light gray for de-emphasized
    },
    typography={
        "primary_font": "Georgia, serif",  # Classic, readable serif
        "title_font": "Georgia, serif",  # Same for consistency
        "caption_font": "Georgia, serif",
        "number_font": "Georgia, serif",  # Numbers should be clear
        "title_size": "18-24px",
        "label_size": "10-12px",
        "annotation_size": "10-11px",
        "line_height": "1.4",
        "title_weight": "normal",  # Not bold - Tufte avoids shouting
    },
    layout_principles=[
        "Remove all borders and boxes around charts",
        "Use white space instead of lines to separate elements",
        "Direct labeling on data points instead of legends",
        "Axis lines only if absolutely necessary - often remove them",
        "No grid lines unless essential for reading values",
        "Small multiples arranged in tight grids with shared axes",
        "Data should touch the frame or float in white space",
        "No 3D effects ever",
        "No shadows or gradients",
        "Aspect ratio should match the nature of the data",
    ],
    annotation_style="""
    Minimal and precise. Labels directly on data points where possible.
    No separate legend if elements can be labeled directly.
    Annotations as small, precise notes without boxes or callouts.
    Use thin lines to connect annotations to data if needed.
    Words integrated with graphics, not separated.
    """,
    gemini_modifiers="""
STYLE INSTRUCTIONS (Tufte / Classic Statistical Graphics):
- Use MAXIMUM data-ink ratio: every pixel should convey information
- NO chartjunk: no decorative grids, 3D effects, shadows, or gradients
- Color palette: near-black (#1a1a1a) for data, white background, minimal gray for structure
- Remove all unnecessary borders, boxes, and frames
- Direct label data points instead of using legends
- Axis lines minimal or absent if scale is clear from data
- Use small multiples if showing comparisons
- Typography: Georgia or classic serif, normal weight, no bold titles
- White space as separator, not lines
- Precision over decoration
- If uncertain, remove rather than add
    """,
    best_for=[
        "Statistical analysis results",
        "Time series with multiple variables",
        "Comparisons across many categories",
        "Academic and research presentations",
        "Financial data for sophisticated audiences",
        "Small multiples showing change over time",
        "Data-dense displays for expert readers",
    ],
    avoid_for=[
        "General public audiences needing guidance",
        "Emotional or narrative-driven content",
        "Brand-heavy corporate communications",
        "Content requiring visual impact over precision",
    ],
)


NYT_COX_STYLE = StyleGuide(
    school=StyleSchool.NYT_COX,
    name="NYT / Amanda Cox / Explanatory Graphics",
    philosophy="""
    The New York Times graphics desk tradition, refined by Amanda Cox:
    - Graphics that EXPLAIN, not just display
    - Reader's eye is guided through a story
    - Annotations as teaching moments
    - Accessibility for general audiences while respecting intelligence
    - Interactive principles even in static form
    - The 'aha moment' - visualization reveals insight
    - Clarity is kindness to the reader
    - Every chart answers a question
    """,
    color_palette={
        "primary": "#333333",  # NYT gray for text and data
        "secondary": "#666666",
        "tertiary": "#aaaaaa",
        "accent": "#d63232",  # NYT red for emphasis
        "accent_alt": "#1e90ff",  # Blue for secondary emphasis
        "background": "#f7f7f5",  # Warm off-white (NYT paper feel)
        "highlight": "#ffcc00",  # Yellow for callouts
        "positive": "#3a9d23",  # Green for positive values
        "negative": "#d63232",  # Red for negative values
        "text": "#333333",
        "annotation_bg": "#fff9e5",  # Pale yellow for annotation boxes
    },
    typography={
        "primary_font": "NYT Franklin, Franklin Gothic, Arial, sans-serif",
        "title_font": "NYT Cheltenham, Georgia, serif",
        "caption_font": "NYT Franklin, Arial, sans-serif",
        "number_font": "NYT Franklin, Arial, sans-serif",
        "title_size": "22-28px",
        "label_size": "11-13px",
        "annotation_size": "12-14px",
        "line_height": "1.5",
        "title_weight": "700",  # Bold headlines
    },
    layout_principles=[
        "Clear headline that states the finding",
        "Subhead or dek that elaborates on the insight",
        "Annotations guide the eye through key points",
        "Source and credit prominently displayed",
        "Generous white space for breathing room",
        "Legend only when direct labeling impossible",
        "Call-out boxes highlight crucial data points",
        "Responsive sizing in mind",
        "Clear visual hierarchy: what to read first is obvious",
        "Axes labeled clearly with units",
    ],
    annotation_style="""
    Generous and explanatory. Use callout boxes with subtle backgrounds.
    Annotations should TEACH the reader something - not just label.
    'What does this mean?' answered at key inflection points.
    Use arrows or thin lines to connect annotations to data.
    Write in complete sentences when explaining significance.
    Header + body pattern for major annotations.
    Keep tone conversational but authoritative.
    """,
    gemini_modifiers="""
STYLE INSTRUCTIONS (NYT / Amanda Cox / Explanatory Graphics):
- Create an EXPLANATORY visualization that teaches the reader
- Include a clear HEADLINE that states the key finding
- Use ANNOTATIONS to guide the reader's eye through important points
- Color palette: #333333 for data, #d63232 for emphasis, warm #f7f7f5 background
- Typography: Sans-serif (Franklin Gothic style) for labels, serif for titles
- Add callout boxes with pale backgrounds to highlight crucial insights
- Write annotations as teaching moments, not just labels
- Include clear axis labels with units
- Source attribution visible at bottom
- Visual hierarchy guides: what to read first, second, third
- Think about what question this chart answers
- Generous white space between elements
    """,
    best_for=[
        "General public audiences",
        "News and journalism contexts",
        "Complex data that needs explanation",
        "Stories with a clear narrative arc",
        "Interactive-inspired static graphics",
        "Election results, economic indicators",
        "Scientific findings for lay audiences",
    ],
    avoid_for=[
        "Expert audiences who find annotations patronizing",
        "Data-dense displays for specialists",
        "Abstract conceptual relationships",
        "Speed-reading contexts (too much annotation)",
    ],
)


FT_BURN_MURDOCH_STYLE = StyleGuide(
    school=StyleSchool.FT_BURN_MURDOCH,
    name="FT / John Burn-Murdoch / Restrained Elegance",
    philosophy="""
    Financial Times graphics tradition, exemplified by John Burn-Murdoch:
    - Extreme restraint - let the data command attention
    - The famous FT salmon pink as signature
    - Line charts as primary vehicle for showing change
    - Multiple series compared with clear differentiation
    - Annotations minimal but precisely placed
    - Professional gravitas appropriate for financial audiences
    - The chart should work in print at 2 columns wide
    - Emphasis through prominence, not decoration
    """,
    color_palette={
        "primary": "#0f5499",  # FT blue for primary series
        "secondary": "#990f3d",  # FT deep red
        "tertiary": "#66a8cd",  # Light blue
        "accent": "#ff7f0e",  # Orange for emphasis
        "background": "#fff1e5",  # FT salmon pink (signature)
        "paper": "#ffffff",  # White for chart area
        "text": "#333333",
        "muted": "#cec6b9",  # Muted for grids
        "highlight_line": "#000000",  # Black for emphasized series
        "series_palette": ["#0f5499", "#990f3d", "#ff7f0e", "#4d8076", "#96cccc"],
    },
    typography={
        "primary_font": "Metric, Financier Sans, sans-serif",
        "title_font": "Financier Display, Georgia, serif",
        "caption_font": "Metric, Arial, sans-serif",
        "number_font": "Metric, sans-serif",
        "title_size": "18-22px",
        "label_size": "10-12px",
        "annotation_size": "10-11px",
        "line_height": "1.35",
        "title_weight": "600",
    },
    layout_principles=[
        "Chart area on white, framed by salmon background",
        "Minimal but precise grid lines in muted tones",
        "Line charts prioritized for time series",
        "Direct labeling at line endpoints",
        "Axis labels rotated if needed for readability",
        "Source: FT always credited bottom left",
        "Compact but not cramped",
        "Works at print column width (small)",
        "Few colors - usually 2-3 series maximum",
        "One series in black or bold if it's the focus",
    ],
    annotation_style="""
    Minimal, surgical. Annotations only at critical inflection points.
    Short phrases, not sentences. Data point labeled directly.
    Use thin rules to connect annotation to data.
    No boxes or backgrounds on annotations.
    Let the data speak - annotations are whispers, not shouts.
    End of line series labels are preferred over legends.
    """,
    gemini_modifiers="""
STYLE INSTRUCTIONS (FT / John Burn-Murdoch / Restrained Elegance):
- Use the SIGNATURE FT color scheme: salmon pink background (#fff1e5), white chart area
- Primary series in FT blue (#0f5499), secondary in FT red (#990f3d)
- RESTRAINT is key - minimal annotation, let data speak
- Prioritize LINE CHARTS for time series data
- Direct labeling at line endpoints, no legend if possible
- Thin, muted grid lines (#cec6b9) if needed
- Sans-serif typography (Metric or similar)
- Source attribution bottom left
- Works at small size - readable at 300px width
- Maximum 3-4 series for clarity
- One series can be emphasized in black/bold
- Annotations as short phrases at inflection points only
- Professional, financial gravitas
- No decorative elements whatsoever
    """,
    best_for=[
        "Time series and line charts",
        "Financial and economic data",
        "Professional/business audiences",
        "Print-ready graphics (column width)",
        "Comparisons across countries/entities over time",
        "COVID-style trajectory charts",
        "Market movements and indicators",
    ],
    avoid_for=[
        "General public needing more context",
        "Network and relationship data",
        "Part-of-whole compositions",
        "Emotional or narrative-driven content",
    ],
)


LUPI_DATA_HUMANISM_STYLE = StyleGuide(
    school=StyleSchool.LUPI_DATA_HUMANISM,
    name="Giorgia Lupi / Data Humanism",
    philosophy="""
    Data Humanism as articulated by Giorgia Lupi:
    - Data represents REAL people and phenomena
    - Imperfection and hand-crafted feel is valuable
    - Complex is not complicated - embrace richness
    - Personal and emotional dimensions of data
    - Each data point has a story
    - Visual complexity that rewards exploration
    - Beauty as a form of respect for the data
    - Context and narrative as primary
    - The 'Dear Data' principle: data as intimate diary
    """,
    color_palette={
        "primary": "#2c2c2c",  # Ink-like dark
        "secondary": "#8b4513",  # Sienna/brown for warmth
        "tertiary": "#c0a080",  # Tan
        "accent": "#c41e3a",  # Deep red for emphasis
        "accent_alt": "#1e5631",  # Forest green
        "background": "#faf7f2",  # Warm cream paper
        "highlight": "#e6b800",  # Gold
        "text": "#2c2c2c",
        "muted": "#b8a99a",
        "organic_palette": ["#c41e3a", "#1e5631", "#1e4d8c", "#c0a080", "#8b4513"],
    },
    typography={
        "primary_font": "Garamond, Georgia, serif",
        "title_font": "Didot, Bodoni, serif",  # Elegant display serif
        "caption_font": "Gill Sans, Optima, sans-serif",
        "number_font": "Garamond, serif",
        "title_size": "24-32px",
        "label_size": "9-11px",
        "annotation_size": "10-12px",
        "line_height": "1.6",
        "title_weight": "normal",  # Elegant, not bold
    },
    layout_principles=[
        "Organic, flowing layouts - not rigid grids",
        "Hand-drawn aesthetic even in digital form",
        "Radial and circular arrangements",
        "Each element can be unique - no strict template",
        "Negative space as compositional element",
        "Layering for depth and discovery",
        "Key/legend as beautiful design element",
        "Small details reward close reading",
        "Title as almost calligraphic element",
        "Works as art piece on the wall",
    ],
    annotation_style="""
    Narrative and personal. Write as if telling a story.
    Annotations can be longer, more contemplative.
    Hand-lettered aesthetic even if typeset.
    Legend as a beautiful decoded key, placed thoughtfully.
    Encourage exploration: 'if you look closely...'
    Connect data to human experience in annotations.
    Poetic register is acceptable.
    """,
    gemini_modifiers="""
STYLE INSTRUCTIONS (Giorgia Lupi / Data Humanism):
- Create a visualization that feels HUMAN and CRAFTED
- Warm, organic color palette: cream background (#faf7f2), deep reds, forest greens, gold accents
- Hand-drawn aesthetic - not rigid or mechanical
- Organic, flowing layouts - radial, circular, or freeform
- Each element can have unique character
- Typography: elegant serifs (Garamond, Didot style)
- Think of this as DATA ART - beautiful enough to hang on a wall
- Legend/key as a beautiful design element, not just utilitarian
- Embrace complexity - layers reward exploration
- Annotations can be narrative, even poetic
- Small details for those who look closely
- NO stark, corporate, or mechanical feel
- Warm and inviting, not cold and clinical
    """,
    best_for=[
        "Personal or human-scale data",
        "Narrative and story-driven presentations",
        "Cultural and humanities subjects",
        "Data art and installations",
        "Annual reports and brand storytelling",
        "Conceptual relationships",
        "When data represents individual humans",
        "Thematic and qualitative analysis",
    ],
    avoid_for=[
        "Financial/technical precision requirements",
        "Fast-consumption news graphics",
        "Expert audiences wanting efficiency",
        "Large-scale statistical comparisons",
    ],
)


STEFANER_TRUTH_BEAUTY_STYLE = StyleGuide(
    school=StyleSchool.STEFANER_TRUTH_BEAUTY,
    name="Moritz Stefaner / Truth & Beauty",
    philosophy="""
    Moritz Stefaner's 'Truth and Beauty Operator' approach:
    - Complex systems deserve complex visualizations
    - Interactivity as a core principle (even in static hints)
    - Scientific rigor meets aesthetic ambition
    - Large-scale pattern revelation
    - Cultural data visualization - music, art, society
    - Network thinking applied everywhere
    - The 'Aha!' moment when structure emerges from chaos
    - Technical sophistication visible in the result
    - Visualization as scientific instrument
    """,
    color_palette={
        "primary": "#2c3e50",  # Deep blue-gray
        "secondary": "#e74c3c",  # Coral red
        "tertiary": "#3498db",  # Bright blue
        "accent": "#9b59b6",  # Purple
        "background": "#ecf0f1",  # Light cool gray
        "background_dark": "#1a1a2e",  # Deep navy (for dramatic effect)
        "highlight": "#f1c40f",  # Yellow
        "text": "#2c3e50",
        "node_palette": ["#e74c3c", "#3498db", "#2ecc71", "#f1c40f", "#9b59b6", "#e67e22"],
    },
    typography={
        "primary_font": "Source Sans Pro, Roboto, sans-serif",
        "title_font": "Montserrat, sans-serif",
        "caption_font": "Source Sans Pro, sans-serif",
        "number_font": "Source Code Pro, monospace",  # Technical feel
        "title_size": "20-28px",
        "label_size": "10-12px",
        "annotation_size": "11-13px",
        "line_height": "1.4",
        "title_weight": "300",  # Light weight for modern feel
    },
    layout_principles=[
        "Force-directed and physics-based layouts",
        "Reveal structure through computation",
        "Dense information displays that emerge from exploration",
        "Interactive affordances even in static form",
        "Dark mode can be powerful for drama",
        "Precise geometric constructions",
        "Hierarchy through size and position",
        "Technical sophistication visible",
        "Works at multiple zoom levels",
        "Credits computational method used",
    ],
    annotation_style="""
    Technical but accessible. Explain the visualization method.
    Include 'how to read this' guidance.
    Reference computational approach where relevant.
    Annotations can discuss the emergence of patterns.
    Scale and context information prominent.
    Hover-state-style annotations for key elements.
    Technical precision in language.
    """,
    gemini_modifiers="""
STYLE INSTRUCTIONS (Moritz Stefaner / Truth & Beauty):
- Create a visualization that reveals EMERGENT STRUCTURE
- Technical sophistication is visible and appreciated
- Color palette: deep blue-gray (#2c3e50), coral (#e74c3c), bright blue (#3498db)
- Light gray background (#ecf0f1) OR dramatic dark (#1a1a2e)
- Force-directed, physics-based, or algorithmically-derived layouts
- Network and systems thinking applied
- Dense information that rewards exploration
- Modern sans-serif typography (Montserrat, Source Sans Pro)
- Precise geometric construction
- Include 'how to read this' annotation
- Works at multiple zoom levels
- Scientific instrument aesthetic
- Patterns should EMERGE from data, not be imposed
- Technical language acceptable
    """,
    best_for=[
        "Complex networks and systems",
        "Large-scale pattern discovery",
        "Cultural and social data (music, art, society)",
        "Scientific and research contexts",
        "Interactive design (even static hints)",
        "Citation and influence networks",
        "Competitive landscape mapping",
        "Emergence and self-organization",
    ],
    avoid_for=[
        "Simple data that doesn't need complexity",
        "Traditional business/financial audiences",
        "Readers uncomfortable with technical aesthetics",
        "Quick-read news contexts",
    ],
)


# Consolidated style guide registry
STYLE_GUIDES = {
    StyleSchool.TUFTE: TUFTE_STYLE,
    StyleSchool.NYT_COX: NYT_COX_STYLE,
    StyleSchool.FT_BURN_MURDOCH: FT_BURN_MURDOCH_STYLE,
    StyleSchool.LUPI_DATA_HUMANISM: LUPI_DATA_HUMANISM_STYLE,
    StyleSchool.STEFANER_TRUTH_BEAUTY: STEFANER_TRUTH_BEAUTY_STYLE,
}


# Engine-to-style affinity mapping
ENGINE_STYLE_AFFINITY = {
    # Stakeholder/Power analysis → FT restrained elegance
    "stakeholder_power_interest": [StyleSchool.FT_BURN_MURDOCH, StyleSchool.NYT_COX],
    "opportunity_vulnerability_matrix": [StyleSchool.FT_BURN_MURDOCH, StyleSchool.TUFTE],
    "competitive_landscape": [StyleSchool.STEFANER_TRUTH_BEAUTY, StyleSchool.FT_BURN_MURDOCH],

    # Flow/Resource analysis → Tufte precision or Stefaner complexity
    "resource_flow_asymmetry": [StyleSchool.TUFTE, StyleSchool.STEFANER_TRUTH_BEAUTY],
    "value_chain_dynamics": [StyleSchool.TUFTE, StyleSchool.FT_BURN_MURDOCH],

    # Temporal analysis → FT (line charts) or NYT (explanatory)
    "event_timeline_causal": [StyleSchool.FT_BURN_MURDOCH, StyleSchool.NYT_COX],
    "concept_evolution": [StyleSchool.STEFANER_TRUTH_BEAUTY, StyleSchool.LUPI_DATA_HUMANISM],

    # Network/Relationship analysis → Stefaner complexity
    "citation_network": [StyleSchool.STEFANER_TRUTH_BEAUTY, StyleSchool.TUFTE],
    "intellectual_genealogy": [StyleSchool.STEFANER_TRUTH_BEAUTY, StyleSchool.LUPI_DATA_HUMANISM],
    "metaphor_analogy_network": [StyleSchool.LUPI_DATA_HUMANISM, StyleSchool.STEFANER_TRUTH_BEAUTY],

    # Argument/Logic analysis → Tufte clarity or NYT explanation
    "argument_architecture": [StyleSchool.TUFTE, StyleSchool.NYT_COX],
    "dialectical_structure": [StyleSchool.LUPI_DATA_HUMANISM, StyleSchool.STEFANER_TRUTH_BEAUTY],
    "assumption_challenge": [StyleSchool.TUFTE, StyleSchool.NYT_COX],

    # Thematic/Narrative analysis → Lupi humanism
    "thematic_synthesis": [StyleSchool.LUPI_DATA_HUMANISM, StyleSchool.NYT_COX],
    "narrative_structure": [StyleSchool.LUPI_DATA_HUMANISM, StyleSchool.NYT_COX],
    "emotional_arc": [StyleSchool.LUPI_DATA_HUMANISM, StyleSchool.NYT_COX],

    # Evidence/Analytical → Tufte precision
    "evidence_quality_assessment": [StyleSchool.TUFTE, StyleSchool.FT_BURN_MURDOCH],
    "confidence_calibration": [StyleSchool.TUFTE, StyleSchool.FT_BURN_MURDOCH],
    "source_verification": [StyleSchool.TUFTE, StyleSchool.NYT_COX],

    # Default for unknown engines
    "_default": [StyleSchool.NYT_COX, StyleSchool.TUFTE],
}


# Format-to-style affinity mapping
FORMAT_STYLE_AFFINITY = {
    # Relational formats
    "network_graph": [StyleSchool.STEFANER_TRUTH_BEAUTY, StyleSchool.TUFTE],
    "chord_diagram": [StyleSchool.LUPI_DATA_HUMANISM, StyleSchool.STEFANER_TRUTH_BEAUTY],
    "hierarchical_tree": [StyleSchool.TUFTE, StyleSchool.NYT_COX],
    "radial_tree": [StyleSchool.LUPI_DATA_HUMANISM, StyleSchool.STEFANER_TRUTH_BEAUTY],

    # Flow formats
    "sankey": [StyleSchool.TUFTE, StyleSchool.FT_BURN_MURDOCH],
    "alluvial": [StyleSchool.STEFANER_TRUTH_BEAUTY, StyleSchool.LUPI_DATA_HUMANISM],
    "flowchart": [StyleSchool.NYT_COX, StyleSchool.TUFTE],
    "value_stream": [StyleSchool.TUFTE, StyleSchool.FT_BURN_MURDOCH],

    # Temporal formats
    "timeline": [StyleSchool.NYT_COX, StyleSchool.FT_BURN_MURDOCH],
    "gantt": [StyleSchool.TUFTE, StyleSchool.FT_BURN_MURDOCH],
    "parallel_timelines": [StyleSchool.STEFANER_TRUTH_BEAUTY, StyleSchool.NYT_COX],
    "cycle_diagram": [StyleSchool.LUPI_DATA_HUMANISM, StyleSchool.NYT_COX],

    # Comparative formats
    "matrix_heatmap": [StyleSchool.TUFTE, StyleSchool.STEFANER_TRUTH_BEAUTY],
    "quadrant_chart": [StyleSchool.FT_BURN_MURDOCH, StyleSchool.NYT_COX],
    "radar_chart": [StyleSchool.TUFTE, StyleSchool.LUPI_DATA_HUMANISM],
    "bar_chart": [StyleSchool.FT_BURN_MURDOCH, StyleSchool.TUFTE],

    # Part-of-whole formats
    "treemap": [StyleSchool.TUFTE, StyleSchool.NYT_COX],
    "sunburst": [StyleSchool.LUPI_DATA_HUMANISM, StyleSchool.STEFANER_TRUTH_BEAUTY],
    "stacked_bar": [StyleSchool.FT_BURN_MURDOCH, StyleSchool.TUFTE],
    "waterfall": [StyleSchool.FT_BURN_MURDOCH, StyleSchool.NYT_COX],

    # Evidence/Analytical formats
    "ach_matrix": [StyleSchool.TUFTE, StyleSchool.NYT_COX],
    "confidence_thermometer": [StyleSchool.NYT_COX, StyleSchool.FT_BURN_MURDOCH],
    "indicator_dashboard": [StyleSchool.FT_BURN_MURDOCH, StyleSchool.NYT_COX],
    "gap_analysis_visual": [StyleSchool.NYT_COX, StyleSchool.TUFTE],

    # Argumentative formats
    "argument_tree": [StyleSchool.TUFTE, StyleSchool.NYT_COX],
    "scenario_cone": [StyleSchool.NYT_COX, StyleSchool.STEFANER_TRUTH_BEAUTY],
    "dialectical_map": [StyleSchool.LUPI_DATA_HUMANISM, StyleSchool.STEFANER_TRUTH_BEAUTY],

    # Default
    "_default": [StyleSchool.NYT_COX, StyleSchool.TUFTE],
}


# Audience-to-style mapping
AUDIENCE_STYLE_AFFINITY = {
    "executive": [StyleSchool.NYT_COX, StyleSchool.FT_BURN_MURDOCH],
    "analyst": [StyleSchool.TUFTE, StyleSchool.FT_BURN_MURDOCH],
    "researcher": [StyleSchool.TUFTE, StyleSchool.STEFANER_TRUTH_BEAUTY],
    "general_public": [StyleSchool.NYT_COX, StyleSchool.LUPI_DATA_HUMANISM],
    "journalist": [StyleSchool.FT_BURN_MURDOCH, StyleSchool.NYT_COX],
    "academic": [StyleSchool.TUFTE, StyleSchool.STEFANER_TRUTH_BEAUTY],
    "creative": [StyleSchool.LUPI_DATA_HUMANISM, StyleSchool.STEFANER_TRUTH_BEAUTY],
    "_default": [StyleSchool.NYT_COX, StyleSchool.TUFTE],
}


# ============================================================================
# STYLE CURATOR AGENT
# ============================================================================

class StyleCurator:
    """
    LLM-powered agent that recommends visualization style based on content.

    Uses Claude Opus 4.5 with extended thinking to select the optimal
    style school for a given engine + format + audience combination.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        thinking_budget: int = 10000,
    ):
        """
        Initialize the Style Curator.

        Args:
            api_key: Anthropic API key (defaults to ANTHROPIC_API_KEY env var)
            thinking_budget: Token budget for extended thinking (default 10000)
        """
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("Anthropic API key required")

        self.client = anthropic.Anthropic(api_key=self.api_key)
        self.model = "claude-opus-4-5-20251101"
        self.thinking_budget = thinking_budget

    def curate_style(
        self,
        engine_key: str,
        format_key: str,
        extracted_data: Optional[dict[str, Any]] = None,
        audience: str = "analyst",
        document_context: Optional[str] = None,
    ) -> StyleCuratorOutput:
        """
        Select the optimal visualization style for the given context.

        Args:
            engine_key: The engine that produced the data
            format_key: The visual format being used
            extracted_data: Optional sample of the actual data
            audience: Target audience type
            document_context: Optional context about the source document

        Returns:
            StyleCuratorOutput with style recommendation and rationale
        """
        logger.info(f"Curating style for engine={engine_key}, format={format_key}, audience={audience}")

        # Build the prompt
        prompt = self._build_prompt(engine_key, format_key, extracted_data, audience, document_context)

        # Call Opus 4.5 with extended thinking
        thinking_content = ""
        response_content = ""

        max_tokens = self.thinking_budget + 4000

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

        logger.info(f"Style curator thinking: {len(thinking_content)} chars")
        logger.info(f"Style curator response: {len(response_content)} chars")

        # Parse the response
        return self._parse_response(response_content, thinking_content)

    def _build_prompt(
        self,
        engine_key: str,
        format_key: str,
        extracted_data: Optional[dict[str, Any]],
        audience: str,
        document_context: Optional[str],
    ) -> str:
        """Build the prompt for style curation."""

        # Get style affinity hints
        engine_styles = ENGINE_STYLE_AFFINITY.get(engine_key, ENGINE_STYLE_AFFINITY["_default"])
        format_styles = FORMAT_STYLE_AFFINITY.get(format_key, FORMAT_STYLE_AFFINITY["_default"])
        audience_styles = AUDIENCE_STYLE_AFFINITY.get(audience, AUDIENCE_STYLE_AFFINITY["_default"])

        # Build style knowledge summary
        style_knowledge = ""
        for school, guide in STYLE_GUIDES.items():
            style_knowledge += f"""
## {guide.name}
Philosophy: {guide.philosophy[:500]}...
Best for: {', '.join(guide.best_for[:4])}
Avoid for: {', '.join(guide.avoid_for[:3])}
"""

        # Data sample
        data_str = ""
        if extracted_data:
            data_str = json.dumps(extracted_data, indent=2, default=str)[:2000]

        prompt = f"""You are the Style Curator, an expert in data visualization aesthetics.

Your task: Select the optimal visualization STYLE (not format) for this specific content.

## THE 5 STYLE SCHOOLS

{style_knowledge}

## AFFINITY HINTS (weighted suggestions, not requirements)

Based on the engine "{engine_key}": consider {[s.value for s in engine_styles]}
Based on the format "{format_key}": consider {[s.value for s in format_styles]}
Based on the audience "{audience}": consider {[s.value for s in audience_styles]}

## CONTEXT

Engine: {engine_key}
Visual Format: {format_key}
Audience: {audience}
{f'Document Context: {document_context}' if document_context else ''}

{f'Sample Data:\n```json\n{data_str}\n```' if data_str else ''}

## YOUR TASK

1. Consider the nature of the data and analysis
2. Consider the audience's needs and sophistication
3. Consider what story or insight the visualization should convey
4. Select the PRIMARY style school that best fits
5. Optionally select a SECONDARY style if hybridization would help

Think deeply about WHY each style would or wouldn't work for this specific case.

## OUTPUT FORMAT

```json
{{
  "primary_style": {{
    "school": "tufte|nyt_cox|ft_burn_murdoch|lupi_data_humanism|stefaner_truth_beauty",
    "confidence": 0.85,
    "rationale": "Why this style fits the content and audience"
  }},
  "secondary_style": {{
    "school": "...",
    "confidence": 0.6,
    "rationale": "Why this could complement the primary"
  }},
  "style_rationale": "Overall analysis of how style choice serves the data",
  "thinking_summary": "Brief summary of your reasoning process"
}}
```

Choose wisely - the style will fundamentally shape how readers understand the data.
"""
        return prompt

    def _parse_response(
        self,
        response_content: str,
        thinking_content: str,
    ) -> StyleCuratorOutput:
        """Parse the style curator's response."""

        try:
            # Extract JSON from response
            if "```json" in response_content:
                json_str = response_content.split("```json")[1].split("```")[0]
            elif "```" in response_content:
                json_str = response_content.split("```")[1].split("```")[0]
            else:
                json_str = response_content

            data = json.loads(json_str.strip())

            # Parse primary style
            primary_data = data.get("primary_style", {})
            school_str = primary_data.get("school", "nyt_cox")
            primary_school = StyleSchool(school_str)
            primary_guide = STYLE_GUIDES[primary_school]

            primary_rec = StyleRecommendation(
                school=primary_school,
                style_guide=primary_guide,
                confidence=primary_data.get("confidence", 0.7),
                rationale=primary_data.get("rationale", ""),
                gemini_style_instructions=primary_guide.gemini_modifiers,
            )

            # Parse secondary style if present
            secondary_rec = None
            if "secondary_style" in data and data["secondary_style"]:
                secondary_data = data["secondary_style"]
                if secondary_data.get("school"):
                    secondary_school = StyleSchool(secondary_data["school"])
                    secondary_guide = STYLE_GUIDES[secondary_school]
                    secondary_rec = StyleRecommendation(
                        school=secondary_school,
                        style_guide=secondary_guide,
                        confidence=secondary_data.get("confidence", 0.5),
                        rationale=secondary_data.get("rationale", ""),
                        gemini_style_instructions=secondary_guide.gemini_modifiers,
                    )

            return StyleCuratorOutput(
                primary_style=primary_rec,
                secondary_style=secondary_rec,
                style_rationale=data.get("style_rationale", ""),
                thinking_summary=data.get("thinking_summary", ""),
                raw_thinking=thinking_content if thinking_content else None,
            )

        except Exception as e:
            logger.error(f"Failed to parse style curator response: {e}")

            # Default to NYT/Cox style
            default_guide = STYLE_GUIDES[StyleSchool.NYT_COX]
            return StyleCuratorOutput(
                primary_style=StyleRecommendation(
                    school=StyleSchool.NYT_COX,
                    style_guide=default_guide,
                    confidence=0.5,
                    rationale="Default style due to parsing error",
                    gemini_style_instructions=default_guide.gemini_modifiers,
                ),
                secondary_style=None,
                style_rationale=f"Error occurred: {str(e)}",
                thinking_summary="Parsing failed, using default NYT/Cox style",
                raw_thinking=thinking_content,
            )

    def get_quick_style(
        self,
        engine_key: str,
        format_key: str,
        audience: str = "analyst",
    ) -> StyleRecommendation:
        """
        Get a quick style recommendation without LLM call.

        Uses affinity mappings to select the most appropriate style.
        Use this for high-volume processing where LLM calls are too expensive.

        Args:
            engine_key: The engine that produced the data
            format_key: The visual format being used
            audience: Target audience type

        Returns:
            StyleRecommendation with the best-fit style
        """
        # Score each style based on affinities
        style_scores = {school: 0.0 for school in StyleSchool}

        # Engine affinity (weight: 3)
        engine_styles = ENGINE_STYLE_AFFINITY.get(engine_key, ENGINE_STYLE_AFFINITY["_default"])
        for i, school in enumerate(engine_styles):
            style_scores[school] += (3 - i) * 3  # First choice gets 9, second gets 6

        # Format affinity (weight: 2)
        format_styles = FORMAT_STYLE_AFFINITY.get(format_key, FORMAT_STYLE_AFFINITY["_default"])
        for i, school in enumerate(format_styles):
            style_scores[school] += (2 - i) * 2  # First choice gets 4, second gets 2

        # Audience affinity (weight: 1)
        audience_styles = AUDIENCE_STYLE_AFFINITY.get(audience, AUDIENCE_STYLE_AFFINITY["_default"])
        for i, school in enumerate(audience_styles):
            style_scores[school] += (2 - i)  # First choice gets 2, second gets 1

        # Find the winner
        best_school = max(style_scores, key=lambda s: style_scores[s])
        best_guide = STYLE_GUIDES[best_school]

        # Calculate confidence based on margin
        scores = sorted(style_scores.values(), reverse=True)
        margin = (scores[0] - scores[1]) / scores[0] if scores[0] > 0 else 0
        confidence = min(0.95, 0.6 + margin * 0.35)

        return StyleRecommendation(
            school=best_school,
            style_guide=best_guide,
            confidence=confidence,
            rationale=f"Selected based on affinity scoring: engine={engine_key}, format={format_key}, audience={audience}",
            gemini_style_instructions=best_guide.gemini_modifiers,
        )


def merge_gemini_prompt_with_style(
    base_prompt: str,
    style_instructions: str,
) -> str:
    """
    Merge a base Gemini prompt with style-specific instructions.

    Args:
        base_prompt: The original Gemini prompt (data mapping, format type)
        style_instructions: The style-specific instructions from StyleGuide.gemini_modifiers

    Returns:
        Combined prompt with style instructions integrated
    """
    # Get display formatting instructions (anti-score, anti-snake_case)
    display_instructions = get_display_instructions()

    return f"""{base_prompt}

{style_instructions}
{display_instructions}

IMPORTANT: Apply BOTH the style instructions AND display formatting rules above.
The style and proper label formatting are as important as the data accuracy.
"""


# Convenience functions
def curate_style(
    engine_key: str,
    format_key: str,
    extracted_data: Optional[dict[str, Any]] = None,
    audience: str = "analyst",
    document_context: Optional[str] = None,
    api_key: Optional[str] = None,
    thinking_budget: int = 10000,
) -> StyleCuratorOutput:
    """
    Convenience function to curate style for a visualization.

    Args:
        engine_key: The engine that produced the data
        format_key: The visual format being used
        extracted_data: Optional sample of the actual data
        audience: Target audience type
        document_context: Optional context about the source document
        api_key: Anthropic API key
        thinking_budget: Token budget for extended thinking

    Returns:
        StyleCuratorOutput with style recommendation
    """
    curator = StyleCurator(api_key=api_key, thinking_budget=thinking_budget)
    return curator.curate_style(
        engine_key=engine_key,
        format_key=format_key,
        extracted_data=extracted_data,
        audience=audience,
        document_context=document_context,
    )


def get_quick_style(
    engine_key: str,
    format_key: str,
    audience: str = "analyst",
) -> StyleRecommendation:
    """
    Get a quick style recommendation without LLM call.

    Uses affinity mappings for fast, deterministic selection.
    """
    # No API key needed for quick mode
    curator = StyleCurator.__new__(StyleCurator)
    return curator.get_quick_style(engine_key, format_key, audience)


# Format-to-prompt templates for generating base Gemini prompts
FORMAT_BASE_PROMPTS = {
    "network_graph": """Create a professional network graph visualization.

LAYOUT: Force-directed or hierarchical layout showing nodes and their connections.
- Nodes represent entities (actors, concepts, organizations)
- Edges represent relationships with descriptive labels (e.g., "Influenced", "Derived from")
- Node size can encode importance/centrality
- Edge thickness encodes relationship strength visually (NEVER with numeric labels)

CRITICAL REQUIREMENTS:
- NEVER display "THICKNESS: 0.85" or any numeric values on edges - this is FORBIDDEN
- Edge labels must be relationship TYPES only (e.g., "Conceptual Extension"), NOT numbers
- NEVER show weight, strength, or confidence scores as text on the graph
- Use line thickness alone to show strength - no text annotation of thickness values
- Convert any snake_case identifiers to Title Case with spaces
- Clear visual hierarchy with most important nodes prominent
- Legend explaining what edge thickness means (e.g., "Thicker = stronger influence")
- 4K resolution (3840 x 2160)
- Professional, clean aesthetic""",

    "sankey": """Create a professional Sankey flow diagram.

LAYOUT: Left-to-right flow showing how quantities move between categories.
- Nodes as vertical bars representing sources/destinations
- Flows as curved bands whose width encodes quantity
- Color-coded by source or category

REQUIREMENTS:
- Clear flow direction from left to right
- Readable labels on all nodes (convert snake_case to Title Case)
- Flow widths proportional to values (do NOT label flows with numbers)
- NEVER display decimal scores on flows - width alone shows magnitude
- 4K resolution (3840 x 2160)
- Professional, clean aesthetic""",

    "quadrant_chart": """Create a professional 2x2 quadrant matrix visualization.

LAYOUT: Four quadrants with labeled axes.
- X-axis and Y-axis clearly labeled with the two dimensions
- Items positioned based on their values on both dimensions
- Quadrant labels in each section

REQUIREMENTS:
- Clear axis labels and scale
- Items as labeled points or bubbles with readable names
- Convert any snake_case identifiers to Title Case with spaces
- NEVER display raw numeric scores (0.85, 0.75) next to items
- Distinct quadrant backgrounds or borders
- 4K resolution (3840 x 2160)
- Professional, clean aesthetic""",

    "timeline": """Create a professional timeline visualization.

LAYOUT: Horizontal or vertical timeline showing events in sequence.
- Time axis with clear markers
- Events as labeled points or cards
- Connections between related events if applicable

REQUIREMENTS:
- Clear chronological flow
- Readable event labels and dates (format snake_case to Title Case)
- Visual hierarchy for major vs minor events
- 4K resolution (3840 x 2160)
- Professional, clean aesthetic""",

    "matrix_heatmap": """Create a professional matrix heatmap visualization.

LAYOUT: Grid showing values across two dimensions.
- Rows and columns clearly labeled
- Color intensity encoding values
- Optional annotations in cells

REQUIREMENTS:
- Clear row and column headers (convert snake_case to Title Case)
- Color legend showing value scale
- NEVER show raw decimal scores like 0.85 - use color only
- 4K resolution (3840 x 2160)
- Professional, clean aesthetic""",

    "radar_chart": """Create a professional radar/spider chart visualization.

LAYOUT: Radial chart showing multiple dimensions from center.
- Axes radiating from center, one per dimension
- Area or line connecting values across dimensions
- Multiple series can be overlaid

REQUIREMENTS:
- All axis labels readable (convert snake_case to Title Case)
- Clear scale on each axis (do NOT show decimal scores as labels)
- Legend if multiple series
- 4K resolution (3840 x 2160)
- Professional, clean aesthetic""",

    "tree_hierarchy": """Create a professional hierarchical tree visualization.

LAYOUT: Top-down or left-right tree showing parent-child relationships.
- Root at top/left
- Branches connecting parent to children
- Node labels clearly visible

REQUIREMENTS:
- Clear hierarchical structure
- Convert all snake_case labels to Title Case with spaces
- No overlapping labels
- NEVER show numeric scores on connections
- Consistent spacing between levels
- 4K resolution (3840 x 2160)
- Professional, clean aesthetic""",

    "bubble_chart": """Create a professional bubble chart visualization.

LAYOUT: 2D scatter plot with sized circles.
- X and Y axes showing two dimensions
- Bubble size encoding a third dimension
- Color can encode a fourth dimension

REQUIREMENTS:
- Clear axis labels and scales
- Convert snake_case labels to Title Case with spaces
- Legend for size and color if used
- Labels on major bubbles (NOT numeric scores)
- 4K resolution (3840 x 2160)
- Professional, clean aesthetic""",

    "parallel_coordinates": """Create a professional parallel coordinates visualization.

LAYOUT: Vertical parallel axes with lines connecting values.
- Each axis represents one dimension
- Lines trace each item across all dimensions
- Color can encode category or value

REQUIREMENTS:
- All axis labels readable (convert snake_case to Title Case)
- Lines clearly distinguishable
- NEVER label lines with raw decimal scores
- Legend for color coding
- 4K resolution (3840 x 2160)
- Professional, clean aesthetic""",

    "chord_diagram": """Create a professional chord diagram visualization.

LAYOUT: Circular layout showing flows between categories.
- Arcs around circle for each category
- Chords connecting categories with flow between them
- Chord width encodes flow magnitude (NOT numeric labels)

REQUIREMENTS:
- Category labels around perimeter (Title Case, no underscores)
- Clear color coding
- NEVER show decimal scores on chords - use width only
- Legend explaining colors
- 4K resolution (3840 x 2160)
- Professional, clean aesthetic""",

    "argument_tree": """Create a professional argument tree/map visualization.

LAYOUT: Hierarchical or flow-based structure showing logical argument components.
- Main CLAIM/THESIS at the top or center
- GROUNDS (evidence/premises) branching below supporting claims
- WARRANTS connecting grounds to claims (show reasoning)
- REBUTTALS/counterarguments shown as opposing branches
- Use Toulmin model structure: Claim → Grounds → Warrant → Backing → Rebuttal

VISUAL ENCODING:
- Claims in prominent boxes (bold, larger)
- Grounds in supporting boxes below
- Warrants as connecting labels or intermediate nodes
- Rebuttals in contrasting color (red/orange)
- Flow arrows showing logical direction

CRITICAL REQUIREMENTS:
- Show logical FLOW from evidence to conclusion
- Each argument component should be clearly labeled
- NO numeric scores or confidence values - use color/size to show strength
- Convert snake_case identifiers to Title Case
- Clear visual distinction between supporting and opposing elements
- 4K resolution (3840 x 2160)
- Clean, professional aesthetic suitable for academic/analytical context""",

    "flowchart": """Create a professional flowchart visualization.

LAYOUT: Sequential flow showing process, decision tree, or logical progression.
- Start/end nodes clearly marked
- Decision nodes as diamonds
- Process/action nodes as rectangles
- Clear directional arrows showing flow

REQUIREMENTS:
- Logical left-to-right or top-to-bottom flow
- All labels in Title Case (no snake_case)
- Clear node shapes for different element types
- NEVER include numeric scores on nodes or edges
- 4K resolution (3840 x 2160)
- Professional, clean aesthetic""",
}

# Default prompt for unknown formats
DEFAULT_FORMAT_PROMPT = """Create a professional data visualization.

LAYOUT: Choose the most appropriate layout for the data structure.
- Clear visual hierarchy
- Readable labels and annotations
- Logical organization of elements

REQUIREMENTS:
- 4K resolution (3840 x 2160)
- Professional, clean aesthetic
- Convert ALL snake_case identifiers to Title Case with spaces
- NEVER display raw numeric scores (like 0.85, 0.75) - use visual encoding instead
- Clear legend if needed
- All text readable"""


def generate_base_prompt(format_key: str, format_name: str = None, include_display_rules: bool = True) -> str:
    """
    Generate a base Gemini prompt for a given format.

    This is used when the LLM curator doesn't provide a gemini_prompt
    (e.g., in batch mode where prompts aren't generated).

    Args:
        format_key: The format key (e.g., 'network_graph', 'sankey')
        format_name: Optional human-readable name for the format
        include_display_rules: Whether to include display formatting rules (default True)

    Returns:
        Base Gemini prompt for the format with display rules
    """
    base = FORMAT_BASE_PROMPTS.get(format_key, DEFAULT_FORMAT_PROMPT)

    # Add format name header if provided
    if format_name and format_name.lower() != format_key.replace("_", " "):
        base = f"## {format_name}\n\n{base}"

    # Add display formatting rules
    if include_display_rules:
        display_instructions = get_display_instructions()
        base = f"{base}\n{display_instructions}"

    return base


# Exports
__all__ = [
    "StyleSchool",
    "StyleGuide",
    "StyleRecommendation",
    "StyleCuratorOutput",
    "StyleCurator",
    "STYLE_GUIDES",
    "ENGINE_STYLE_AFFINITY",
    "FORMAT_STYLE_AFFINITY",
    "AUDIENCE_STYLE_AFFINITY",
    "curate_style",
    "get_quick_style",
    "merge_gemini_prompt_with_style",
    "generate_base_prompt",
    "FORMAT_BASE_PROMPTS",
    "DEFAULT_FORMAT_PROMPT",
]
