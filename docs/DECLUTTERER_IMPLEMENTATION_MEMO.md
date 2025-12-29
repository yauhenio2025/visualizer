# DECLUTTERER Implementation Memo

**Purpose:** Enable a context-free LLM session to fully implement text density control for the Visualizer system.

**Problem:** Visualizations are cluttered with 50-100+ text elements. Gemini renders everything it receives, overwhelming readers.

**Solution:** Insert a DECLUTTERER stage (Claude Sonnet) that intelligently compresses text content before Gemini image generation.

---

## ARCHITECTURE OVERVIEW

### Current Flow (Problem)
```
Document
  → Analyzer API (localhost:8847) extracts structured data
  → Output Curator recommends format + generates base Gemini prompt
  → Style Curator merges style instructions (Tufte/NYT/FT/Lupi/Stefaner)
  → GeminiImageRenderer._format_content_for_prompt() converts ALL data to text
  → Final prompt = styled_prompt + "DATA TO VISUALIZE" + content (70-100 elements)
  → Gemini generates 4K image with ALL text crammed in
```

### Target Flow (Solution)
```
Same as above, but INSERT after _format_content_for_prompt():
  → content = _format_content_for_prompt(canonical, engine_key)
  → content = DECLUTTERER.compress(content, engine_key, max_elements=15)  ← NEW
  → Final prompt = styled_prompt + compressed_content (15-20 elements)
  → Gemini generates clean, legible visualization
```

---

## KEY FILES TO MODIFY

### 1. Primary Integration Point
**File:** `/home/evgeny/projects/analyzer/src/renderers/gemini_image.py`
**Line:** ~1022-1032 (the render() method)

**Current code:**
```python
content = cls._format_content_for_prompt(canonical, engine_key)
# Build prompt by appending content to the styled prompt
prompt = f"""{custom_gemini_prompt}

## DATA TO VISUALIZE

{content}

IMPORTANT: Apply ALL style instructions above to the visualization of this data.
"""
```

**Target code:**
```python
content = cls._format_content_for_prompt(canonical, engine_key)

# DECLUTTER: Compress content for visual clarity
from src.renderers.declutterer import Declutterer
declutterer = Declutterer()
content = await declutterer.compress(
    content=content,
    engine_key=engine_key,
    max_elements=config.get("max_visual_elements", 15),
    strategy=config.get("declutter_strategy", "auto"),
)

prompt = f"""{custom_gemini_prompt}

## DATA TO VISUALIZE

{content}

IMPORTANT: Apply ALL style instructions above to the visualization of this data.
"""
```

### 2. New File to Create
**File:** `/home/evgeny/projects/analyzer/src/renderers/declutterer.py`

---

## DECLUTTERER CLASS SPECIFICATION

```python
"""
Declutterer Agent

LLM-powered text compression for visualization clarity.
Uses Claude Sonnet for speed (not Opus - this is compression, not analysis).

Design principles:
1. Preserve meaning: 90% of insight with 50% of text
2. Prioritize by importance: Keep top-level, collapse details
3. Engine-aware: Different strategies for different data types
4. Non-destructive: Full extraction preserved for textual outputs
"""

import os
from typing import Optional
import anthropic

# Engine archetype mappings for strategy selection
ENGINE_ARCHETYPES = {
    # Argumentative engines → hierarchical compression
    "argument_architecture": "argumentative",
    "dialectical_structure": "argumentative",
    "assumption_excavation": "argumentative",

    # Network engines → top-n compression
    "citation_network": "network",
    "intellectual_genealogy": "network",
    "metaphor_analogy_network": "network",
    "stakeholder_power_interest": "network",

    # Temporal engines → clustering compression
    "event_timeline_causal": "temporal",
    "concept_evolution": "temporal",
    "reception_history": "temporal",

    # Flow engines → aggregation compression
    "resource_flow_asymmetry": "flow",

    # Matrix engines → filtering compression
    "competitive_landscape": "matrix",
    "opportunity_vulnerability_matrix": "matrix",
}

# Strategy prompts for each archetype
STRATEGY_PROMPTS = {
    "argumentative": """You are compressing argument data for a visualization.

STRATEGY: Hierarchical compression
- Keep ALL main claims/theses (these are the core structure)
- For each claim, show ONLY: the claim text + count of supporting evidence
- Remove individual grounds/warrants/rebuttals - just note how many exist
- Preserve rebuttals as brief one-liners (they're visually important as counterpoints)

EXAMPLE TRANSFORMATION:
BEFORE:
  ARGUMENT: Rentier Framework
    CLAIM: Rentier capitalism correctly describes...
    GROUND: Tech companies engage in fierce competition...
    GROUND: Technofeudalism proponents concede...
    GROUND: Marx's rent theory provides...
    WARRANT: The Marxian category of rent...
    REBUTTAL: Varoufakis argues capitalism has been "killed"...

AFTER:
  ARGUMENT: Rentier Framework
    CLAIM: "Rentier capitalism correctly describes the digital economy"
    [3 supporting grounds, 1 rebuttal]
    REBUTTAL: "Varoufakis: capitalism has been 'killed'"
""",

    "network": """You are compressing network/relationship data for a visualization.

STRATEGY: Top-N with clustering
- Keep the TOP 5-7 most central/important nodes
- For edges, keep only edges connecting these top nodes
- Group minor nodes into categories (e.g., "5 other stakeholders")
- Preserve edge labels but remove numeric weights

EXAMPLE TRANSFORMATION:
BEFORE: 15 stakeholders with 30 relationships

AFTER:
  KEY ACTORS (top 5 by influence):
  1. Platform Giants (Amazon, Google, Apple, Meta, Microsoft)
  2. Rentier Capitalists (investors extracting platform rent)
  3. Platform Workers (gig workers, content creators)
  4. Regulatory Bodies (EU, national governments)
  5. Civil Society (unions, advocacy groups)

  [10 additional actors grouped as: Financial Capital (3), Academia (2), Users (5)]

  KEY RELATIONSHIPS:
  - Platform Giants → extract rent from → Platform Workers
  - Regulatory Bodies → constrain → Platform Giants
  - Civil Society → advocates for → Platform Workers
""",

    "temporal": """You are compressing timeline/temporal data for a visualization.

STRATEGY: Period clustering
- Group events into 3-5 major periods/phases
- For each period, show 1-2 key events only
- Remove minor events but note count
- Preserve causal arrows between periods

EXAMPLE TRANSFORMATION:
BEFORE: 20 events from 1970-2025

AFTER:
  PERIOD 1: Industrial Foundation (1970s-1990s)
    Key: "Post-war industrial capitalism establishes mass production base"
    [4 other events in this period]

  PERIOD 2: Financialization (1990s-2008)
    Key: "Financial capital begins dominating productive capital"
    [5 other events]

  PERIOD 3: Platform Rise (2008-2020)
    Key: "Digital platforms emerge as dominant rent extraction mechanism"
    [6 other events]

  PERIOD 4: Crisis & Contestation (2020-present)
    Key: "Accumulation crisis meets systemic resistance"
    [3 other events]
""",

    "flow": """You are compressing flow/resource data for a visualization.

STRATEGY: Aggregation
- Keep the top 5 largest flows
- Aggregate small flows into "Other" category
- Preserve flow direction and relative magnitude (large/medium/small)
- Remove exact numeric values

EXAMPLE:
BEFORE: 15 flows with values

AFTER:
  MAJOR FLOWS:
  1. Users → Platform Giants [LARGE: data extraction]
  2. Platform Giants → Shareholders [LARGE: rent distribution]
  3. Workers → Platform Giants [MEDIUM: labor value]
  4. Platform Giants → Infrastructure [MEDIUM: operational costs]
  5. Advertisers → Platform Giants [MEDIUM: attention monetization]

  [10 smaller flows aggregated as "Other operational flows"]
""",

    "matrix": """You are compressing matrix/positioning data for a visualization.

STRATEGY: Corner + center focus
- Show items in the 4 corners (extremes on both axes)
- Show 1-2 items near the center (balanced/neutral)
- Group middle-ground items by quadrant
- Preserve axis labels and quadrant meanings

EXAMPLE:
BEFORE: 20 items positioned on power-interest matrix

AFTER:
  HIGH POWER, HIGH INTEREST (top-right):
    - Platform Giants
    - Financial Capital

  HIGH POWER, LOW INTEREST (top-left):
    - Regulatory Bodies

  LOW POWER, HIGH INTEREST (bottom-right):
    - Platform Workers
    - Civil Society

  LOW POWER, LOW INTEREST (bottom-left):
    - General Public

  [14 other actors positioned in middle zones]
""",
}

DEFAULT_STRATEGY = """You are compressing data for a visualization.

GENERAL COMPRESSION RULES:
1. Keep TOP 5-7 most important items
2. For each item, keep only the primary label/description
3. Remove secondary details but note they exist
4. Group remaining items into categories
5. Remove ALL numeric scores - use only descriptive terms
6. Convert snake_case to Title Case

Target: Reduce to 15-20 text elements maximum while preserving core structure.
"""


class Declutterer:
    """
    LLM-powered text compression for visualization clarity.
    """

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY required for Declutterer")

        self.client = anthropic.Anthropic(api_key=self.api_key)
        self.model = "claude-sonnet-4-5-20250929"  # Fast, good at compression

    async def compress(
        self,
        content: str,
        engine_key: str,
        max_elements: int = 15,
        strategy: str = "auto",
    ) -> str:
        """
        Compress formatted content for clearer visualization.

        Args:
            content: Formatted text from _format_content_for_prompt()
            engine_key: Engine that produced the data (for strategy selection)
            max_elements: Target maximum visual elements
            strategy: "auto" | "argumentative" | "network" | "temporal" | "flow" | "matrix"

        Returns:
            Compressed content string
        """
        # Skip compression for already-short content
        if len(content) < 500:
            return content

        # Auto-select strategy based on engine archetype
        if strategy == "auto":
            archetype = ENGINE_ARCHETYPES.get(engine_key, "default")
            strategy_prompt = STRATEGY_PROMPTS.get(archetype, DEFAULT_STRATEGY)
        else:
            strategy_prompt = STRATEGY_PROMPTS.get(strategy, DEFAULT_STRATEGY)

        prompt = f"""{strategy_prompt}

---

CONTENT TO COMPRESS:

{content}

---

OUTPUT REQUIREMENTS:
1. Maximum {max_elements} distinct visual elements (boxes, labels, nodes)
2. Preserve the analytical structure and hierarchy
3. Use "[X other items]" notation for grouped/omitted content
4. NO numeric scores or percentages - use descriptive terms only
5. All labels in Title Case (no snake_case or kebab-case)
6. Add note at end: "[Full details available in textual output]"

Return ONLY the compressed content, no explanations."""

        response = self.client.messages.create(
            model=self.model,
            max_tokens=2000,
            messages=[{"role": "user", "content": prompt}]
        )

        return response.content[0].text

    def estimate_elements(self, content: str) -> int:
        """
        Estimate number of visual elements in content.
        Heuristic based on line patterns.
        """
        lines = content.strip().split('\n')
        element_patterns = [
            'ARGUMENT:', 'CLAIM:', 'GROUND:', 'WARRANT:', 'REBUTTAL:',
            'STAKEHOLDER:', 'ACTOR:', 'NODE:', 'EDGE:',
            'EVENT:', 'PERIOD:', 'PHASE:',
            'FLOW:', 'SOURCE:', 'TARGET:',
            'METAPHOR:', 'MAPPING:', 'ANALOGY:',
        ]

        count = 0
        for line in lines:
            line_upper = line.strip().upper()
            if any(pattern in line_upper for pattern in element_patterns):
                count += 1
            elif line.strip().startswith('-') or line.strip().startswith('•'):
                count += 1

        return max(count, len([l for l in lines if l.strip()]) // 3)


# Convenience function
async def declutter_content(
    content: str,
    engine_key: str,
    max_elements: int = 15,
    strategy: str = "auto",
    api_key: Optional[str] = None,
) -> str:
    """Convenience function for decluttering content."""
    declutterer = Declutterer(api_key=api_key)
    return await declutterer.compress(content, engine_key, max_elements, strategy)
```

---

## INTEGRATION STEPS

### Step 1: Create the Declutterer module
Create `/home/evgeny/projects/analyzer/src/renderers/declutterer.py` with the code above.

### Step 2: Modify GeminiImageRenderer.render()
In `/home/evgeny/projects/analyzer/src/renderers/gemini_image.py`, find the render() method (around line 1000-1082).

**Find this block (around line 1022):**
```python
content = cls._format_content_for_prompt(canonical, engine_key)
# Build prompt by appending content to the styled prompt
prompt = f"""{custom_gemini_prompt}
```

**Replace with:**
```python
content = cls._format_content_for_prompt(canonical, engine_key)

# DECLUTTER: Compress for visual clarity
from src.renderers.declutterer import declutter_content
content = await declutter_content(
    content=content,
    engine_key=engine_key,
    max_elements=config.get("max_visual_elements", 15),
    strategy=config.get("declutter_strategy", "auto"),
)

# Build prompt by appending compressed content to the styled prompt
prompt = f"""{custom_gemini_prompt}
```

### Step 3: Also apply to fallback path
The render() method has TWO paths - one for custom_gemini_prompt and one for template-based prompts. Apply decluttering to BOTH:

**Around line 1129-1133, find:**
```python
# Format content for the prompt
content = cls._format_content_for_prompt(canonical, engine_key)

# Build the full prompt with quality preamble
prompt = cls.QUALITY_PREAMBLE + template["prompt_template"].format(content=content)
```

**Replace with:**
```python
# Format content for the prompt
content = cls._format_content_for_prompt(canonical, engine_key)

# DECLUTTER: Compress for visual clarity
from src.renderers.declutterer import declutter_content
content = await declutter_content(
    content=content,
    engine_key=engine_key,
    max_elements=config.get("max_visual_elements", 15) if config else 15,
    strategy=config.get("declutter_strategy", "auto") if config else "auto",
)

# Build the full prompt with quality preamble
prompt = cls.QUALITY_PREAMBLE + template["prompt_template"].format(content=content)
```

### Step 4: Add config options to API
In `/home/evgeny/projects/analyzer/src/api/routes/analyze.py`, add to the request schema:
- `max_visual_elements` (int, default 15)
- `declutter_strategy` (str, default "auto")

Pass these through to the renderer config.

### Step 5: Expose in visualizer frontend (optional)
In `/home/evgeny/projects/visualizer/app.py`, add UI controls for:
- Slider: "Visual density" (5-30 elements)
- Dropdown: "Auto" / "Detailed" / "Minimal"

---

## TESTING

### Test 1: Argument Architecture
```bash
# Run analysis on the rentier capitalism paper
# Engine: argument_architecture
# Verify: Output has 6-8 argument boxes, not 10+ with all grounds/warrants
```

### Test 2: Metaphor Network
```bash
# Same paper
# Engine: metaphor_analogy_network
# Verify: 5-7 metaphor bridges, grouped mappings, not 10+ with all details
```

### Test 3: Stakeholder Matrix
```bash
# Engine: stakeholder_power_interest
# Verify: Corner actors prominent, middle actors grouped
```

### Test 4: Content preservation
```bash
# Compare textual output (should have FULL content)
# vs visual output (should have COMPRESSED content)
# Verify: Textual has everything, visual is summarized
```

---

## CONFIGURATION OPTIONS

Add to job submission:
```json
{
  "documents": [...],
  "engine": "argument_architecture",
  "output_mode": "gemini_image",
  "config": {
    "max_visual_elements": 15,
    "declutter_strategy": "auto",
    "declutter_enabled": true
  }
}
```

Strategy options:
- `"auto"` - Select based on engine archetype
- `"argumentative"` - Hierarchical compression for claim-evidence
- `"network"` - Top-N nodes/edges
- `"temporal"` - Period clustering
- `"flow"` - Aggregate small flows
- `"matrix"` - Corner + center focus
- `"none"` - Skip decluttering (for debugging)

---

## SUCCESS CRITERIA

| Metric | Before | After |
|--------|--------|-------|
| Elements per visualization | 50-100 | 15-20 |
| Legibility (subjective 1-10) | 5-6 | 8-9 |
| Information preserved | 100% | 90%+ |
| Added latency | 0s | 1-3s |
| Textual output affected | N/A | NO (unchanged) |

---

## EDGE CASES

1. **Very short content (<500 chars)**: Skip decluttering
2. **Unknown engine**: Use DEFAULT_STRATEGY
3. **API key missing**: Raise clear error, don't silently fail
4. **Claude timeout**: Return original content with warning log
5. **Config missing**: Use defaults (15 elements, auto strategy)

---

## RELATED FILES FOR CONTEXT

- `/home/evgeny/projects/visualizer/analyzer/output_curator.py` - Generates base Gemini prompts
- `/home/evgeny/projects/visualizer/analyzer/style_curator.py` - Style instructions (Tufte, etc.)
- `/home/evgeny/projects/visualizer/analyzer/display_utils.py` - Label formatting rules
- `/home/evgeny/projects/analyzer/src/llm/claude.py` - Claude client (model reference)
- `/home/evgeny/projects/analyzer/src/core/engine_archetypes.py` - Engine type mappings

---

## IMPLEMENTATION CHECKLIST

- [ ] Create `/home/evgeny/projects/analyzer/src/renderers/declutterer.py`
- [ ] Modify `gemini_image.py` render() - custom prompt path (~line 1022)
- [ ] Modify `gemini_image.py` render() - template path (~line 1129)
- [ ] Add config options to analyze.py route
- [ ] Test argument_architecture engine
- [ ] Test metaphor_analogy_network engine
- [ ] Test stakeholder_power_interest engine
- [ ] Verify textual outputs unchanged
- [ ] Add UI controls (optional)
- [ ] Deploy and monitor

---

## DOCUMENT USED FOR TESTING

`/home/evgeny/Downloads/Rentier capitalism technofascism and the destruction of the common.pdf`

This academic paper about rentier capitalism produces particularly dense visualizations due to its complex argumentative structure and multiple metaphor systems.
