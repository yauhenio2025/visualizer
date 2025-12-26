# Task Brief: Engine-to-Format Compatibility Mapping

## Problem Statement

The Visualizer has 70 analysis engines but only 28 engine-specific visualization templates. When an engine doesn't have a dedicated template, the system falls back to a generic "choose whatever you want" prompt for Gemini, resulting in inconsistent and often inappropriate visualizations.

## Current Architecture (Flawed)

```
Engine selected → Check VISUALIZATION_TEMPLATES[engine_key]
                     ↓ found? → Use engine-specific template
                     ↓ not found? → Check FORMAT_TEMPLATES[format_key] (from curator)
                                      ↓ found? → Use format template
                                      ↓ not found? → Use _generic template (bad!)
```

**The _generic template says:**
> "Choose the most appropriate diagram type for the data"

This gives Gemini no guidance and produces random visualization types.

## Proposed Architecture (Smart)

Each engine should declare which visualization formats are **compatible** with its output data structure. The curator then picks the **best** format from that constrained list based on the actual extracted content.

```
Engine selected → Look up ENGINE_COMPATIBLE_FORMATS[engine_key]
                     ↓
                  Returns: ["sankey", "alluvial", "chord_diagram", "flowchart"]
                     ↓
                  Curator analyzes extracted data
                     ↓
                  Curator picks best format FROM THAT LIST
                     ↓
                  Renderer uses FORMAT_TEMPLATES[chosen_format]
```

## Key Files to Modify

### 1. Analyzer - Renderer (`/home/evgeny/projects/analyzer/src/renderers/gemini_image.py`)

**Current structure (lines ~96-1120):**
- `VISUALIZATION_TEMPLATES` - 28 engine-specific templates
- `FORMAT_TEMPLATES` - 25 visualization type templates
- `_generic` fallback template

**Add new structure:**
```python
ENGINE_COMPATIBLE_FORMATS = {
    "engine_key": ["format1", "format2", "format3"],
    # ... all 70 engines
}
```

**Modify render logic (~line 1851):**
```python
# Current priority: format_key > engine_key > _generic
# New priority: format_key > engine_key > default_from_compatible_list > _generic
```

### 2. Visualizer - Output Curator (`/home/evgeny/projects/visualizer/analyzer/output_curator.py`)

**Modify curator to:**
1. Accept the list of compatible formats for the engine
2. Analyze the extracted data structure
3. Pick the best format from the compatible list (not from all 25 formats)

**Current curator endpoint:** `/api/analyzer/curate-output` and `/api/analyzer/curate-output/batch`

### 3. Visualizer - App (`/home/evgeny/projects/visualizer/app.py`)

**Modify to:**
1. Look up compatible formats for selected engine(s)
2. Pass compatible formats to curator API
3. Ensure format_key flows through to analyzer

## Existing Architecture: Engine Archetypes

**IMPORTANT**: There's already an archetype system in `/home/evgeny/projects/analyzer/src/core/engine_archetypes.py`

This groups engines by visualization TYPE they produce:

```python
ENGINE_ARCHETYPES = {
    "causal": [
        "event_timeline_causal",
        "escalation_trajectory_analysis",
        "causal_model",
    ],
    "network": [
        "citation_network",
        "stakeholder_power_interest",
        "relational_topology",
        "scholarly_debate_map",
        "interdisciplinary_connection",
        "intellectual_genealogy",
        "science_studies_network",
    ],
    "temporal": [
        "reception_history",
        "emerging_trend_detector",
        "concept_evolution",
        "chronology_simultaneity",
        "temporal_multiscale",
        "temporal_discontinuity_finder",
    ],
    "spatial_matrix": [
        "competitive_landscape",
        "cross_cultural_variation",
        "comparative_framework",
        "epistemic_stance",
    ],
    "structural": [
        "argument_architecture",
        "dialectical_structure",
        "assumption_excavation",
        "conceptual_framework_extraction",
        "structural_pattern_detector",
    ],
    "flow": [
        "resource_flow_asymmetry",
        "deal_flow_tracker",
        "value_chain_decomposition",
    ],
    "assessment": [
        "evidence_quality_assessment",
        "rhetorical_strategy",
        "opportunity_vulnerability_matrix",
    ],
    "anomaly": [
        "conceptual_anomaly_detector",
        "contrarian_concept_generation",
        "literature_gap_identifier",
    ],
    "catalog": [
        "exemplar_catalog",
        "entity_extraction",
        "quote_attribution_voice",
        "statistical_evidence",
    ],
    "thematic": [
        "thematic_synthesis",
        "metaphor_analogy_network",
        "value_ethical_framework",
    ],
}
```

## Proposed: Archetype-to-Format Mapping

Instead of mapping each engine individually, map each ARCHETYPE to compatible formats:

```python
ARCHETYPE_COMPATIBLE_FORMATS = {
    "causal": ["timeline", "flowchart", "sankey", "parallel_timelines"],
    "network": ["network_graph", "chord_diagram", "radial_tree", "matrix_heatmap"],
    "temporal": ["timeline", "alluvial", "gantt", "parallel_timelines"],
    "spatial_matrix": ["quadrant_chart", "matrix_heatmap", "radar_chart", "treemap"],
    "structural": ["argument_tree", "hierarchical_tree", "radial_tree", "flowchart"],
    "flow": ["sankey", "alluvial", "chord_diagram", "flowchart"],
    "assessment": ["radar_chart", "confidence_thermometer", "indicator_dashboard", "bar_chart"],
    "anomaly": ["matrix_heatmap", "radar_chart", "gap_analysis_visual", "network_graph"],
    "catalog": ["treemap", "bar_chart", "matrix_heatmap", "sunburst"],
    "thematic": ["radial_tree", "network_graph", "treemap", "sunburst"],
}
```

Then lookup becomes:
```python
def get_compatible_formats(engine_key: str) -> list[str]:
    archetype = ENGINE_TO_ARCHETYPE.get(engine_key)
    if archetype:
        return ARCHETYPE_COMPATIBLE_FORMATS.get(archetype, [])
    return ["network_graph", "hierarchical_tree", "matrix_heatmap"]  # sensible defaults
```

## Full Engine List by Archetype

From `engine_archetypes.py` (partial - need to verify against actual engine registry):

### causal (→ timeline, flowchart, sankey, parallel_timelines)
- `event_timeline_causal`
- `escalation_trajectory_analysis`
- `causal_model`

### network (→ network_graph, chord_diagram, radial_tree, matrix_heatmap)
- `citation_network`
- `stakeholder_power_interest`
- `relational_topology`
- `scholarly_debate_map`
- `interdisciplinary_connection`
- `intellectual_genealogy`
- `science_studies_network`

### temporal (→ timeline, alluvial, gantt, parallel_timelines)
- `reception_history`
- `emerging_trend_detector`
- `concept_evolution`
- `chronology_simultaneity`
- `temporal_multiscale`
- `temporal_discontinuity_finder`

### spatial_matrix (→ quadrant_chart, matrix_heatmap, radar_chart, treemap)
- `competitive_landscape`
- `cross_cultural_variation`
- `comparative_framework`
- `epistemic_stance`

### structural (→ argument_tree, hierarchical_tree, radial_tree, flowchart)
- `argument_architecture`
- `dialectical_structure`
- `assumption_excavation`
- `conceptual_framework_extraction`
- `structural_pattern_detector`

### flow (→ sankey, alluvial, chord_diagram, flowchart)
- `resource_flow_asymmetry`
- `deal_flow_tracker`
- `value_chain_decomposition`

### assessment (→ radar_chart, confidence_thermometer, indicator_dashboard, bar_chart)
- `evidence_quality_assessment`
- `rhetorical_strategy`
- `opportunity_vulnerability_matrix`

### anomaly (→ matrix_heatmap, radar_chart, gap_analysis_visual, network_graph)
- `conceptual_anomaly_detector`
- `contrarian_concept_generation`
- `literature_gap_identifier`

### catalog (→ treemap, bar_chart, matrix_heatmap, sunburst)
- `exemplar_catalog`
- `entity_extraction`
- `quote_attribution_voice`
- `statistical_evidence`

### thematic (→ radial_tree, network_graph, treemap, sunburst)
- `thematic_synthesis`
- `metaphor_analogy_network`
- `value_ethical_framework`

### UNMAPPED ENGINES (need to add to archetypes)
Check the full engine registry and add any missing engines to appropriate archetypes.

```bash
# Get full engine list from API or database
curl -s https://analyzer-3wsg.onrender.com/v1/engines | jq '.engines[].engine_key'
```

## Implementation Steps

### Step 1: Add ARCHETYPE_COMPATIBLE_FORMATS to engine_archetypes.py
Add the archetype-to-format mapping in `/home/evgeny/projects/analyzer/src/core/engine_archetypes.py`:

```python
ARCHETYPE_COMPATIBLE_FORMATS = {
    "causal": ["timeline", "flowchart", "sankey", "parallel_timelines"],
    "network": ["network_graph", "chord_diagram", "radial_tree", "matrix_heatmap"],
    "temporal": ["timeline", "alluvial", "gantt", "parallel_timelines"],
    "spatial_matrix": ["quadrant_chart", "matrix_heatmap", "radar_chart", "treemap"],
    "structural": ["argument_tree", "hierarchical_tree", "radial_tree", "flowchart"],
    "flow": ["sankey", "alluvial", "chord_diagram", "flowchart"],
    "assessment": ["radar_chart", "confidence_thermometer", "indicator_dashboard", "bar_chart"],
    "anomaly": ["matrix_heatmap", "radar_chart", "gap_analysis_visual", "network_graph"],
    "catalog": ["treemap", "bar_chart", "matrix_heatmap", "sunburst"],
    "thematic": ["radial_tree", "network_graph", "treemap", "sunburst"],
}

def get_compatible_formats(engine_key: str) -> list[str]:
    """Get list of visualization formats compatible with an engine."""
    archetype = ENGINE_TO_ARCHETYPE.get(engine_key)
    if archetype:
        return ARCHETYPE_COMPATIBLE_FORMATS.get(archetype, [])
    return ["network_graph", "hierarchical_tree", "matrix_heatmap"]  # fallback
```

### Step 2: Expose compatible formats via API
Add endpoint in analyzer API (`/home/evgeny/projects/analyzer/src/api/routes/`):

```python
@router.get("/engines/{engine_key}/compatible-formats")
def get_engine_compatible_formats(engine_key: str):
    from src.core.engine_archetypes import get_compatible_formats, ENGINE_TO_ARCHETYPE
    formats = get_compatible_formats(engine_key)
    archetype = ENGINE_TO_ARCHETYPE.get(engine_key, "unknown")
    return {
        "engine_key": engine_key,
        "archetype": archetype,
        "compatible_formats": formats
    }
```

### Step 3: Update OutputCurator
Modify `/home/evgeny/projects/visualizer/analyzer/output_curator.py`:

```python
def curate(self, engine_key: str, extracted_data: dict, compatible_formats: list[str], ...):
    """
    Now accepts compatible_formats list and constrains recommendation to those.
    """
    # Update system prompt to include:
    # "Choose the best visualization format from these options: {compatible_formats}"
    # "Explain why this format best represents the extracted data"
```

### Step 4: Update Visualizer Frontend
Modify `/home/evgeny/projects/visualizer/app.py`:

1. When user selects an engine, fetch its compatible formats
2. Pass compatible_formats to curator API
3. Display compatible formats in curator panel (let user override if desired)

```javascript
// When engine selected:
const compatibleFormats = await fetch(`/api/analyzer/engines/${engineKey}/compatible-formats`);
// Pass to curator:
callOutputCurator(engineKey, extractedData, compatibleFormats.compatible_formats);
```

### Step 5: Update Renderer Fallback
Modify `/home/evgeny/projects/analyzer/src/renderers/gemini_image.py`:

```python
# In render() method, after checking format_key and engine_key templates:
if not template:
    # Try first compatible format as fallback
    from src.core.engine_archetypes import get_compatible_formats
    compatible = get_compatible_formats(engine_key)
    if compatible:
        default_format = compatible[0]  # First is "primary" recommendation
        template = cls.FORMAT_TEMPLATES.get(default_format)
        logger.info("using_archetype_default_format", engine_key=engine_key, format_key=default_format)

if not template:
    template = cls.VISUALIZATION_TEMPLATES.get("_generic")  # Last resort
```

### Step 6: Audit & Complete Archetype Coverage
1. Get full list of all engines from database/registry
2. Ensure every engine is in ENGINE_ARCHETYPES (add missing ones)
3. Verify each archetype has sensible format mappings

### Step 7: Test
- Test each archetype category with sample documents
- Verify curator picks appropriate format from constrained list
- Verify renderer uses correct template
- Verify fallback chain works: format_key → engine template → archetype default → _generic

## Existing Format Templates (25 total)

These are the visualization types available in FORMAT_TEMPLATES:

**Relational:**
- `network_graph` - Force-directed node-link diagram
- `chord_diagram` - Circular bilateral relationships
- `hierarchical_tree` - Top-down tree structure
- `radial_tree` - Concept map from center

**Flow:**
- `sankey` - Flow quantities between nodes
- `alluvial` - Category changes over time/stages
- `flowchart` - Process steps and decisions

**Temporal:**
- `timeline` - Events on time axis
- `gantt` - Duration bars on timeline
- `parallel_timelines` - Multiple synchronized timelines

**Cyclical:**
- `cycle_diagram` - Recurring processes

**Comparative:**
- `matrix_heatmap` - 2D grid with color intensity
- `quadrant_chart` - 2x2 positioning matrix
- `radar_chart` - Multi-axis comparison
- `bar_chart` - Categorical comparison

**Hierarchical:**
- `treemap` - Nested rectangles by size
- `sunburst` - Radial hierarchy

**Quantitative:**
- `waterfall` - Cumulative effect visualization

**Assessment:**
- `ach_matrix` - Analysis of competing hypotheses
- `confidence_thermometer` - Confidence/certainty scale
- `indicator_dashboard` - Multiple metrics display
- `gap_analysis_visual` - Current vs target state

**Argumentative:**
- `argument_tree` - Claims, premises, evidence
- `scenario_cone` - Futures/possibilities fan
- `dialectical_map` - Thesis/antithesis/synthesis

## Success Criteria

1. No engine ever falls through to `_generic` template
2. Curator recommendations are constrained to sensible options
3. Same engine can produce different visualizations based on content
4. All 70 engines have 2-5 compatible formats defined
5. Logs show which format was selected and why

## Reference: Current Template Selection Logic

From `gemini_image.py` line ~1851:
```python
# Priority: format_key > engine_key > _generic
template = None
if format_key and format_key in cls.FORMAT_TEMPLATES:
    template = cls.FORMAT_TEMPLATES[format_key]
    logger.info("gemini_using_format_template", format_key=format_key, engine_key=engine_key)
if not template and engine_key in cls.VISUALIZATION_TEMPLATES:
    template = cls.VISUALIZATION_TEMPLATES[engine_key]
if not template:
    template = cls.VISUALIZATION_TEMPLATES.get("_generic")
```
