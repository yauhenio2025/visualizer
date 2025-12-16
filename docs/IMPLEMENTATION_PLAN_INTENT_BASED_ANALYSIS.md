# Implementation Plan: Intent-Based Multi-Output Analysis System

## Executive Summary

Transform the visualizer from an engine-centric to an **intent-centric** system where:
1. Users express what they want to **understand** (not which engines to run)
2. AI detects what analyses the **document affords**
3. AI selects optimal **engines + output formats** automatically
4. Results arrive in **multiple media** (image + table + text) per analysis

**Key Principle**: Let LLMs figure out the best output format, don't engineer rigid typologies.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           USER INTERFACE                                │
│  "I want to MAP the key players in this policy debate"                  │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    1. DOCUMENT AFFORDANCE DETECTOR                       │
│  Input: Document content (first 3000 chars)                             │
│  Output: {                                                              │
│    "domain": "policy/regulation",                                       │
│    "entity_density": "high",                                            │
│    "temporal_content": "medium",                                        │
│    "quantitative_content": "low",                                       │
│    "suitable_analyses": ["stakeholder", "comparative", "timeline"],     │
│    "unsuitable_analyses": ["deal_flow", "intellectual_genealogy"]       │
│  }                                                                      │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    2. INTENT CLASSIFIER                                  │
│  Input: User's natural language request                                 │
│  Output: {                                                              │
│    "primary_verb": "MAP",                                               │
│    "primary_noun": "ACTORS",                                            │
│    "secondary_intents": ["understand power dynamics", "identify sides"]  │
│  }                                                                      │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    3. ENGINE SELECTOR                                    │
│  Input: Affordances + Intent                                            │
│  Output: {                                                              │
│    "selected_engines": ["stakeholder_power_interest"],                  │
│    "rationale": "User wants to map actors; document has high entity     │
│                  density; stakeholder engine matches MAP + ACTORS"      │
│  }                                                                      │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    4. MULTI-OUTPUT PLANNER                               │
│  Input: Engine + Canonical Schema + Intent                              │
│  Output: {                                                              │
│    "outputs": [                                                         │
│      {"type": "gemini_image", "purpose": "Quadrant visualization"},     │
│      {"type": "table", "purpose": "Sortable list of all stakeholders"}, │
│      {"type": "text", "purpose": "Executive summary of power dynamics"} │
│    ]                                                                    │
│  }                                                                      │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    5. PARALLEL RENDERER                                  │
│  Runs canonical data through multiple renderers simultaneously          │
│  Output: 3 files (image.png, data.html, summary.md)                     │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Preserving Existing Functionality

### MCP Server Tools (UNCHANGED)
These tools will continue to work exactly as before:
- `get_ai_recommendations()` → still works, calls `/api/analyzer/curator/recommend`
- `submit_analysis()` → still works, submits to `/api/analyzer/analyze`
- `check_job_status()` → still works
- `get_results()` → still works
- `list_bundles()`, `list_pipelines()` → still work

### New Tools (ADDED)
New tools that provide the intent-based interface:
- `analyze_with_intent()` → NEW high-level tool
- `get_document_affordances()` → NEW diagnostic tool

---

## Intent Taxonomy

### Verbs (Actions)
| Verb | Description | Maps To |
|------|-------------|---------|
| MAP | See who/what exists and relationships | stakeholder, citation_network, intellectual_genealogy |
| COMPARE | Understand differences | comparative_framework, cross_cultural_variation |
| TRACE | Follow evolution over time | concept_evolution, event_timeline, reception_history |
| EVALUATE | Assess strength/quality | evidence_quality, argument_architecture |
| SYNTHESIZE | Find patterns/themes | thematic_synthesis, structural_pattern_detector |
| DECONSTRUCT | Uncover assumptions/power | assumption_excavation, conditions_of_possibility |
| TRACK | Follow money/resources | resource_flow, deal_flow_tracker |
| IDENTIFY | Find gaps/anomalies/trends | literature_gap, emerging_trend, conceptual_anomaly |
| EXTRACT | Pull key facts/quotes | entity_extraction, quote_attribution, statistical_evidence |
| EXPLAIN | Understand mechanisms | explanatory_pattern, dialectical_structure |

### Nouns (Objects of Analysis)
| Noun | What It Covers |
|------|----------------|
| ACTORS | People, organizations, stakeholders, power |
| ARGUMENTS | Claims, evidence, reasoning |
| CONCEPTS | Frameworks, theories, ideas |
| EVENTS | Timeline, causation, turning points |
| FLOWS | Money, resources, information |
| PATTERNS | Recurring structures, trends |
| GAPS | What's missing, unsaid |
| VOICES | Who speaks, how, authority |

---

## Document Affordance Detection

### Signals to Detect
1. **Domain**: policy, finance, philosophy, technology, history, science
2. **Entity Density**: High (many named entities) vs Low (abstract concepts)
3. **Temporal Content**: Current events vs Historical analysis vs Timeless theory
4. **Quantitative Content**: Numbers/stats present vs Qualitative only
5. **Genre**: News, academic paper, policy document, opinion piece

### Affordance → Engine Mapping
```yaml
high_entity_density:
  enables: [stakeholder_power_interest, entity_extraction, citation_network]

high_temporal_content:
  enables: [event_timeline_causal, chronology_cycle, concept_evolution]

high_quantitative_content:
  enables: [statistical_evidence, resource_flow_asymmetry, deal_flow_tracker]

domain_finance:
  enables: [deal_flow_tracker, resource_flow_asymmetry, competitive_landscape]
  disables: [intellectual_genealogy, conditions_of_possibility]

domain_philosophy:
  enables: [intellectual_genealogy, assumption_excavation, dialectical_structure]
  disables: [deal_flow_tracker, regulatory_pulse]
```

---

## Multi-Output Planning

### Output Media Types
1. **gemini_image** - 4K visualization (networks, quadrants, timelines, flows)
2. **table** - Sortable/filterable HTML table (entity lists, comparison matrices)
3. **text** - Executive memo, structured report
4. **mermaid** - Code-based diagram (flowcharts, sequences)

### When to Use Each
| Engine Output Shape | Best Primary Media | Secondary Media |
|--------------------|-------------------|-----------------|
| Entity list with scores | Quadrant image | Sortable table |
| Relationship network | Network image | Edge table |
| Comparison matrix | Grid image | Comparison table |
| Timeline/chronology | Timeline image | Event table |
| Taxonomy/hierarchy | Tree image | Category table |
| Patterns/findings | Structured text | Key points table |

### LLM-Based Selection (Not Rules)
The Multi-Output Planner uses an LLM to decide, not hardcoded rules:

```
Given this canonical schema for {engine_name}:
{canonical_schema}

And this user intent: {intent}

What output formats would best serve the user?
Consider:
- Visual impact (is this data naturally visual?)
- Data density (too many items for an image?)
- User's likely use case (presentation vs analysis vs quick scan)
- Complementary outputs (image for overview + table for detail)

Recommend 1-3 output formats with rationale.
```

---

## Implementation Phases

### Phase 1: Document Affordance Detector (Analyzer)
**Files to modify**: `analyzer/src/core/curator.py` (NEW)

Create a new curator module that:
1. Takes document sample text
2. Uses Claude to detect domain, entity density, temporal content, quantitative content
3. Returns list of suitable/unsuitable engine categories

**New endpoint**: `POST /api/curator/affordances`
```json
{
  "sample_text": "...",
  "max_chars": 3000
}
```

**Response**:
```json
{
  "domain": "policy",
  "entity_density": "high",
  "temporal_content": "medium",
  "quantitative_content": "low",
  "genre": "news_analysis",
  "suitable_engine_categories": ["power", "argument", "temporal"],
  "unsuitable_engine_categories": ["scholarly", "financial"],
  "reasoning": "This appears to be a policy analysis piece..."
}
```

---

### Phase 2: Intent Classifier (Analyzer)
**Files to modify**: `analyzer/src/core/curator.py`

Add intent classification that:
1. Takes user's natural language request
2. Maps to verb + noun taxonomy
3. Returns structured intent

**New endpoint**: `POST /api/curator/classify-intent`
```json
{
  "user_request": "Show me who the key players are and how they relate"
}
```

**Response**:
```json
{
  "primary_verb": "MAP",
  "primary_noun": "ACTORS",
  "secondary_intents": ["relationships", "power_dynamics"],
  "confidence": 0.92
}
```

---

### Phase 3: Enhanced Engine Selector (Analyzer)
**Files to modify**: `analyzer/src/core/curator.py`

Enhance existing curator to:
1. Take affordances + intent (not just document text)
2. Return engine recommendations with output format suggestions

**Enhanced endpoint**: `POST /api/curator/recommend` (modified)
```json
{
  "sample_text": "...",
  "intent": {
    "verb": "MAP",
    "noun": "ACTORS"
  },
  "affordances": {...}  // Optional, auto-detected if not provided
}
```

**Enhanced Response**:
```json
{
  "recommended_engines": [
    {
      "engine": "stakeholder_power_interest",
      "confidence": 0.95,
      "recommended_outputs": [
        {"type": "gemini_image", "purpose": "Power-interest quadrant"},
        {"type": "table", "purpose": "Sortable stakeholder list"}
      ],
      "rationale": "..."
    }
  ]
}
```

---

### Phase 4: Multi-Output Pipeline (Analyzer)
**Files to modify**:
- `analyzer/src/workers/pipeline.py`
- `analyzer/src/renderers/` (add smart table renderer)

Create ability to:
1. Run single engine extraction + curation
2. Pass canonical output through multiple renderers
3. Return multiple outputs per job

**New job type**: `multi_output`
```json
{
  "documents": [...],
  "engine": "stakeholder_power_interest",
  "output_modes": ["gemini_image", "table", "text"],
  "collection_mode": "single"
}
```

**Response structure**:
```json
{
  "outputs": {
    "stakeholder_power_interest": {
      "gemini_image": {"url": "s3://..."},
      "table": {"html": "..."},
      "text": {"content": "..."}
    }
  }
}
```

---

### Phase 5: Smart Table Renderer (Analyzer)
**Files to modify**: `analyzer/src/renderers/table.py`

Replace hardcoded table templates with LLM-based table generation:
1. LLM analyzes canonical schema
2. LLM decides table structure (columns, sorting, highlighting)
3. LLM generates table HTML

This parallels how Gemini renderer works - LLM decides visualization, not code.

---

### Phase 6: MCP Server Enhancement (Visualizer)
**Files to modify**: `visualizer/mcp_server/mcp_server.py`

Add new high-level tool:

```python
@mcp.tool()
def analyze_with_intent(
    document_path: str,
    intent: str,  # Natural language: "Map the key players"
    anthropic_api_key: Optional[str] = None,
    gemini_api_key: Optional[str] = None
) -> str:
    """
    Analyze a document based on what you want to understand.

    Instead of selecting engines, describe your intent:
    - "Map the key players and their relationships"
    - "Trace how this concept evolved over time"
    - "Evaluate the strength of the arguments"
    - "Find gaps in the current research"

    The AI will:
    1. Detect what analyses your document supports
    2. Select the best engine(s) for your intent
    3. Generate outputs in optimal formats (image, table, text)

    Returns: Multiple outputs tailored to your request.
    """
```

---

### Phase 7: Visualizer UI (Optional)
**Files to modify**: `visualizer/app.py`, `visualizer/templates/`

Add intent-based UI:
1. Upload document
2. See detected affordances
3. Choose from suggested intents (or type custom)
4. Receive multi-format results

---

## API Changes Summary

### New Endpoints (Analyzer)
| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/api/curator/affordances` | POST | Detect document affordances |
| `/api/curator/classify-intent` | POST | Classify user intent |

### Modified Endpoints (Analyzer)
| Endpoint | Change |
|----------|--------|
| `/api/curator/recommend` | Add intent parameter, return output format recommendations |
| `/api/analyze` | Support `output_modes` array for multi-output |

### New MCP Tools (Visualizer)
| Tool | Purpose |
|------|---------|
| `analyze_with_intent()` | High-level intent-based analysis |
| `get_document_affordances()` | Diagnostic: see what document supports |

### Unchanged (Backward Compatible)
- All existing `/api/analyzer/*` endpoints
- All existing MCP tools (`get_ai_recommendations`, `submit_analysis`, etc.)

---

## File Changes Summary

### Analyzer (`/home/evgeny/projects/analyzer/`)
```
src/
├── core/
│   └── curator.py          # NEW: Affordance detector, intent classifier
├── workers/
│   └── pipeline.py         # MODIFY: Add multi-output support
├── renderers/
│   └── table.py            # MODIFY: LLM-based smart table generation
│   └── multi_output.py     # NEW: Orchestrate multiple renderers
└── api/
    └── routes/
        └── curator.py      # NEW: New curator endpoints
```

### Visualizer (`/home/evgeny/projects/visualizer/`)
```
mcp_server/
└── mcp_server.py           # MODIFY: Add analyze_with_intent() tool

app.py                      # MODIFY: Add intent-based UI routes (optional)
```

---

## Testing Plan

### Unit Tests
1. Affordance detection on known document types
2. Intent classification on sample queries
3. Engine selection given affordance + intent combinations
4. Multi-output generation from canonical data

### Integration Tests
1. End-to-end: document → intent → engines → multi-output
2. MCP tool: `analyze_with_intent()` produces expected outputs
3. Backward compatibility: existing tools still work

### Manual Tests
1. FT article → should suggest deal_flow, stakeholder
2. Philosophy paper → should suggest intellectual_genealogy, assumption_excavation
3. Policy document → should suggest comparative_framework, stakeholder

---

## Rollout Strategy

1. **Phase 1-2**: Deploy affordance detector + intent classifier (no breaking changes)
2. **Phase 3**: Deploy enhanced curator (backward compatible)
3. **Phase 4-5**: Deploy multi-output pipeline + smart tables
4. **Phase 6**: Deploy new MCP tool (additive)
5. **Phase 7**: Deploy UI changes (optional)

Each phase can be deployed independently without breaking existing functionality.

---

## Success Metrics

1. **User doesn't need to know engine names** - Intent-based queries work
2. **Multiple useful outputs per analysis** - Image + table + text
3. **Appropriate engine selection** - Philosophy docs don't get deal_flow
4. **No regressions** - Existing MCP tools continue to work

---

## Notes for Implementation Session

1. Start with Phase 1 (Affordance Detector) - it's the foundation
2. Use Claude Sonnet for affordance/intent detection (fast, cheap)
3. Test with diverse documents: FT article, philosophy paper, policy doc
4. Keep existing curator logic as fallback
5. Multi-output can reuse existing renderers (just run multiple)

**Critical**: All changes are ADDITIVE. The existing `get_ai_recommendations` → `submit_analysis` → `get_results` flow must continue to work unchanged.
