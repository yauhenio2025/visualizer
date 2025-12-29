# Strategic Analysis: DECLUTTERER Stage for Visualization Pipeline

## Executive Summary

The visualizer system produces text-heavy visualizations because the complete pipeline transmits ALL extracted data to Gemini without any text density control. This document analyzes where text overload originates and recommends inserting a **DECLUTTERER** stage to intelligently reduce visual clutter while preserving analytical value.

---

## 1. Current Pipeline Architecture

```
Document
    ↓
Analyzer API (Claude extraction)
    ↓
Structured Data (canonical format: arguments, nodes, edges, etc.)
    ↓
Output Curator (Claude Opus) → Format Recommendation + Base Gemini Prompt
    ↓
Style Curator → Styled Gemini Prompt (Tufte/NYT/FT/Lupi/Stefaner)
    ↓
Gemini Image Renderer:
    ├─ _format_content_for_prompt(canonical, engine_key)
    │       → Converts ALL extracted data to text
    │
    ├─ prompt = styled_gemini_prompt + "DATA TO VISUALIZE" + content
    │
    └─ Gemini generate_image(prompt)
            → 4K Image with ALL text elements
```

**Key files:**
- `/home/evgeny/projects/visualizer/analyzer/output_curator.py` - Format recommendation
- `/home/evgeny/projects/visualizer/analyzer/style_curator.py` - Style application
- `/home/evgeny/projects/analyzer/src/renderers/gemini_image.py` - Image generation

---

## 2. Where Text Overload Originates

### 2.1 Source: Extraction Stage
Claude extracts comprehensive data from documents:
- 10-15 arguments with claims, grounds, warrants, rebuttals
- 10+ metaphors with source/target domains and mappings
- 20+ stakeholders with positions, interests, power scores
- Multiple clusters, timelines, relationships

### 2.2 Amplification: Format Functions
Each `_format_*` method in `gemini_image.py` converts extracted data to text:

**Example: `_format_arguments()` (line 1280-1317)**
```python
for arg in arguments[:10]:  # Up to 10 arguments
    lines.append(f"ARGUMENT: {arg_name}")
    lines.append(f"  CLAIM: {claim.get('text')}")
    for g in grounds[:3]:  # Up to 3 grounds each
        lines.append(f"  GROUND: {g.get('text')}")
    lines.append(f"  WARRANT: {warrant.get('text')}")
    lines.append(f"  REBUTTAL: {rebuttal.get('text')}")
```
**Result:** 10 args × 7 elements = **70 text elements** for one engine

**Example: `_format_metaphor_analogy_network()` (line 2993+)**
- 10 metaphors × (name + source→target + effect + 3 mappings)
- Plus analogies, clusters, competing framings
- **Result:** 100+ text elements

### 2.3 No Reduction Before Gemini
The styled prompt gets ALL this content appended:
```python
prompt = f"""{custom_gemini_prompt}

## DATA TO VISUALIZE

{content}  ← ALL 70-100+ text elements

IMPORTANT: Apply ALL style instructions above...
"""
```

Gemini receives instructions about style but no guidance on **text density** or **prioritization**.

---

## 3. Evidence: Screenshots Analysis

| Visualization | Problem | Text Elements |
|--------------|---------|---------------|
| Argument Architecture | Every claim, warrant, ground, rebuttal shown | ~50+ boxes |
| Missing Pieces | Each absence has Expected/Why/Possible Reasons | ~27 text blocks |
| Metaphor Network | Source→target mappings with all details | ~40+ labels |
| Syndrome Indicator | Two-column detailed analysis | ~30+ sections |
| Conceptual Evolution | Dense legend, multiple relationship labels | ~25 annotations |

Even with Tufte-style instructions for "minimal decoration," Gemini tries to render ALL the data.

---

## 4. Insertion Points for DECLUTTERER

### Option A: Extraction Stage
**Location:** Claude extraction prompts
**Action:** Extract fewer elements (top 5 not top 15)
```
CON: Loses potentially valuable information
CON: User might want full extraction for textual outputs
CON: Requires changes to 70+ engine prompts
```

### Option B: Format Functions (Current Limits)
**Location:** `_format_*` methods in `gemini_image.py`
**Action:** Reduce hardcoded limits (e.g., `arguments[:10]` → `arguments[:5]`)
```
PRO: Simple to implement
CON: Arbitrary cutoffs don't prioritize by importance
CON: Same content sent regardless of visualization complexity
```

### Option C: New DECLUTTERER Stage ⭐ RECOMMENDED
**Location:** After format functions, before Gemini prompt assembly
**Action:** Claude analyzes formatted content and intelligently compresses

```python
# In gemini_image.py render() method, before line 1024:
content = cls._format_content_for_prompt(canonical, engine_key)

# NEW STAGE
if should_declutter(engine_key, len(content)):
    content = await declutter_content(
        content=content,
        engine_key=engine_key,
        target_elements=15,  # Max visual elements
        preserve_hierarchy=True,
    )

prompt = f"""{custom_gemini_prompt}
## DATA TO VISUALIZE
{content}
...
"""
```

### Option D: Gemini Prompt Instructions
**Location:** Style modifiers or display instructions
**Action:** Tell Gemini to prioritize/omit
```
CON: Gemini will still try to include everything
CON: No intelligent prioritization of what's important
```

### Option E: Hybrid - Format Limits + Optional Claude Refinement
**Location:** Format functions + optional Claude pass
**Action:** Tighter limits by default, Claude for complex engines
```
PRO: Balanced approach
PRO: Saves API calls for simple cases
CON: Requires tuning per engine
```

---

## 5. Recommended Architecture: DECLUTTERER Agent

### 5.1 Design Principles
1. **Preserve meaning:** 90% of insight with 50% of text
2. **Prioritize by importance:** Keep top-level structure, collapse details
3. **Engine-aware:** Different strategies for argument vs network vs timeline
4. **Non-destructive:** Full extraction preserved for textual outputs

### 5.2 DECLUTTERER Agent Specification

```python
class Declutterer:
    """
    LLM-powered agent that reduces text density for visualization.

    Uses Claude Sonnet for speed (not Opus - this is compression, not analysis).
    """

    async def declutter(
        self,
        content: str,
        engine_key: str,
        max_elements: int = 15,
        strategy: str = "hierarchical",  # hierarchical | top_n | clustering
    ) -> str:
        """
        Compress formatted content for clearer visualization.

        Strategies:
        - hierarchical: Keep top-level, collapse children
        - top_n: Keep N most important items
        - clustering: Group similar items with representative labels
        """
```

### 5.3 Declutter Strategies by Engine Type

| Engine Type | Strategy | Details |
|------------|----------|---------|
| **Argument** (claim-evidence) | hierarchical | Keep claims, show grounds count, omit details |
| **Network** (nodes-edges) | top_n | Top 10 nodes by centrality, top 15 edges |
| **Timeline** (events) | clustering | Group events by period, show key events |
| **Matrix** (2D positioning) | filtering | Show corners and center, omit middle |
| **Flow** (source→target) | simplification | Aggregate small flows into "Other" |

### 5.4 Example: Argument Declutter

**Before (full content):**
```
ARGUMENTS:
ARGUMENT: Rentier Framework Superiority
  CLAIM: Rentier capitalism, not technofeudalism, correctly describes...
  GROUND: Tech companies engage in fierce competition...
  GROUND: Technofeudalism proponents concede their usage involves...
  GROUND: Marx's rent theory in Capital Volume III provides...
  WARRANT: The Marxian category of rent—fixed surplus profits...
  REBUTTAL: Varoufakis argues capitalism has been "killed"...

ARGUMENT: Digital Commons Destruction
  CLAIM: Rentier capitalism accelerates the destruction...
  GROUND: Digital platforms monopolize non-reproducible resources...
  GROUND: Platforms expropriate social connections...
  WARRANT: Rent extraction requires monopolizing resources...
  REBUTTAL: Some argue digital platforms create rather than destroy...

[... 8 more arguments with full details ...]
```

**After declutter (compressed):**
```
ARGUMENTS (6 total, showing 4 primary):

1. MAIN THESIS: Rentier Framework Superiority
   "Rentier capitalism correctly describes the contemporary digital economy"
   → Supported by 3 evidence points, 1 rebuttal

2. SUPPORTING: Digital Commons Destruction
   "Rentier capitalism accelerates destruction of the commons"
   → Supported by 2 evidence points, 1 rebuttal

3. SUPPORTING: Revolutionary Foreclosure
   "Total subsumption forecloses traditional paths to revolution"
   → Supported by 3 evidence points, 1 rebuttal

4. CONCLUSION: Terminal Crisis
   "The current crisis is capitalism itself, not American hegemony"
   → Supported by 2 evidence points, 1 rebuttal

[2 additional arguments omitted - use textual output for full detail]
```

**Impact:** 70 text elements → 20 text elements (71% reduction)

---

## 6. Implementation Plan

### Phase 1: Quick Win - Tighten Format Limits (1-2 hours)
**Location:** `/home/evgeny/projects/analyzer/src/renderers/gemini_image.py`
**Changes:**
- `arguments[:10]` → `arguments[:6]`
- `grounds[:3]` → `grounds[:2]`
- `metaphors[:10]` → `metaphors[:6]`
- Add "See textual output for complete details" note

**Risk:** Low - just tightening existing limits
**Impact:** 30-40% reduction

### Phase 2: Declutter Agent (4-6 hours)
**New file:** `/home/evgeny/projects/analyzer/src/renderers/declutterer.py`
**Changes:**
1. Create `Declutterer` class with Claude Sonnet
2. Add strategy methods for each engine archetype
3. Integrate into `GeminiImageRenderer.render()`
4. Add `declutter_threshold` config option

### Phase 3: Engine-Specific Strategies (2-3 hours per engine type)
**Priority engines:**
1. `argument_architecture` - Most text-heavy
2. `metaphor_analogy_network` - Complex nested structure
3. `stakeholder_power_interest` - Many actors with details
4. `concept_evolution` - Dense legend problem

### Phase 4: Smart Density Control (Future)
- Curator recommends `text_density: low|medium|high`
- Format functions respect density setting
- Declutter runs only when density mismatch detected

---

## 7. Metrics for Success

| Metric | Current | Target |
|--------|---------|--------|
| Avg text elements per visualization | 50+ | 15-20 |
| Visualization legibility (1-10) | 5-6 | 8-9 |
| Information loss | 0% | <10% |
| Added latency per image | 0s | <3s |
| User complaints about clutter | Ongoing | Zero |

---

## 8. Alternative Approaches Considered

### 8.1 Pre-render Preview
Show user a preview with text density indicator, let them choose detail level.
```
CON: Adds user friction
CON: Users don't know optimal density
```

### 8.2 Multi-resolution Output
Generate both detailed and summary visualizations.
```
CON: 2x cost
CON: User must choose which to view
```

### 8.3 Interactive Visualization
Generate D3/SVG with collapsible sections.
```
PRO: Best of both worlds
CON: Requires different renderer entirely
CON: Static image output is a key feature
```

---

## 9. Recommendation

**Implement in order:**
1. **Immediate:** Phase 1 (tighten limits) - 30 min
2. **This week:** Phase 2 (Declutterer agent) - 4-6 hours
3. **Next sprint:** Phase 3 (engine-specific strategies) - ongoing

The DECLUTTERER agent approach is the most robust because:
- Intelligent prioritization vs arbitrary cutoffs
- Preserves full extraction for other outputs
- Can be tuned per engine
- Single insertion point in architecture
- Non-breaking change (optional processing step)

---

## Appendix: Key Code Locations

| Component | File | Line |
|-----------|------|------|
| Format functions | `/home/evgeny/projects/analyzer/src/renderers/gemini_image.py` | 1223-3100 |
| Prompt assembly | `/home/evgeny/projects/analyzer/src/renderers/gemini_image.py` | 1024-1032 |
| Output Curator | `/home/evgeny/projects/visualizer/analyzer/output_curator.py` | 233-295 |
| Style Curator | `/home/evgeny/projects/visualizer/analyzer/style_curator.py` | 685-737 |
| Display Utils | `/home/evgeny/projects/visualizer/analyzer/display_utils.py` | 234-267 |
