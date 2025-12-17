# Dennett Upgrade Implementation Progress

## Quick Start for New Sessions

**IMPORTANT:** If you're a new Claude session continuing this work, read this file FIRST.

### What This Project Is

We're upgrading the Visualizer/Analyzer document intelligence platform to incorporate Daniel Dennett's "Intuition Pumps" as analytical tools. This involves:

1. **New Engines** - 10 Dennett-inspired analysis engines
2. **Prompt Enhancements** - Adding Dennett tools to all existing engines
3. **UI Overhaul** - Radical new "Analytical Canvas" paradigm
4. **Renaming** - Academic framing instead of intelligence/spycraft language

### Key Files to Read

1. **This file** - `docs/DENNETT_UPGRADE_PROGRESS.md` - Current progress and next steps
2. **Main plan** - `docs/IMPLEMENTATION_PLAN_DENNETT_UPGRADE.md` - Full design specification
3. **Original audit** - `docs/STRATEGIC_AUDIT_CIA_PERSPECTIVE.md` - Initial gap analysis

### Architecture Overview

```
VISUALIZER (Frontend/Proxy)     ANALYZER (Backend/Engines)
/home/evgeny/projects/          /home/evgeny/projects/
visualizer/                     analyzer/
├── app.py (Flask UI)           ├── src/engines/*.py (Engine definitions)
├── mcp_server/                 ├── src/bundles/*.py (Bundle definitions)
└── docs/                       ├── src/pipelines/*.py (Pipeline definitions)
                                └── src/core/schemas.py (Data models)
```

**Engines live in Analyzer, not Visualizer.** To add an engine:
1. Create `/home/evgeny/projects/analyzer/src/engines/{engine_key}.py`
2. Register it in the engine registry
3. Define extraction_prompt, curation_prompt, canonical_schema

---

## Current Implementation Status

**Last Updated:** 2025-12-17 09:30 UTC
**Last Session:** Implemented 4 core Dennett engines + dennett_toolkit bundle

### Phase 1: Dennett Core Engines

| Engine | Status | File | Notes |
|--------|--------|------|-------|
| `surely_alarm` | ✅ COMPLETE | `/analyzer/src/engines/surely_alarm.py` | Detects rhetorical confidence markers |
| `occams_broom` | ✅ COMPLETE | `/analyzer/src/engines/occams_broom.py` | Detects strategic omissions |
| `boom_crutch_finder` | ✅ COMPLETE | `/analyzer/src/engines/boom_crutch_finder.py` | Finds "then magic happens" gaps |
| `deepity_detector` | ✅ COMPLETE | `/analyzer/src/engines/deepity_detector.py` | Pseudo-profundity scanner |
| `steelman_generator` | 🔴 NOT STARTED | — | Strongest argument version |
| `jootsing_analyzer` | 🔴 NOT STARTED | — | System boundary explorer |
| `philosophers_syndrome_detector` | 🔴 NOT STARTED | — | Imagination vs necessity |
| `boundary_probe` | 🔴 NOT STARTED | — | Sortes paradox analysis |
| `provenance_audit` | 🔴 NOT STARTED | — | Source quality mapping |
| `epistemic_calibration` | 🔴 NOT STARTED | — | Certainty gradient |

### Phase 2: Existing Engine Enhancements

| Engine | Dennett Header Added | Enhanced Prompt | Notes |
|--------|---------------------|-----------------|-------|
| `argument_architecture` | 🔴 NO | 🔴 NO | Priority 1 |
| `assumption_excavation` | 🔴 NO | 🔴 NO | Priority 1 |
| `rhetorical_strategy` | 🔴 NO | 🔴 NO | Priority 1 |
| `evidence_quality_assessment` | 🔴 NO | 🔴 NO | Priority 1 |
| `absent_center` | 🔴 NO | 🔴 NO | Related to occams_broom |
| `contrarian_concept_generation` | 🔴 NO | 🔴 NO | Related to steelman |
| [Other 41 engines] | 🔴 NO | 🔴 NO | Phase 2 |

### Phase 3: Bundles

| Bundle | Status | File | Notes |
|--------|--------|------|-------|
| `dennett_toolkit` | ✅ COMPLETE | `/analyzer/src/bundles/dennett_toolkit.py` | Core 4 Dennett engines |
| `epistemic_rigor_suite` | 🔴 NOT STARTED | — | Renamed from intelligence_tradecraft |
| `persuasion_archaeology` | 🔴 NOT STARTED | — | Rhetoric analysis |

### Phase 4: Pipelines

| Pipeline | Status | Notes |
|----------|--------|-------|
| `dennett_diagnostic` | 🔴 NOT STARTED | 4-stage Dennett sweep |
| `epistemic_stress_test` | 🔴 NOT STARTED | Arguments → confidence |
| `complete_epistemic_audit` | 🔴 NOT STARTED | 7-stage comprehensive |

### Phase 5: UI Overhaul

| Component | Status | Notes |
|-----------|--------|-------|
| Canvas architecture | 🔴 NOT STARTED | Infinite canvas base |
| Node/edge system | 🔴 NOT STARTED | Document/analysis nodes |
| Lens system | 🔴 NOT STARTED | Analytical overlays |
| Hypothesis workspace | 🔴 NOT STARTED | ACH-style matrix |
| Certainty dashboard | 🔴 NOT STARTED | Confidence visualization |

---

## Implementation Order

### Current Sprint: Phase 1.1 - First 4 Dennett Engines

**IMMEDIATE NEXT STEPS:**

1. ✅ Read this progress file
2. ⏳ Implement `surely_alarm` engine
3. ⏳ Implement `occams_broom` engine
4. ⏳ Implement `boom_crutch_finder` engine
5. ⏳ Implement `deepity_detector` engine
6. ⏳ Create `dennett_toolkit` bundle
7. ⏳ Update this progress file

### Implementation Pattern for Each Engine

Each engine needs:

```python
# /home/evgeny/projects/analyzer/src/engines/{engine_key}.py

from typing import Any, Optional
from src.core.schemas import AnalysisContext, EngineCategory, EngineKind
from src.engines.base import BaseEngine, EngineRegistry

class {ClassName}Engine(BaseEngine):
    engine_key = "{engine_key}"
    engine_name = "{Human Name}"
    description = "..."
    kind = EngineKind.SYNTHESIS  # or RELATIONAL, etc.
    category = EngineCategory.RHETORIC  # or ARGUMENT, EPISTEMOLOGY, etc.
    reasoning_domain = "..."
    researcher_question = "..."
    version = 1

    extraction_focus = [...]
    primary_output_modes = ["structured_text_report", "table"]

    @classmethod
    def get_canonical_schema(cls) -> dict[str, Any]:
        return {...}

    @classmethod
    def get_extraction_prompt(cls, context: Optional[AnalysisContext] = None) -> str:
        return "..."

    @classmethod
    def get_curation_prompt(cls, context: Optional[AnalysisContext] = None) -> str:
        return "..."

# Register the engine
EngineRegistry.register({ClassName}Engine)
```

Then add import to `/home/evgeny/projects/analyzer/src/engines/__init__.py`

---

## Session Handoff Checklist

When ending a session, ensure:

- [ ] This PROGRESS.md is updated with exact status
- [ ] Any partially-written files are noted
- [ ] Next immediate step is clearly stated
- [ ] Any blockers or issues are documented

When starting a new session:

1. Read `docs/DENNETT_UPGRADE_PROGRESS.md` (this file)
2. Check "Current Implementation Status" table
3. Find the first 🔴 NOT STARTED item
4. Read relevant section in `docs/IMPLEMENTATION_PLAN_DENNETT_UPGRADE.md`
5. Implement following the pattern above
6. Update this file before ending session

---

## Technical Context

### EngineCategory Enum

Located at `/home/evgeny/projects/analyzer/src/core/schemas.py`:

```python
class EngineCategory(str, Enum):
    ARGUMENT = "argument"
    CONCEPTS = "concepts"
    TEMPORAL = "temporal"
    POWER = "power"
    EVIDENCE = "evidence"
    RHETORIC = "rhetoric"
    EPISTEMOLOGY = "epistemology"
    SCHOLARLY = "scholarly"
    MARKET = "market"
```

### EngineKind Enum

```python
class EngineKind(str, Enum):
    PRIMITIVE = "primitive"      # Single-purpose extraction
    RELATIONAL = "relational"    # Maps relationships
    SYNTHESIS = "synthesis"      # Aggregates/synthesizes
```

### Output Modes Available

- `structured_text_report` - Markdown narrative
- `table` / `comparative_matrix_table` - HTML tables
- `gemini_*` - Various 4K visualizations
- `mermaid` - Diagram code

---

## Notes and Decisions

### Design Decisions Made

1. **Dennett engines are SYNTHESIS type** - They analyze/judge, not just extract
2. **Category assignment:**
   - `surely_alarm` → RHETORIC (analyzes language use)
   - `occams_broom` → EPISTEMOLOGY (analyzes knowledge gaps)
   - `boom_crutch_finder` → ARGUMENT (analyzes reasoning)
   - `deepity_detector` → RHETORIC (analyzes language)
3. **Primary outputs:** Most Dennett engines output `structured_text_report` + `table`

### Open Questions

- [ ] Should Dennett engines have Gemini visualizations? (Probably later)
- [ ] Exact scoring scales for vulnerability/severity ratings

### Known Issues

(None yet - just starting implementation)

---

## Change Log

| Date | Session | Changes |
|------|---------|---------|
| 2025-12-17 | Initial | Created implementation plan, progress tracker, began implementation |
| 2025-12-17 | Session 1 | Implemented 4 core Dennett engines: surely_alarm, occams_broom, boom_crutch_finder, deepity_detector. Created dennett_toolkit bundle. All registered in __init__.py files. |

---

**NEXT SESSION: Continue with `steelman_generator` engine, then `dennett_diagnostic` pipeline**

## Immediate Next Steps (for next session)

1. ✅ Read this progress file
2. ⏳ Implement `steelman_generator` engine (Rapoport's Rules)
3. ⏳ Implement `jootsing_analyzer` engine (System boundary explorer)
4. ⏳ Create `dennett_diagnostic` pipeline (chains the 4 core engines)
5. ⏳ Add Dennett header to existing engines (argument_architecture, etc.)

## Files Created This Session

**Engines (in /home/evgeny/projects/analyzer/src/engines/):**
- `surely_alarm.py` - ~250 lines, fully functional
- `occams_broom.py` - ~280 lines, fully functional
- `boom_crutch_finder.py` - ~300 lines, fully functional
- `deepity_detector.py` - ~290 lines, fully functional

**Bundles (in /home/evgeny/projects/analyzer/src/bundles/):**
- `dennett_toolkit.py` - ~150 lines, bundles all 4 core engines

**Updated:**
- `/analyzer/src/engines/__init__.py` - Added imports and registrations
- `/analyzer/src/bundles/__init__.py` - Added import and registration

## Verification Commands

To verify engines are registered correctly:
```bash
cd /home/evgeny/projects/analyzer
python -c "from src.engines import EngineRegistry; print([e.engine_key for e in EngineRegistry.list_engines() if 'surely' in e.engine_key or 'occam' in e.engine_key or 'boom' in e.engine_key or 'deepity' in e.engine_key])"
```

To verify bundle is registered:
```bash
cd /home/evgeny/projects/analyzer
python -c "from src.bundles import BundleRegistry; print([b.bundle_key for b in BundleRegistry.list_bundles() if 'dennett' in b.bundle_key])"
```
