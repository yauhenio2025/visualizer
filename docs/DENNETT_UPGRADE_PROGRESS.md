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

**Last Updated:** 2025-12-17 11:00 UTC
**Last Session:** Implemented 4 new Dennett engines (philosophers_syndrome_detector, boundary_probe, provenance_audit, epistemic_calibration) + epistemic_stress_test pipeline

### Phase 1: Dennett Core Engines

| Engine | Status | File | Notes |
|--------|--------|------|-------|
| `surely_alarm` | ✅ COMPLETE | `/analyzer/src/engines/surely_alarm.py` | Detects rhetorical confidence markers |
| `occams_broom` | ✅ COMPLETE | `/analyzer/src/engines/occams_broom.py` | Detects strategic omissions |
| `boom_crutch_finder` | ✅ COMPLETE | `/analyzer/src/engines/boom_crutch_finder.py` | Finds "then magic happens" gaps |
| `deepity_detector` | ✅ COMPLETE | `/analyzer/src/engines/deepity_detector.py` | Pseudo-profundity scanner |
| `steelman_generator` | ✅ COMPLETE | `/analyzer/src/engines/steelman_generator.py` | Rapoport's Rules: strongest argument version |
| `jootsing_analyzer` | ✅ COMPLETE | `/analyzer/src/engines/jootsing_analyzer.py` | System boundary explorer - jumping out of the system |
| `philosophers_syndrome_detector` | ✅ COMPLETE | `/analyzer/src/engines/philosophers_syndrome_detector.py` | Imagination failure detector |
| `boundary_probe` | ✅ COMPLETE | `/analyzer/src/engines/boundary_probe.py` | Sortes paradox analysis |
| `provenance_audit` | ✅ COMPLETE | `/analyzer/src/engines/provenance_audit.py` | Source quality mapping with heterophenomenology |
| `epistemic_calibration` | ✅ COMPLETE | `/analyzer/src/engines/epistemic_calibration.py` | Certainty gradient with uncertainty types |

### Phase 2: Existing Engine Enhancements

| Engine | Dennett Header Added | Enhanced Prompt | Notes |
|--------|---------------------|-----------------|-------|
| `argument_architecture` | ✅ YES | ✅ YES | Toulmin + Dennett toolkit |
| `assumption_excavation` | ✅ YES | ✅ YES | Dennett archaeological probes |
| `rhetorical_strategy` | ✅ YES | ✅ YES | Rhetorical + Dennett fairness checks |
| `evidence_quality_assessment` | ✅ YES | ✅ YES | Epistemic rigor + provenance |
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
| `dennett_diagnostic` | ✅ COMPLETE | 4-stage Dennett sweep: surely_alarm → boom_crutch → deepity → occams_broom |
| `epistemic_stress_test` | ✅ COMPLETE | 4-stage: argument_architecture → steelman → philosophers_syndrome → epistemic_calibration |
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
| 2025-12-17 | Session 2 | Implemented steelman_generator + jootsing_analyzer engines. Created dennett_diagnostic pipeline. Added Dennett headers to 4 priority engines (argument_architecture, assumption_excavation, rhetorical_strategy, evidence_quality_assessment). |
| 2025-12-17 | Session 3 | Implemented philosophers_syndrome_detector, boundary_probe, provenance_audit, epistemic_calibration engines. Created epistemic_stress_test pipeline. All 10 core Dennett engines now complete! |

---

**NEXT SESSION: Phase 2 - Enhance existing engines with Dennett headers + create remaining bundles**

## Immediate Next Steps (for next session)

### All 10 Core Dennett Engines are COMPLETE! 🎉

1. ✅ Read this progress file
2. ⏳ Add Dennett headers to remaining priority engines:
   - `absent_center` (related to occams_broom)
   - `contrarian_concept_generation` (related to steelman)
   - Other high-value engines
3. ⏳ Create `epistemic_rigor_suite` bundle (provenance_audit + epistemic_calibration + steelman_generator)
4. ⏳ Create `persuasion_archaeology` bundle (surely_alarm + deepity_detector + rhetorical_strategy)
5. ⏳ Create `complete_epistemic_audit` pipeline (7-stage comprehensive)
6. ⏳ Begin Phase 5: UI Overhaul planning

## Files Created/Modified This Session (Session 3)

**New Engines (in /home/evgeny/projects/analyzer/src/engines/):**
- `philosophers_syndrome_detector.py` - ~280 lines, Imagination failure vs impossibility detection
- `boundary_probe.py` - ~300 lines, Sortes paradox analysis for concept boundaries
- `provenance_audit.py` - ~320 lines, Source quality mapping with heterophenomenology
- `epistemic_calibration.py` - ~340 lines, Certainty gradient with uncertainty type classification

**New Pipelines (in /home/evgeny/projects/analyzer/src/pipelines/):**
- `epistemic_stress_test.py` - 4-stage pipeline: argument_architecture → steelman_generator → philosophers_syndrome_detector → epistemic_calibration

**Updated:**
- `/analyzer/src/engines/__init__.py` - Added 4 new Dennett engines
- `/analyzer/src/pipelines/__init__.py` - Added epistemic_stress_test pipeline

## Previous Session Files (Session 2)

**New Engines (in /home/evgeny/projects/analyzer/src/engines/):**
- `steelman_generator.py` - ~280 lines, Rapoport's Rules for steelmanning arguments
- `jootsing_analyzer.py` - ~300 lines, System boundary explorer

**New Pipelines (in /home/evgeny/projects/analyzer/src/pipelines/):**
- `dennett_diagnostic.py` - 4-stage pipeline: surely_alarm → boom_crutch_finder → deepity_detector → occams_broom

**Modified Engines (Dennett headers added):**
- `argument_architecture.py` - Enhanced with Dennett toolkit integration
- `assumption_excavation.py` - Enhanced with Dennett archaeological probes
- `rhetorical_strategy.py` - Enhanced with Dennett fairness checks
- `evidence_quality_assessment.py` - Enhanced with epistemic rigor tools

## Previous Session Files

**Engines (in /home/evgeny/projects/analyzer/src/engines/):**
- `surely_alarm.py` - ~250 lines, fully functional
- `occams_broom.py` - ~280 lines, fully functional
- `boom_crutch_finder.py` - ~300 lines, fully functional
- `deepity_detector.py` - ~290 lines, fully functional

**Bundles (in /home/evgeny/projects/analyzer/src/bundles/):**
- `dennett_toolkit.py` - ~150 lines, bundles all 4 core engines

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
