# Dennett + CIA Tradecraft Upgrade Progress

## ⚠️ CRITICAL CONTEXT FOR NEW SESSIONS

**READ THIS FIRST.** We are building a TWO-LAYER analytical enhancement:

```
┌─────────────────────────────────────────────────────────────────┐
│  LAYER 2: DENNETT INTUITION PUMPS (Epistemological Critique)   │
│  ✅ COMPLETE - All 10 engines implemented                       │
├─────────────────────────────────────────────────────────────────┤
│  LAYER 1: CIA TRADECRAFT (Analytical Rigor) - REBRANDED        │
│  ⚠️ ONLY 2 OF 15 ENGINES DONE - THIS IS THE PRIORITY NOW       │
├─────────────────────────────────────────────────────────────────┤
│  FOUNDATION: Existing 47 Engines (Document Intelligence)       │
│  ✅ EXISTS                                                      │
└─────────────────────────────────────────────────────────────────┘
```

**The Problem:** We got enchanted by Dennett and completed all 10 Intuition Pump engines, but only 2 of the 15 CIA tradecraft engines (rebranded to academic terms). The CIA layer provides the analytical RIGOR that the Dennett layer CRITIQUES.

**The Fix:** Next sessions must prioritize CIA tradecraft engines BEFORE UI work.

---

## Quick Reference: What We're Building

### Source Documents (READ THESE)

| Document | Purpose | Priority |
|----------|---------|----------|
| `docs/STRATEGIC_AUDIT_CIA_PERSPECTIVE.md` | Original gap analysis - lists ALL missing tradecraft engines | **READ FIRST** |
| `docs/IMPLEMENTATION_PLAN_DENNETT_UPGRADE.md` | Dennett tools + renaming scheme | Reference |
| This file | Current progress and next steps | Working doc |

### Architecture

```
VISUALIZER (Frontend/Proxy)     ANALYZER (Backend/Engines)
/home/evgeny/projects/          /home/evgeny/projects/
visualizer/                     analyzer/
├── app.py (Flask UI)           ├── src/engines/*.py (Engine definitions)
├── mcp_server/                 ├── src/bundles/*.py (Bundle definitions)
└── docs/                       ├── src/pipelines/*.py (Pipeline definitions)
                                └── src/core/schemas.py (Data models)
```

**Engines live in Analyzer, not Visualizer.**

---

## Current Implementation Status

**Last Updated:** 2025-12-17 11:30 UTC
**Last Session:** Session 3 - Completed all 10 Dennett engines + epistemic_stress_test pipeline

### LAYER 2: Dennett Intuition Pumps ✅ COMPLETE

All 10 engines implemented and registered:

| Engine | Status | CIA Equivalent | Notes |
|--------|--------|----------------|-------|
| `surely_alarm` | ✅ | — | Rhetorical confidence markers |
| `occams_broom` | ✅ | — | Strategic omissions |
| `boom_crutch_finder` | ✅ | — | "Then magic happens" gaps |
| `deepity_detector` | ✅ | — | Pseudo-profundity scanner |
| `steelman_generator` | ✅ | `red_team_challenge` (partial) | Rapoport's Rules |
| `jootsing_analyzer` | ✅ | — | System boundary explorer |
| `philosophers_syndrome_detector` | ✅ | — | Imagination failure detector |
| `boundary_probe` | ✅ | — | Sortes paradox analysis |
| `provenance_audit` | ✅ | **`source_credibility_assessment`** | Heterophenomenology + source mapping |
| `epistemic_calibration` | ✅ | **`analytic_confidence_levels`** | Certainty gradient |

### LAYER 1: CIA Tradecraft (Rebranded) ⚠️ INCOMPLETE

**TIER 1 - MISSION CRITICAL (2 of 5 done):**

| CIA Original | Dennett Rebrand | Status | Priority |
|--------------|-----------------|--------|----------|
| `source_credibility_assessment` | `provenance_audit` | ✅ DONE | — |
| `analytic_confidence_levels` | `epistemic_calibration` | ✅ DONE | — |
| `competing_hypotheses_analysis` | `hypothesis_tournament` | 🔴 **NOT DONE** | **#1 PRIORITY** |
| `deception_indicator_detection` | `authenticity_forensics` | 🔴 NOT DONE | #2 PRIORITY |
| `information_gaps_analysis` | `terra_incognita_mapper` | 🔴 NOT DONE | #3 PRIORITY |

**TIER 2 - HIGH PRIORITY (0 of 5 done):**

| CIA Original | Dennett Rebrand | Status | Notes |
|--------------|-----------------|--------|-------|
| `indicators_warnings_tracker` | `signal_sentinel` | 🔴 NOT DONE | I&W tracking |
| `scenario_futures_matrix` | `possibility_space_explorer` | 🔴 NOT DONE | Scenario planning |
| `network_centrality_analysis` | `relational_topology` | 🔴 NOT DONE | Graph-theoretic analysis |
| `decision_maker_profiling` | `rational_actor_modeling` | 🔴 NOT DONE | Leader psychology |
| `timeline_anomaly_detection` | `temporal_discontinuity_finder` | 🔴 NOT DONE | Pattern breaks |

**TIER 3 - ENHANCED CAPABILITY (0 of 5 done):**

| CIA Original | Dennett Rebrand | Status | Notes |
|--------------|-----------------|--------|-------|
| `red_team_challenge` | `steelman_stress_test` | 🟡 PARTIAL | `steelman_generator` covers some |
| `escalation_trajectory_analysis` | — | 🔴 NOT DONE | Crisis progression |
| `opportunity_vulnerability_matrix` | — | 🔴 NOT DONE | Exploitable gaps |
| `influence_attribution_analysis` | — | 🔴 NOT DONE | Campaign attribution |
| `key_intelligence_questions_mapper` | — | 🔴 NOT DONE | KIQ/EEI alignment |

### Bundles Status

| Bundle | Status | Member Engines |
|--------|--------|----------------|
| `dennett_toolkit` | ✅ COMPLETE | surely_alarm, occams_broom, boom_crutch_finder, deepity_detector |
| `epistemic_rigor_suite` | 🔴 NOT DONE | provenance_audit + epistemic_calibration + hypothesis_tournament + steelman_generator |
| `persuasion_archaeology` | 🔴 NOT DONE | surely_alarm + deepity_detector + rhetorical_strategy + authenticity_forensics |
| `strategic_warning` | 🔴 NOT DONE | signal_sentinel + possibility_space_explorer + temporal_discontinuity_finder |
| `network_intelligence` | 🔴 NOT DONE | stakeholder_power_interest + relational_topology + rational_actor_modeling |

### Pipelines Status

| Pipeline | Status | Stages |
|----------|--------|--------|
| `dennett_diagnostic` | ✅ COMPLETE | surely_alarm → boom_crutch → deepity → occams_broom |
| `epistemic_stress_test` | ✅ COMPLETE | argument_architecture → steelman → philosophers_syndrome → epistemic_calibration |
| `source_to_confidence` | 🔴 NOT DONE | provenance_audit → epistemic_calibration → terra_incognita_mapper |
| `analytic_rigor_pipeline` | 🔴 NOT DONE | argument_architecture → hypothesis_tournament → steelman_stress_test → epistemic_calibration |
| `complete_epistemic_audit` | 🔴 NOT DONE | 7-stage comprehensive |

---

## 🎯 IMMEDIATE NEXT STEPS (Session 4+)

### Priority Order: Complete CIA Tradecraft Layer

**Session 4 should implement these 3 engines (completes TIER 1):**

1. **`hypothesis_tournament`** (was `competing_hypotheses_analysis`)
   - ACH is THE gold standard intelligence methodology
   - Category: ARGUMENT
   - See CIA audit Section VII for schema
   - ~350 lines estimated

2. **`authenticity_forensics`** (was `deception_indicator_detection`)
   - Manipulation/disinfo detection
   - Category: RHETORIC
   - See CIA audit for manipulation indicators
   - ~300 lines estimated

3. **`terra_incognita_mapper`** (was `information_gaps_analysis`)
   - What we DON'T know - critical for collection tasking
   - Category: EPISTEMOLOGY
   - ~280 lines estimated

**Then create:**
4. `epistemic_rigor_suite` bundle
5. `source_to_confidence` pipeline

### After TIER 1 Complete: TIER 2 Engines

Session 5-6:
- `signal_sentinel` (indicators & warnings)
- `possibility_space_explorer` (scenario futures)
- `relational_topology` (network centrality)

### UI Work Comes AFTER Tradecraft Layer

Do NOT start UI overhaul until at least TIER 1 + TIER 2 engines are complete.

---

## Engine Implementation Pattern

```python
# /home/evgeny/projects/analyzer/src/engines/{engine_key}.py

from typing import Any, Optional
from src.core.schemas import AnalysisContext, EngineCategory, EngineKind
from src.engines.base import BaseEngine, EngineRegistry

class {ClassName}Engine(BaseEngine):
    engine_key = "{engine_key}"
    engine_name = "{Human Name}"
    description = "..."
    kind = EngineKind.SYNTHESIS
    category = EngineCategory.{CATEGORY}
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

## Key Schemas from CIA Audit

### `hypothesis_tournament` (ACH) Schema

```json
{
  "hypotheses": [
    {
      "id": "string",
      "description": "string",
      "initial_likelihood": "string"
    }
  ],
  "evidence_items": [
    {
      "id": "string",
      "description": "string",
      "source": "string",
      "evaluations": {
        "H1": "CONSISTENT | INCONSISTENT | NEUTRAL",
        "H2": "CONSISTENT | INCONSISTENT | NEUTRAL"
      },
      "diagnosticity": "HIGH | MEDIUM | LOW"
    }
  ],
  "analysis": {
    "most_supported_hypothesis": "string",
    "confidence": "string",
    "key_discriminating_evidence": ["string"],
    "critical_uncertainties": ["string"],
    "collection_priorities": ["string"]
  }
}
```

### Source Reliability Scale (IC Standard)

```
A: Reliable (no doubt of authenticity, trustworthiness, competency)
B: Usually Reliable (minor doubt)
C: Fairly Reliable (doubt in some instances)
D: Not Usually Reliable (significant doubt)
E: Unreliable (lacking authenticity, trustworthiness, competency)
F: Cannot Be Judged
```

### Information Validity Scale (1-6)

```
1: Confirmed by independent sources
2: Probably true (consistent with other information)
3: Possibly true (not confirmed, not contradicted)
4: Doubtfully true (inconsistent with other information)
5: Improbable (contradicted by other information)
6: Cannot be judged
```

### Confidence Language (IC Standard)

```
"almost certain" = 95%+ probability
"highly likely" = 80-95% probability
"likely" = 60-80% probability
"roughly even chance" = 40-60% probability
"unlikely" = 20-40% probability
"highly unlikely" = 5-20% probability
"remote" = <5% probability
```

---

## Change Log

| Date | Session | Changes |
|------|---------|---------|
| 2025-12-17 | Initial | Created implementation plan, progress tracker |
| 2025-12-17 | Session 1 | 4 core Dennett engines + dennett_toolkit bundle |
| 2025-12-17 | Session 2 | steelman_generator + jootsing_analyzer + dennett_diagnostic pipeline + Dennett headers on 4 engines |
| 2025-12-17 | Session 3 | philosophers_syndrome_detector, boundary_probe, provenance_audit, epistemic_calibration + epistemic_stress_test pipeline. **All 10 Dennett engines complete.** |
| 2025-12-17 | Session 3 (end) | **CRITICAL INSIGHT:** Identified that CIA tradecraft layer (13 of 15 engines) still missing. Reprioritized roadmap. |

---

## Verification Commands

```bash
# Verify all 10 Dennett engines registered
cd /home/evgeny/projects/analyzer
python -c "
from src.engines import EngineRegistry
dennett = ['surely_alarm', 'occams_broom', 'boom_crutch_finder', 'deepity_detector',
           'steelman_generator', 'jootsing_analyzer', 'philosophers_syndrome_detector',
           'boundary_probe', 'provenance_audit', 'epistemic_calibration']
registered = [e.engine_key for e in EngineRegistry.list_engines()]
for d in dennett:
    print(f'{d}: {\"✅\" if d in registered else \"❌\"}')"

# Verify pipelines
python -c "
from src.pipelines import MetaEngineRegistry
for p in MetaEngineRegistry.list_pipelines():
    if 'dennett' in p.pipeline_key or 'epistemic' in p.pipeline_key:
        print(f'{p.pipeline_key}: {len(p.stages)} stages')"
```

---

## Files Created Across All Sessions

### Session 1
- `surely_alarm.py`, `occams_broom.py`, `boom_crutch_finder.py`, `deepity_detector.py`
- `dennett_toolkit.py` (bundle)

### Session 2
- `steelman_generator.py`, `jootsing_analyzer.py`
- `dennett_diagnostic.py` (pipeline)
- Enhanced: argument_architecture, assumption_excavation, rhetorical_strategy, evidence_quality_assessment

### Session 3
- `philosophers_syndrome_detector.py`, `boundary_probe.py`, `provenance_audit.py`, `epistemic_calibration.py`
- `epistemic_stress_test.py` (pipeline)

---

## Session 4 Checklist

When starting Session 4:

- [ ] Read this file
- [ ] Read `STRATEGIC_AUDIT_CIA_PERSPECTIVE.md` Section VII for schemas
- [ ] Implement `hypothesis_tournament` engine
- [ ] Implement `authenticity_forensics` engine
- [ ] Implement `terra_incognita_mapper` engine
- [ ] Create `epistemic_rigor_suite` bundle
- [ ] Create `source_to_confidence` pipeline
- [ ] Update this progress file
- [ ] Commit to both repos

**DO NOT start UI work until CIA TIER 1 + TIER 2 engines are complete.**
