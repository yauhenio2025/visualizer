# Dennett + CIA Tradecraft Upgrade Progress

## ⚠️ CRITICAL CONTEXT FOR NEW SESSIONS

**READ THIS FIRST.** We are building a TWO-LAYER analytical enhancement:

```
┌─────────────────────────────────────────────────────────────────┐
│  LAYER 2: DENNETT INTUITION PUMPS (Epistemological Critique)   │
│  ✅ COMPLETE - All 10 engines implemented                       │
├─────────────────────────────────────────────────────────────────┤
│  LAYER 1: CIA TRADECRAFT (Analytical Rigor) - REBRANDED        │
│  ✅ TIER 1 COMPLETE (5/5) + ✅ TIER 2 COMPLETE (5/5)            │
│  ✅ TIER 3 COMPLETE (5/5) - All 15 tradecraft engines done     │
├─────────────────────────────────────────────────────────────────┤
│  FOUNDATION: Existing 47 Engines (Document Intelligence)       │
│  ✅ EXISTS - Now 70 total engines                               │
└─────────────────────────────────────────────────────────────────┘
```

**Session 6 Achievement:** Completed CIA TIER 3 — all 5 optional enhancement engines + 2 additional bundles. Full tradecraft layer is now operational.

**Next Priority:** UI work (keyboard shortcuts, information density, workflow optimization).

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

**Last Updated:** 2025-12-17 14:30 UTC
**Last Session:** Session 8 - Schema audit and curation validation

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

### LAYER 1: CIA Tradecraft (Rebranded)

**TIER 1 - MISSION CRITICAL (5 of 5 done) ✅ COMPLETE:**

| CIA Original | Dennett Rebrand | Status | Notes |
|--------------|-----------------|--------|-------|
| `source_credibility_assessment` | `provenance_audit` | ✅ DONE | Session 3 |
| `analytic_confidence_levels` | `epistemic_calibration` | ✅ DONE | Session 3 |
| `competing_hypotheses_analysis` | `hypothesis_tournament` | ✅ DONE | **Session 4** - ACH methodology |
| `deception_indicator_detection` | `authenticity_forensics` | ✅ DONE | **Session 4** - Manipulation detection |
| `information_gaps_analysis` | `terra_incognita_mapper` | ✅ DONE | **Session 4** - Collection requirements |

**TIER 2 - HIGH PRIORITY (5 of 5 done) ✅ COMPLETE:**

| CIA Original | Dennett Rebrand | Status | Notes |
|--------------|-----------------|--------|-------|
| `indicators_warnings_tracker` | `signal_sentinel` | ✅ DONE | **Session 5** - I&W tracking |
| `scenario_futures_matrix` | `possibility_space_explorer` | ✅ DONE | **Session 5** - Scenario planning |
| `network_centrality_analysis` | `relational_topology` | ✅ DONE | **Session 5** - Graph-theoretic analysis |
| `decision_maker_profiling` | `rational_actor_modeling` | ✅ DONE | **Session 5** - Leader psychology |
| `timeline_anomaly_detection` | `temporal_discontinuity_finder` | ✅ DONE | **Session 5** - Pattern breaks |

**TIER 3 - ENHANCED CAPABILITY (5 of 5 done) ✅ COMPLETE:**

| CIA Original | Dennett Rebrand | Status | Notes |
|--------------|-----------------|--------|-------|
| `red_team_challenge` | `steelman_stress_test` | ✅ DONE | **Session 6** - Enhanced adversarial testing |
| `escalation_trajectory_analysis` | `escalation_trajectory_analysis` | ✅ DONE | **Session 6** - Crisis progression modeling |
| `opportunity_vulnerability_matrix` | `opportunity_vulnerability_matrix` | ✅ DONE | **Session 6** - Exploitable gaps assessment |
| `influence_attribution_analysis` | `influence_attribution_analysis` | ✅ DONE | **Session 6** - Campaign attribution |
| `key_intelligence_questions_mapper` | `key_intelligence_questions_mapper` | ✅ DONE | **Session 6** - KIQ/EEI alignment |

### Bundles Status

| Bundle | Status | Member Engines |
|--------|--------|----------------|
| `dennett_toolkit` | ✅ COMPLETE | surely_alarm, occams_broom, boom_crutch_finder, deepity_detector |
| `epistemic_rigor_suite` | ✅ COMPLETE | provenance_audit, epistemic_calibration, hypothesis_tournament, steelman_generator |
| `strategic_warning` | ✅ COMPLETE | signal_sentinel, possibility_space_explorer, temporal_discontinuity_finder |
| `persuasion_archaeology` | ✅ COMPLETE | surely_alarm + deepity_detector + rhetorical_strategy + authenticity_forensics |
| `network_intelligence` | ✅ COMPLETE | stakeholder_power_interest + relational_topology + rational_actor_modeling |

### Pipelines Status

| Pipeline | Status | Stages |
|----------|--------|--------|
| `dennett_diagnostic` | ✅ COMPLETE | surely_alarm → boom_crutch → deepity → occams_broom |
| `epistemic_stress_test` | ✅ COMPLETE | argument_architecture → steelman → philosophers_syndrome → epistemic_calibration |
| `source_to_confidence` | ✅ COMPLETE | provenance_audit → epistemic_calibration → terra_incognita_mapper |
| `warning_assessment_complete` | ✅ COMPLETE | signal_sentinel → temporal_discontinuity_finder → possibility_space_explorer |
| `analytic_rigor_pipeline` | 🔴 NOT DONE | argument_architecture → hypothesis_tournament → steelman_stress_test → epistemic_calibration |
| `complete_epistemic_audit` | 🔴 NOT DONE | 7-stage comprehensive |

---

## 🎯 IMMEDIATE NEXT STEPS (Session 5+)

### Priority Order: Complete TIER 2 (5 engines)

**Session 5 Target: All 5 TIER 2 engines + strategic_warning bundle**

---

### ENGINE 1: `signal_sentinel` (was `indicators_warnings_tracker`)

**Purpose:** Track early warning indicators against established signposts. What observable changes would signal that a scenario is unfolding?

**Category:** TEMPORAL

**Core Concept:** I&W (Indications & Warnings) methodology
- Define SIGNPOSTS: Observable events that would indicate change
- Track INDICATORS: Current evidence for/against each signpost
- Assess TRAJECTORY: Direction of change
- Flag THRESHOLD CROSSINGS: When indicators suggest action needed

**Key Schema Elements:**
```
signposts: [
  {id, description, scenario_it_signals, observable_indicators[], threshold_for_action}
]
indicators: [
  {id, signpost_id, current_status, trend, last_observed, evidence}
]
warning_assessment: {
  overall_posture: NORMAL | ELEVATED | HIGH | CRITICAL,
  most_active_signposts: [],
  trajectory: STABLE | ESCALATING | DE-ESCALATING
}
```

**Use Case:** "What signs should we watch for? What do current indicators tell us?"

---

### ENGINE 2: `possibility_space_explorer` (was `scenario_futures_matrix`)

**Purpose:** Map multiple plausible futures and their drivers. Cone of plausibility analysis.

**Category:** SYNTHESIS (or ARGUMENT)

**Core Concept:** Scenario planning methodology
- Identify KEY DRIVERS: Forces that shape the future
- Define SCENARIOS: Plausible future states (not predictions)
- Map PATHWAYS: How we get from here to each scenario
- Assess PROBABILITIES: Relative likelihood (without false precision)

**Key Schema Elements:**
```
drivers: [
  {id, name, description, current_state, range_of_outcomes, uncertainty_level}
]
scenarios: [
  {id, name, description, driver_configuration, probability_assessment,
   implications, indicators_this_is_happening}
]
scenario_matrix: {
  axes: [driver1, driver2],  # Usually 2x2 or 2x3 matrix
  quadrants: [{scenario_id, position}]
}
pathways: [
  {from_current, to_scenario, key_events, branch_points}
]
```

**Use Case:** "What futures are possible? How would we get there? What should we watch?"

---

### ENGINE 3: `relational_topology` (was `network_centrality_analysis`)

**Purpose:** Graph-theoretic analysis of actor networks. Find key nodes, structural holes, influence pathways.

**Category:** POWER

**Core Concept:** Social network analysis + graph theory
- Calculate CENTRALITY: Degree, betweenness, eigenvector, closeness
- Find STRUCTURAL HOLES: Bridges between clusters
- Identify GATEKEEPERS: Who controls information/resource flow
- Map INFLUENCE PATHWAYS: How does influence propagate?

**Key Schema Elements:**
```
nodes: [
  {id, label, node_type, attributes,
   centrality_scores: {degree, betweenness, eigenvector, closeness}}
]
edges: [
  {source, target, relationship_type, weight, direction}
]
clusters: [
  {id, members[], internal_density, label}
]
structural_analysis: {
  key_brokers: [],  # High betweenness
  influencers: [],  # High eigenvector
  gatekeepers: [],  # Control flow between clusters
  vulnerabilities: []  # Single points of failure
}
```

**Use Case:** "Who are the key players? Who bridges groups? Where are vulnerabilities?"

**Note:** Extends existing `stakeholder_power_interest` with formal graph metrics.

---

### ENGINE 4: `rational_actor_modeling` (was `decision_maker_profiling`)

**Purpose:** Model how key decision-makers think and decide. Cognitive profiling for prediction.

**Category:** POWER

**Core Concept:** Decision-maker psychology + rational actor theory
- Map WORLDVIEW: How do they see the situation?
- Identify DECISION STYLE: Risk tolerance, time horizon, information preferences
- Model CONSTRAINTS: What limits their options?
- Predict RESPONSES: How would they react to scenarios?

**Key Schema Elements:**
```
actors: [
  {id, name, role,
   worldview: {core_beliefs[], threat_perception, opportunity_perception},
   decision_style: {risk_tolerance, time_horizon, information_preference,
                    consensus_vs_decisive, ideological_vs_pragmatic},
   constraints: {institutional, political, resource, domestic},
   track_record: [{decision, context, outcome, what_it_reveals}],
   predicted_responses: [{scenario, likely_action, confidence, rationale}]
  }
]
interaction_dynamics: [
  {actor1, actor2, relationship, influence_direction, trust_level}
]
```

**Use Case:** "How does this leader think? How would they respond to X?"

---

### ENGINE 5: `temporal_discontinuity_finder` (was `timeline_anomaly_detection`)

**Purpose:** Detect suspicious timing, unusual gaps, pattern breaks in timelines.

**Category:** TEMPORAL

**Core Concept:** Anomaly detection in temporal sequences
- Find GAPS: Missing time periods, unexplained silences
- Detect CLUSTERING: Suspicious bunching of events
- Flag PATTERN BREAKS: Deviations from established rhythms
- Identify ANACHRONISMS: Things that don't fit their supposed time

**Key Schema Elements:**
```
timeline: [
  {timestamp, event, source, confidence}
]
anomalies: [
  {anomaly_type: GAP | CLUSTER | PATTERN_BREAK | ANACHRONISM | SUSPICIOUS_TIMING,
   description, location_in_timeline, severity, possible_explanations[]}
]
patterns: [
  {pattern_type, description, instances[], breaks[]}
]
assessment: {
  timeline_integrity: HIGH | MEDIUM | LOW,
  most_suspicious_anomalies: [],
  investigation_priorities: []
}
```

**Use Case:** "Is this timeline consistent? Are there suspicious patterns or gaps?"

---

### BUNDLE: `strategic_warning`

**Members:** signal_sentinel + possibility_space_explorer + temporal_discontinuity_finder

**Purpose:** Complete early warning and futures analysis package.

**Unified Extraction Focus:**
- Indicators and signposts
- Future scenarios and drivers
- Timeline events and patterns
- Warning thresholds and trajectories

---

### PIPELINE: `warning_assessment_complete`

**Stages:** signal_sentinel → temporal_discontinuity_finder → possibility_space_explorer

**Synergy:** "Track indicators, flag timeline anomalies, project futures"

---

### After TIER 2: TIER 3 Engines (Session 6+)

- `steelman_stress_test` (enhanced red team — builds on steelman_generator)
- `escalation_trajectory_analysis` (crisis progression modeling)
- `opportunity_vulnerability_matrix` (exploitable gaps)
- `influence_attribution_analysis` (campaign attribution)
- `key_intelligence_questions_mapper` (KIQ/EEI alignment)

### UI Work Comes AFTER Tradecraft Layer

Do NOT start UI overhaul until TIER 2 is complete.

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

### `hypothesis_tournament` (ACH) Schema ✅ IMPLEMENTED

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
| 2025-12-17 | Session 4 | **hypothesis_tournament, authenticity_forensics, terra_incognita_mapper + epistemic_rigor_suite bundle + source_to_confidence pipeline. TIER 1 COMPLETE.** |
| 2025-12-17 | Session 5 | **signal_sentinel, possibility_space_explorer, relational_topology, rational_actor_modeling, temporal_discontinuity_finder + strategic_warning bundle + warning_assessment_complete pipeline. TIER 2 COMPLETE.** |
| 2025-12-17 | Session 6 | **steelman_stress_test, escalation_trajectory_analysis, opportunity_vulnerability_matrix, influence_attribution_analysis, key_intelligence_questions_mapper + network_intelligence bundle + persuasion_archaeology bundle. TIER 3 COMPLETE. ALL TRADECRAFT ENGINES DONE.** |
| 2025-12-17 | Session 7 | **UI optimization for 70+ engines scale: Quick Actions cards, compact lists, filtering, action icons.** |
| 2025-12-17 | Session 8 | **Schema audit: Audited all 70 engines, added curation validation, documented type system architecture.** |

---

## Verification Commands

```bash
# Verify all engines registered
cd /home/evgeny/projects/analyzer
python -c "
from src.engines import EngineRegistry
engines = [e.engine_key for e in EngineRegistry.list_engines()]
print(f'Total engines: {len(engines)}')

# Check CIA TIER 1
tier1 = ['provenance_audit', 'epistemic_calibration', 'hypothesis_tournament',
         'authenticity_forensics', 'terra_incognita_mapper']
for e in tier1:
    print(f'{e}: {\"✅\" if e in engines else \"❌\"}')"

# Verify bundles
python -c "
from src.bundles import BundleRegistry
bundles = [b.bundle_key for b in BundleRegistry.list_bundles()]
print(f'epistemic_rigor_suite: {\"✅\" if \"epistemic_rigor_suite\" in bundles else \"❌\"}')"

# Verify pipelines
python -c "
from src.pipelines import MetaEngineRegistry
pipelines = [p.pipeline_key for p in MetaEngineRegistry.list_pipelines()]
print(f'source_to_confidence: {\"✅\" if \"source_to_confidence\" in pipelines else \"❌\"}')"
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

### Session 4
- `hypothesis_tournament.py` (ACH methodology)
- `authenticity_forensics.py` (deception/manipulation detection)
- `terra_incognita_mapper.py` (information gaps + collection requirements)
- `epistemic_rigor_suite.py` (bundle)
- `source_to_confidence.py` (pipeline)

### Session 5
- `signal_sentinel.py` (I&W tracking - indicators & warnings)
- `possibility_space_explorer.py` (scenario futures - planning methodology)
- `relational_topology.py` (network centrality - graph theory)
- `rational_actor_modeling.py` (decision maker profiling)
- `temporal_discontinuity_finder.py` (timeline anomaly detection)
- `strategic_warning.py` (bundle)
- `warning_assessment_complete.py` (pipeline)

### Session 6
- `steelman_stress_test.py` (enhanced red team - adversarial testing)
- `escalation_trajectory_analysis.py` (crisis progression modeling)
- `opportunity_vulnerability_matrix.py` (exploitable gaps assessment)
- `influence_attribution_analysis.py` (campaign attribution)
- `key_intelligence_questions_mapper.py` (KIQ/EEI alignment)
- `network_intelligence.py` (bundle)
- `persuasion_archaeology.py` (bundle)

---

## Session 6 Checklist (TIER 3 / Optional Enhancements) ✅ COMPLETE

**ALL TIERS COMPLETE!** Full tradecraft layer is operational.

**TIER 3 engines (all done):**
- [x] `steelman_stress_test` — Enhanced red team (builds on steelman_generator)
- [x] `escalation_trajectory_analysis` — Crisis progression modeling
- [x] `opportunity_vulnerability_matrix` — Exploitable gaps assessment
- [x] `influence_attribution_analysis` — Campaign attribution
- [x] `key_intelligence_questions_mapper` — KIQ/EEI alignment

**Additional bundles (all done):**
- [x] `network_intelligence` bundle (stakeholder_power_interest + relational_topology + rational_actor_modeling)
- [x] `persuasion_archaeology` bundle (surely_alarm + deepity_detector + rhetorical_strategy + authenticity_forensics)

**Current Stats:**
- Total engines: 70
- Total bundles: 18
- Total pipelines: 21

---

## Session 7 Checklist (UI Work) ✅ COMPLETE

Implemented UI optimizations for 70+ engines scale:
- [x] Card-based Quick Actions system
- [x] Compact engine display in lists
- [x] Filtering/search functionality
- [x] Action icons for immediate analysis
- [x] Visual distinction between engines/bundles/pipelines

---

## Session 8 Checklist (Schema Audit) ✅ COMPLETE

**Purpose:** Audit all 70 engines for schema/type consistency.

**Part 1 - Audit:**
- [x] Comprehensive audit of all 70 engines
- [x] Identified 25 engines with custom types (by design)
- [x] Added runtime curation validation
- [x] Created audit report with recommendations

**Part 2 - Fixes (all recommendations implemented):**
- [x] Fixed entity_extraction: `location` → `place`
- [x] Extended core enums: +4 EntityType, +11 RelationType
- [x] Created `src/core/extended_types.py` with 16 domain-specific enums
- [x] Created `scripts/validate_schemas.py` for CI validation
- [x] Updated audit report with all fixes

**Key Findings:**
- Engine canonical schemas use domain-specific types (intentional)
- Extraction phase validates against UES enums
- Curation phase had NO validation — now fixed
- Renderers are type-agnostic (structure-based)

**Files Created (in analyzer repo):**
- `audit_schemas.py` — Script to audit all engines
- `schema_audit_results.json` — Complete audit data
- `docs/SCHEMA_AUDIT_REPORT.md` — Full findings + fixes
- `src/core/curation.py` — Added validation functions
- `src/core/schemas.py` — Extended EntityType, RelationType
- `src/core/extended_types.py` — 16 domain-specific enum classes
- `scripts/validate_schemas.py` — CI validation script

**Validation Results:**
```
Validated: 70 engines, 18 bundles, 21 pipelines
All validations passed!
Warnings: 15 engines with custom types (expected)
```

**Next Session Priority:** Additional UI work or engine improvements as needed.
