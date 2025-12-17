# Session 9: CIA-Level Prompt Consistency Audit

## Mission Brief

**Classification:** HIGH PRIORITY - SYSTEM INTEGRITY
**Objective:** Comprehensive audit of all engine prompts, visualization templates, and curation instructions to identify and resolve logical inconsistencies, conflicting requirements, and impossible instructions.

## Background: The Problem We Just Found

In Session 8, we discovered a critical flaw in the `event_timeline_causal` visualization template that caused Gemini to produce nonsensical outputs:

**The Bug:**
```
LAYOUT:
- Events positioned by date          ← CHRONOLOGICAL axis
- Causal arrows connecting events    ← CAUSAL flow direction
```

These instructions are **logically incompatible** when:
- Event A (Dec 2025) is CAUSED BY Event B (2020)
- Chronological: A should be RIGHT of B (later date)
- Causal flow: A should be RIGHT of B (effect after cause)
- BUT: If Event C (Dec 2025) CAUSES Event D (policy change in Jan 2026), then:
  - Chronological: C is LEFT of D
  - Causal: C is LEFT of D ← OK!
- BUT: If root cause Event E (1970s China grid expansion) eventually leads to Event F (Dec 2025 data center boom):
  - We need to show E → ... → F
  - Timeline says E at 1970s, F at 2025
  - Causal says E left, F right
  - **CONFLICT** if intermediate events from 2024 are causes of F but chronologically after E

**Result:** Gemini tried to satisfy both requirements and produced a diagram where Dec 2025 events appeared at 1990s positions.

## Your Mission

Conduct a **comprehensive audit** of the entire prompt/instruction ecosystem to find similar logical inconsistencies. This is a CIA-level analytical review - assume nothing, question everything, trace logical implications.

## Audit Scope

### 1. Gemini Visualization Templates
**Location:** `/home/evgeny/projects/analyzer/src/renderers/gemini_image.py`

For EACH template in `VISUALIZATION_TEMPLATES`, analyze:

1. **Layout Instructions**
   - Are all positioning requirements compatible?
   - Can the layout physically accommodate all the data types?
   - Do any instructions conflict (e.g., "position by X" AND "position by Y")?

2. **Style Instructions**
   - Are color assignments unambiguous?
   - Do any visual encodings conflict (e.g., "size by importance" AND "size by frequency")?
   - Are there enough visual channels for all the data dimensions?

3. **Data Requirements**
   - Does the template expect data structures that the engine actually produces?
   - Are field names in the template consistent with canonical schema field names?
   - Does the formatter (`_format_*` method) pass the right data structure?

4. **Logical Soundness**
   - Can Gemini actually execute all instructions simultaneously?
   - Are there any physically impossible requirements?
   - Do any "CRITICAL" or "IMPORTANT" notes contradict the main instructions?

### 2. Engine Extraction Prompts
**Location:** `/home/evgeny/projects/analyzer/src/engines/*.py` - `get_extraction_prompt()`

For EACH engine, analyze:

1. **Instruction Clarity**
   - Are extraction instructions unambiguous?
   - Could the same text be interpreted multiple contradictory ways?
   - Are all requested data types clearly defined?

2. **Scope Consistency**
   - Do instructions ask for data the engine schema can't represent?
   - Are there requests for fields not in the canonical schema?
   - Do any instructions ask for incompatible things?

3. **UES Alignment**
   - Does the prompt request types that exist in the UES enums?
   - Are role/type instructions consistent with `src/core/schemas.py`?

### 3. Engine Curation Prompts
**Location:** `/home/evgeny/projects/analyzer/src/engines/*.py` - `get_curation_prompt()`

For EACH engine, analyze:

1. **Aggregation Logic**
   - Are merge/consolidation instructions logically sound?
   - Do any instructions create impossible aggregation requirements?
   - Are conflict resolution rules clear and consistent?

2. **Schema Alignment**
   - Does the curation prompt produce the `canonical_schema` structure?
   - Are all requested output fields present in the schema?
   - Do type instructions match the schema type definitions?

3. **Cross-Reference Consistency**
   - Do ID references work correctly (e.g., "event_id" referenced in causal_relationships)?
   - Are all referenced IDs guaranteed to exist?

### 4. Engine Concretization Prompts
**Location:** `/home/evgeny/projects/analyzer/src/engines/*.py` - `get_concretization_prompt()`

For EACH engine, analyze:

1. **Transformation Soundness**
   - Are all transformation instructions reversible/traceable?
   - Do any transformations lose critical information?
   - Are ID transformations consistent?

### 5. Data Formatters
**Location:** `/home/evgeny/projects/analyzer/src/renderers/gemini_image.py` - `_format_*` methods

For EACH formatter, analyze:

1. **Data Completeness**
   - Does the formatter pass all data the template needs?
   - Are any critical fields omitted?
   - Does the formatter expect fields the engine doesn't produce?

2. **Instruction Embedding**
   - Are embedded instructions (like "[SIZE_GUIDE: 0.85]") clear and actionable?
   - Do any embedded instructions conflict with template instructions?

## Specific Engines to Scrutinize

Based on complexity and potential for similar issues, prioritize these engines:

### HIGH PRIORITY (complex temporal/spatial/relational logic):
1. **event_timeline_causal** - JUST FIXED, verify fix is complete
2. **chronology_simultaneity** - Temporal relationships, same risk
3. **temporal_multiscale** - Multiple time scales, high complexity
4. **temporal_discontinuity_finder** - Temporal anomalies
5. **relational_topology** - Network positioning vs. semantic positioning
6. **citation_network** - Citation flow vs. chronological publication
7. **intellectual_genealogy** - Influence flow vs. temporal order
8. **concept_evolution** - Evolution direction vs. timeline
9. **reception_history** - Reception timeline vs. impact flow
10. **emerging_trend_detector** - Trend curves vs. hype cycles

### MEDIUM PRIORITY (positioning/layout complexity):
11. **stakeholder_power_interest** - 2x2 matrix positioning
12. **competitive_landscape** - Strategic positioning matrix
13. **scholarly_debate_map** - Network layout + camp clustering
14. **resource_flow_asymmetry** - Sankey flow direction
15. **interdisciplinary_connection** - Multi-domain layout

### STANDARD PRIORITY (verify consistency):
- All remaining engines

## Audit Methodology

### Step 1: Template Analysis
For each Gemini visualization template:
```
1. Read the LAYOUT section
2. List all positioning/arrangement requirements
3. Check for pairwise conflicts between requirements
4. Read the STYLE section
5. List all visual encoding requirements
6. Check for encoding channel conflicts
7. Compare template expectations vs. engine canonical schema
8. Compare template expectations vs. formatter output
```

### Step 2: Prompt Chain Analysis
For each engine, trace the full prompt chain:
```
Extraction Prompt → Curation Prompt → Concretization Prompt → Formatter → Template

At each transition, verify:
- Output structure matches next stage's input expectations
- Type/field names are consistent
- No data is lost that's needed downstream
- No conflicting instructions accumulate
```

### Step 3: Conflict Detection
Use this checklist for each instruction set:

**Spatial Conflicts:**
- [ ] "Position by X" + "Position by Y" where X ≠ Y
- [ ] "Left-to-right by A" + "Top-to-bottom by B" conflicts
- [ ] Fixed grid + variable node count
- [ ] Timeline axis + non-temporal ordering

**Visual Encoding Conflicts:**
- [ ] "Size by X" + "Size by Y"
- [ ] "Color by category" + "Color by value"
- [ ] "Thickness by A" + "Thickness by B"

**Data Structure Conflicts:**
- [ ] Template expects array, schema defines object
- [ ] Template expects nested structure, formatter flattens
- [ ] Field name mismatch between stages

**Logical Impossibilities:**
- [ ] "Show all X" when X could be 1000+ items
- [ ] "Connect all pairs" in dense networks
- [ ] Circular reference requirements

## Deliverables

### 1. PROMPT_AUDIT_REPORT.md
Create comprehensive audit report documenting:
- All inconsistencies found
- Severity rating (CRITICAL / HIGH / MEDIUM / LOW)
- Root cause analysis for each
- Recommended fix for each

### 2. Fix Implementation
For each CRITICAL and HIGH severity issue:
- Implement the fix
- Document before/after
- Explain why the fix resolves the conflict

### 3. Audit Checklist
Create reusable checklist for future prompt development:
- Questions to ask before finalizing any prompt
- Common pitfalls to avoid
- Validation steps to run

## Example Audit Entry

```markdown
### Issue: concept_evolution Template Conflict

**Severity:** HIGH
**Location:** `gemini_image.py` lines 282-324

**Conflict:**
Template says:
- "Horizontal timeline as base axis (left=earlier, right=later)"
- "Branching lines showing how concepts split or merge"

**Problem:**
If Concept A (2020) spawns Concept B (2022) AND Concept C (2019 - a retrospective reinterpretation), then:
- Timeline: A at 2020, B at 2022 (right of A), C at 2019 (LEFT of A)
- Branching: A spawns both B and C (both should be right/below A)
- CONFLICT: C must be both LEFT (timeline) and RIGHT (branching) of A

**Fix:**
Change to conceptual branching diagram without timeline axis, OR
Add explicit instruction for handling retrospective concepts

**Implementation:**
[code changes here]
```

## Success Criteria

The audit is complete when:

1. **Every engine** has been reviewed for prompt consistency
2. **Every visualization template** has been checked for logical soundness
3. **All CRITICAL/HIGH issues** have been fixed
4. **PROMPT_AUDIT_REPORT.md** documents all findings
5. **Reusable checklist** exists for future development
6. **All changes committed** with clear commit messages

## Time Estimate

This is a thorough audit. Expect:
- Template analysis: 2-3 hours (47 engines × detailed review)
- Prompt chain analysis: 1-2 hours
- Fix implementation: 1-2 hours
- Documentation: 30 minutes

## Starting Point

Begin with the HIGH PRIORITY engines listed above, especially those with temporal or spatial positioning logic. The `event_timeline_causal` fix provides a template for how to resolve these issues.

## Final Note

This audit is about **logical soundness** - can the instructions actually be followed? Not about style or preference. Every instruction set should be executable without contradiction.

Think like an adversarial tester: "If I were an LLM trying to follow these instructions perfectly, where would I get stuck? Where would I have to make an arbitrary choice because the instructions conflict?"

Find those places. Fix them.
