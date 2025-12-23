# Textual Output Types

8 differentiated output types, each serving different consumers with different needs.

## Quick Reference

| # | Type | Icon | Length | Core Question |
|---|------|------|--------|---------------|
| 1 | **Snapshot** | ⚡ | ~400 words | "What do I need to know RIGHT NOW?" |
| 2 | **Deep Dive** | 🔬 | 2000-5000 words | "What's the full picture?" |
| 3 | **Evidence Pack** | 📁 | Variable | "What's the evidence chain?" |
| 4 | **Signal Report** | 📡 | ~800 words | "What signals indicate change?" |
| 5 | **Status Brief** | 📋 | ~1200 words | "What happened and where are we?" |
| 6 | **Stakeholder Profile** | 👤 | ~1500 words/actor | "Who is this and how will they act?" |
| 7 | **Gap Analysis** | 🎯 | ~1500 words | "Where are the weaknesses?" |
| 8 | **Options Brief** | ⚖️ | ~1200 words | "What should I choose?" |

---

## 1. Snapshot ⚡

**Purpose**: Immediate situational awareness for time-starved decision-makers

**Audience**: Executives, leaders who have 2 minutes maximum

**Structure**:
- Bottom Line (1 sentence)
- Key Finding (2-3 sentences)
- Implications (3 bullets, one line each)
- Confidence level with reasoning
- What to watch for

**Best For Engines**:
- `signal_sentinel`, `anomaly_detector`, `emerging_trend_detector`
- Any engine where "what changed?" matters most

**Visual Complement**: Visual shows the anomaly/signal; Snapshot says what to DO about it

---

## 2. Deep Dive 🔬

**Purpose**: Comprehensive synthesis with calibrated confidence levels

**Audience**: Analysts, researchers, subject matter experts

**Structure**:
- Scope Note
- Key Judgments (each with confidence level + evidence for/against)
- Competing Hypotheses (if applicable)
- Detailed Analysis sections
- What We Don't Know (substantive, not perfunctory)
- Outlook with confidence
- Methodology note

**Confidence Framework**:
- **HIGH**: Multiple independent sources agree; logical consistency
- **MODERATE**: Plausible but limited sources; some uncertainty
- **LOW**: Single source; significant gaps; alternatives equally viable

**Best For Engines**:
- `hypothesis_tournament`, `evidence_quality_assessment`, `thematic_synthesis`
- Any engine producing complex, multi-faceted findings

**Visual Complement**: Visual shows structure; Deep Dive explains WHY and HOW CONFIDENT

---

## 3. Evidence Pack 📁

**Purpose**: Complete source documentation for verification and audit

**Audience**: Due diligence teams, fact-checkers, analysts verifying claims

**Structure**:
- Evidence Index (ID, claim, source, reliability)
- Detailed Evidence Items (verbatim quotes, context, corroboration)
- Contradictions Log
- Gaps Register (what we looked for but didn't find)
- Source Evaluation Notes

**Source Reliability Scale**:
- **A**: Completely reliable (verified, corroborated)
- **B**: Usually reliable (strong track record)
- **C**: Fairly reliable (some verification possible)
- **D**: Not usually reliable (use with caution)
- **E**: Unreliable (contradicted by other sources)
- **F**: Cannot be judged

**Best For Engines**:
- `provenance_audit`, `statistical_evidence`, `quote_attribution_voice`
- Any engine extracting specific claims from sources

**Visual Complement**: Visual shows patterns; Evidence Pack provides the receipts

---

## 4. Signal Report 📡

**Purpose**: Alert to emerging patterns requiring attention or monitoring

**Audience**: Strategic planners, risk managers

**Structure**:
- Signal Summary (what we're detecting)
- Indicators Detected (status table: triggered/elevated/normal)
- Pattern Analysis
- Threshold Analysis (baseline vs. current)
- Timeline Assessment (when decision may be required)
- Recommended Responses
- Watch List (next 48-72 hours)
- False Positive Assessment

**Best For Engines**:
- `signal_sentinel`, `temporal_discontinuity_finder`, `escalation_trajectory_analysis`
- Any engine detecting change or emergence

**Visual Complement**: Visual shows the timeline/pattern; Signal Report says what it MEANS

---

## 5. Status Brief 📋

**Purpose**: Comprehensive update on current situation

**Audience**: Operations teams, stakeholders tracking developments

**Structure**:
- Situation Summary (current state)
- Period Developments (chronological table)
- Key Actors Update
- Change from Previous (escalated/stable/de-escalated)
- Resource & Flow Update
- Open Questions (specific)
- Near-Term Outlook (falsifiable)
- Analyst Assessment

**Best For Engines**:
- `event_timeline_causal`, `stakeholder_power_interest`, `resource_flow_asymmetry`
- Any engine tracking evolution over time

**Visual Complement**: Visual shows the landscape; Status Brief explains what CHANGED

---

## 6. Stakeholder Profile 👤

**Purpose**: Deep understanding of key actors for engagement strategy

**Audience**: Negotiators, strategists, anyone who needs to influence or understand actors

**Structure**:
- Summary (who they are, why they matter)
- Basic Information
- Interests & Motivations (with evidence)
- Red Lines
- Decision-Making Pattern
- Network Position
- Track Record
- Statements & Positions (quotes + actions vs. words)
- Behavioral Prediction
- Engagement Recommendations

**Best For Engines**:
- `rational_actor_modeling`, `stakeholder_power_interest`, `relational_topology`
- Any engine analyzing actors

**Visual Complement**: Visual shows network position; Profile explains HOW TO ENGAGE

---

## 7. Gap Analysis 🎯

**Purpose**: Systematic identification of weaknesses for strengthening

**Audience**: Critics, red teams, quality assurance, anyone stress-testing

**Structure**:
- Executive Summary
- Vulnerability Inventory (severity/exploitability/impact matrix)
- Detailed Vulnerability Analysis (each with attack vector + mitigation)
- Hidden Assumptions
- Logical Gaps
- Evidence Weaknesses
- Steelman Counterarguments (strongest attacks)
- Mitigation Roadmap (prioritized)
- Residual Risk (what can't be eliminated)

**Best For Engines**:
- `assumption_excavation`, `steelman_stress_test`, `boom_crutch_finder`
- Any engine finding flaws or gaps

**Visual Complement**: Visual shows structure; Gap Analysis shows WHERE IT'S WEAK

---

## 8. Options Brief ⚖️

**Purpose**: Decision support with clear options and trade-offs

**Audience**: Decision-makers choosing between alternatives

**Structure**:
- Decision Required (one clear sentence)
- Background (minimal)
- Options (each with pros, cons, risks, resources)
- Comparison Matrix
- Recommendation with Rationale
- Implementation Steps (if accepted)
- Decision Triggers (when to revisit)

**Best For Engines**:
- `comparative_framework`, `possibility_space_explorer`, `steelman_generator`
- Any engine comparing alternatives

**Visual Complement**: Visual shows comparison landscape; Options Brief says WHICH TO CHOOSE

---

## Engine → Output Type Mapping

### Quick Reference: Best Output by Engine Category

| Engine Category | Primary Output | Secondary Output |
|-----------------|----------------|------------------|
| Detection/Warning | Signal Report | Snapshot |
| Evidence/Verification | Deep Dive | Evidence Pack |
| Actor/Network | Stakeholder Profile | Status Brief |
| Temporal/Evolution | Status Brief | Deep Dive |
| Argument/Logic | Gap Analysis | Options Brief |
| Comparison | Options Brief | Deep Dive |
| Deception/Credibility | Deep Dive | Gap Analysis |
| Thematic/Conceptual | Deep Dive | Status Brief |

---

## The Visual-Text Contract

**Rule**: Text NEVER describes what the visual shows. They complement, not duplicate.

| Visual Shows | Text Provides |
|--------------|---------------|
| Structure | Why it matters |
| Relationships | How to navigate them |
| Patterns | What they mean |
| Comparisons | Which to choose |
| Timeline | Causal explanation |
| Anomalies | What to do about them |

**Test**: Could you understand each independently? Does each add something the other can't?

---

## Implementation

Templates are in: `analyzer/prompts/textual_output_templates.py`

Each template includes:
- Structural skeleton
- Generation instructions
- Quality criteria
- Complementarity rules

Use `get_template(output_type)` to retrieve.
Use `get_recommended_outputs(engine_key)` to get best outputs for an engine.
