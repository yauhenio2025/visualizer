# Intelligence-Grade Textual Outputs

## What Would the CIA Do?

The US Intelligence Community has spent decades perfecting how to turn raw data into actionable intelligence products. They've learned that **different consumers need different products** - not just different lengths, but fundamentally different structures, emphases, and purposes.

This document maps our 70 engines to intelligence-grade textual output formats.

---

## CIA Intelligence Product Types (Our Inspiration)

| IC Product | Purpose | Consumer | Our Equivalent |
|------------|---------|----------|----------------|
| **President's Daily Brief (PDB)** | Top-level situational awareness | POTUS (5 min max) | **Flash Assessment** |
| **National Intelligence Estimate (NIE)** | Comprehensive multi-source synthesis | Senior policymakers | **Intelligence Assessment** |
| **Intelligence Information Report (IIR)** | Single-source raw intelligence | Analysts for fusion | **Evidence Dossier** |
| **Warning Intelligence** | Early indicators of change | Decision-makers | **Warning Bulletin** |
| **Current Intelligence** | Timely updates on evolving situations | Operations staff | **Situation Report (SITREP)** |
| **Intelligence Memorandum** | In-depth single-topic analysis | Subject matter experts | **Deep Dive Analysis** |
| **Biographical Intelligence** | Actor profiles and motivations | Case officers, diplomats | **Actor Profile** |
| **Vulnerability Assessment** | Gaps that could be exploited | Security teams | **Vulnerability Assessment** |

---

## Our 8 Intelligence-Grade Textual Output Types

### 1. FLASH ASSESSMENT
**CIA Equivalent**: President's Daily Brief item
**Length**: 1 page maximum (300-400 words)
**Reading Time**: 2 minutes
**Purpose**: Immediate situational awareness for time-starved decision-makers

**Structure**:
```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FLASH ASSESSMENT: [Topic in 5 words]
Classification: [UNCLASSIFIED / SENSITIVE]
Date: [YYYY-MM-DD]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

BOTTOM LINE
[One sentence. What you need to know right now.]

KEY DEVELOPMENT
[2-3 sentences. What changed and why it matters.]

IMPLICATIONS
• [Implication 1 - one line]
• [Implication 2 - one line]
• [Implication 3 - one line]

CONFIDENCE: [HIGH/MODERATE/LOW] - [One sentence explaining why]

WATCH FOR
[One sentence: What signal would change this assessment]

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

**Engines Best Suited**:
- `signal_sentinel` - Early warning detection
- `anomaly_detector` - Unusual pattern identification
- `emerging_trend_detector` - Trend emergence
- `temporal_discontinuity_finder` - Timeline breaks
- `escalation_trajectory_analysis` - Escalation dynamics

**What Visual Shows vs. What Flash Assessment Adds**:
| Visual | Flash Assessment |
|--------|------------------|
| Pattern/anomaly highlighted | Why it matters NOW |
| Timeline with break point | What decision this requires |
| Escalation trajectory | What to watch for next |

---

### 2. INTELLIGENCE ASSESSMENT
**CIA Equivalent**: National Intelligence Estimate (NIE)
**Length**: 5-15 pages
**Reading Time**: 30-60 minutes
**Purpose**: Comprehensive synthesis with calibrated confidence

**Structure**:
```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
INTELLIGENCE ASSESSMENT: [Topic]
Classification: [UNCLASSIFIED / SENSITIVE]
Date: [YYYY-MM-DD]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

SCOPE NOTE
[What this assessment covers, what it excludes, and why]

KEY JUDGMENTS
1. [Judgment with confidence level]
   - We assess with [HIGH/MODERATE/LOW] confidence that...
   - Evidence supporting this judgment: [brief]
   - Evidence against this judgment: [brief]

2. [Judgment with confidence level]
   ...

CONFIDENCE FRAMEWORK
┌────────────────────────────────────────────────────────────┐
│ HIGH CONFIDENCE: Multiple independent sources agree;      │
│                  logical consistency; analytic consensus  │
│ MODERATE CONFIDENCE: Plausible but limited sources;       │
│                      some uncertainty in interpretation   │
│ LOW CONFIDENCE: Single source; significant gaps;          │
│                 alternative explanations equally viable   │
└────────────────────────────────────────────────────────────┘

DETAILED ANALYSIS

[Section 1: Major Finding]
Assessment: [What we judge to be true]
Evidence Base:
  • [Evidence item 1 with source reliability]
  • [Evidence item 2 with source reliability]
  • [Evidence item 3 with source reliability]
Analytic Line: [How we interpret this evidence]
Alternative View: [What dissenting analysts argue]
Gaps: [What we don't know that matters]

[Section 2: Major Finding]
...

COMPETING HYPOTHESES
┌─────────────────┬────────────────┬────────────────────────┐
│ Hypothesis      │ Consistency    │ Key Evidence For/Against│
├─────────────────┼────────────────┼────────────────────────┤
│ H1: [...]       │ [High/Med/Low] │ For: [...] Against: [...]│
│ H2: [...]       │ [High/Med/Low] │ For: [...] Against: [...]│
│ H3: [...]       │ [High/Med/Low] │ For: [...] Against: [...]│
└─────────────────┴────────────────┴────────────────────────┘

WHAT WE DON'T KNOW
[Critical gaps that would change this assessment if filled]

OUTLOOK
[Where this is heading based on current trajectory]

APPENDIX: SOURCE EVALUATION
[Source reliability ratings and methodology notes]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

**Engines Best Suited**:
- `hypothesis_tournament` - Competing hypotheses analysis
- `evidence_quality_assessment` - Evidence evaluation
- `epistemic_calibration` - Confidence calibration
- `terra_incognita_mapper` - Knowledge gaps
- `provenance_audit` - Source chain verification
- `assumption_excavation` - Hidden assumptions

**What Visual Shows vs. What Intelligence Assessment Adds**:
| Visual | Intelligence Assessment |
|--------|------------------------|
| Hypothesis comparison matrix | Why we favor one over others |
| Evidence quality radar | Source-by-source reliability |
| Knowledge gap map | Implications of each gap |

---

### 3. EVIDENCE DOSSIER
**CIA Equivalent**: Intelligence Information Report (IIR)
**Length**: 10-30 pages (depends on evidence volume)
**Reading Time**: Reference document (not linear reading)
**Purpose**: Complete evidence chain for verification and audit

**Structure**:
```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
EVIDENCE DOSSIER: [Topic]
Classification: [UNCLASSIFIED / SENSITIVE]
Date: [YYYY-MM-DD]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

EVIDENCE INDEX
┌─────┬────────────────────┬────────────────┬────────────────┐
│ ID  │ Claim              │ Source         │ Reliability    │
├─────┼────────────────────┼────────────────┼────────────────┤
│ E001│ [Claim summary]    │ [Source ref]   │ [A/B/C/D/E/F]  │
│ E002│ [Claim summary]    │ [Source ref]   │ [A/B/C/D/E/F]  │
│ ... │ ...                │ ...            │ ...            │
└─────┴────────────────────┴────────────────┴────────────────┘

SOURCE RELIABILITY KEY
┌───────────────────────────────────────────────────────────┐
│ A - Completely reliable (verified, corroborated)         │
│ B - Usually reliable (strong track record)               │
│ C - Fairly reliable (some verification possible)         │
│ D - Not usually reliable (limited verification)          │
│ E - Unreliable (contradicted by other sources)           │
│ F - Cannot be judged (insufficient basis)                │
└───────────────────────────────────────────────────────────┘

DETAILED EVIDENCE ITEMS

────────────────────────────────────────────────────────────
ITEM E001
────────────────────────────────────────────────────────────
Claim: [Full claim text]
Source: [Document title, author, date, page/section]
Reliability: [Letter rating]
Verbatim Extract:
  "[Direct quote from source with page reference]"
Context: [What was happening when this was said/written]
Corroboration: [Other sources that support/contradict]
Analyst Note: [Any caveats or interpretive notes]

────────────────────────────────────────────────────────────
ITEM E002
────────────────────────────────────────────────────────────
...

CONTRADICTIONS LOG
[Where sources disagree, with analysis of which to credit]

PROVENANCE CHAINS
[For key claims: Source → Cited By → Cited By → Our Document]

GAPS REGISTER
[Evidence we looked for but did not find]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

**Engines Best Suited**:
- `statistical_evidence` - Quantitative evidence extraction
- `quote_attribution_voice` - Quote extraction with attribution
- `exemplar_catalog` - Case study catalog
- `citation_network` - Citation chain mapping
- `provenance_audit` - Source verification

**What Visual Shows vs. What Evidence Dossier Adds**:
| Visual | Evidence Dossier |
|--------|------------------|
| Citation network graph | Full quotes with page numbers |
| Evidence quality radar | Source-by-source reliability ratings |
| Statistical summary | Raw numbers with methodology notes |

---

### 4. WARNING BULLETIN
**CIA Equivalent**: Warning Intelligence / Indications & Warnings
**Length**: 1-2 pages
**Reading Time**: 5 minutes
**Purpose**: Alert decision-makers to emerging threats/opportunities

**Structure**:
```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
⚠️ WARNING BULLETIN: [Threat/Opportunity in 5 words]
Priority: [CRITICAL / HIGH / ELEVATED / ROUTINE]
Date: [YYYY-MM-DD]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

THREAT/OPPORTUNITY SUMMARY
[2-3 sentences: What we're seeing and why it matters]

INDICATORS DETECTED
┌────────────┬──────────────────┬───────────────────────────┐
│ Indicator  │ Status           │ What It Suggests          │
├────────────┼──────────────────┼───────────────────────────┤
│ [Signal 1] │ 🔴 TRIGGERED     │ [Interpretation]          │
│ [Signal 2] │ 🟡 ELEVATED      │ [Interpretation]          │
│ [Signal 3] │ 🟢 NORMAL        │ [Would indicate if...]    │
└────────────┴──────────────────┴───────────────────────────┘

THRESHOLD ANALYSIS
• Previous baseline: [What was normal]
• Current reading: [What we're seeing now]
• Deviation: [How significant the change is]

TIMELINE TO DECISION
[How long before situation crystallizes / window closes]

RECOMMENDED ACTIONS
□ [Immediate action 1]
□ [Near-term action 2]
□ [Contingency to prepare]

WATCH LIST (Next 48-72 hours)
• [Specific indicator to monitor]
• [Specific indicator to monitor]
• [Trigger that would elevate priority]

CONFIDENCE: [HIGH/MODERATE/LOW]
[One sentence: What could make this a false alarm]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

**Engines Best Suited**:
- `signal_sentinel` - Indications & warnings
- `temporal_discontinuity_finder` - Pattern breaks
- `escalation_trajectory_analysis` - Escalation modeling
- `anomaly_detector` - Outlier detection
- `emerging_trend_detector` - Trend emergence

**What Visual Shows vs. What Warning Bulletin Adds**:
| Visual | Warning Bulletin |
|--------|-----------------|
| Anomaly highlighted on timeline | What to DO about it |
| Escalation curve | When decision point arrives |
| Signal dashboard | Which signals to watch next |

---

### 5. SITUATION REPORT (SITREP)
**CIA Equivalent**: Current Intelligence
**Length**: 2-3 pages
**Reading Time**: 10 minutes
**Purpose**: Comprehensive status update on evolving situation

**Structure**:
```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SITUATION REPORT: [Topic]
Report Period: [Date range]
Date: [YYYY-MM-DD]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

SITUATION SUMMARY
[3-4 sentences: Current state of play]

PERIOD DEVELOPMENTS
┌──────────────┬─────────────────────────────────────────────┐
│ Date         │ Development                                 │
├──────────────┼─────────────────────────────────────────────┤
│ [YYYY-MM-DD] │ [What happened and significance]            │
│ [YYYY-MM-DD] │ [What happened and significance]            │
│ [YYYY-MM-DD] │ [What happened and significance]            │
└──────────────┴─────────────────────────────────────────────┘

KEY ACTORS UPDATE
• [Actor 1]: [Current posture and recent moves]
• [Actor 2]: [Current posture and recent moves]
• [Actor 3]: [Current posture and recent moves]

CHANGE FROM LAST REPORT
↑ ESCALATED: [What got worse/more urgent]
↔ STABLE: [What stayed the same]
↓ DE-ESCALATED: [What improved]

RESOURCE FLOWS (This Period)
[Money, personnel, materiel movements observed]

OPEN QUESTIONS
• [Unresolved question 1]
• [Unresolved question 2]

NEAR-TERM OUTLOOK (Next reporting period)
[What we expect to see and why]

ANALYST ASSESSMENT
[Interpretive judgment on trajectory]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

**Engines Best Suited**:
- `event_timeline_causal` - Event sequencing
- `stakeholder_power_interest` - Actor tracking
- `resource_flow_asymmetry` - Resource movement
- `concept_evolution` - How positions are shifting
- `chronology_simultaneity` - Parallel developments

**What Visual Shows vs. What SITREP Adds**:
| Visual | SITREP |
|--------|--------|
| Timeline of events | Causal connections between events |
| Actor network | What each actor did THIS period |
| Resource flow Sankey | Commentary on flow significance |

---

### 6. ACTOR PROFILE
**CIA Equivalent**: Biographical Intelligence
**Length**: 3-5 pages per actor
**Reading Time**: 15-20 minutes
**Purpose**: Deep understanding of key decision-maker or entity

**Structure**:
```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ACTOR PROFILE: [Name/Entity]
Type: [Individual / Organization / State Actor / Other]
Date: [YYYY-MM-DD]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

SUMMARY ASSESSMENT
[2-3 sentences: Who they are and why they matter]

BASIC DATA
• Role/Position: [Current role]
• Affiliation: [Organization, network, faction]
• Location: [Where they operate]
• Influence Level: [High/Medium/Low] in [domain]

INTERESTS & MOTIVATIONS
Primary Interests:
  1. [Interest 1] - [Evidence for this]
  2. [Interest 2] - [Evidence for this]
  3. [Interest 3] - [Evidence for this]

Underlying Motivations:
  • [What drives them at a deeper level]

Red Lines:
  • [What they won't accept]

DECISION-MAKING PATTERN
• Worldview: [How they see the situation]
• Risk Tolerance: [Risk-averse / Moderate / Risk-seeking]
• Time Horizon: [Short-term / Long-term thinker]
• Information Sources: [Who they listen to]
• Decision Style: [Deliberative / Intuitive / Consensus-driven]

NETWORK POSITION
Key Relationships:
  • [Ally 1]: [Nature of relationship]
  • [Ally 2]: [Nature of relationship]
  • [Rival 1]: [Nature of conflict]

Structural Position:
  • [Broker / Hub / Peripheral / Bridging]
  • Power sources: [What gives them influence]
  • Vulnerabilities: [What could weaken them]

TRACK RECORD
Past Behavior in Similar Situations:
  • [Situation 1]: [How they responded]
  • [Situation 2]: [How they responded]
Pattern: [What this suggests about future behavior]

STATEMENTS & POSITIONS
Key Quotes:
  • "[Direct quote]" - [Context, date]
  • "[Direct quote]" - [Context, date]
Stated Positions: [What they claim to want]
Actions vs. Words: [Where actions diverge from rhetoric]

PREDICTION
Most Likely Behavior:
  [What they will probably do and why]
Alternative Scenarios:
  • If [condition], they might [behavior]
  • If [condition], they might [behavior]

ENGAGEMENT RECOMMENDATIONS
• [How to approach / influence / work with this actor]
• [What NOT to do]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

**Engines Best Suited**:
- `rational_actor_modeling` - Decision-maker profiling
- `stakeholder_power_interest` - Power/interest mapping
- `quote_attribution_voice` - What they've said
- `relational_topology` - Network position
- `value_ethical_framework` - Values and beliefs

**What Visual Shows vs. What Actor Profile Adds**:
| Visual | Actor Profile |
|--------|--------------|
| Network position in graph | WHY they're positioned there |
| Power/interest quadrant | HOW to engage them |
| Quote network | CONTEXT for what they've said |

---

### 7. VULNERABILITY ASSESSMENT
**CIA Equivalent**: Red Team / Vulnerability Analysis
**Length**: 3-5 pages
**Reading Time**: 15-20 minutes
**Purpose**: Identify weaknesses, gaps, and attack surfaces

**Structure**:
```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
VULNERABILITY ASSESSMENT: [Subject]
Assessment Type: [Argument / Position / Organization / Strategy]
Date: [YYYY-MM-DD]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

EXECUTIVE SUMMARY
[2-3 sentences: Key vulnerabilities identified]

VULNERABILITY INVENTORY
┌────────────────┬────────────┬────────────────┬────────────┐
│ Vulnerability  │ Severity   │ Exploitability │ Impact     │
├────────────────┼────────────┼────────────────┼────────────┤
│ [Vuln 1]       │ [H/M/L]    │ [H/M/L]        │ [H/M/L]    │
│ [Vuln 2]       │ [H/M/L]    │ [H/M/L]        │ [H/M/L]    │
│ [Vuln 3]       │ [H/M/L]    │ [H/M/L]        │ [H/M/L]    │
└────────────────┴────────────┴────────────────┴────────────┘

DETAILED VULNERABILITY ANALYSIS

────────────────────────────────────────────────────────────
VULNERABILITY 1: [Name]
Severity: [HIGH/MEDIUM/LOW]
────────────────────────────────────────────────────────────
Description: [What the vulnerability is]
Location: [Where in argument/structure it exists]
Why It Matters: [Consequences if exploited]
Evidence: [How we identified this]
Attack Vector: [How an adversary would exploit it]
Mitigation: [How to address this vulnerability]

────────────────────────────────────────────────────────────
VULNERABILITY 2: [Name]
────────────────────────────────────────────────────────────
...

HIDDEN ASSUMPTIONS
[Assumptions that, if challenged, would undermine the position]
  1. [Assumption 1] - Vulnerability if false: [...]
  2. [Assumption 2] - Vulnerability if false: [...]

LOGICAL GAPS
[Where the reasoning has holes]
  1. [Gap 1]: Between [premise] and [conclusion]
  2. [Gap 2]: Between [premise] and [conclusion]

EVIDENCE WEAKNESSES
[Where the evidence is thin or questionable]
  • [Weakness 1]: [What's missing or unreliable]
  • [Weakness 2]: [What's missing or unreliable]

STEELMAN COUNTERARGUMENTS
[Strongest attacks an adversary could mount]
  1. [Counterargument 1] - How to respond: [...]
  2. [Counterargument 2] - How to respond: [...]

MITIGATION ROADMAP
Priority Order:
  1. [Address vulnerability X] - [Why first]
  2. [Address vulnerability Y] - [Why second]
  3. [Address vulnerability Z] - [Why third]

RESIDUAL RISK
[Vulnerabilities that cannot be fully mitigated]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

**Engines Best Suited**:
- `assumption_excavation` - Hidden assumptions
- `steelman_stress_test` - Adversarial testing
- `boom_crutch_finder` - Logical gaps
- `opportunity_vulnerability_matrix` - Systematic vulnerability ID
- `terra_incognita_mapper` - Knowledge gaps
- `occams_broom` - Inconvenient facts

**What Visual Shows vs. What Vulnerability Assessment Adds**:
| Visual | Vulnerability Assessment |
|--------|-------------------------|
| Argument structure map | WHERE the weak points are |
| Evidence quality radar | HOW weak each point is |
| Gap visualization | WHAT to do about each gap |

---

### 8. DECISION MEMO
**CIA Equivalent**: Policy Options Paper
**Length**: 2-4 pages
**Reading Time**: 10-15 minutes
**Purpose**: Frame decision with options, trade-offs, and recommendations

**Structure**:
```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DECISION MEMO: [Decision Required]
Decision Deadline: [Date or "As Soon As Practical"]
Date: [YYYY-MM-DD]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

DECISION REQUIRED
[One sentence: What decision needs to be made]

BACKGROUND
[2-3 sentences: Context necessary to understand the decision]

OPTIONS

────────────────────────────────────────────────────────────
OPTION A: [Name]
────────────────────────────────────────────────────────────
Description: [What this option involves]

Advantages:
  ✓ [Pro 1]
  ✓ [Pro 2]
  ✓ [Pro 3]

Disadvantages:
  ✗ [Con 1]
  ✗ [Con 2]

Risks:
  ⚠ [Risk 1]: Likelihood [H/M/L], Impact [H/M/L]
  ⚠ [Risk 2]: Likelihood [H/M/L], Impact [H/M/L]

Resource Requirements:
  [Time, money, personnel, political capital]

────────────────────────────────────────────────────────────
OPTION B: [Name]
────────────────────────────────────────────────────────────
...

────────────────────────────────────────────────────────────
OPTION C: [Name] (if applicable)
────────────────────────────────────────────────────────────
...

COMPARISON MATRIX
┌─────────────────┬────────────┬────────────┬────────────┐
│ Criterion       │ Option A   │ Option B   │ Option C   │
├─────────────────┼────────────┼────────────┼────────────┤
│ Effectiveness   │ [H/M/L]    │ [H/M/L]    │ [H/M/L]    │
│ Cost            │ [H/M/L]    │ [H/M/L]    │ [H/M/L]    │
│ Risk            │ [H/M/L]    │ [H/M/L]    │ [H/M/L]    │
│ Reversibility   │ [H/M/L]    │ [H/M/L]    │ [H/M/L]    │
│ Timeline        │ [H/M/L]    │ [H/M/L]    │ [H/M/L]    │
└─────────────────┴────────────┴────────────┴────────────┘

RECOMMENDATION
[Clear statement of which option to choose and why]

IMPLEMENTATION (If Recommendation Accepted)
• Immediate: [Action within 24-48 hours]
• Short-term: [Action within 1-2 weeks]
• Medium-term: [Action within 1-3 months]

DECISION TRIGGERS
[What would cause us to revisit this decision]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

**Engines Best Suited**:
- `comparative_framework` - Options comparison
- `hypothesis_tournament` - Evaluating alternatives
- `possibility_space_explorer` - Scenario exploration
- `steelman_generator` - Best case for each option
- `opportunity_vulnerability_matrix` - Risk assessment

**What Visual Shows vs. What Decision Memo Adds**:
| Visual | Decision Memo |
|--------|--------------|
| Comparison quadrant | WHICH option to choose |
| Scenario tree | WHAT to do and WHEN |
| Trade-off matrix | WHY this recommendation |

---

## Engine-to-Output Type Mapping

### Matrix: Which Outputs for Which Engines

| Engine Category | Flash | Intel Assess | Evidence | Warning | SITREP | Actor | Vuln | Decision |
|-----------------|-------|--------------|----------|---------|--------|-------|------|----------|
| **Signal/Warning** | ★★★ | ★☆☆ | ☆☆☆ | ★★★ | ★★☆ | ☆☆☆ | ☆☆☆ | ★☆☆ |
| **Hypothesis/Evidence** | ☆☆☆ | ★★★ | ★★★ | ☆☆☆ | ★☆☆ | ☆☆☆ | ★★☆ | ★★☆ |
| **Actor/Network** | ★☆☆ | ★★☆ | ★☆☆ | ☆☆☆ | ★★★ | ★★★ | ★☆☆ | ★☆☆ |
| **Temporal/Evolution** | ★★☆ | ★★☆ | ★★☆ | ★★☆ | ★★★ | ★☆☆ | ☆☆☆ | ★☆☆ |
| **Argument/Logic** | ☆☆☆ | ★★★ | ★★☆ | ☆☆☆ | ☆☆☆ | ☆☆☆ | ★★★ | ★★★ |
| **Comparison** | ★☆☆ | ★★☆ | ★☆☆ | ☆☆☆ | ★☆☆ | ★☆☆ | ★★☆ | ★★★ |
| **Deception/Credibility** | ★★★ | ★★★ | ★★★ | ★★☆ | ★☆☆ | ★★☆ | ★★★ | ★☆☆ |

★★★ = Ideal fit | ★★☆ = Good fit | ★☆☆ = Possible fit | ☆☆☆ = Not recommended

---

## Detailed Engine → Output Mapping

### Group 1: Detection & Warning Engines → Flash Assessment + Warning Bulletin

| Engine | Primary Output | Secondary Output |
|--------|---------------|------------------|
| `signal_sentinel` | Warning Bulletin | Flash Assessment |
| `anomaly_detector` | Warning Bulletin | Flash Assessment |
| `temporal_discontinuity_finder` | Warning Bulletin | SITREP |
| `emerging_trend_detector` | Flash Assessment | Warning Bulletin |
| `escalation_trajectory_analysis` | Warning Bulletin | Decision Memo |
| `surely_alarm` | Vulnerability Assessment | Flash Assessment |

### Group 2: Evidence & Verification Engines → Intelligence Assessment + Evidence Dossier

| Engine | Primary Output | Secondary Output |
|--------|---------------|------------------|
| `hypothesis_tournament` | Intelligence Assessment | Decision Memo |
| `evidence_quality_assessment` | Intelligence Assessment | Evidence Dossier |
| `epistemic_calibration` | Intelligence Assessment | Evidence Dossier |
| `provenance_audit` | Evidence Dossier | Intelligence Assessment |
| `statistical_evidence` | Evidence Dossier | Intelligence Assessment |
| `authenticity_forensics` | Intelligence Assessment | Warning Bulletin |

### Group 3: Actor & Network Engines → Actor Profile + SITREP

| Engine | Primary Output | Secondary Output |
|--------|---------------|------------------|
| `stakeholder_power_interest` | Actor Profile | SITREP |
| `rational_actor_modeling` | Actor Profile | Decision Memo |
| `relational_topology` | Actor Profile | Intelligence Assessment |
| `quote_attribution_voice` | Actor Profile | Evidence Dossier |
| `resource_flow_asymmetry` | SITREP | Actor Profile |
| `deal_flow_tracker` | SITREP | Flash Assessment |

### Group 4: Temporal & Evolution Engines → SITREP + Intelligence Assessment

| Engine | Primary Output | Secondary Output |
|--------|---------------|------------------|
| `event_timeline_causal` | SITREP | Intelligence Assessment |
| `concept_evolution` | Intelligence Assessment | SITREP |
| `chronology_cycle` | Intelligence Assessment | Warning Bulletin |
| `chronology_simultaneity` | SITREP | Intelligence Assessment |
| `reception_history` | Intelligence Assessment | SITREP |
| `temporal_multiscale` | Intelligence Assessment | SITREP |

### Group 5: Argument & Logic Engines → Vulnerability Assessment + Decision Memo

| Engine | Primary Output | Secondary Output |
|--------|---------------|------------------|
| `argument_architecture` | Vulnerability Assessment | Intelligence Assessment |
| `assumption_excavation` | Vulnerability Assessment | Intelligence Assessment |
| `steelman_generator` | Decision Memo | Vulnerability Assessment |
| `steelman_stress_test` | Vulnerability Assessment | Decision Memo |
| `boom_crutch_finder` | Vulnerability Assessment | Intelligence Assessment |
| `dialectical_structure` | Intelligence Assessment | Decision Memo |

### Group 6: Comparison & Framework Engines → Decision Memo + Intelligence Assessment

| Engine | Primary Output | Secondary Output |
|--------|---------------|------------------|
| `comparative_framework` | Decision Memo | Intelligence Assessment |
| `possibility_space_explorer` | Decision Memo | Intelligence Assessment |
| `competitive_landscape` | Decision Memo | SITREP |
| `opportunity_vulnerability_matrix` | Vulnerability Assessment | Decision Memo |

### Group 7: Deception & Credibility Engines → Intelligence Assessment + Vulnerability Assessment

| Engine | Primary Output | Secondary Output |
|--------|---------------|------------------|
| `authenticity_forensics` | Intelligence Assessment | Warning Bulletin |
| `occams_broom` | Vulnerability Assessment | Intelligence Assessment |
| `deepity_detector` | Vulnerability Assessment | Intelligence Assessment |
| `influence_attribution_analysis` | Intelligence Assessment | Warning Bulletin |
| `terra_incognita_mapper` | Intelligence Assessment | Vulnerability Assessment |

### Group 8: Thematic & Conceptual Engines → Intelligence Assessment

| Engine | Primary Output | Secondary Output |
|--------|---------------|------------------|
| `thematic_synthesis` | Intelligence Assessment | SITREP |
| `conceptual_framework_extraction` | Intelligence Assessment | Evidence Dossier |
| `structural_pattern_detector` | Intelligence Assessment | Vulnerability Assessment |
| `interdisciplinary_connection` | Intelligence Assessment | Evidence Dossier |
| `jootsing_analyzer` | Vulnerability Assessment | Intelligence Assessment |

---

## Visual-Text Complementarity Rules

### Rule 1: Never Describe What the Visual Shows
❌ "The network graph shows that Actor A is connected to Actor B..."
✓ "Actor A's connection to Actor B enables them to bypass normal channels because..."

### Rule 2: Text Provides the "So What"
❌ [Visual shows timeline] + [Text lists events in order]
✓ [Visual shows timeline] + [Text explains causal chains and turning points]

### Rule 3: Text Handles Uncertainty
Visual: Clean, definitive-looking structure
Text: "We assess with MODERATE confidence... Alternative interpretations include..."

### Rule 4: Text Provides Actionability
Visual: Shows current state
Text: "Given this situation, recommend: 1) ... 2) ... 3) ..."

### Rule 5: Cross-Reference, Don't Duplicate
Text should say: "See accompanying network visualization for full relationship map. Key findings from that analysis:"
NOT: [Repeat everything the visual shows]

---

## Quality Rubric for Each Output Type

### Flash Assessment Quality Checklist
- [ ] Can be read in under 2 minutes
- [ ] BLUF is exactly one sentence
- [ ] No more than 3 implications listed
- [ ] Confidence level is stated with reasoning
- [ ] "Watch for" item is specific and measurable
- [ ] Would change someone's day if they read it

### Intelligence Assessment Quality Checklist
- [ ] Key judgments use calibrated confidence language
- [ ] Each judgment has evidence for AND against
- [ ] Competing hypotheses are genuinely competing (not strawmen)
- [ ] "What we don't know" section is substantive
- [ ] Source reliability is documented
- [ ] Alternative views are presented fairly

### Evidence Dossier Quality Checklist
- [ ] Every claim has a source citation
- [ ] Source reliability is rated
- [ ] Direct quotes are verbatim with page numbers
- [ ] Contradictions are documented
- [ ] Provenance chains are traceable
- [ ] Gaps register shows what's missing

### Warning Bulletin Quality Checklist
- [ ] Priority level is justified
- [ ] Indicators are specific and measurable
- [ ] Threshold deviation is quantified
- [ ] Timeline to decision is realistic
- [ ] Watch list is actionable for next 48-72 hours
- [ ] False alarm possibility is acknowledged

### SITREP Quality Checklist
- [ ] Developments are in chronological order
- [ ] Actor updates are current
- [ ] Change from last report is clear
- [ ] Open questions are specific
- [ ] Near-term outlook is falsifiable
- [ ] Assessment adds interpretive value beyond facts

### Actor Profile Quality Checklist
- [ ] Interests are supported by evidence
- [ ] Decision-making pattern is based on track record
- [ ] Network position is structurally analyzed
- [ ] Predictions are specific enough to be wrong
- [ ] Engagement recommendations are practical
- [ ] Red lines are clearly stated

### Vulnerability Assessment Quality Checklist
- [ ] Vulnerabilities are ranked by severity/exploitability/impact
- [ ] Attack vectors are realistic
- [ ] Mitigations are actionable
- [ ] Hidden assumptions are non-obvious
- [ ] Steelman counterarguments are genuinely strong
- [ ] Residual risk is acknowledged

### Decision Memo Quality Checklist
- [ ] Decision required is crystal clear
- [ ] Options are genuinely different (not strawman vs. real option)
- [ ] Pros/cons are balanced
- [ ] Risks use likelihood/impact framework
- [ ] Recommendation is clear and justified
- [ ] Implementation steps are actionable

---

## Implementation Notes

### Template System
Each output type needs:
1. **Structural template** (the markdown skeleton above)
2. **Generation prompt** (instructions for LLM to fill template)
3. **Quality check prompt** (LLM self-evaluation against rubric)
4. **Complementarity check** (what visual shows, what text should add)

### Engine Metadata Extension
Each engine should declare:
```yaml
engine: signal_sentinel
...
preferred_text_outputs:
  primary: warning_bulletin
  secondary: flash_assessment
  avoid: evidence_dossier  # not a good fit
text_complement_focus:
  - "what_to_do"
  - "timeline_to_decision"
  - "watch_indicators"
```

### Multi-Output Orchestration
When user requests visual + text:
1. Generate visual first
2. Analyze visual output for key elements shown
3. Generate text with explicit instruction to COMPLEMENT not DUPLICATE
4. Include "See accompanying visual for [X]" cross-references
5. Quality check for complementarity

---

## Summary: The 8 Output Types

| # | Output Type | Length | Purpose | Key Question Answered |
|---|-------------|--------|---------|----------------------|
| 1 | Flash Assessment | 1 page | Immediate awareness | "What do I need to know RIGHT NOW?" |
| 2 | Intelligence Assessment | 5-15 pages | Comprehensive synthesis | "What's the full picture and how confident are we?" |
| 3 | Evidence Dossier | 10-30 pages | Evidence chain | "What's the evidence and how reliable is it?" |
| 4 | Warning Bulletin | 1-2 pages | Early warning | "What signals indicate change is coming?" |
| 5 | SITREP | 2-3 pages | Status update | "What happened and where are we now?" |
| 6 | Actor Profile | 3-5 pages | Decision-maker analysis | "Who is this actor and how will they behave?" |
| 7 | Vulnerability Assessment | 3-5 pages | Weakness identification | "Where are the gaps and how can they be exploited?" |
| 8 | Decision Memo | 2-4 pages | Decision support | "What are my options and which should I choose?" |

Each serves a different consumer with different needs. None is a "better" or "worse" version of another - they are fundamentally different products for different purposes.
