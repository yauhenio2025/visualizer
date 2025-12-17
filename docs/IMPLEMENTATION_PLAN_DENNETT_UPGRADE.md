# The Visualizer 2.0: A Dennettian Document Intelligence Platform
## Strategic Implementation Plan

**Prepared:** 17 December 2025
**Philosophy:** Daniel Dennett's Intuition Pumps as Analytical Framework
**Paradigm:** From "Intelligence Analysis" to "Critical Inquiry & Epistemic Cartography"

---

## Part I: Philosophical Foundation — Dennett's Intuition Pumps

### What Are Intuition Pumps?

Daniel Dennett describes intuition pumps as "imagination extenders and focus-holders" — thought experiments and thinking tools that help us probe concepts, test assumptions, and generate insights. They are not proofs but provocations that reveal hidden structure in our thinking.

**Core Dennett Tools We Will Integrate:**

| # | Intuition Pump | Description | Analytical Application |
|---|----------------|-------------|------------------------|
| 1 | **The "Surely" Operator** | When someone says "surely" or "obviously," that's where the argument is weakest | Flag rhetorical confidence markers that mask weak reasoning |
| 2 | **Occam's Broom** | What's being swept under the rug? The opposite of Occam's Razor | Identify what's conspicuously NOT mentioned |
| 3 | **Jootsing** | "Jumping Out Of The System" — understand rules before creatively violating them | Identify implicit rules/constraints, then explore violations |
| 4 | **Rathering** | The "would you rather" frame that exposes hidden assumptions | Surface forced choices and false dilemmas |
| 5 | **Deepity** | Statements that seem profound but are trivially true OR importantly false | Detect pseudo-profundity |
| 6 | **Boom Crutch** | Explanations with a magic step ("then a miracle occurs") | Identify explanatory gaps in causal chains |
| 7 | **Philosophers' Syndrome** | Mistaking failure of imagination for insight into necessity | Distinguish "I can't imagine how" from "It's impossible" |
| 8 | **Sturgeon's Law** | 90% of everything is crap | Triage for quality, don't treat all claims equally |
| 9 | **Rapoport's Rules** | How to compose a successful critical commentary | Steelman before attacking |
| 10 | **The Sortes Paradox** | Where do you draw the line? | Probe boundary cases in concepts |
| 11 | **The Curse of Knowledge** | Experts forget what it's like not to know | Identify jargon-as-argument |
| 12 | **The Intentional Stance** | Predict behavior by assuming rationality | Model stakeholder reasoning |
| 13 | **Heterophenomenology** | Take reports at face value as data | Separate claims from truth of claims |
| 14 | **The Two-Bitser** | Context determines meaning/function | Note how meaning shifts by context |
| 15 | **Cartesian Gravity** | The temptation to locate "where it all comes together" | Resist central-controller fallacies |

---

## Part II: The Great Renaming — From Spycraft to Scholarship

### Naming Philosophy

We replace intelligence/military framing with:
- **Epistemic cartography** (mapping knowledge landscapes)
- **Critical inquiry** (rigorous questioning)
- **Cognitive archaeology** (excavating hidden assumptions)
- **Discourse anatomy** (dissecting argument structures)

### Engine Renaming Table

| Old Name (CIA-speak) | New Name (Academic) | Key Dennett Pump |
|---------------------|---------------------|------------------|
| `source_credibility_assessment` | `provenance_audit` | Heterophenomenology |
| `analytic_confidence_levels` | `epistemic_calibration` | Philosophers' Syndrome |
| `competing_hypotheses_analysis` | `hypothesis_tournament` | Rapoport's Rules |
| `deception_indicator_detection` | `authenticity_forensics` | Deepity Detector |
| `information_gaps_analysis` | `terra_incognita_mapper` | Occam's Broom |
| `indicators_warnings_tracker` | `signal_sentinel` | Sturgeon's Law |
| `scenario_futures_matrix` | `possibility_space_explorer` | Philosophers' Syndrome |
| `network_centrality_analysis` | `relational_topology` | Cartesian Gravity |
| `decision_maker_profiling` | `rational_actor_modeling` | Intentional Stance |
| `timeline_anomaly_detection` | `temporal_discontinuity_finder` | Boom Crutch |
| `red_team_challenge` | `steelman_stress_test` | Rapoport's Rules |
| `intelligence_tradecraft` (bundle) | `epistemic_rigor_suite` | Multiple |
| `strategic_warning` (bundle) | `horizon_scanning_suite` | Jootsing |
| `influence_forensics` (bundle) | `persuasion_archaeology` | "Surely" Operator |
| `collection_requirements` | `inquiry_agenda` | Occam's Broom |
| `confidence_calibration` | `certainty_gradient` | Philosophers' Syndrome |
| `bias_detection` | `cognitive_audit` | Curse of Knowledge |
| `briefing_distillation` | `insight_crystallization` | Sturgeon's Law |

### Bundle Renaming

| Old Bundle | New Bundle | Description |
|------------|------------|-------------|
| `intelligence_tradecraft` | `epistemic_rigor_suite` | Core analytical discipline |
| `strategic_warning` | `horizon_scanning_suite` | Future-oriented pattern detection |
| `influence_forensics` | `persuasion_archaeology` | Rhetoric and manipulation analysis |
| `network_intelligence` | `relational_cartography` | Actor network mapping |
| `decision_support` | `inquiry_synthesis` | Policy-ready analytical output |

### Pipeline Renaming

| Old Pipeline | New Pipeline | Stages |
|--------------|--------------|--------|
| `source_to_confidence` | `provenance_to_certainty` | provenance_audit → epistemic_calibration → terra_incognita_mapper |
| `warning_assessment_complete` | `horizon_scan_complete` | signal_sentinel → temporal_discontinuity_finder → possibility_space_explorer |
| `analytic_rigor_pipeline` | `epistemic_stress_test` | argument_architecture → hypothesis_tournament → steelman_stress_test → epistemic_calibration |

---

## Part III: New Engines — Dennett-Infused

### Engine 1: `surely_alarm` — The Confidence Marker Detector

**Philosophy:** Dennett's "Surely" Operator — when an author says "surely," "obviously," "clearly," "of course," "it goes without saying," these are often where arguments are weakest.

```python
class SurelyAlarmEngine(BaseEngine):
    engine_key = "surely_alarm"
    engine_name = "The 'Surely' Alarm"
    description = (
        "Detects rhetorical confidence markers ('surely', 'obviously', 'clearly', "
        "'of course') that often mask weak reasoning or undefended assumptions. "
        "Based on Dennett's observation that these words are epistemological alarm bells."
    )
    category = EngineCategory.RHETORIC
    researcher_question = "Where does the text assert confidence without earning it?"
```

**Extraction Prompt:**
```
You are applying Dennett's "Surely" Operator — identifying where rhetorical
confidence substitutes for argumentative rigor.

Scan for these CONFIDENCE MARKERS (and synonyms):
- "surely" / "certainly" / "undoubtedly"
- "obviously" / "clearly" / "evidently"
- "of course" / "naturally" / "it goes without saying"
- "everyone knows" / "common sense dictates"
- "any reasonable person would agree"
- "it stands to reason"

For each marker found:
1. QUOTE: The sentence containing the marker
2. CLAIM: What's being asserted as obvious?
3. DEFENSE: Is this claim actually defended elsewhere? Or merely asserted?
4. VULNERABILITY: If someone challenged this, how would the argument hold up?
5. FUNCTION: Is the marker:
   - COVERING a weak point (masking)
   - BUILDING consensus (rhetorical)
   - ABBREVIATING a complex argument (legitimate shorthand)
   - BULLYING disagreement (intimidation)

Rate vulnerability: 0.0 (well-defended elsewhere) to 1.0 (pure assertion).

The most important findings are markers with HIGH vulnerability scores.
```

**Canonical Schema:**
```json
{
  "confidence_markers": [
    {
      "marker_id": "string",
      "marker_text": "string",
      "full_quote": "string",
      "implicit_claim": "string",
      "defense_status": "defended_elsewhere | partially_defended | undefended | assertion_only",
      "vulnerability_score": "number (0-1)",
      "marker_function": "masking | rhetorical | shorthand | intimidation",
      "challenge_vector": "string - how this could be questioned",
      "source_article": "string"
    }
  ],
  "patterns": {
    "most_vulnerable_claims": ["string"],
    "marker_frequency_by_author": {"author": "count"},
    "marker_frequency_by_function": {"function": "count"}
  },
  "meta": {
    "total_markers": "number",
    "average_vulnerability": "number",
    "most_common_marker_type": "string"
  }
}
```

---

### Engine 2: `occams_broom` — The Strategic Silence Detector

**Philosophy:** Dennett's "Occam's Broom" — the phenomenon of sweeping inconvenient facts under the rug. Unlike Occam's Razor (don't multiply entities unnecessarily), Occam's Broom hides entities that complicate the story.

```python
class OccamsBroomEngine(BaseEngine):
    engine_key = "occams_broom"
    engine_name = "Occam's Broom Detector"
    description = (
        "Identifies what's being swept under the rug — inconvenient facts, "
        "counterexamples, or complications that are suspiciously absent. "
        "Detects strategic omissions that simplify the narrative at the cost of accuracy."
    )
    category = EngineCategory.EPISTEMOLOGY
    researcher_question = "What inconvenient facts are being swept under the rug?"
```

**Extraction Prompt:**
```
You are applying Dennett's "Occam's Broom" — detecting what's conspicuously absent.

Unlike Occam's Razor (parsimony), Occam's Broom is about HIDING complexity.

Look for these SWEEP PATTERNS:

1. **MISSING COUNTEREXAMPLES**
   - The argument makes broad claims
   - What obvious exceptions are not mentioned?
   - What famous cases would complicate this story?

2. **ABSENT STAKEHOLDERS**
   - Who is affected but not consulted?
   - Whose perspective would complicate the narrative?
   - Who benefits from this absence?

3. **INCONVENIENT DATA**
   - What statistics/studies would challenge this?
   - What time periods are conveniently excluded?
   - What geographic scope is suspiciously narrow?

4. **UNASKED QUESTIONS**
   - What follow-up questions are not posed?
   - What "and then what happens?" is avoided?
   - What long-term consequences are ignored?

5. **DEFINITIONAL GERRYMANDERING**
   - Are terms defined to exclude problematic cases?
   - Does the categorization hide what it should reveal?

For each sweep:
1. WHAT'S MISSING: Describe the absent element
2. WHY RELEVANT: Why should it be present?
3. WHOSE INTEREST: Who benefits from this omission?
4. IMPACT: How would including it change the argument?
5. SWEEP CONFIDENCE: How certain are you this is intentional vs. oversight?
```

---

### Engine 3: `deepity_detector` — Pseudo-Profundity Scanner

**Philosophy:** A "deepity" (Dennett's term) is a statement that seems profound but is either trivially true or, if taken as profound, false.

**Example deepities:**
- "Love is just a word" (trivial: yes, it's a word; profound: false, it's much more)
- "Beauty is only skin deep" (trivial; but implies beauty is shallow, which is a separate claim)
- "Everything happens for a reason" (trivial: causality; profound: cosmic purpose — very different claims)

```python
class DeepityDetectorEngine(BaseEngine):
    engine_key = "deepity_detector"
    engine_name = "Deepity Detector"
    description = (
        "Identifies statements that seem profound but are either trivially true "
        "or importantly false. Reveals pseudo-profundity masquerading as insight — "
        "a key vector for intellectual confusion."
    )
    category = EngineCategory.RHETORIC
    researcher_question = "What statements trade on ambiguity between trivial and profound?"
```

**Extraction Prompt:**
```
You are detecting DEEPITIES — statements that seem profound but collapse on inspection.

A deepity has TWO readings:
1. TRIVIAL READING: True but uninteresting
2. PROFOUND READING: Interesting but false (or at least undefended)

The power of a deepity comes from EQUIVOCATING between these readings.

Scan for statements that:
- Sound wise or philosophical
- Use metaphor or aphorism form
- Seem to "explain" complex phenomena simply
- Provoke head-nodding without critical examination

For each candidate:

1. QUOTE the statement
2. TRIVIAL READING: How is this obviously true? (usually tautological)
3. PROFOUND READING: What bold claim is being implied?
4. EQUIVOCATION: Does the text slide between readings?
5. DEEPITY SCORE:
   - 0.0 = Genuinely insightful (both readings valid and defended)
   - 0.5 = Ambiguous but probably meaningful
   - 1.0 = Pure deepity (gains power only from confusion)

EXAMPLES:
- "Technology is neither good nor bad"
  - Trivial: True, it's just tools
  - Profound: Implies we shouldn't regulate or judge — contested claim

- "The market knows best"
  - Trivial: Markets aggregate information
  - Profound: Markets always produce optimal outcomes — empirically false
```

---

### Engine 4: `boom_crutch_finder` — Explanatory Gap Detector

**Philosophy:** A "boom crutch" (Dennett) is a step in an explanation where something magical happens — "and then a miracle occurs" — often disguised by vague verbs or nominalizations.

```python
class BoomCrutchFinderEngine(BaseEngine):
    engine_key = "boom_crutch_finder"
    engine_name = "Boom Crutch Finder"
    description = (
        "Identifies explanatory gaps where causal chains invoke magic steps — "
        "'then understanding emerges', 'the market adjusts', 'society transforms'. "
        "Reveals where hand-waving substitutes for mechanism."
    )
    category = EngineCategory.ARGUMENT
    researcher_question = "Where do explanations hide a 'then magic happens' step?"
```

**Extraction Prompt:**
```
You are hunting for BOOM CRUTCHES — the "then a miracle occurs" steps in explanations.

A boom crutch is:
- A step that seems to explain but actually SKIPS the hard part
- Often hidden by nominalizations ("transformation occurs" instead of "X does Y")
- Frequently disguised by passive voice ("understanding is achieved")
- Particularly common in discussions of consciousness, markets, society, institutions

BOOM CRUTCH INDICATORS:

1. **NOMINALIZATIONS**
   - "Learning happens" (WHO learns WHAT and HOW?)
   - "Innovation emerges" (from WHERE through WHAT process?)
   - "Value is created" (by WHOM through WHAT mechanism?)

2. **PASSIVE VOICE MAGIC**
   - "The problem is solved"
   - "Consensus is reached"
   - "The transition is made"

3. **VAGUE AGENTS**
   - "The market adjusts" (which agents, what decisions?)
   - "Society evolves" (what people doing what?)
   - "Technology disrupts" (which actors using what?)

4. **COMPRESSED CAUSATION**
   - "Education leads to prosperity" (what's the mechanism?)
   - "Trust enables cooperation" (how exactly?)
   - "Data drives decisions" (through what process?)

For each boom crutch:
1. QUOTE: The sentence with the gap
2. HIDDEN QUESTION: What's actually being skipped?
3. MECHANISTIC DEMAND: What mechanism would fill this gap?
4. BOOM SCORE: How much weight does the argument place on this gap?
   0.0 = Minor handwave in otherwise detailed explanation
   1.0 = The entire argument rests on this magic step
```

---

### Engine 5: `steelman_generator` — Rapoport's Rules Engine

**Philosophy:** Dennett's version of "Rapoport's Rules" for critical commentary:
1. Attempt to re-express your target's position so clearly they say "Thanks, I wish I'd thought of putting it that way"
2. List points of agreement (especially if not widely shared)
3. Mention anything you learned from your target
4. Only THEN are you permitted to say any critical words

```python
class SteelmanGeneratorEngine(BaseEngine):
    engine_key = "steelman_generator"
    engine_name = "Steelman Generator"
    description = (
        "Generates the strongest possible version of each argument before critique. "
        "Applies Dennett's version of Rapoport's Rules: understand and strengthen "
        "before attacking. Prevents strawmanning and deepens analytical rigor."
    )
    category = EngineCategory.ARGUMENT
    researcher_question = "What is the strongest version of each argument present?"
```

**Extraction Prompt:**
```
You are generating STEELMAN versions of every argument — the strongest possible
formulation, even (especially) for positions you might disagree with.

This is Dennett's application of Rapoport's Rules:
1. Express the position better than its proponent did
2. Find points of agreement
3. Note what you learned
4. Only then, critique

For each significant argument or position:

1. **ORIGINAL FORMULATION**: How does the text state it?

2. **STEELMAN VERSION**: Reformulate to be:
   - Clearer and more precise
   - More defensible (remove obvious weaknesses)
   - More charitable (interpret ambiguities favorably)
   - More powerful (add unstated but available support)

3. **POINTS OF AGREEMENT**: What parts are genuinely insightful?

4. **STRONGEST EVIDENCE**: What's the best support for this position?

5. **LEARNING VALUE**: What does engaging this position teach us?

6. **REMAINING VULNERABILITIES**: Only after steelmanning — what weaknesses persist?

7. **STEELMAN DELTA**: How much did you have to add/repair?
   - LOW: Position was already well-stated
   - MEDIUM: Needed clarification but core was solid
   - HIGH: Required significant reconstruction
```

---

### Engine 6: `jootsing_analyzer` — System Boundary Explorer

**Philosophy:** Dennett's "Jootsing" = "Jumping Out Of The System." To be creative, you must first understand the rules, then identify where they can be productively violated.

```python
class JootsingAnalyzerEngine(BaseEngine):
    engine_key = "jootsing_analyzer"
    engine_name = "Jootsing Analyzer"
    description = (
        "Identifies the implicit rules, constraints, and boundaries of a discourse — "
        "then explores what becomes possible when those constraints are violated. "
        "Based on Dennett's 'Jumping Out Of The System' creativity principle."
    )
    category = EngineCategory.CONCEPTS
    researcher_question = "What are the implicit rules here, and what if we broke them?"
```

**Extraction Prompt:**
```
You are performing JOOTSING analysis — identifying rules to productively violate.

JOOTSING = "Jumping Out Of The System"

Creativity requires:
1. Understanding the rules of the system
2. Identifying which rules are TRULY necessary vs. merely conventional
3. Exploring what happens when you break the conventional ones

For this text:

## RULE DISCOVERY

1. **EXPLICIT RULES**: What constraints are stated?
   - Definitions used
   - Boundaries set
   - Exclusions declared

2. **IMPLICIT RULES**: What constraints are assumed?
   - Categories taken for granted
   - Methods considered acceptable
   - Questions considered askable
   - Stakeholders considered relevant

3. **BOUNDARY CONDITIONS**: What's treated as fixed?
   - Time horizons
   - Geographic scope
   - Institutional framework
   - Technology assumptions

## JOOTSING OPPORTUNITIES

For each identified rule:

1. RULE: State the constraint clearly
2. NECESSITY: Is this NECESSARILY true or merely CONVENTIONAL?
3. WHAT IF VIOLATED: What would happen if we dropped this constraint?
4. CREATIVE POTENTIAL: What new possibilities open up?
5. HISTORICAL PRECEDENT: Has this rule been broken before? What happened?

Rate each rule:
- NECESSARY: Violating destroys coherence (keep)
- CONVENTIONAL: Violating enables new thinking (candidate for jootsing)
- ARBITRARY: No good reason it's there (jootsing may reveal why)
```

---

### Engine 7: `philosophers_syndrome_detector` — Imagination vs. Necessity

**Philosophy:** Dennett warns against "mistaking a failure of imagination for an insight into necessity." Just because you can't imagine how X could work doesn't mean X is impossible.

```python
class PhilosophersSyndromeDetector(BaseEngine):
    engine_key = "philosophers_syndrome_detector"
    engine_name = "Imagination Failure Detector"
    description = (
        "Identifies where arguments treat failure of imagination as proof of "
        "impossibility. Detects 'I can't see how X could work, therefore X is "
        "impossible' reasoning — a common but fallacious pattern."
    )
    category = EngineCategory.EPISTEMOLOGY
    researcher_question = "Where is failure of imagination being mistaken for impossibility?"
```

**Extraction Prompt:**
```
You are detecting PHILOSOPHER'S SYNDROME — failure of imagination disguised
as insight into necessity.

This fallacy takes the form:
"I cannot imagine how X could be true, therefore X is impossible/false."

INDICATORS:

1. **IMPOSSIBILITY CLAIMS**
   - "X is impossible" / "X could never work" / "There's no way X"
   - Check: Is this based on logical necessity or failure to imagine?

2. **INCONCEIVABILITY ARGUMENTS**
   - "It's inconceivable that..." / "No one could imagine how..."
   - Red flag: Inconceivability ≠ impossibility

3. **HISTORICAL BLINDNESS**
   - Claims that ignore past failures of imagination
   - "Heavier-than-air flight is impossible" (Lord Kelvin, 1895)
   - Check: What similar claims were wrong before?

4. **COMPLEXITY SURRENDER**
   - "This is too complex to be explained by X"
   - Often masks: "I don't understand how X could explain this"

5. **INTUITION ANCHORING**
   - "Intuitively, X cannot..."
   - Check: Is intuition reliable in this domain?

For each instance:

1. QUOTE: The impossibility/inconceivability claim
2. TYPE: Which indicator pattern?
3. ACTUAL BASIS: What would PROVE this impossible?
4. IMAGINATION TEST: Has something similar been imagined/achieved before?
5. SYNDROME SEVERITY:
   - LOW: Claim has some basis beyond imagination failure
   - MEDIUM: Mainly imagination-based but acknowledges uncertainty
   - HIGH: Pure "I can't imagine it therefore impossible"
```

---

### Engine 8: `boundary_probe` — The Sortes Paradox Engine

**Philosophy:** The Sortes paradox asks where you draw the line (how many grains make a "heap"?). Dennett uses this to probe concept boundaries and reveal hidden arbitrariness.

```python
class BoundaryProbeEngine(BaseEngine):
    engine_key = "boundary_probe"
    engine_name = "Boundary Probe"
    description = (
        "Applies Sortes-style analysis to key concepts — where exactly is the "
        "boundary? Reveals hidden arbitrariness in categories and thresholds "
        "that arguments depend upon."
    )
    category = EngineCategory.CONCEPTS
    researcher_question = "Where are the boundary lines, and are they defensible?"
```

---

### Engine 9: `provenance_audit` — Source Quality Map

**Philosophy:** Uses Dennett's "Heterophenomenology" — take reports at face value AS DATA, then investigate separately whether they're true.

```python
class ProvenanceAuditEngine(BaseEngine):
    engine_key = "provenance_audit"
    engine_name = "Provenance Audit"
    description = (
        "Maps the sources of claims and evaluates their epistemic quality. "
        "Separates the claim-as-data from the claim-as-true. Traces how "
        "information enters the discourse and transforms along the way."
    )
    category = EngineCategory.EVIDENCE
    researcher_question = "Where do these claims come from, and how reliable are those sources?"
```

**Canonical Schema:**
```json
{
  "sources": [
    {
      "source_id": "string",
      "name": "string",
      "type": "primary | secondary | tertiary | unknown",
      "access_type": "direct_observation | interview | document | hearsay | inference",
      "credibility_factors": {
        "expertise_relevant": "number (0-1)",
        "track_record": "number (0-1)",
        "independence": "number (0-1)",
        "corroboration": "number (0-1)"
      },
      "bias_indicators": ["string"],
      "transformation_chain": "string - how info traveled from origin to this text",
      "heterophenomenological_note": "string - what we learn from the fact this claim was made, regardless of truth"
    }
  ],
  "provenance_chains": [
    {
      "claim": "string",
      "original_source": "source_id",
      "transformation_steps": ["string"],
      "fidelity_degradation": "number (0-1) - how much signal loss"
    }
  ],
  "patterns": {
    "echo_chamber_indicators": ["string"],
    "circular_citation": ["string"],
    "single_point_of_failure": ["string"]
  }
}
```

---

### Engine 10: `epistemic_calibration` — Certainty Gradient

**Philosophy:** Replace binary confidence with calibrated uncertainty. Integrate the philosophers' syndrome check — distinguish "we don't know" from "it's unknowable."

```python
class EpistemicCalibrationEngine(BaseEngine):
    engine_key = "epistemic_calibration"
    engine_name = "Epistemic Calibration"
    description = (
        "Assigns calibrated certainty levels to claims. Distinguishes empirical "
        "uncertainty (we don't know yet) from deep uncertainty (we can't know) "
        "from imagination failure (we merely can't imagine how)."
    )
    category = EngineCategory.EPISTEMOLOGY
    researcher_question = "How certain should we actually be about these claims?"
```

**Certainty Spectrum:**
```
CERTAINTY GRADIENT (not binary):

├─ ESTABLISHED FACT (0.95-1.0)
│   Well-replicated, consilience across methods
│
├─ STRONG EVIDENCE (0.80-0.94)
│   Good evidence, some room for doubt
│
├─ MODERATE SUPPORT (0.60-0.79)
│   More likely than not, but significant uncertainty
│
├─ CONTESTED (0.40-0.59)
│   Reasonable people disagree, evidence mixed
│
├─ WEAK SUPPORT (0.20-0.39)
│   Some evidence, but mostly speculative
│
├─ SPECULATIVE (0.05-0.19)
│   Interesting possibility, little evidence
│
└─ UNKNOWN (below 0.05)
    Could go either way, we simply don't know

UNCERTAINTY TYPES:
- EMPIRICAL: We don't have the data yet (potentially resolvable)
- FUNDAMENTAL: May be unknowable in principle
- IMAGINATION FAILURE: "Can't see how" ≠ "Is false"
- MODEL UNCERTAINTY: Depends on which framework you use
```

---

## Part IV: Radical UI Redesign — The Analytical Canvas

### Current Limitations

The current UI is a **linear workflow**: select documents → select engine → run → view results. This is adequate for single analyses but fails for:
- Iterative exploration
- Comparative analysis
- Building arguments across multiple analyses
- Maintaining analytical context over time

### Paradigm Shift: The Analytical Canvas

**Concept:** Transform from a *tool* to a *thinking environment*.

#### 4.1 The Investigation Board

Replace the linear flow with a **spatial canvas** inspired by:
- Detective investigation boards (pins, strings, photos)
- Mind mapping tools (infinite canvas, clusters)
- Research annotation tools (marginalia, highlighting)

```
┌────────────────────────────────────────────────────────────────────────┐
│                    THE ANALYTICAL CANVAS                               │
├────────────────────────────────────────────────────────────────────────┤
│                                                                        │
│   📄──────────────📄                    ┌─────────────────┐            │
│   Doc1              Doc2                │ Stakeholder Map │            │
│     │                 │                 │  [Visualization] │           │
│     └────────┬────────┘                 └────────┬────────┘            │
│              │                                   │                     │
│              ▼                                   │                     │
│        ┌───────────┐                            │                     │
│        │ Theme Map │◄───────────────────────────┘                     │
│        │ [Result]  │                                                  │
│        └─────┬─────┘                                                  │
│              │                                                        │
│         ┌────┴────┐                                                   │
│         ▼         ▼                                                   │
│    📝 Note 1   📝 Note 2                                              │
│    "Key insight..." "Follow up on..."                                 │
│                                                                        │
│   ═══════════════════════════════════════════════════════════════     │
│   [Zoom: 100%] [Pan Mode] [Connect Mode] [Annotate Mode] [Export]     │
└────────────────────────────────────────────────────────────────────────┘
```

**Features:**
- **Nodes:** Documents, analyses, visualizations, notes, questions
- **Edges:** Connections (explicitly drawn or auto-inferred)
- **Clusters:** Groupable regions for organizing related items
- **Infinite canvas:** Pan and zoom, no fixed boundaries
- **Persistent state:** Save and restore canvases

#### 4.2 The Lens System

Instead of running engines one at a time, apply **analytical lenses** as overlays:

```
┌─────────────────────────────────────────────────────┐
│                   DOCUMENT VIEW                      │
│                                                     │
│  This is the document text. The author             │
│  argues that surely[🔴] technology will             │
│  transform education. Innovation                    │
│  emerges[🟡] naturally from competitive             │
│  markets, leading to better outcomes[🔵].           │
│                                                     │
├─────────────────────────────────────────────────────┤
│ ACTIVE LENSES: [✓ Surely Alarm] [✓ Boom Crutch]    │
│                [  Deepity     ] [  Stakeholders ]   │
├─────────────────────────────────────────────────────┤
│ 🔴 "Surely" Markers: 3 found                        │
│ 🟡 Boom Crutches: 1 found ("emerges naturally")     │
│ 🔵 Undefended Claims: 2 found                       │
└─────────────────────────────────────────────────────┘
```

**Lens Types:**
- **Rhetorical lenses:** Surely Alarm, Deepity Detector
- **Structural lenses:** Argument map, Claim-Evidence links
- **Actor lenses:** Stakeholder highlighting, Voice attribution
- **Temporal lenses:** Timeline markers, Change points
- **Quality lenses:** Evidence strength, Confidence levels

#### 4.3 The Hypothesis Workspace

A dedicated workspace for testing competing explanations:

```
┌─────────────────────────────────────────────────────────────────────┐
│                    HYPOTHESIS TOURNAMENT                             │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌─────────────┐   ┌─────────────┐   ┌─────────────┐               │
│  │ Hypothesis 1│   │ Hypothesis 2│   │ Hypothesis 3│               │
│  │             │   │             │   │             │               │
│  │ "Market     │   │ "Regulatory │   │ "Technology │               │
│  │  forces     │   │  capture    │   │  lock-in    │               │
│  │  dominate"  │   │  explains"  │   │  primary"   │               │
│  └──────┬──────┘   └──────┬──────┘   └──────┬──────┘               │
│         │                 │                 │                       │
├─────────┼─────────────────┼─────────────────┼───────────────────────┤
│ Evidence│    H1    │    H2    │    H3    │ Diagnosticity           │
├─────────┼──────────┼──────────┼──────────┼────────────────────────┤
│ E1: ... │    ✓     │    ✓     │    -     │ LOW                     │
│ E2: ... │    ✗     │    ✓     │    -     │ HIGH ◄── Most useful    │
│ E3: ... │    ✓     │    -     │    ✓     │ MEDIUM                  │
│ E4: ... │    -     │    ✗     │    ✓     │ HIGH                    │
├─────────┴──────────┴──────────┴──────────┴────────────────────────┤
│ SCORE:      -1          +1          +1                             │
│ STATUS:  Weakened    Supported   Supported                         │
└─────────────────────────────────────────────────────────────────────┘
│ [Add Hypothesis] [Add Evidence] [What would change this?]          │
└─────────────────────────────────────────────────────────────────────┘
```

#### 4.4 The Inquiry Queue

Instead of one-off analyses, maintain a **persistent inquiry agenda**:

```
┌─────────────────────────────────────────────────────┐
│              INQUIRY AGENDA                          │
├─────────────────────────────────────────────────────┤
│                                                     │
│ 🔥 ACTIVE QUESTIONS                                  │
│ ├─ Q1: Who benefits from current regulatory         │
│ │      framework? [3 analyses run]                  │
│ ├─ Q2: What assumptions underlie the consensus?     │
│ │      [1 analysis run, 2 pending]                  │
│ └─ Q3: What's being swept under the rug?            │
│        [Not started]                                │
│                                                     │
│ 📋 BACKGROUND QUESTIONS                              │
│ ├─ Q4: Historical precedents?                       │
│ └─ Q5: Cross-cultural variation?                    │
│                                                     │
│ ✅ RESOLVED                                          │
│ └─ Q0: What are the main themes? [Answered]         │
│                                                     │
│ [+ New Question] [Link to Canvas] [Export Brief]    │
└─────────────────────────────────────────────────────┘
```

#### 4.5 The Steelman Challenge Mode

When viewing any analysis or claim, a one-click "Steelman This" button:

```
┌─────────────────────────────────────────────────────┐
│ CLAIM: "AI regulation will stifle innovation"       │
├─────────────────────────────────────────────────────┤
│                                                     │
│ 🎯 ORIGINAL FORMULATION                             │
│ "Regulation always lags technology, creating        │
│  compliance burdens that benefit incumbents."       │
│                                                     │
│ 💪 STEELMANNED VERSION                              │
│ "Given the pace of AI development, regulatory       │
│  frameworks may struggle to remain relevant,        │
│  potentially creating compliance costs that         │
│  disproportionately burden smaller innovators       │
│  while established players have resources to        │
│  adapt. Historical examples like GDPR suggest       │
│  implementation costs can be substantial."          │
│                                                     │
│ ✅ AGREEMENTS                                        │
│ • Regulation does often lag technology              │
│ • Compliance costs are real                         │
│ • Smaller players face disadvantages                │
│                                                     │
│ 🎓 LEARNING POINTS                                  │
│ • Forces consideration of pace asymmetry            │
│ • Highlights incumbent advantage mechanism          │
│                                                     │
│ ⚠️ REMAINING VULNERABILITIES (after steelmanning)   │
│ • Assumes regulation is always burdensome           │
│ • Ignores trust/stability benefits                  │
│ • Selection bias in "stifled innovation" claims     │
│                                                     │
│ [Challenge This] [Accept Steelman] [Add to Canvas]  │
└─────────────────────────────────────────────────────┘
```

#### 4.6 The Certainty Dashboard

Visual display of confidence levels across all findings:

```
┌─────────────────────────────────────────────────────────────────────┐
│                    CERTAINTY DASHBOARD                               │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ESTABLISHED  ████████████████████████████████████ 4 claims         │
│  STRONG       ████████████████████ 8 claims                         │
│  MODERATE     ████████████████████████████ 12 claims                │
│  CONTESTED    ██████████████ 6 claims                               │
│  WEAK         ██████ 3 claims                                       │
│  SPECULATIVE  ██ 1 claim                                            │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │ MOST CONSEQUENTIAL UNCERTAINTIES                            │   │
│  ├─────────────────────────────────────────────────────────────┤   │
│  │ 1. "Market efficiency assumption" - CONTESTED               │   │
│  │    Impact if wrong: HIGH | Type: Model uncertainty          │   │
│  │                                                             │   │
│  │ 2. "Technology adoption timeline" - SPECULATIVE             │   │
│  │    Impact if wrong: HIGH | Type: Empirical (resolvable)     │   │
│  │                                                             │   │
│  │ 3. "Regulatory effectiveness" - MODERATE                    │   │
│  │    Impact if wrong: MEDIUM | Type: Empirical                │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
│  [Export Confidence Summary] [What Would Change These?]             │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Part V: Sharpening Existing Engines

### Global Prompt Enhancements

Add to ALL extraction prompts:

```python
DENNETT_HEADER = """
Apply Dennett's analytical tools throughout:

• SURELY ALARM: Flag confident assertions ("surely", "obviously") — probe these
• OCCAM'S BROOM: Note what's suspiciously absent or swept aside
• BOOM CRUTCH: Mark explanatory gaps ("then magic happens")
• STEELMAN: Before critiquing, express arguments in their strongest form
• JOOTSING: Identify implicit rules — could they be productively violated?
"""
```

### Specific Engine Improvements

#### `argument_architecture` — Enhanced Prompt

```python
IMPROVED_EXTRACTION = """
You are analyzing argument structure using the Toulmin model + Dennett tools.

For each argument:

TOULMIN COMPONENTS:
1. CLAIM: Main assertion
   - Apply SURELY ALARM: Does text use confidence markers without defense?

2. GROUNDS: Evidence offered
   - PROVENANCE CHECK: Where does this evidence actually come from?

3. WARRANT: Reasoning link
   - BOOM CRUTCH CHECK: Is the warrant a magic step?
   - If implicit, is it ACTUALLY warranted?

4. BACKING: Support for warrant
   - STURGEON'S LAW: Is backing high-quality or filler?

5. QUALIFIER: Strength modifiers
   - PHILOSOPHER'S SYNDROME: Are impossibility claims imagination-based?

6. REBUTTAL: Counterarguments
   - STEELMAN: Are counters addressed in strongest form?
   - OCCAM'S BROOM: What counters are suspiciously absent?

ADDITIONAL:
- DEEPITY CHECK: Any claims that equivocate between trivial and profound?
- JOOTSING OPPORTUNITY: What implicit rules constrain this argument?
"""
```

#### `assumption_excavation` — Enhanced with Dennett

```python
IMPROVED_EXTRACTION = """
You are excavating hidden assumptions using Dennett's archaeological tools.

ASSUMPTION TYPES + DENNETT PROBES:

1. PREMISE ASSUMPTIONS (what must be true for argument to work)
   - BOOM CRUTCH: Any premise that's actually a magic step?
   - SURELY ALARM: Premises asserted with unearned confidence?

2. DEFINITIONAL ASSUMPTIONS (how terms are implicitly defined)
   - BOUNDARY PROBE (Sortes): Where exactly are the lines?
   - TWO-BITSER: Does meaning shift by context?

3. WORLDVIEW ASSUMPTIONS (beliefs about how things work)
   - PHILOSOPHER'S SYNDROME: "Can't imagine otherwise" ≠ "Must be true"
   - JOOTSING: Which worldview constraints are merely conventional?

4. METHODOLOGICAL ASSUMPTIONS (what counts as valid knowledge)
   - OCCAM'S BROOM: What methods/evidence types are swept aside?
   - CURSE OF KNOWLEDGE: Is expert jargon substituting for argument?

5. NORMATIVE ASSUMPTIONS (values taken as given)
   - DEEPITY CHECK: Are value claims disguised as factual claims?
   - RATHERING: Are false dilemmas hiding value choices?

For each assumption:
- STATE IT: Express the assumption clearly
- DENNETT PROBE: Which tool reveals its vulnerability?
- CHALLENGE VECTOR: How would you stress-test this assumption?
"""
```

#### `rhetorical_strategy` — Integrated Dennett Detection

```python
IMPROVED_EXTRACTION = """
You are analyzing persuasion through Dennett's philosophical lens.

ENHANCED RHETORICAL ANALYSIS:

1. ARISTOTELIAN APPEALS + DENNETT
   - ETHOS: Is authority legitimate or appeal to false authority?
   - PATHOS: Legitimate emotional engagement or manipulation?
   - LOGOS: Valid reasoning or BOOM CRUTCH disguised as logic?

2. RHETORICAL DEVICES + QUALITY CHECK
   - REPETITION: Emphasis or hypnosis?
   - ANALOGY: Illuminating or misleading?
   - Apply DEEPITY DETECTOR to aphorisms

3. ARGUMENT TACTICS + FAIRNESS CHECK
   - STEELMAN TEST: Does text engage opposing views in strongest form?
   - Or STRAWMAN: Weakened versions attacked?

4. FRAMING STRATEGIES + JOOTSING
   - What frames are offered?
   - What frames are implicitly excluded?
   - What would JOOTSING to a different frame reveal?

5. SURELY ALARM SCAN
   - Flag ALL confidence markers
   - Rate whether confidence is earned

6. OCCAM'S BROOM SCAN
   - What counterarguments are suspiciously absent?
   - What evidence is swept aside?
"""
```

#### `evidence_quality_assessment` — Provenance Focus

```python
IMPROVED_EXTRACTION = """
You are assessing evidence quality with epistemic rigor.

EVIDENCE EVALUATION PROTOCOL:

1. PROVENANCE AUDIT
   - WHERE does this evidence actually originate?
   - TRANSFORMATION CHAIN: How did it get from origin to here?
   - TELEPHONE GAME CHECK: How much signal loss?

2. HETEROPHENOMENOLOGICAL SEPARATION
   - Treat claim AS DATA: What do we learn from the fact this claim was made?
   - Then evaluate: Is the claim actually true?

3. QUALITY MARKERS
   - STURGEON'S LAW: Is this in the 90% crap or 10% good?
   - Apply quality filter BEFORE counting citations

4. CERTAINLY GRADIENT
   - Assign probability, not binary true/false
   - Distinguish:
     * Empirical uncertainty (we don't know YET)
     * Fundamental uncertainty (we CAN'T know)
     * Imagination failure (we can't SEE how)

5. BOOM CRUTCH IN EVIDENCE
   - Does evidence actually EXPLAIN or just DESCRIBE?
   - "Studies show" without mechanism = weak evidence
"""
```

---

## Part VI: New Bundles

### Bundle 1: `dennett_toolkit`

The core Dennett analytical suite:

```python
BUNDLE = {
    "bundle_key": "dennett_toolkit",
    "name": "Dennett's Philosophical Toolkit",
    "description": "Core thinking tools from Dennett's Intuition Pumps: detect weak confidence, hidden gaps, pseudo-profundity, and magic explanations.",
    "member_engines": [
        "surely_alarm",
        "occams_broom",
        "deepity_detector",
        "boom_crutch_finder"
    ],
    "shared_extraction": True,
    "unified_prompt": "Apply Dennett's four core diagnostic tools simultaneously..."
}
```

### Bundle 2: `epistemic_rigor_suite`

(Renamed from `intelligence_tradecraft`)

```python
BUNDLE = {
    "bundle_key": "epistemic_rigor_suite",
    "name": "Epistemic Rigor Suite",
    "description": "Comprehensive analytical discipline: source quality, confidence calibration, hypothesis testing, and gap identification.",
    "member_engines": [
        "provenance_audit",
        "epistemic_calibration",
        "hypothesis_tournament",
        "terra_incognita_mapper",
        "steelman_generator"
    ]
}
```

### Bundle 3: `persuasion_archaeology`

(Renamed from `influence_forensics`)

```python
BUNDLE = {
    "bundle_key": "persuasion_archaeology",
    "name": "Persuasion Archaeology",
    "description": "Deep excavation of rhetorical strategies, manipulation techniques, and the machinery of persuasion.",
    "member_engines": [
        "surely_alarm",
        "deepity_detector",
        "rhetorical_strategy",
        "authenticity_forensics",
        "steelman_generator"
    ]
}
```

### Bundle 4: `relational_cartography`

(Renamed from `network_intelligence`)

```python
BUNDLE = {
    "bundle_key": "relational_cartography",
    "name": "Relational Cartography",
    "description": "Mapping actors, relationships, power dynamics, and network structure.",
    "member_engines": [
        "stakeholder_power_interest",
        "relational_topology",
        "resource_flow_asymmetry",
        "rational_actor_modeling"
    ]
}
```

### Bundle 5: `horizon_scanning_suite`

(Renamed from `strategic_warning`)

```python
BUNDLE = {
    "bundle_key": "horizon_scanning_suite",
    "name": "Horizon Scanning Suite",
    "description": "Future-oriented pattern detection, scenario exploration, and early signal identification.",
    "member_engines": [
        "signal_sentinel",
        "temporal_discontinuity_finder",
        "possibility_space_explorer",
        "jootsing_analyzer"
    ]
}
```

---

## Part VII: New Pipelines

### Pipeline 1: `epistemic_stress_test`

The core rigorous analysis pipeline:

```
argument_architecture
    → steelman_generator
    → hypothesis_tournament
    → epistemic_calibration
```

**Synergy:** Extract arguments → strengthen them → test against alternatives → calibrate confidence

### Pipeline 2: `dennett_diagnostic`

Full Dennett toolkit sweep:

```
surely_alarm
    → boom_crutch_finder
    → deepity_detector
    → occams_broom
```

**Synergy:** Flag weak confidence → find magic steps → detect pseudo-profundity → reveal what's hidden

### Pipeline 3: `provenance_to_certainty`

Source quality to confidence:

```
provenance_audit
    → epistemic_calibration
    → terra_incognita_mapper
```

**Synergy:** Map where claims come from → calibrate confidence → identify what we still need

### Pipeline 4: `persuasion_to_substance`

Strip rhetoric to find core argument:

```
rhetorical_strategy
    → surely_alarm
    → steelman_generator
    → argument_architecture
```

**Synergy:** Identify persuasion tactics → flag unearned confidence → extract strongest version → map actual logic

### Pipeline 5: `assumption_to_alternative`

From excavation to exploration:

```
assumption_excavation
    → jootsing_analyzer
    → contrarian_concept_generation
```

**Synergy:** Dig up hidden premises → identify violable rules → generate alternatives

### Pipeline 6: `complete_epistemic_audit`

The comprehensive pipeline:

```
provenance_audit
    → surely_alarm
    → boom_crutch_finder
    → steelman_generator
    → hypothesis_tournament
    → epistemic_calibration
    → terra_incognita_mapper
```

**Synergy:** Full epistemic workout from sources through confidence to gaps

---

## Part VIII: Implementation Roadmap

### Phase 1: Foundation (Weeks 1-3)

**Backend:**
1. Implement `surely_alarm` engine
2. Implement `occams_broom` engine
3. Implement `boom_crutch_finder` engine
4. Add Dennett header to ALL existing extraction prompts

**Frontend:**
1. Design canvas architecture
2. Prototype node/edge system
3. Implement basic pan/zoom

### Phase 2: Core Dennett Suite (Weeks 4-6)

**Backend:**
1. Implement `deepity_detector` engine
2. Implement `steelman_generator` engine
3. Implement `jootsing_analyzer` engine
4. Create `dennett_toolkit` bundle

**Frontend:**
1. Implement lens system architecture
2. Add inline highlighting for lens markers
3. Build lens toggle UI

### Phase 3: Epistemic Engines (Weeks 7-9)

**Backend:**
1. Implement `provenance_audit` engine
2. Implement `epistemic_calibration` engine
3. Implement `hypothesis_tournament` engine
4. Create `epistemic_rigor_suite` bundle

**Frontend:**
1. Build hypothesis workspace
2. Implement evidence-hypothesis matrix
3. Create certainty dashboard

### Phase 4: UI Revolution (Weeks 10-13)

**Frontend:**
1. Complete analytical canvas implementation
2. Build inquiry queue system
3. Implement steelman challenge mode
4. Add annotation layer
5. Build export/share features

### Phase 5: Integration & Polish (Weeks 14-16)

1. Create new pipelines
2. Full integration testing
3. Performance optimization
4. Documentation
5. User testing and iteration

---

## Part IX: Success Metrics

### Analytical Quality
- % of arguments steelmanned before critique
- Average vulnerability score reduction (through better evidence)
- Deepity detection rate vs. false positive rate

### User Engagement
- Time spent in hypothesis workspace
- Number of lens toggles per session
- Canvas nodes created per analysis

### Insight Generation
- New questions generated per analysis
- Alternative hypotheses considered
- Information gaps identified

---

## Appendix A: Complete Engine Renaming Table

| Current Key | New Key | New Name | Category |
|-------------|---------|----------|----------|
| — | `surely_alarm` | The "Surely" Alarm | RHETORIC |
| — | `occams_broom` | Occam's Broom Detector | EPISTEMOLOGY |
| — | `deepity_detector` | Deepity Detector | RHETORIC |
| — | `boom_crutch_finder` | Boom Crutch Finder | ARGUMENT |
| — | `steelman_generator` | Steelman Generator | ARGUMENT |
| — | `jootsing_analyzer` | Jootsing Analyzer | CONCEPTS |
| — | `philosophers_syndrome_detector` | Imagination Failure Detector | EPISTEMOLOGY |
| — | `boundary_probe` | Boundary Probe | CONCEPTS |
| — | `provenance_audit` | Provenance Audit | EVIDENCE |
| — | `epistemic_calibration` | Epistemic Calibration | EPISTEMOLOGY |
| — | `hypothesis_tournament` | Hypothesis Tournament | ARGUMENT |
| — | `terra_incognita_mapper` | Terra Incognita Mapper | EPISTEMOLOGY |
| — | `rational_actor_modeling` | Rational Actor Modeling | POWER |
| — | `relational_topology` | Relational Topology | POWER |
| — | `signal_sentinel` | Signal Sentinel | TEMPORAL |
| — | `temporal_discontinuity_finder` | Temporal Discontinuity Finder | TEMPORAL |
| — | `possibility_space_explorer` | Possibility Space Explorer | ARGUMENT |
| — | `authenticity_forensics` | Authenticity Forensics | RHETORIC |

---

## Appendix B: Keyboard Shortcuts (Retained)

While the canvas paradigm is primary, power users still want keyboard efficiency:

| Shortcut | Action |
|----------|--------|
| `Space` | Toggle hand/select mode on canvas |
| `C` | Create new node |
| `E` | Create edge between selected nodes |
| `D` | Delete selected |
| `L` | Open lens panel |
| `H` | Open hypothesis workspace |
| `Q` | Open inquiry queue |
| `S` | Steelman selected claim |
| `/` | Open command palette |
| `Cmd+K` | Quick search across all content |
| `Cmd+Z/Y` | Undo/redo |

---

*End of Implementation Plan*
