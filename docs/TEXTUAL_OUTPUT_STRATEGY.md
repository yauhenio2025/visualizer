# Textual Output Strategy: Complementing Visual Analysis

## The Problem

Our visual outputs (Gemini 4K images) excel at showing structure, relationships, and patterns at a glance. But our textual outputs currently:
- Often replicate what the visual shows (describing the same relationships in words)
- Don't differentiate meaningfully between "Text Report" vs "Executive Memo" vs "Research Report"
- Miss the opportunity to provide what visuals cannot: explanation, reasoning, nuance, actionability

## Core Principle: Visual Shows, Text Explains

| **Visual Strength** | **Textual Complement** |
|---------------------|------------------------|
| Spatial relationships | Why these relationships matter |
| Patterns at a glance | What the patterns mean |
| Structure overview | Causal reasoning behind structure |
| Quantitative comparison | Qualitative interpretation |
| "What is" | "So what?" and "Now what?" |

---

## Textual Output Types: Differentiated Purposes

### 1. Executive Memo (Decision-Focused)
**Audience**: Decision-maker with 5 minutes
**Length**: 1-2 pages
**Purpose**: Inform decision, recommend action

**Structure**:
```
BOTTOM LINE UP FRONT (BLUF)
- Single sentence: What decision-makers need to know

KEY FINDINGS (3-5 bullets)
- Most consequential insights only
- Each bullet = one actionable fact

IMPLICATIONS
- What this means for [the organization/strategy/decision]

RECOMMENDED ACTIONS
- Specific, prioritized steps to take
- Who should do what, when

WATCH ITEMS / UNCERTAINTIES
- What could change this assessment
- Confidence levels on key claims
```

**Tone**: Direct, confident, action-oriented
**Complements Visual**: Visual shows the landscape; Memo says what to do about it

---

### 2. Research Report (Evidence-Focused)
**Audience**: Analyst, researcher, due diligence team
**Length**: 5-15 pages
**Purpose**: Deep understanding with full evidence chain

**Structure**:
```
EXECUTIVE SUMMARY (1 page)
- Condensed findings for those who won't read full report

METHODOLOGY
- What engines were used, what they extract
- Document corpus description
- Limitations of this analysis

DETAILED FINDINGS
- Section per major finding
- Each finding includes:
  - The claim
  - Supporting evidence (direct quotes, data points)
  - Confidence level (certain/likely/speculative)
  - Alternative interpretations

ANALYSIS & INTERPRETATION
- What the findings mean taken together
- Patterns, contradictions, gaps
- Historical/contextual framing

LIMITATIONS & CAVEATS
- What this analysis cannot tell us
- Data gaps, methodological constraints
- Areas requiring further investigation

APPENDIX
- Full evidence table
- Source reliability assessments
- Methodology technical details
```

**Tone**: Academic, evidence-based, appropriately hedged
**Complements Visual**: Visual shows structure; Report provides the evidence behind it

---

### 3. Briefing Document (Narrative-Focused)
**Audience**: Stakeholder who needs to understand, not decide
**Length**: 2-5 pages
**Purpose**: Narrative understanding of the situation

**Structure**:
```
THE SITUATION
- Narrative overview (what's happening, why it matters)

KEY INSIGHTS
- 5-7 findings explained in accessible language
- Each insight includes "why this matters"

CONTEXTUAL BACKGROUND
- Historical context
- Related developments
- Relevant comparisons

LOOKING AHEAD
- Emerging trends
- Potential scenarios
- Questions to monitor
```

**Tone**: Clear, accessible, informative
**Complements Visual**: Visual shows the "what"; Briefing explains the "why"

---

### 4. Q&A Dossier (Question-Focused)
**Audience**: User with specific questions
**Length**: Variable
**Purpose**: Direct, sourced answers to anticipated questions

**Structure**:
```
For each anticipated question:

Q: [Natural language question]

A: [Direct answer - 1-2 sentences]

Evidence: [Specific quotes/data supporting this]

Confidence: [High/Medium/Low + why]

Nuance: [What the simple answer misses]
```

**Tone**: Direct, specific, helpful
**Complements Visual**: Visual gives overview; Q&A gives precision

---

## Engine-Specific Text Strategies

### Map Actors & Networks Engines
**Visual shows**: Network graph - who connects to whom, power positions
**Text should provide**:
- WHY these relationships exist (interests, history)
- WHAT the implications are for strategy
- WHO requires attention (ranked by priority)
- HOW to engage key actors

**Executive Memo Focus**: Top 3 actors to engage, key alliances to monitor
**Research Report Focus**: Complete stakeholder mapping with evidence
**Briefing Focus**: Narrative of the power landscape
**Q&A Focus**: "Who has most influence over X?" "What does Y want?"

---

### Evaluate Arguments Engines
**Visual shows**: Argument map - claims, premises, logical flow
**Text should provide**:
- STRENGTH assessment of each argument
- GAPS and vulnerabilities
- COUNTERARGUMENTS (strongest form)
- RECOMMENDATION on which position is best supported

**Executive Memo Focus**: Key vulnerabilities, strongest counterpoints, recommended position
**Research Report Focus**: Complete argument analysis with evidence quality ratings
**Briefing Focus**: Plain-language explanation of competing views
**Q&A Focus**: "What's the strongest argument for X?" "What's the main weakness?"

---

### Trace Evolution Engines
**Visual shows**: Timeline - events, sequence, phases
**Text should provide**:
- CAUSAL explanations (why each transition happened)
- TURNING POINTS (what made the difference)
- COUNTERFACTUALS (what could have been different)
- FUTURE trajectories (where this is heading)

**Executive Memo Focus**: Critical inflection points, emerging trends, watch items
**Research Report Focus**: Detailed chronological analysis with causal chains
**Briefing Focus**: Story of how we got here
**Q&A Focus**: "When did X change?" "Why did Y happen?"

---

### Find Patterns Engines
**Visual shows**: Concept tree/clusters - themes, hierarchies
**Text should provide**:
- MEANING of each pattern (why it exists)
- SIGNIFICANCE (what it implies)
- GAPS (what's conspicuously absent)
- CONNECTIONS (how patterns relate)

**Executive Memo Focus**: Key themes, implications, action items
**Research Report Focus**: Comprehensive thematic analysis with evidence
**Briefing Focus**: The major themes explained
**Q&A Focus**: "What are the main themes?" "Is X mentioned?"

---

### Assess Credibility Engines
**Visual shows**: Radar/matrix - credibility dimensions, scores
**Text should provide**:
- SPECIFIC concerns (what raises red flags)
- VERIFICATION status (what can/can't be confirmed)
- METHODOLOGY critique (how sources gathered evidence)
- TRUST recommendations (what to rely on)

**Executive Memo Focus**: Trust levels, red flags, recommended due diligence
**Research Report Focus**: Complete source assessment with verification chain
**Briefing Focus**: How much to trust what you're reading
**Q&A Focus**: "Can we trust X?" "What's the evidence for Y?"

---

### Compare Positions Engines
**Visual shows**: Quadrant/matrix - positions mapped on dimensions
**Text should provide**:
- TRADE-OFFS between positions (what you gain/lose)
- DECISION CRITERIA (how to choose)
- SYNTHESIS possibilities (can positions be combined?)
- RECOMMENDATION (which position is strongest)

**Executive Memo Focus**: Recommended position, key differentiators, decision framework
**Research Report Focus**: Comprehensive comparison with full analysis
**Briefing Focus**: The main camps and their arguments
**Q&A Focus**: "How does X compare to Y?" "What's the best option?"

---

## Extraction/Curation Strategies

### For Executive Memo
**Extraction**: Full canonical schema from engines
**Curation Rules**:
- Keep only findings with HIGH actionability
- Maximum 5 key findings
- Every finding must connect to a possible action
- Strip methodology, keep implications

**Concretization**:
- Add specific "do this" recommendations
- Include timeline (this week, this quarter)
- Flag items requiring immediate attention

---

### For Research Report
**Extraction**: Full canonical schema + raw evidence
**Curation Rules**:
- Preserve ALL evidence chains
- Include confidence levels for each claim
- Document methodology and limitations
- Keep alternative interpretations

**Concretization**:
- Add direct quotes from sources
- Include source reliability ratings
- Build full citation system
- Document what ISN'T in the evidence

---

### For Briefing Document
**Extraction**: Curated insights + narrative context
**Curation Rules**:
- Balance breadth and depth
- Prioritize understandability over completeness
- Include historical context
- Focus on "story arc"

**Concretization**:
- Add analogies and comparisons
- Include "plain English" explanations
- Connect to things reader already knows
- Avoid jargon

---

### For Q&A Dossier
**Extraction**: Intent-driven subset of findings
**Curation Rules**:
- Anticipate 10-15 likely questions
- Each answer must be self-contained
- Include counter-questions reader should ask
- Prioritize precision over comprehensiveness

**Concretization**:
- Answer first, then explain
- Every claim gets a source
- Confidence levels on every answer
- "What this doesn't tell you" sections

---

## Implementation Requirements

### Phase 1: Differentiated Templates
Create distinct prompt templates for each output type that:
- Specify audience and purpose
- Enforce structure
- Apply curation rules
- Include concretization instructions

### Phase 2: Extraction Layer
Modify engines to output:
- `findings_for_executive` - Pre-curated for decision-makers
- `findings_for_research` - Full evidence chains
- `findings_for_briefing` - Narrative-ready
- `anticipated_questions` - Q&A seeds

### Phase 3: Complementarity Check
Before generating text output, analyze the visual output and:
- Identify what visual already communicates well
- Focus text on gaps in visual communication
- Cross-reference to avoid redundancy
- Add "See visual for [X]" pointers

### Phase 4: Multi-Output Orchestration
When user requests both visual + text:
- Visual generated first
- Visual summary fed to text generation
- Text explicitly complements, not replicates
- Cross-references embedded in both

---

## Quality Metrics

### Executive Memo Quality
- [ ] Can be read in 5 minutes
- [ ] BLUF is one sentence
- [ ] Every finding is actionable
- [ ] Actions are specific (who/what/when)
- [ ] Uncertainty is flagged appropriately

### Research Report Quality
- [ ] Every claim has evidence citation
- [ ] Confidence levels on all findings
- [ ] Alternative interpretations considered
- [ ] Limitations explicitly stated
- [ ] Reproducible methodology documented

### Briefing Document Quality
- [ ] Non-specialist can understand
- [ ] Story arc is clear
- [ ] Context helps interpretation
- [ ] Jargon is explained or avoided
- [ ] Reader knows why this matters

### Q&A Dossier Quality
- [ ] Questions match what reader would ask
- [ ] Answers are direct (lead with answer)
- [ ] Sources are specific
- [ ] Confidence is clear
- [ ] Gaps are acknowledged

---

## The Visual-Text Contract

For every engine output, we commit to:

1. **Visual**: Shows the WHAT (structure, relationships, patterns)
2. **Text**: Explains the WHY, SO WHAT, and NOW WHAT

They should never merely repeat each other. A good test:
- Could you understand the text without seeing the visual? (Yes - self-contained)
- Could you understand the visual without reading the text? (Yes - self-explanatory)
- Does the text add insight the visual cannot show? (Must be yes)
- Does the visual show something the text cannot efficiently describe? (Must be yes)

If both show the same thing, one is redundant. Eliminate redundancy, maximize complementarity.
