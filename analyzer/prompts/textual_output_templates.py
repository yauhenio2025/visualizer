"""
Textual Output Templates - 8 Differentiated Output Types

Each output type serves a different consumer with different needs.
These are not different "lengths" of the same thing - they are
fundamentally different products.

Output Types:
1. Snapshot - 1-page immediate awareness
2. Deep Dive - Comprehensive synthesis with calibrated confidence
3. Evidence Pack - Complete source documentation
4. Signal Report - Early indicators of change
5. Status Brief - Current state summary
6. Stakeholder Profile - Decision-maker/entity analysis
7. Gap Analysis - Weakness and vulnerability identification
8. Options Brief - Decision support with trade-offs
"""

from typing import Any
from dataclasses import dataclass
from enum import Enum


class TextualOutputType(Enum):
    """The 8 differentiated textual output types."""
    SNAPSHOT = "snapshot"
    DEEP_DIVE = "deep_dive"
    EVIDENCE_PACK = "evidence_pack"
    SIGNAL_REPORT = "signal_report"
    STATUS_BRIEF = "status_brief"
    STAKEHOLDER_PROFILE = "stakeholder_profile"
    GAP_ANALYSIS = "gap_analysis"
    OPTIONS_BRIEF = "options_brief"


@dataclass
class OutputTypeMetadata:
    """Metadata for each output type."""
    key: str
    name: str
    icon: str
    description: str
    length: str
    reading_time: str
    audience: str
    core_question: str


OUTPUT_TYPE_METADATA: dict[str, OutputTypeMetadata] = {
    "snapshot": OutputTypeMetadata(
        key="snapshot",
        name="Snapshot",
        icon="⚡",
        description="1-page executive summary for immediate awareness",
        length="~400 words",
        reading_time="2 minutes",
        audience="Decision-makers with no time",
        core_question="What do I need to know RIGHT NOW?"
    ),
    "deep_dive": OutputTypeMetadata(
        key="deep_dive",
        name="Deep Dive",
        icon="🔬",
        description="Comprehensive synthesis with calibrated confidence levels",
        length="2000-5000 words",
        reading_time="20-40 minutes",
        audience="Analysts and researchers",
        core_question="What's the full picture and how confident are we?"
    ),
    "evidence_pack": OutputTypeMetadata(
        key="evidence_pack",
        name="Evidence Pack",
        icon="📁",
        description="Complete source documentation with reliability ratings",
        length="Variable (reference document)",
        reading_time="Reference use",
        audience="Due diligence teams, fact-checkers",
        core_question="What's the evidence and how reliable is it?"
    ),
    "signal_report": OutputTypeMetadata(
        key="signal_report",
        name="Signal Report",
        icon="📡",
        description="Early indicators and emerging patterns requiring attention",
        length="~800 words",
        reading_time="5 minutes",
        audience="Strategic planners, risk managers",
        core_question="What signals indicate change is coming?"
    ),
    "status_brief": OutputTypeMetadata(
        key="status_brief",
        name="Status Brief",
        icon="📋",
        description="Current state summary with recent developments",
        length="~1200 words",
        reading_time="8 minutes",
        audience="Operations teams, stakeholders",
        core_question="What happened and where are we now?"
    ),
    "stakeholder_profile": OutputTypeMetadata(
        key="stakeholder_profile",
        name="Stakeholder Profile",
        icon="👤",
        description="Deep analysis of key actors, motivations, and likely behavior",
        length="~1500 words per actor",
        reading_time="10 minutes per actor",
        audience="Negotiators, strategists",
        core_question="Who is this actor and how will they behave?"
    ),
    "gap_analysis": OutputTypeMetadata(
        key="gap_analysis",
        name="Gap Analysis",
        icon="🎯",
        description="Systematic identification of weaknesses and vulnerabilities",
        length="~1500 words",
        reading_time="10 minutes",
        audience="Critics, red teams, quality assurance",
        core_question="Where are the gaps and how can they be addressed?"
    ),
    "options_brief": OutputTypeMetadata(
        key="options_brief",
        name="Options Brief",
        icon="⚖️",
        description="Decision framework with options, trade-offs, and recommendations",
        length="~1200 words",
        reading_time="8 minutes",
        audience="Decision-makers choosing between alternatives",
        core_question="What are my options and which should I choose?"
    ),
}


# =============================================================================
# PROMPT TEMPLATES
# =============================================================================

SNAPSHOT_TEMPLATE = '''You are generating a SNAPSHOT - a 1-page executive summary for immediate awareness.

AUDIENCE: Decision-makers with 2 minutes maximum
PURPOSE: Immediate situational awareness - what they need to know RIGHT NOW
LENGTH: 400 words maximum - ruthlessly concise

CRITICAL RULES:
- Bottom line FIRST - lead with the single most important insight
- No background or methodology - they don't have time
- Every sentence must earn its place
- Implications must be actionable
- Confidence statement is REQUIRED

STRUCTURE (follow exactly):

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SNAPSHOT: [Topic in 5 words or fewer]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

BOTTOM LINE
[One sentence. The single most important thing to know.]

KEY FINDING
[2-3 sentences maximum. What the analysis revealed and why it matters.]

IMPLICATIONS
• [Implication 1 - one line, actionable]
• [Implication 2 - one line, actionable]
• [Implication 3 - one line, actionable]

CONFIDENCE: [HIGH/MODERATE/LOW]
[One sentence explaining the basis for this confidence level]

WATCH FOR
[One sentence: What specific signal would change this assessment]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

ANALYSIS DATA:
{analysis_data}

VISUAL OUTPUT SUMMARY (if available):
{visual_summary}

COMPLEMENTARITY RULE: Do NOT describe what the visual shows. The visual shows the WHAT. Your job is the SO WHAT and NOW WHAT.

Generate the Snapshot now:'''


DEEP_DIVE_TEMPLATE = '''You are generating a DEEP DIVE - a comprehensive synthesis with calibrated confidence.

AUDIENCE: Analysts, researchers, and subject matter experts who need the full picture
PURPOSE: Comprehensive understanding with explicit confidence levels and alternative interpretations
LENGTH: 2000-5000 words depending on complexity

CRITICAL RULES:
- Every major judgment MUST have a confidence level (HIGH/MODERATE/LOW)
- Every judgment MUST acknowledge evidence for AND against
- Alternative interpretations MUST be presented fairly, not as strawmen
- "What we don't know" section MUST be substantive, not perfunctory
- Sources must be cited and reliability assessed

CONFIDENCE FRAMEWORK:
- HIGH: Multiple independent sources agree; logical consistency; strong analytic consensus
- MODERATE: Plausible but limited sources; some uncertainty in interpretation
- LOW: Single source; significant gaps; alternative explanations equally viable

STRUCTURE (follow exactly):

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DEEP DIVE: [Topic]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

SCOPE
[What this analysis covers, what it excludes, and why]

KEY JUDGMENTS

1. [First major judgment]
   Confidence: [HIGH/MODERATE/LOW]

   We assess that [clear statement of judgment].

   Supporting evidence:
   • [Evidence point 1 with source]
   • [Evidence point 2 with source]

   Countervailing evidence:
   • [Evidence that complicates or challenges this judgment]

   Alternative interpretation: [What a reasonable analyst might conclude differently]

2. [Second major judgment]
   Confidence: [HIGH/MODERATE/LOW]
   [Same structure as above]

[Continue for all major judgments]

COMPETING HYPOTHESES
[If applicable - multiple explanations evaluated against evidence]

| Hypothesis | Consistency with Evidence | Key Support | Key Challenge |
|------------|---------------------------|-------------|---------------|
| H1: [...]  | [High/Medium/Low]         | [...]       | [...]         |
| H2: [...]  | [High/Medium/Low]         | [...]       | [...]         |

DETAILED ANALYSIS

[Section 1: First major theme/finding]
[Deep exploration with evidence, analysis, and interpretation]

[Section 2: Second major theme/finding]
[Deep exploration with evidence, analysis, and interpretation]

[Continue as needed]

WHAT WE DON'T KNOW
[Substantive gaps that matter - not just "more research needed"]
• [Gap 1]: Why it matters and what would fill it
• [Gap 2]: Why it matters and what would fill it

OUTLOOK
[Where this is heading based on current trajectory - with confidence level]

METHODOLOGY NOTE
[Brief description of analytical approach and limitations]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

ANALYSIS DATA:
{analysis_data}

VISUAL OUTPUT SUMMARY (if available):
{visual_summary}

COMPLEMENTARITY RULE: The visual shows structure and relationships. Your job is to explain WHY those structures exist, WHAT they mean, and HOW confident we should be.

Generate the Deep Dive now:'''


EVIDENCE_PACK_TEMPLATE = '''You are generating an EVIDENCE PACK - complete source documentation for verification and audit.

AUDIENCE: Due diligence teams, fact-checkers, analysts who need to verify claims
PURPOSE: Full evidence chain with source reliability assessments
LENGTH: Variable - this is a reference document, not linear reading

CRITICAL RULES:
- EVERY claim must have a source citation
- Source reliability MUST be rated using the standard scale
- Direct quotes must be VERBATIM with exact location (page, paragraph, timestamp)
- Contradictions between sources MUST be documented
- Gaps (evidence sought but not found) MUST be logged

SOURCE RELIABILITY SCALE:
- A: Completely reliable (verified, independently corroborated)
- B: Usually reliable (strong track record, minor caveats)
- C: Fairly reliable (some verification possible, context matters)
- D: Not usually reliable (limited verification, use with caution)
- E: Unreliable (contradicted by other sources)
- F: Cannot be judged (insufficient basis to assess)

STRUCTURE (follow exactly):

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
EVIDENCE PACK: [Topic]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

EVIDENCE INDEX
| ID   | Claim Summary              | Source         | Reliability |
|------|----------------------------|----------------|-------------|
| E001 | [Brief claim]              | [Source ref]   | [A-F]       |
| E002 | [Brief claim]              | [Source ref]   | [A-F]       |
[Continue for all evidence items]

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DETAILED EVIDENCE ITEMS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

──────────────────────────────────────────────────────────
ITEM E001
──────────────────────────────────────────────────────────
Claim: [Full claim text]
Source: [Document title, author, date, exact location]
Reliability: [Letter rating] - [Brief justification]

Verbatim Extract:
> "[Direct quote from source - exact words]"
> — [Attribution, page/paragraph/timestamp]

Context: [What was happening when this was written/said]
Corroboration: [Other sources that support or contradict]
Analyst Note: [Any caveats, interpretive issues, or flags]

[Repeat for each evidence item]

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CONTRADICTIONS LOG
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[Where sources disagree, with analysis of which to credit and why]

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
GAPS REGISTER
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[Evidence we looked for but did not find]
• [Gap 1]: [What we sought] - [Why it matters]
• [Gap 2]: [What we sought] - [Why it matters]

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SOURCE EVALUATION NOTES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[Overall assessment of source quality and any systematic biases]
```

ANALYSIS DATA:
{analysis_data}

Generate the Evidence Pack now:'''


SIGNAL_REPORT_TEMPLATE = '''You are generating a SIGNAL REPORT - early indicators of emerging changes.

AUDIENCE: Strategic planners, risk managers, anyone who needs early warning
PURPOSE: Alert to emerging patterns that may require action or monitoring
LENGTH: ~800 words - concise but substantive

CRITICAL RULES:
- Focus on CHANGE - what's different from baseline
- Indicators must be SPECIFIC and MEASURABLE
- Threshold deviations must be QUANTIFIED where possible
- Timeline to decision point must be realistic
- Watch list must be actionable for next 48-72 hours
- Acknowledge false alarm possibility

STRUCTURE (follow exactly):

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📡 SIGNAL REPORT: [What's emerging - 5 words]
Priority: [CRITICAL / HIGH / ELEVATED / ROUTINE]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

SIGNAL SUMMARY
[2-3 sentences: What we're detecting and why it matters]

INDICATORS DETECTED

| Indicator | Status | Baseline | Current | Significance |
|-----------|--------|----------|---------|--------------|
| [Signal 1] | 🔴 TRIGGERED | [Normal] | [Now] | [What it suggests] |
| [Signal 2] | 🟡 ELEVATED | [Normal] | [Now] | [What it suggests] |
| [Signal 3] | 🟢 NORMAL | [Normal] | [Now] | [Would indicate if changed] |

PATTERN ANALYSIS
[What the combination of signals suggests - pattern recognition]

THRESHOLD ANALYSIS
• Previous baseline: [What was normal]
• Current reading: [What we're seeing]
• Deviation significance: [How unusual this is]
• Historical precedent: [When we've seen similar patterns]

TIMELINE ASSESSMENT
[How long before situation crystallizes or window closes]
• Decision point: [When action may be required]
• Observation window: [How long we have to gather more data]

RECOMMENDED RESPONSES
□ [Immediate action - within 24-48 hours]
□ [Near-term preparation - within 1-2 weeks]
□ [Contingency to prepare - if signal strengthens]

WATCH LIST (Next 48-72 Hours)
• [Specific indicator to monitor - what would it mean]
• [Specific indicator to monitor - what would it mean]
• [Trigger that would elevate priority level]

CONFIDENCE: [HIGH/MODERATE/LOW]
[What could make this a false alarm]

FALSE POSITIVE ASSESSMENT
[Probability this is noise rather than signal, and why]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

ANALYSIS DATA:
{analysis_data}

VISUAL OUTPUT SUMMARY (if available):
{visual_summary}

COMPLEMENTARITY RULE: If visual shows a timeline or anomaly, don't redescribe it. Explain what it MEANS and what to DO.

Generate the Signal Report now:'''


STATUS_BRIEF_TEMPLATE = '''You are generating a STATUS BRIEF - a comprehensive update on the current situation.

AUDIENCE: Operations teams, stakeholders who need to know where things stand
PURPOSE: Clear picture of current state with recent developments
LENGTH: ~1200 words

CRITICAL RULES:
- Developments in CHRONOLOGICAL order
- Clearly distinguish FACTS from INTERPRETATION
- "Change from last period" must be explicit
- Open questions must be specific, not vague
- Near-term outlook must be falsifiable (could be proven wrong)

STRUCTURE (follow exactly):

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STATUS BRIEF: [Topic]
Report Period: [Date range covered]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

SITUATION SUMMARY
[3-4 sentences: Current state of play - where we are right now]

PERIOD DEVELOPMENTS

| Date | Development | Significance |
|------|-------------|--------------|
| [Date] | [What happened] | [Why it matters] |
| [Date] | [What happened] | [Why it matters] |
| [Date] | [What happened] | [Why it matters] |

KEY ACTORS UPDATE
[Current posture and recent moves of major players]

• [Actor 1]: [What they did, what they said, what it suggests]
• [Actor 2]: [What they did, what they said, what it suggests]
• [Actor 3]: [What they did, what they said, what it suggests]

CHANGE FROM PREVIOUS STATE
↑ ESCALATED/INCREASED:
  • [What got more intense/urgent/significant]

→ STABLE/UNCHANGED:
  • [What stayed the same]

↓ DE-ESCALATED/DECREASED:
  • [What improved or reduced in intensity]

RESOURCE & FLOW UPDATE
[If applicable: Money, personnel, materiel, attention flows observed]

OPEN QUESTIONS
• [Unresolved question 1 - specific, not "what will happen"]
• [Unresolved question 2 - specific, not "what will happen"]
• [Unresolved question 3 - specific, not "what will happen"]

NEAR-TERM OUTLOOK
[What we expect to see in the next reporting period and why]

Confidence: [HIGH/MODERATE/LOW]
Key assumption: [What would have to be true for this outlook to hold]
Alternative scenario: [What could happen instead]

ANALYST ASSESSMENT
[Interpretive judgment on overall trajectory - clearly labeled as interpretation]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

ANALYSIS DATA:
{analysis_data}

VISUAL OUTPUT SUMMARY (if available):
{visual_summary}

COMPLEMENTARITY RULE: Visual shows the landscape. You explain what CHANGED and what it MEANS.

Generate the Status Brief now:'''


STAKEHOLDER_PROFILE_TEMPLATE = '''You are generating a STAKEHOLDER PROFILE - deep analysis of a key actor.

AUDIENCE: Negotiators, strategists, anyone who needs to understand and potentially influence this actor
PURPOSE: Predict behavior, identify pressure points, guide engagement
LENGTH: ~1500 words per actor

CRITICAL RULES:
- Interests must be supported by EVIDENCE (what they've said/done)
- Decision patterns must be based on TRACK RECORD, not stereotypes
- Predictions must be specific enough to be WRONG
- Engagement recommendations must be PRACTICAL
- Distinguish between STATED positions and REVEALED preferences

STRUCTURE (follow exactly):

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STAKEHOLDER PROFILE: [Name/Entity]
Type: [Individual / Organization / Coalition / Other]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

SUMMARY
[2-3 sentences: Who they are and why they matter to this situation]

BASIC INFORMATION
• Role/Position: [Current role or organizational identity]
• Affiliation: [Networks, factions, organizations]
• Sphere of Influence: [Where they have power]
• Influence Level: [High/Medium/Low] in [specific domain]

INTERESTS & MOTIVATIONS

Primary Interests (in order of priority):
1. [Interest 1]
   Evidence: [What they've said/done that reveals this]

2. [Interest 2]
   Evidence: [What they've said/done that reveals this]

3. [Interest 3]
   Evidence: [What they've said/done that reveals this]

Underlying Drivers:
• [What motivates them at a deeper level - values, fears, aspirations]

Red Lines (what they won't accept):
• [Non-negotiable 1] - Evidence: [...]
• [Non-negotiable 2] - Evidence: [...]

DECISION-MAKING PATTERN

Worldview: [How they see the situation/world]
Risk Tolerance: [Risk-averse / Moderate / Risk-seeking] - Evidence: [...]
Time Horizon: [Short-term / Long-term oriented] - Evidence: [...]
Information Sources: [Who they listen to, what they read]
Decision Style: [Deliberative / Intuitive / Consensus-driven / Autocratic]

NETWORK POSITION

Key Relationships:
• [Ally 1]: [Nature of relationship, strength, basis]
• [Ally 2]: [Nature of relationship, strength, basis]
• [Rival/Adversary 1]: [Nature of conflict, history]

Structural Position:
• Network role: [Hub / Broker / Peripheral / Bridge]
• Power sources: [What gives them influence]
• Dependencies: [What they need from others]
• Vulnerabilities: [What could weaken their position]

TRACK RECORD

Past Behavior in Analogous Situations:
• [Situation 1]: [How they responded] → [What it suggests]
• [Situation 2]: [How they responded] → [What it suggests]

Pattern: [What their track record suggests about future behavior]

STATEMENTS & POSITIONS

Key Quotes:
> "[Direct quote 1]" — [Context, date]
> "[Direct quote 2]" — [Context, date]

Stated Positions: [What they claim to want]
Actions vs. Words: [Where behavior diverges from rhetoric]

BEHAVIORAL PREDICTION

Most Likely Behavior:
[What they will probably do in this situation and why]
Confidence: [HIGH/MODERATE/LOW]

Alternative Scenarios:
• If [condition], they might [behavior]
• If [condition], they might [behavior]

ENGAGEMENT RECOMMENDATIONS

Approach Strategy:
• [How to engage effectively with this actor]
• [What appeals to them / what language to use]

Avoid:
• [What NOT to do - what would backfire]
• [Topics or approaches that would alienate them]

Leverage Points:
• [What could motivate them to cooperate]
• [What pressure could be applied if needed]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

ANALYSIS DATA:
{analysis_data}

VISUAL OUTPUT SUMMARY (if available):
{visual_summary}

COMPLEMENTARITY RULE: Visual shows network position. You explain WHY they're there and HOW to engage them.

Generate the Stakeholder Profile now:'''


GAP_ANALYSIS_TEMPLATE = '''You are generating a GAP ANALYSIS - systematic identification of weaknesses and vulnerabilities.

AUDIENCE: Critics, quality assurance, anyone stress-testing an argument or position
PURPOSE: Find weaknesses before adversaries do; identify what needs strengthening
LENGTH: ~1500 words

CRITICAL RULES:
- Vulnerabilities must be GENUINE, not nitpicks
- Severity/exploitability/impact must be assessed for each
- Attack vectors must be REALISTIC
- Mitigations must be ACTIONABLE
- Steelman counterarguments must be STRONG (not strawmen)
- Acknowledge residual risk that can't be eliminated

STRUCTURE (follow exactly):

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
GAP ANALYSIS: [Subject Being Analyzed]
Analysis Type: [Argument / Position / Strategy / Plan / Organization]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

EXECUTIVE SUMMARY
[2-3 sentences: Key vulnerabilities identified and overall assessment]

VULNERABILITY INVENTORY

| # | Vulnerability | Severity | Exploitability | Impact |
|---|---------------|----------|----------------|--------|
| 1 | [Brief name] | [H/M/L] | [H/M/L] | [H/M/L] |
| 2 | [Brief name] | [H/M/L] | [H/M/L] | [H/M/L] |
| 3 | [Brief name] | [H/M/L] | [H/M/L] | [H/M/L] |

RATING KEY:
- Severity: How fundamental is this weakness?
- Exploitability: How easily could it be attacked?
- Impact: How much damage if exploited?

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DETAILED VULNERABILITY ANALYSIS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

──────────────────────────────────────────────────────────
VULNERABILITY 1: [Name]
Severity: [HIGH/MEDIUM/LOW]
──────────────────────────────────────────────────────────

Description:
[What the vulnerability is - clear and specific]

Location:
[Where in the argument/structure this exists]

Why It Matters:
[Consequences if this is exploited or exposed]

Evidence:
[How we identified this weakness]

Attack Vector:
[How a critic or adversary would exploit this]

Mitigation:
[How to address this - specific and actionable]

[Repeat for each vulnerability]

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
HIDDEN ASSUMPTIONS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[Assumptions that, if challenged, would undermine the position]

1. Assumption: [What's being taken for granted]
   If false: [What happens to the argument]
   Likelihood false: [How plausible is this challenge]

2. Assumption: [What's being taken for granted]
   If false: [What happens to the argument]
   Likelihood false: [How plausible is this challenge]

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
LOGICAL GAPS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[Where the reasoning has holes]

1. Gap: Between [premise] and [conclusion]
   Missing step: [What's needed to bridge this]

2. Gap: Between [premise] and [conclusion]
   Missing step: [What's needed to bridge this]

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
EVIDENCE WEAKNESSES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[Where the evidence is thin or questionable]

• [Weakness 1]: [What's missing or unreliable]
• [Weakness 2]: [What's missing or unreliable]

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEELMAN COUNTERARGUMENTS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[Strongest attacks a sophisticated critic could mount]

1. Counterargument: [Strong version of opposing view]
   How to respond: [If there's a good response]
   Residual damage: [What remains even after response]

2. Counterargument: [Strong version of opposing view]
   How to respond: [If there's a good response]
   Residual damage: [What remains even after response]

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MITIGATION ROADMAP
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Priority order for addressing vulnerabilities:

1. [First priority] - Why: [Highest severity/exploitability]
2. [Second priority] - Why: [...]
3. [Third priority] - Why: [...]

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RESIDUAL RISK
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[Vulnerabilities that cannot be fully eliminated - must be accepted]

• [Residual risk 1]: [Why it can't be eliminated]
• [Residual risk 2]: [Why it can't be eliminated]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

ANALYSIS DATA:
{analysis_data}

VISUAL OUTPUT SUMMARY (if available):
{visual_summary}

COMPLEMENTARITY RULE: Visual shows structure. You show WHERE the weak points are and HOW to fix them.

Generate the Gap Analysis now:'''


OPTIONS_BRIEF_TEMPLATE = '''You are generating an OPTIONS BRIEF - decision support with trade-offs and recommendations.

AUDIENCE: Decision-makers choosing between alternatives
PURPOSE: Frame decision clearly, present options fairly, recommend with justification
LENGTH: ~1200 words

CRITICAL RULES:
- Decision required must be CRYSTAL CLEAR
- Options must be GENUINELY different (not strawman vs. real option)
- Pros/cons must be BALANCED (not rigged toward one option)
- Risks must use likelihood/impact framework
- Recommendation must be CLEAR and JUSTIFIED
- Implementation steps must be ACTIONABLE

STRUCTURE (follow exactly):

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OPTIONS BRIEF: [Decision Required]
Decision Deadline: [Date or "As circumstances require"]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

DECISION REQUIRED
[One clear sentence: What choice needs to be made]

BACKGROUND
[2-3 sentences: Context necessary to understand the decision - no more]

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OPTIONS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

──────────────────────────────────────────────────────────
OPTION A: [Descriptive Name]
──────────────────────────────────────────────────────────

Description:
[What this option involves - concrete and specific]

Advantages:
✓ [Pro 1]
✓ [Pro 2]
✓ [Pro 3]

Disadvantages:
✗ [Con 1]
✗ [Con 2]

Risks:
⚠ [Risk 1]: Likelihood [H/M/L] × Impact [H/M/L]
⚠ [Risk 2]: Likelihood [H/M/L] × Impact [H/M/L]

Resource Requirements:
[Time, money, personnel, political capital, opportunity cost]

──────────────────────────────────────────────────────────
OPTION B: [Descriptive Name]
──────────────────────────────────────────────────────────

[Same structure as Option A]

──────────────────────────────────────────────────────────
OPTION C: [Descriptive Name] (if applicable)
──────────────────────────────────────────────────────────

[Same structure]

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
COMPARISON MATRIX
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

| Criterion | Option A | Option B | Option C |
|-----------|----------|----------|----------|
| Effectiveness | [H/M/L] | [H/M/L] | [H/M/L] |
| Cost | [H/M/L] | [H/M/L] | [H/M/L] |
| Risk | [H/M/L] | [H/M/L] | [H/M/L] |
| Speed | [H/M/L] | [H/M/L] | [H/M/L] |
| Reversibility | [H/M/L] | [H/M/L] | [H/M/L] |

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RECOMMENDATION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Recommended Option: [OPTION X]

Rationale:
[Clear statement of why this option is recommended - what makes it superior]

Key Trade-off Accepted:
[What we're giving up by not choosing other options]

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
IMPLEMENTATION (If Recommendation Accepted)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Immediate (24-48 hours):
□ [Specific action]

Short-term (1-2 weeks):
□ [Specific action]

Medium-term (1-3 months):
□ [Specific action]

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DECISION TRIGGERS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[What would cause us to revisit this decision]

• If [condition], reconsider [option]
• If [condition], escalate to [authority]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

ANALYSIS DATA:
{analysis_data}

VISUAL OUTPUT SUMMARY (if available):
{visual_summary}

COMPLEMENTARITY RULE: Visual shows comparison landscape. You say WHICH to choose and WHY.

Generate the Options Brief now:'''


# =============================================================================
# TEMPLATE REGISTRY
# =============================================================================

TEXTUAL_OUTPUT_TEMPLATES: dict[str, str] = {
    "snapshot": SNAPSHOT_TEMPLATE,
    "deep_dive": DEEP_DIVE_TEMPLATE,
    "evidence_pack": EVIDENCE_PACK_TEMPLATE,
    "signal_report": SIGNAL_REPORT_TEMPLATE,
    "status_brief": STATUS_BRIEF_TEMPLATE,
    "stakeholder_profile": STAKEHOLDER_PROFILE_TEMPLATE,
    "gap_analysis": GAP_ANALYSIS_TEMPLATE,
    "options_brief": OPTIONS_BRIEF_TEMPLATE,
}


# =============================================================================
# ENGINE → OUTPUT TYPE AFFINITY
# =============================================================================

# Which output types work best for which engines
# ★★★ = Ideal | ★★☆ = Good | ★☆☆ = Possible | ☆☆☆ = Not recommended

ENGINE_OUTPUT_AFFINITY: dict[str, dict[str, int]] = {
    # Detection & Warning Engines
    "signal_sentinel": {"snapshot": 3, "signal_report": 3, "status_brief": 2},
    "anomaly_detector": {"snapshot": 3, "signal_report": 3, "gap_analysis": 2},
    "temporal_discontinuity_finder": {"signal_report": 3, "status_brief": 2},
    "emerging_trend_detector": {"snapshot": 3, "signal_report": 3},
    "escalation_trajectory_analysis": {"signal_report": 3, "options_brief": 2},
    "surely_alarm": {"gap_analysis": 3, "snapshot": 2},

    # Evidence & Verification Engines
    "hypothesis_tournament": {"deep_dive": 3, "options_brief": 2},
    "evidence_quality_assessment": {"deep_dive": 3, "evidence_pack": 3},
    "epistemic_calibration": {"deep_dive": 3, "evidence_pack": 2},
    "provenance_audit": {"evidence_pack": 3, "deep_dive": 2},
    "statistical_evidence": {"evidence_pack": 3, "deep_dive": 2},
    "authenticity_forensics": {"deep_dive": 3, "signal_report": 2},

    # Actor & Network Engines
    "stakeholder_power_interest": {"stakeholder_profile": 3, "status_brief": 2},
    "rational_actor_modeling": {"stakeholder_profile": 3, "options_brief": 2},
    "relational_topology": {"stakeholder_profile": 3, "deep_dive": 2},
    "quote_attribution_voice": {"stakeholder_profile": 2, "evidence_pack": 3},
    "resource_flow_asymmetry": {"status_brief": 3, "stakeholder_profile": 2},
    "deal_flow_tracker": {"status_brief": 3, "snapshot": 2},

    # Temporal & Evolution Engines
    "event_timeline_causal": {"status_brief": 3, "deep_dive": 2},
    "concept_evolution": {"deep_dive": 3, "status_brief": 2},
    "chronology_cycle": {"deep_dive": 3, "signal_report": 2},
    "chronology_simultaneity": {"status_brief": 3, "deep_dive": 2},
    "reception_history": {"deep_dive": 3, "status_brief": 2},
    "temporal_multiscale": {"deep_dive": 3, "status_brief": 2},

    # Argument & Logic Engines
    "argument_architecture": {"gap_analysis": 3, "deep_dive": 2},
    "assumption_excavation": {"gap_analysis": 3, "deep_dive": 2},
    "steelman_generator": {"options_brief": 3, "gap_analysis": 2},
    "steelman_stress_test": {"gap_analysis": 3, "options_brief": 2},
    "boom_crutch_finder": {"gap_analysis": 3, "deep_dive": 2},
    "dialectical_structure": {"deep_dive": 3, "options_brief": 2},

    # Comparison & Framework Engines
    "comparative_framework": {"options_brief": 3, "deep_dive": 2},
    "possibility_space_explorer": {"options_brief": 3, "deep_dive": 2},
    "competitive_landscape": {"options_brief": 3, "status_brief": 2},
    "opportunity_vulnerability_matrix": {"gap_analysis": 3, "options_brief": 2},

    # Thematic & Conceptual Engines
    "thematic_synthesis": {"deep_dive": 3, "status_brief": 2},
    "conceptual_framework_extraction": {"deep_dive": 3, "evidence_pack": 2},
    "structural_pattern_detector": {"deep_dive": 3, "gap_analysis": 2},
    "interdisciplinary_connection": {"deep_dive": 3, "evidence_pack": 2},
    "jootsing_analyzer": {"gap_analysis": 3, "deep_dive": 2},

    # Deception & Credibility Engines
    "occams_broom": {"gap_analysis": 3, "deep_dive": 2},
    "deepity_detector": {"gap_analysis": 3, "deep_dive": 2},
    "influence_attribution_analysis": {"deep_dive": 3, "signal_report": 2},
    "terra_incognita_mapper": {"deep_dive": 3, "gap_analysis": 2},
}


def get_recommended_outputs(engine_key: str) -> list[str]:
    """Get recommended output types for an engine, sorted by affinity."""
    if engine_key not in ENGINE_OUTPUT_AFFINITY:
        # Default recommendations for unknown engines
        return ["deep_dive", "snapshot"]

    affinities = ENGINE_OUTPUT_AFFINITY[engine_key]
    sorted_outputs = sorted(affinities.items(), key=lambda x: x[1], reverse=True)
    return [output for output, score in sorted_outputs if score >= 2]


def get_template(output_type: str) -> str:
    """Get the prompt template for an output type."""
    if output_type not in TEXTUAL_OUTPUT_TEMPLATES:
        raise ValueError(f"Unknown output type: {output_type}")
    return TEXTUAL_OUTPUT_TEMPLATES[output_type]


def get_output_metadata(output_type: str) -> OutputTypeMetadata:
    """Get metadata for an output type."""
    if output_type not in OUTPUT_TYPE_METADATA:
        raise ValueError(f"Unknown output type: {output_type}")
    return OUTPUT_TYPE_METADATA[output_type]


def format_template(
    output_type: str,
    analysis_data: dict[str, Any],
    visual_summary: str | None = None
) -> str:
    """Format a template with analysis data."""
    template = get_template(output_type)

    # Convert analysis data to formatted string
    import json
    analysis_str = json.dumps(analysis_data, indent=2, default=str)

    visual_str = visual_summary or "No visual output available for this analysis."

    return template.format(
        analysis_data=analysis_str,
        visual_summary=visual_str
    )
