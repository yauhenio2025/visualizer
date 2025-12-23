# Engine-Output Ontology

**The fundamental separation of concerns in document intelligence.**

---

## Core Distinction

```
ENGINE (What to Extract/Analyze)     OUTPUT FORMAT (How to Present)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Analytical operation that           Presentation style that
transforms input into               visualizes or articulates
structured extracted data           the extracted data
```

**The Equation:**
```
ANALYSIS = ENGINE(document) → EXTRACTED_DATA → OUTPUT_FORMAT → final_output
```

---

## I. Engines: Analytical Operations

Engines perform **extraction and analysis**. They take documents as input and produce **structured data** as output.

### Engine Categories & Data Types Produced

| Category | Engine | Data Type Produced |
|----------|--------|-------------------|
| **DETECTION & WARNING** | | |
| | `signal_sentinel` | `{signals[], indicators[], threshold_status}` |
| | `anomaly_detector` | `{anomalies[], baseline_deviation, significance}` |
| | `temporal_discontinuity_finder` | `{discontinuities[], breaks[], pattern_changes}` |
| | `emerging_trend_detector` | `{trends[], velocity, trajectory}` |
| | `escalation_trajectory_analysis` | `{stages[], triggers[], thresholds}` |
| | `surely_alarm` | `{certainties[], overconfidence_markers}` |
| **EVIDENCE & VERIFICATION** | | |
| | `hypothesis_tournament` | `{hypotheses[], evidence_mapping[], rankings}` |
| | `evidence_quality_assessment` | `{evidence_items[], quality_scores, reliability}` |
| | `epistemic_calibration` | `{claims[], confidence_levels, uncertainty_map}` |
| | `provenance_audit` | `{sources[], chains_of_custody, attributions}` |
| | `statistical_evidence` | `{statistics[], methods, validity_assessment}` |
| | `authenticity_forensics` | `{markers[], red_flags, authenticity_score}` |
| **ACTOR & NETWORK** | | |
| | `stakeholder_power_interest` | `{actors[], power_scores, interest_positions}` |
| | `rational_actor_modeling` | `{actors[], incentives[], predicted_behaviors}` |
| | `relational_topology` | `{nodes[], edges[], network_metrics}` |
| | `quote_attribution_voice` | `{quotes[], speakers[], voice_analysis}` |
| | `resource_flow_asymmetry` | `{flows[], asymmetries[], dependencies}` |
| | `deal_flow_tracker` | `{deals[], parties[], terms[], timeline}` |
| **TEMPORAL & EVOLUTION** | | |
| | `event_timeline_causal` | `{events[], causal_links[], timeline}` |
| | `concept_evolution` | `{concepts[], versions[], transformations}` |
| | `chronology_cycle` | `{cycles[], periods[], recurrences}` |
| | `chronology_simultaneity` | `{concurrent_events[], synchronicities}` |
| | `reception_history` | `{interpretations[], shifts[], audiences}` |
| | `temporal_multiscale` | `{short_term[], medium_term[], long_term[]}` |
| **ARGUMENT & LOGIC** | | |
| | `argument_architecture` | `{claims[], premises[], logical_structure}` |
| | `assumption_excavation` | `{assumptions[], hidden_premises[], dependencies}` |
| | `steelman_generator` | `{strongest_versions[], improvements[]}` |
| | `steelman_stress_test` | `{vulnerabilities[], attack_vectors[]}` |
| | `boom_crutch_finder` | `{dependencies[], single_points_of_failure}` |
| | `dialectical_structure` | `{thesis[], antithesis[], synthesis[]}` |
| **COMPARISON & FRAMEWORK** | | |
| | `comparative_framework` | `{items[], dimensions[], comparisons}` |
| | `possibility_space_explorer` | `{options[], constraints[], tradeoffs}` |
| | `competitive_landscape` | `{competitors[], positions[], dynamics}` |
| | `opportunity_vulnerability_matrix` | `{opportunities[], vulnerabilities[], priorities}` |
| **THEMATIC & CONCEPTUAL** | | |
| | `thematic_synthesis` | `{themes[], evidence[], synthesis}` |
| | `conceptual_framework_extraction` | `{concepts[], relationships[], framework}` |
| | `structural_pattern_detector` | `{patterns[], instances[], significance}` |
| | `interdisciplinary_connection` | `{bridges[], analogies[], transfers}` |
| | `jootsing_analyzer` | `{rules[], violations[], innovations}` |
| **RHETORICAL & LINGUISTIC** | | |
| | `rhetorical_strategy` | `{strategies[], techniques[], effects}` |
| | `occams_broom` | `{omissions[], swept_under[], implications}` |
| | `deepity_detector` | `{deepities[], true_but_trivial[], profound_claims}` |
| | `influence_attribution_analysis` | `{influence_patterns[], attributions[]}` |
| | `terra_incognita_mapper` | `{unknowns[], gaps[], blind_spots}` |

### What Makes a Good Engine

1. **Pure analytical operation** - extracts/transforms, doesn't present
2. **Produces structured data** - JSON-like output with defined schema
3. **Domain-agnostic** - works on any document type
4. **Composable** - output can feed into multiple presentation formats

### Anti-pattern: Engines That Are Really Formats

Watch for these conflations:

| Looks Like Engine | Actually Is |
|-------------------|-------------|
| `generate_q_and_a` | OUTPUT FORMAT (presents any data as Q&A) |
| `create_executive_summary` | OUTPUT FORMAT (compresses any data) |
| `build_outline` | OUTPUT FORMAT (structures any data hierarchically) |
| `write_narrative` | OUTPUT FORMAT (linearizes any data) |

---

## II. Output Formats: Presentation Styles

Output formats take **extracted data** and render it for human consumption.

### A. Visual Formats (Gemini/Image)

Each visual format has specific **data type affinity** - what kinds of extracted data it visualizes best.

| Visual Format | Best For Data Types | Example Use |
|---------------|---------------------|-------------|
| **RELATIONAL** | | |
| `network_graph` | `{nodes[], edges[]}` | Actor networks, concept relationships |
| `chord_diagram` | `{entities[], flows_between[]}` | Mutual dependencies, bilateral flows |
| `hierarchical_tree` | `{nodes[], parent_child[]}` | Org charts, taxonomies, outlines |
| `radial_tree` | `{center, branches[]}` | Concept maps, influence radiating out |
| **FLOW & PROCESS** | | |
| `sankey` | `{sources[], targets[], values[]}` | Resource flows, conversions, funnels |
| `alluvial` | `{stages[], flows[], time_points}` | Evolution over time, categorical shifts |
| `flowchart` | `{steps[], decisions[], paths[]}` | Processes, decision trees |
| `sequence_diagram` | `{actors[], messages[], timeline}` | Interactions over time |
| **TEMPORAL** | | |
| `timeline` | `{events[], dates[], durations}` | Event sequences, history |
| `gantt` | `{tasks[], starts[], ends[], dependencies}` | Project timelines, overlapping periods |
| `sparklines` | `{series[], time_points[]}` | Trend indicators, small multiples |
| **COMPARATIVE** | | |
| `matrix/heatmap` | `{rows[], columns[], values[][]}` | Comparisons, ACH, correlations |
| `quadrant_chart` | `{items[], x_score, y_score}` | Power-interest, impact-effort |
| `bar_chart` | `{categories[], values[]}` | Ranked comparisons |
| `radar/spider` | `{entities[], dimensions[], scores}` | Multi-dimensional comparison |
| **SPATIAL/SET** | | |
| `venn_diagram` | `{sets[], overlaps[]}` | Set relationships, shared attributes |
| `treemap` | `{categories[], sizes[], hierarchy}` | Hierarchical part-of-whole |
| `geographic_map` | `{locations[], attributes[]}` | Spatial distribution |
| **PART-OF-WHOLE** | | |
| `pie/donut` | `{categories[], proportions[]}` | Composition (use sparingly) |
| `stacked_bar` | `{categories[], components[]}` | Composition over categories |
| `waterfall` | `{start, additions[], subtractions[], end}` | Cumulative changes |

### B. Textual Formats (Our 8 Types)

| Format | Core Question | Data Affinity |
|--------|--------------|---------------|
| `snapshot` | "What do I need to know NOW?" | Any - compresses to essentials |
| `deep_dive` | "What's the full picture?" | Complex multi-faceted data |
| `evidence_pack` | "What's the evidence chain?" | Source-heavy data with citations |
| `signal_report` | "What signals indicate change?" | Detection/warning data |
| `status_brief` | "What changed and where are we?" | Temporal/evolution data |
| `stakeholder_profile` | "Who is this and how will they act?" | Actor/network data |
| `gap_analysis` | "Where are the weaknesses?" | Argument/logic data |
| `options_brief` | "What should I choose?" | Comparison data |

### C. Structured Formats (Tables, Lists, Diagrams)

| Format | Structure | Best For |
|--------|-----------|----------|
| `matrix_table` | Rows × Columns with values | Comparisons, ACH, evaluations |
| `smart_table` | Dynamic columns based on data | Any structured extraction |
| `ranked_list` | Ordered items with scores | Prioritized findings |
| `hierarchy_list` | Nested bullet points | Taxonomies, outlines |
| `q_and_a` | Question-Answer pairs | Key points as interrogatives |
| `timeline_table` | Date | Event | Significance | Temporal sequences |
| `evidence_table` | Claim | Source | Reliability | Evidence documentation |

### D. Diagram Formats (Mermaid, D3)

| Format | Mermaid Type | Best For |
|--------|--------------|----------|
| `mermaid_flowchart` | flowchart | Processes, decisions |
| `mermaid_sequence` | sequenceDiagram | Actor interactions over time |
| `mermaid_class` | classDiagram | Taxonomies, type hierarchies |
| `mermaid_state` | stateDiagram | State machines, status flows |
| `mermaid_er` | erDiagram | Entity relationships |
| `mermaid_gantt` | gantt | Project timelines |
| `mermaid_mindmap` | mindmap | Concept clustering |
| `d3_force` | force-directed | Networks with physics |
| `d3_tree` | tree layout | Hierarchies |
| `d3_sankey` | sankey | Flows with interactivity |

---

## III. Engine → Output Affinity Matrix

**Legend:** 3 = Ideal match, 2 = Good fit, 1 = Possible, blank = Poor fit

### Detection & Warning Engines

| Engine | snapshot | signal | status | deep | gap | Visual Best Fit |
|--------|----------|--------|--------|------|-----|-----------------|
| `signal_sentinel` | 3 | 3 | 2 | | | sparklines, dashboard |
| `anomaly_detector` | 3 | 3 | | 2 | | heatmap, scatter |
| `emerging_trend_detector` | 3 | 3 | | | | timeline, sparklines |
| `escalation_trajectory` | 2 | 3 | 2 | | | flowchart, timeline |

### Evidence & Verification Engines

| Engine | evidence | deep | snapshot | Visual Best Fit |
|--------|----------|------|----------|-----------------|
| `evidence_quality` | 3 | 3 | | matrix/heatmap, radar |
| `hypothesis_tournament` | 2 | 3 | | ACH matrix |
| `provenance_audit` | 3 | 2 | | sankey (source flows) |
| `statistical_evidence` | 3 | 3 | | bar charts, distributions |

### Actor & Network Engines

| Engine | stakeholder | status | deep | Visual Best Fit |
|--------|-------------|--------|------|-----------------|
| `stakeholder_power_interest` | 3 | 2 | 2 | quadrant, network |
| `relational_topology` | 3 | | 3 | network graph, chord |
| `resource_flow_asymmetry` | 2 | 3 | 2 | sankey, alluvial |
| `quote_attribution` | 2 | | 2 | evidence table |

### Temporal & Evolution Engines

| Engine | status | deep | signal | Visual Best Fit |
|--------|--------|------|--------|-----------------|
| `event_timeline_causal` | 3 | 2 | | timeline, sequence |
| `concept_evolution` | 2 | 3 | | alluvial, timeline |
| `chronology_cycle` | 2 | 3 | | circular timeline |
| `temporal_multiscale` | 2 | 3 | | nested timelines |

### Argument & Logic Engines

| Engine | gap | deep | options | Visual Best Fit |
|--------|-----|------|---------|-----------------|
| `argument_architecture` | 3 | 2 | | tree, flowchart |
| `assumption_excavation` | 3 | 2 | | hierarchy list |
| `steelman_stress_test` | 3 | | 2 | vulnerability matrix |
| `dialectical_structure` | 2 | 3 | 2 | thesis-antithesis-synthesis flow |

### Comparison & Framework Engines

| Engine | options | deep | gap | Visual Best Fit |
|--------|---------|------|-----|-----------------|
| `comparative_framework` | 3 | 2 | | matrix, radar |
| `possibility_space_explorer` | 3 | 2 | | quadrant, decision tree |
| `competitive_landscape` | 3 | | 2 | positioning map |
| `opportunity_vulnerability` | 2 | | 3 | 2×2 matrix |

---

## IV. Resource Flow Example: Multiple Visual Representations

Your insight about resource flows having multiple visualizations:

**Engine:** `resource_flow_asymmetry`
**Extracted Data Type:** `{sources[], sinks[], flows[], values[], asymmetries[]}`

**Possible Output Formats:**

| Format | When to Use | What It Shows |
|--------|-------------|---------------|
| `sankey` | Showing flow volumes | How much goes where |
| `chord_diagram` | Showing mutual flows | Who exchanges with whom |
| `alluvial` | Showing change over time | How flows shifted |
| `network_graph` | Showing dependencies | Who depends on whom |
| `matrix_heatmap` | Showing all pairs | Complete flow picture |
| `geographic_map` | Showing spatial flows | Where resources move |
| `waterfall` | Showing cumulative effect | Net position changes |

**The engine extracts the same data; the format reveals different aspects.**

---

## V. Cleaning Up Current Conflations

### In Analyzer API Output Modes

| Current Mode | Classification | Recommendation |
|--------------|----------------|----------------|
| `gemini_image` | FORMAT: Visual | Keep - generic visual |
| `smart_table` | FORMAT: Structured | Keep |
| `mermaid` | FORMAT: Diagram | Keep |
| `structured_text_report` | FORMAT: Textual | → Rename to `narrative_report` |
| `executive_memo` | FORMAT: Textual | → Merge into `snapshot` |
| `research_report` | FORMAT: Textual | → Merge into `deep_dive` |
| `text_qna` | FORMAT: Structured | Keep as Q&A format |
| `table` | FORMAT: Structured | Redundant with smart_table |

### Engines That Need Review

| Engine | Issue | Resolution |
|--------|-------|------------|
| None found | - | Current engines appear to be pure analytical operations |

---

## VI. Recommended Output Format Expansion

### New Visual Formats to Add

| Format | For Data Type | Use Case |
|--------|---------------|----------|
| `chord_diagram` | Bilateral flows | Trade, communication, mutual dependencies |
| `alluvial` | Categorical changes | How entities shift categories over time |
| `radar_chart` | Multi-dimensional scores | Comparing entities across many dimensions |
| `treemap` | Hierarchical sizes | Showing nested proportions |
| `bubble_chart` | 3-dimensional comparison | Size + position encoding |
| `arc_diagram` | Sequential relationships | Narrative connections |

### New Textual Formats to Add

| Format | Core Question | For |
|--------|--------------|-----|
| `warning_bulletin` | "What requires immediate attention?" | Urgent findings, escalation |
| `collection_requirements` | "What do we still need to know?" | Information gaps |
| `red_team_brief` | "How could this be wrong?" | Devil's advocate analysis |

### New Structured Formats to Add

| Format | Structure | For |
|--------|-----------|-----|
| `ach_matrix` | Hypotheses × Evidence | Competing hypotheses analysis |
| `indicator_dashboard` | Indicators with status | Warning system tracking |
| `source_reliability_table` | Sources with ratings | Evidence quality tracking |

---

## VII. Implementation Priority

### Phase 1: Clean Ontology (Now)
- [x] Document engine vs. output distinction
- [ ] Audit all engines for purity
- [ ] Remove conflated "engines" that are formats
- [ ] Update ENGINE_OUTPUT_AFFINITY with all engines

### Phase 2: Expand Visual Repertoire
- [ ] Add chord_diagram for bilateral flows
- [ ] Add alluvial for temporal categorical shifts
- [ ] Add radar_chart for multi-dimensional comparison
- [ ] Create visual selection logic based on data type

### Phase 3: Intelligent Format Selection
- [ ] Detect data type from engine output
- [ ] Recommend optimal formats automatically
- [ ] Allow user override with explanation
- [ ] Learn from user preferences

---

## VIII. The Golden Rule

**Engine developers:** Focus on WHAT to extract. Define your output schema. Don't think about presentation.

**Format developers:** Focus on HOW to present. Accept any data matching your input schema. Create beautiful, clear output.

**The pipeline:** Engine → Structured Data → Format Selection → Output

This separation enables:
- Any engine output can go to any compatible format
- New engines automatically work with existing formats
- New formats automatically work with existing engines
- Users can choose how to view the same analysis
