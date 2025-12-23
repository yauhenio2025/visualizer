# Complete Engine-Output Ontology

**70 Engines • 21 Pipelines • 18 Bundles • Complete Output Format Repertoire**

---

## The Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           TWO-CURATOR SYSTEM                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  INPUT/COLLECTION                                                           │
│        │                                                                    │
│        ▼                                                                    │
│  ┌──────────────────┐                                                       │
│  │  ENGINE CURATOR  │  "Given this input and user intent,                   │
│  │                  │   which analytical operations will                    │
│  │  (LLM-powered)   │   extract the most valuable insights?"                │
│  └────────┬─────────┘                                                       │
│           │ selects engines/bundles/pipelines                               │
│           ▼                                                                 │
│  ┌──────────────────┐                                                       │
│  │  ANALYSIS PHASE  │  Engines execute, produce structured data             │
│  │  (70 engines)    │  Each engine has defined output schema                │
│  └────────┬─────────┘                                                       │
│           │ extracted_data (JSON schemas)                                   │
│           ▼                                                                 │
│  ┌──────────────────┐                                                       │
│  │  OUTPUT CURATOR  │  "Given this extracted data type,                     │
│  │                  │   which presentation formats will                     │
│  │  (LLM-powered)   │   best communicate the insights?"                     │
│  └────────┬─────────┘                                                       │
│           │ selects output formats                                          │
│           ▼                                                                 │
│  ┌──────────────────┐                                                       │
│  │  RENDER PHASE    │  Visual, Textual, Structured outputs                  │
│  │  (formats)       │  Multiple formats from same data                      │
│  └──────────────────┘                                                       │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## I. Complete Engine Catalog (70 Engines)

### Category: ARGUMENT (9 engines)

| Engine | Description | Output Data Type | Best Output Formats |
|--------|-------------|------------------|---------------------|
| `argument_architecture` | Maps logical structure of arguments | `{claims[], premises[], warrants[], logical_structure}` | tree diagram, flowchart, deep_dive |
| `assumption_excavation` | Excavates hidden premises and unstated assumptions | `{assumptions[], hidden_premises[], dependencies[]}` | hierarchy list, gap_analysis |
| `dialectical_structure` | Maps thesis/antithesis/synthesis patterns | `{thesis[], antithesis[], synthesis[], tensions[]}` | flow diagram, deep_dive |
| `evidence_quality_assessment` | Evaluates quality of evidence | `{evidence_items[], quality_scores[], reliability_ratings[]}` | matrix table, evidence_pack, radar chart |
| `steelman_generator` | Generates strongest version of arguments | `{original[], steelmanned[], improvements[]}` | comparison table, options_brief |
| `steelman_stress_test` | Subjects arguments to adversarial testing | `{vulnerabilities[], attack_vectors[], resilience_scores[]}` | gap_analysis, vulnerability matrix |
| `hypothesis_tournament` | Implements Analysis of Competing Hypotheses | `{hypotheses[], evidence_mapping[], consistency_matrix}` | ACH matrix, heatmap, deep_dive |
| `contrarian_concept_generation` | Generates counter-positions | `{orthodox_positions[], contrarian_alternatives[], rationales[]}` | comparison table, options_brief |
| `possibility_space_explorer` | Scenario planning for futures analysis | `{scenarios[], probabilities[], drivers[], wild_cards[]}` | scenario cone, decision tree, options_brief |

### Category: RHETORIC (9 engines)

| Engine | Description | Output Data Type | Best Output Formats |
|--------|-------------|------------------|---------------------|
| `surely_alarm` | Detects rhetorical confidence markers | `{markers[], locations[], overconfidence_score}` | annotated text, snapshot, gap_analysis |
| `deepity_detector` | Identifies trivial-profound statements | `{deepities[], true_but_trivial[], profound_claims[]}` | list with examples, gap_analysis |
| `rhetorical_strategy` | Analyzes persuasion techniques | `{strategies[], techniques[], aristotelian_appeals[]}` | breakdown chart, deep_dive |
| `authenticity_forensics` | Detects deception and manipulation | `{markers[], red_flags[], authenticity_score}` | signal_report, dashboard |
| `influence_attribution_analysis` | Traces influence operations | `{campaigns[], attributions[], coordination_markers[]}` | network graph, timeline, deep_dive |
| `metaphor_analogy_network` | Maps metaphor usage and analogies | `{metaphors[], source_domains[], target_domains[], network}` | network graph, concept map |
| `quote_attribution_voice` | Extracts quotes with voice analysis | `{quotes[], speakers[], voice_profiles[], sentiments[]}` | evidence_pack, stakeholder_profile |
| `cultural_landmark_translation` | Identifies culturally-specific references | `{references[], cultural_contexts[], translations[]}` | annotated list, deep_dive |
| `narrative_transformation` | Analyzes using narratological frameworks | `{narrative_elements[], transformations[], story_structure}` | flow diagram, timeline |

### Category: EPISTEMOLOGY (11 engines)

| Engine | Description | Output Data Type | Best Output Formats |
|--------|-------------|------------------|---------------------|
| `occams_broom` | Identifies what's swept under the rug | `{omissions[], inconvenient_facts[], implications[]}` | gap_analysis, annotated list |
| `provenance_audit` | Maps sources and epistemic quality | `{sources[], chains[], reliability_ratings[]}` | evidence_pack, source table |
| `epistemic_calibration` | Assigns calibrated certainty levels | `{claims[], confidence_levels[], uncertainty_map}` | confidence matrix, deep_dive |
| `terra_incognita_mapper` | Maps what we don't know | `{unknowns[], gaps[], blind_spots[], priorities[]}` | gap_analysis, treemap |
| `key_intelligence_questions_mapper` | Maps findings to KIQs | `{questions[], findings_mapped[], coverage_gaps[]}` | matrix table, status_brief |
| `philosophers_syndrome_detector` | Detects failure of imagination as proof | `{instances[], logical_errors[], corrections[]}` | annotated list, gap_analysis |
| `epistemic_stance` | Maps certainty/uncertainty markers | `{hedging[], boosting[], epistemic_markers[]}` | annotated text, bar chart |
| `absent_center` | Detects strategic silences | `{silences[], gaps[], implications[]}` | gap_analysis, deep_dive |
| `methodological_approach` | Analyzes research methods | `{methods[], designs[], validity_assessments[]}` | comparison table, deep_dive |
| `conditions_of_possibility` | Maps preconditions for discourses | `{conditions[], enablers[], constraints[]}` | hierarchy diagram, deep_dive |
| `explanatory_pattern_analysis` | Maps explanatory patterns | `{patterns[], causal_types[], coverage[]}` | taxonomy tree, matrix |

### Category: CONCEPTS (7 engines)

| Engine | Description | Output Data Type | Best Output Formats |
|--------|-------------|------------------|---------------------|
| `jootsing_analyzer` | Identifies implicit rules and boundaries | `{rules[], constraints[], boundary_conditions[]}` | boundary diagram, deep_dive |
| `boundary_probe` | Applies Sortes-style concept analysis | `{concepts[], edge_cases[], fuzzy_boundaries[]}` | spectrum diagram, deep_dive |
| `thematic_synthesis` | Identifies recurring themes | `{themes[], evidence[], synthesis}` | theme cloud, deep_dive, snapshot |
| `conceptual_framework_extraction` | Extracts analytical frameworks | `{frameworks[], components[], relationships[]}` | hierarchy tree, concept map |
| `structural_pattern_detector` | Detects structural patterns | `{patterns[], instances[], frequencies[]}` | pattern matrix, deep_dive |
| `conceptual_anomaly_detector` | Finds conceptual outliers | `{anomalies[], deviations[], significance[]}` | scatter plot, signal_report |
| `composition_taxonomy` | Categorizes types within domain | `{taxonomy[], categories[], exemplars[]}` | tree diagram, taxonomy table |

### Category: TEMPORAL (10 engines)

| Engine | Description | Output Data Type | Best Output Formats |
|--------|-------------|------------------|---------------------|
| `signal_sentinel` | Implements I&W for early warning | `{signals[], indicators[], threshold_status[]}` | dashboard, signal_report, sparklines |
| `temporal_discontinuity_finder` | Detects timeline anomalies | `{discontinuities[], gaps[], pattern_breaks[]}` | timeline with flags, signal_report |
| `escalation_trajectory_analysis` | Models escalation patterns | `{stages[], triggers[], thresholds[], trajectory}` | escalation ladder, flowchart |
| `event_timeline_causal` | Maps events with causal links | `{events[], causal_links[], timeline}` | timeline, sankey, status_brief |
| `concept_evolution` | Tracks concept evolution | `{concepts[], versions[], transformations[]}` | alluvial diagram, timeline |
| `intellectual_genealogy` | Traces intellectual lineages | `{thinkers[], influences[], lineages[]}` | genealogy tree, network |
| `reception_history` | Traces how ideas were received | `{interpretations[], shifts[], audiences[]}` | timeline, deep_dive |
| `temporal_multiscale` | Analyzes at micro/meso/macro scales | `{short_term[], medium_term[], long_term[]}` | nested timeline, deep_dive |
| `chronology_cycle` | Maps boom-bust and recurring patterns | `{cycles[], periods[], recurrences[]}` | cycle diagram, timeline |
| `chronology_simultaneity` | Maps overlapping timelines | `{parallel_events[], synchronicities[], contradictions[]}` | parallel timeline, gantt |
| `emerging_trend_detector` | Identifies emerging trends | `{trends[], velocity[], trajectory[]}` | trend chart, signal_report |

### Category: POWER (7 engines)

| Engine | Description | Output Data Type | Best Output Formats |
|--------|-------------|------------------|---------------------|
| `stakeholder_power_interest` | Maps stakeholders by power/interest | `{actors[], power_scores[], interest_positions[]}` | quadrant chart, network, stakeholder_profile |
| `relational_topology` | Graph analysis of actor networks | `{nodes[], edges[], centrality_metrics[]}` | network graph, chord diagram |
| `rational_actor_modeling` | Profiles decision-makers | `{actors[], incentives[], predicted_behaviors[]}` | stakeholder_profile, options_brief |
| `resource_flow_asymmetry` | Maps resource flows and dependencies | `{flows[], asymmetries[], dependencies[]}` | sankey, alluvial, chord diagram |
| `opportunity_vulnerability_matrix` | Identifies exploitable vulnerabilities | `{opportunities[], vulnerabilities[], priorities[]}` | 2x2 matrix, gap_analysis |
| `power_interest_subtext` | Uncovers hidden power dynamics | `{implicit_power[], hierarchies[], tensions[]}` | network graph, deep_dive |
| `deal_flow_tracker` | Tracks M&A, partnerships | `{deals[], parties[], terms[], timeline}` | timeline, table, sankey |

### Category: EVIDENCE (5 engines)

| Engine | Description | Output Data Type | Best Output Formats |
|--------|-------------|------------------|---------------------|
| `exemplar_catalog` | Catalogs concrete examples | `{examples[], categories[], use_cases[]}` | gallery, evidence_pack |
| `comparative_framework` | Builds comparison frameworks | `{entities[], dimensions[], comparisons[]}` | matrix table, radar chart |
| `statistical_evidence` | Catalogs quantitative data | `{statistics[], methods[], validity[]}` | data table, bar charts, evidence_pack |
| `entity_extraction` | Extracts named entities | `{entities[], types[], relationships[]}` | entity list, network graph |
| `cross_cultural_variation` | Analyzes cultural differences | `{variations[], contexts[], patterns[]}` | comparison matrix, deep_dive |

### Category: SCHOLARLY (5 engines)

| Engine | Description | Output Data Type | Best Output Formats |
|--------|-------------|------------------|---------------------|
| `scholarly_debate_map` | Maps intellectual debates | `{debates[], positions[], scholars[], dynamics}` | network graph, deep_dive |
| `citation_network` | Maps citation relationships | `{citations[], hubs[], clusters[]}` | network graph, chord diagram |
| `interdisciplinary_connection` | Maps cross-disciplinary links | `{bridges[], analogies[], transfers[]}` | network graph, deep_dive |
| `literature_gap_identifier` | Finds knowledge gaps | `{gaps[], significance[], opportunities[]}` | gap_analysis, treemap |
| `science_studies_network` | Actor-Network Theory analysis | `{actors[], networks[], assemblages[]}` | network graph, deep_dive |

### Category: MARKET (4 engines)

| Engine | Description | Output Data Type | Best Output Formats |
|--------|-------------|------------------|---------------------|
| `regulatory_pulse` | Tracks policy changes | `{regulations[], changes[], impacts[]}` | timeline, status_brief |
| `competitive_landscape` | Maps competitive positioning | `{competitors[], positions[], dynamics[]}` | positioning map, quadrant |
| `composition_anatomy` | Decomposes legislation/policy | `{components[], provisions[], implications[]}` | hierarchy tree, deep_dive |
| `pedagogical_pathway` | Extracts learning structures | `{prerequisites[], sequences[], dependencies[]}` | pathway diagram, hierarchy |
| `value_ethical_framework` | Extracts values and ethics | `{values[], positions[], tensions[]}` | radar chart, deep_dive |

---

## II. Complete Pipeline Catalog (21 Pipelines)

### Tier 1: Foundation → Enhancement (6 pipelines)

| Pipeline | Stages | Output Progression |
|----------|--------|-------------------|
| `argument_evidence_audit` | argument_architecture → evidence_quality_assessment | logical structure → evidence quality |
| `argument_assumption_excavation` | argument_architecture → assumption_excavation | logical structure → hidden premises |
| `voice_to_power` | quote_attribution_voice → stakeholder_power_interest | who said what → who has power |
| `statistics_quality_audit` | statistical_evidence → evidence_quality_assessment | numbers → reliability |
| `themes_to_evolution` | thematic_synthesis → concept_evolution | what themes → how they changed |
| `examples_to_comparison` | exemplar_catalog → comparative_framework | examples → systematic comparison |

### Tier 2: Extraction → Relational (4 pipelines)

| Pipeline | Stages | Output Progression |
|----------|--------|-------------------|
| `citations_to_genealogy` | citation_network → intellectual_genealogy | who cites whom → intellectual lineage |
| `citations_to_debates` | citation_network → scholarly_debate_map | citations → debates |
| `frameworks_across_disciplines` | conceptual_framework_extraction → interdisciplinary_connection | frameworks → cross-disciplinary bridges |
| `quotes_to_rhetoric` | quote_attribution_voice → rhetorical_strategy | quotes → persuasion analysis |

### Tier 3: Critique → Generation (5 pipelines)

| Pipeline | Stages | Output Progression |
|----------|--------|-------------------|
| `absence_to_contrarian` | absent_center → contrarian_concept_generation | what's missing → alternative views |
| `gaps_to_trends` | literature_gap_identifier → emerging_trend_detector | gaps → emerging directions |
| `assumptions_to_contrarian` | assumption_excavation → contrarian_concept_generation | assumptions → challenges |
| `source_to_confidence` | provenance_audit → epistemic_calibration → terra_incognita_mapper | sources → confidence → gaps |
| `warning_assessment_complete` | signal_sentinel → temporal_discontinuity_finder → possibility_space_explorer | signals → anomalies → scenarios |

### Tier 4: Multi-Stage Deep (6 pipelines)

| Pipeline | Stages | Output Progression |
|----------|--------|-------------------|
| `deep_argument_analysis` | argument_architecture → assumption_excavation → absent_center → contrarian_concept_generation | full argument deconstruction |
| `evidence_audit_complete` | quote_attribution_voice → statistical_evidence → evidence_quality_assessment → epistemic_stance | full evidence audit |
| `intellectual_history_complete` | citation_network → intellectual_genealogy → reception_history → concept_evolution | full intellectual history |
| `stakeholder_deep_analysis` | quote_attribution_voice → stakeholder_power_interest → resource_flow_asymmetry → power_interest_subtext | full stakeholder analysis |
| `dennett_diagnostic` | surely_alarm → boom_crutch_finder → deepity_detector → occams_broom | Dennett's critical toolkit |
| `epistemic_stress_test` | argument_architecture → steelman_generator → philosophers_syndrome_detector → epistemic_calibration | full epistemic stress test |

---

## III. Complete Bundle Catalog (18 Bundles)

| Bundle | Engines | Use Case |
|--------|---------|----------|
| `dennett_toolkit` | surely_alarm, occams_broom, boom_crutch_finder, deepity_detector | Critical thinking audit |
| `epistemic_rigor_suite` | provenance_audit, epistemic_calibration, hypothesis_tournament, steelman_generator | Evidence & confidence analysis |
| `strategic_warning` | signal_sentinel, possibility_space_explorer, temporal_discontinuity_finder | Early warning & futures |
| `network_intelligence` | stakeholder_power_interest, relational_topology, rational_actor_modeling | Actor network mapping |
| `persuasion_archaeology` | surely_alarm, deepity_detector, rhetorical_strategy, authenticity_forensics | Rhetoric deconstruction |
| `argument_forensics` | argument_architecture, assumption_excavation, dialectical_structure, evidence_quality_assessment, contrarian_concept_generation | Argument analysis |
| `intellectual_landscape` | scholarly_debate_map, concept_evolution, thematic_synthesis, conceptual_framework_extraction, citation_network | Academic mapping |
| `power_politics` | stakeholder_power_interest, resource_flow_asymmetry, event_timeline_causal, power_interest_subtext | Power dynamics |
| `empirical_foundation` | evidence_quality_assessment, exemplar_catalog, comparative_framework, statistical_evidence | Evidence base |
| `conceptual_architecture` | concept_evolution, conceptual_framework_extraction, structural_pattern_detector, exemplar_catalog, composition_taxonomy | Concept mapping |
| `epistemology_critique` | epistemic_stance, evidence_quality_assessment, methodological_approach, absent_center | Epistemological audit |
| `rhetorical_analysis` | rhetorical_strategy, metaphor_analogy_network, quote_attribution_voice | Rhetoric analysis |
| `temporal_genealogy` | event_timeline_causal, intellectual_genealogy, concept_evolution, reception_history, chronology_cycle, temporal_multiscale | Historical analysis |
| `knowledge_frontier` | emerging_trend_detector, literature_gap_identifier, conceptual_anomaly_detector | Research frontiers |
| `market_policy_dynamics` | entity_extraction, deal_flow_tracker, regulatory_pulse, competitive_landscape | Market intelligence |
| `science_technology_studies` | science_studies_network, conditions_of_possibility, interdisciplinary_connection | STS analysis |
| `cultural_translation` | cross_cultural_variation, cultural_landmark_translation, narrative_transformation | Cultural analysis |
| `deep_methodology` | explanatory_pattern_analysis, pedagogical_pathway, value_ethical_framework | Methodological analysis |

---

## IV. Complete Output Format Repertoire

### A. Visual Formats

| Format Key | Type | Best For Data Types | When to Use |
|------------|------|---------------------|-------------|
| **RELATIONAL** |
| `network_graph` | Gemini | nodes[], edges[] | Actor relationships, citations, concepts |
| `chord_diagram` | Gemini | entities[], flows_between[] | Bilateral flows, mutual dependencies |
| `hierarchical_tree` | Gemini | nodes[], parent_child[] | Taxonomies, org charts, outlines |
| `radial_tree` | Gemini | center, branches[] | Concept maps, influence radii |
| **FLOW & PROCESS** |
| `sankey` | Gemini | sources[], targets[], values[] | Resource flows, conversions |
| `alluvial` | Gemini | stages[], flows[], time_points | Evolution over time, categorical shifts |
| `flowchart` | Mermaid | steps[], decisions[], paths[] | Processes, decision trees |
| `sequence_diagram` | Mermaid | actors[], messages[], timeline | Interactions over time |
| **TEMPORAL** |
| `timeline` | Gemini/Mermaid | events[], dates[], durations | Event sequences |
| `gantt` | Mermaid | tasks[], starts[], ends[] | Project timelines, overlaps |
| `sparklines` | D3 | series[], time_points[] | Trend indicators |
| **COMPARATIVE** |
| `matrix_heatmap` | Gemini | rows[], columns[], values[][] | Comparisons, ACH, correlations |
| `quadrant_chart` | Gemini | items[], x_score, y_score | Power-interest, impact-effort |
| `radar_chart` | Gemini | entities[], dimensions[], scores | Multi-dimensional comparison |
| `bar_chart` | Gemini | categories[], values[] | Ranked comparisons |
| **SPATIAL/SET** |
| `venn_diagram` | Gemini | sets[], overlaps[] | Set relationships |
| `treemap` | Gemini | categories[], sizes[], hierarchy | Part-of-whole, hierarchical |
| **SPECIALIZED** |
| `ach_matrix` | Table | hypotheses[], evidence[], consistency | Competing hypotheses |
| `escalation_ladder` | Gemini | stages[], thresholds[] | Escalation dynamics |
| `scenario_cone` | Gemini | futures[], probabilities[] | Scenario planning |

### B. Textual Formats (8 Types)

| Format Key | Core Question | Word Count | Best For |
|------------|--------------|------------|----------|
| `snapshot` | What do I need to know NOW? | ~400 | Any - executive summary |
| `deep_dive` | What's the full picture? | 2000-5000 | Complex multi-faceted analysis |
| `evidence_pack` | What's the evidence chain? | Variable | Source documentation |
| `signal_report` | What signals indicate change? | ~800 | Detection/warning data |
| `status_brief` | What changed and where are we? | ~1200 | Temporal/evolution data |
| `stakeholder_profile` | Who is this and how will they act? | ~1500/actor | Actor analysis |
| `gap_analysis` | Where are the weaknesses? | ~1500 | Argument/logic data |
| `options_brief` | What should I choose? | ~1200 | Comparison/decision data |

### C. Structured Formats

| Format Key | Structure | Best For |
|------------|-----------|----------|
| `smart_table` | Dynamic columns | Any structured extraction |
| `matrix_table` | Rows × Columns | Comparisons, evaluations |
| `evidence_table` | Claim, Source, Rating | Evidence documentation |
| `timeline_table` | Date, Event, Significance | Temporal sequences |
| `text_qna` | Question-Answer pairs | Key findings as interrogatives |
| `ranked_list` | Ordered items with scores | Prioritized findings |
| `hierarchy_list` | Nested bullet points | Taxonomies, outlines |

### D. Diagram Formats (Mermaid/D3)

| Format Key | Mermaid Type | Best For |
|------------|--------------|----------|
| `mermaid_flowchart` | flowchart | Processes, decisions |
| `mermaid_sequence` | sequenceDiagram | Actor interactions |
| `mermaid_class` | classDiagram | Taxonomies |
| `mermaid_state` | stateDiagram | State machines |
| `mermaid_er` | erDiagram | Entity relationships |
| `mermaid_gantt` | gantt | Timelines |
| `mermaid_mindmap` | mindmap | Concept clustering |
| `d3_force` | force-directed | Networks |
| `d3_sankey` | sankey | Interactive flows |

---

## V. Engine-to-Output Affinity Matrix (Complete)

### Legend
- **3** = Ideal match (engine output perfectly suited for format)
- **2** = Good fit (works well)
- **1** = Possible (can work but not optimal)
- **blank** = Poor fit (avoid)

### ARGUMENT Engines

| Engine | snapshot | deep_dive | gap_analysis | options_brief | evidence_pack | Best Visual |
|--------|----------|-----------|--------------|---------------|---------------|-------------|
| argument_architecture | 2 | 3 | 2 | | 2 | tree, flowchart |
| assumption_excavation | | 2 | 3 | | 2 | hierarchy list |
| dialectical_structure | | 3 | 2 | 2 | | flow diagram |
| evidence_quality_assessment | | 3 | 2 | | 3 | matrix, radar |
| steelman_generator | 2 | 2 | | 3 | | comparison table |
| steelman_stress_test | | 2 | 3 | 2 | | vulnerability matrix |
| hypothesis_tournament | | 3 | 2 | 2 | | ACH matrix, heatmap |
| contrarian_concept_generation | 2 | 2 | | 3 | | comparison table |
| possibility_space_explorer | 2 | 2 | | 3 | | scenario cone, tree |

### RHETORIC Engines

| Engine | snapshot | deep_dive | gap_analysis | signal_report | stakeholder | Best Visual |
|--------|----------|-----------|--------------|---------------|-------------|-------------|
| surely_alarm | 2 | | 3 | 2 | | annotated text |
| deepity_detector | | 2 | 3 | | | list with examples |
| rhetorical_strategy | | 3 | 2 | | | breakdown chart |
| authenticity_forensics | 2 | 2 | 2 | 3 | | dashboard |
| influence_attribution_analysis | | 3 | | 3 | | network, timeline |
| metaphor_analogy_network | | 3 | | | | network graph |
| quote_attribution_voice | | 2 | | | 3 | evidence table |
| cultural_landmark_translation | | 3 | | | | annotated list |
| narrative_transformation | | 3 | | | | flow diagram |

### EPISTEMOLOGY Engines

| Engine | snapshot | deep_dive | gap_analysis | evidence_pack | status_brief | Best Visual |
|--------|----------|-----------|--------------|---------------|--------------|-------------|
| occams_broom | | 2 | 3 | | | annotated list |
| provenance_audit | | 2 | | 3 | | sankey, table |
| epistemic_calibration | | 3 | | 2 | | confidence matrix |
| terra_incognita_mapper | | 2 | 3 | | | treemap |
| key_intelligence_questions | | | | | 3 | matrix table |
| philosophers_syndrome | | 2 | 3 | | | annotated list |
| epistemic_stance | | 3 | | | | annotated text |
| absent_center | | 3 | 3 | | | gap list |
| methodological_approach | | 3 | | | | comparison table |
| conditions_of_possibility | | 3 | | | | hierarchy |
| explanatory_pattern | | 3 | | | | taxonomy |

### TEMPORAL Engines

| Engine | snapshot | signal_report | status_brief | deep_dive | Best Visual |
|--------|----------|---------------|--------------|-----------|-------------|
| signal_sentinel | 3 | 3 | 2 | | dashboard, sparklines |
| temporal_discontinuity | 2 | 3 | 2 | | timeline with flags |
| escalation_trajectory | 2 | 3 | 2 | | escalation ladder |
| event_timeline_causal | | | 3 | 2 | timeline, sankey |
| concept_evolution | | | 2 | 3 | alluvial |
| intellectual_genealogy | | | 2 | 3 | genealogy tree |
| reception_history | | | 2 | 3 | timeline |
| temporal_multiscale | | | 2 | 3 | nested timeline |
| chronology_cycle | | 2 | 2 | 3 | cycle diagram |
| chronology_simultaneity | | | 3 | 2 | parallel timeline |
| emerging_trend | 3 | 3 | 2 | | trend chart |

### POWER Engines

| Engine | stakeholder | status_brief | deep_dive | options_brief | Best Visual |
|--------|-------------|--------------|-----------|---------------|-------------|
| stakeholder_power_interest | 3 | 2 | 2 | | quadrant, network |
| relational_topology | 3 | | 3 | | network, chord |
| rational_actor_modeling | 3 | | 2 | 2 | profile cards |
| resource_flow_asymmetry | 2 | 3 | 2 | | sankey, alluvial |
| opportunity_vulnerability | | | 2 | 3 | 2x2 matrix |
| power_interest_subtext | 3 | | 3 | | network |
| deal_flow_tracker | | 3 | | | timeline, sankey |

### CONCEPTS Engines

| Engine | snapshot | deep_dive | gap_analysis | Best Visual |
|--------|----------|-----------|--------------|-------------|
| jootsing_analyzer | | 3 | 2 | boundary diagram |
| boundary_probe | | 3 | | spectrum |
| thematic_synthesis | 3 | 3 | | theme cloud |
| conceptual_framework | | 3 | | hierarchy, concept map |
| structural_pattern | | 3 | | pattern matrix |
| conceptual_anomaly | 2 | 2 | | scatter plot |
| composition_taxonomy | | 3 | | tree diagram |

### EVIDENCE Engines

| Engine | evidence_pack | deep_dive | snapshot | Best Visual |
|--------|---------------|-----------|----------|-------------|
| exemplar_catalog | 3 | 2 | | gallery |
| comparative_framework | 2 | 2 | | matrix, radar |
| statistical_evidence | 3 | 2 | 2 | bar charts |
| entity_extraction | 2 | | 2 | entity list, network |
| cross_cultural_variation | | 3 | | comparison matrix |

### SCHOLARLY Engines

| Engine | deep_dive | evidence_pack | status_brief | Best Visual |
|--------|-----------|---------------|--------------|-------------|
| scholarly_debate_map | 3 | | | network |
| citation_network | 2 | 2 | | network, chord |
| interdisciplinary | 3 | | | bridge diagram |
| literature_gap | 2 | | 2 | gap_analysis, treemap |
| science_studies_network | 3 | | | ANT diagram |

### MARKET Engines

| Engine | status_brief | snapshot | deep_dive | Best Visual |
|--------|--------------|----------|-----------|-------------|
| regulatory_pulse | 3 | 2 | | timeline |
| competitive_landscape | 2 | 2 | 2 | positioning map |
| composition_anatomy | | | 3 | hierarchy |
| pedagogical_pathway | | | 3 | pathway |
| value_ethical_framework | | | 3 | radar |

---

## VI. Two-Curator System Design

### Curator 1: Engine Selection

**Input:** User intent + document collection characteristics
**Output:** Recommended engines/bundles/pipelines with rationale

```
CURATOR 1 DECISION MATRIX

User Intent (Verb)     Document Type        → Recommended Engines
─────────────────────────────────────────────────────────────────
MAP + ACTORS           Policy/Business      → stakeholder_power_interest, relational_topology
MAP + ARGUMENTS        Academic/Legal       → argument_architecture, dialectical_structure
TRACE + EVOLUTION      Historical/Academic  → concept_evolution, intellectual_genealogy
EVALUATE + EVIDENCE    Research/Legal       → evidence_quality_assessment, provenance_audit
DETECT + MANIPULATION  News/Social Media    → authenticity_forensics, influence_attribution
FIND + GAPS            Research/Strategy    → terra_incognita_mapper, literature_gap_identifier
COMPARE + OPTIONS      Strategy/Policy      → comparative_framework, possibility_space_explorer
CHALLENGE + ARGUMENTS  Any                  → steelman_stress_test, hypothesis_tournament
```

### Curator 2: Output Selection

**Input:** Extracted data type from engine + user context
**Output:** Recommended output formats with rationale

```
CURATOR 2 DECISION MATRIX

Data Type              Audience             → Recommended Formats
─────────────────────────────────────────────────────────────────
nodes[], edges[]       Analyst              → network_graph + deep_dive
nodes[], edges[]       Executive            → quadrant_chart + snapshot
flows[], values[]      Financial            → sankey + status_brief
events[], timeline     Operations           → timeline + status_brief
claims[], evidence[]   Legal                → evidence_pack + matrix_table
hypotheses[], matrix   Intelligence         → ACH_matrix + deep_dive
actors[], behaviors[]  Strategy             → stakeholder_profile + quadrant
vulnerabilities[]      Security             → gap_analysis + vulnerability_matrix
scenarios[], futures[] Executive            → scenario_cone + options_brief
```

### System Integration

```python
async def analyze_with_curators(
    documents: list[Document],
    user_intent: str,
    audience: str = "analyst"
) -> AnalysisResult:

    # Phase 1: Engine Curator selects analytical operations
    engine_selection = await engine_curator.select(
        intent=user_intent,
        document_characteristics=analyze_documents(documents),
        available_engines=ALL_70_ENGINES,
        available_bundles=ALL_18_BUNDLES,
        available_pipelines=ALL_21_PIPELINES
    )

    # Phase 2: Execute analysis
    extracted_data = await execute_engines(
        documents=documents,
        engines=engine_selection.engines
    )

    # Phase 3: Output Curator selects presentation formats
    output_selection = await output_curator.select(
        data_types=extracted_data.schemas,
        audience=audience,
        available_visual_formats=VISUAL_FORMATS,
        available_textual_formats=TEXTUAL_FORMATS,
        available_structured_formats=STRUCTURED_FORMATS
    )

    # Phase 4: Render outputs
    outputs = await render_outputs(
        data=extracted_data,
        formats=output_selection.formats
    )

    return AnalysisResult(
        engine_rationale=engine_selection.rationale,
        output_rationale=output_selection.rationale,
        outputs=outputs
    )
```

---

## VII. Implementation Priorities

### Phase 1: Complete the Foundation
- [ ] Add missing engines to ENGINE_OUTPUT_AFFINITY mapping (30 remaining)
- [ ] Define JSON schemas for all 70 engine outputs
- [ ] Document data type → visual format mapping

### Phase 2: Build Output Curator
- [ ] Create output format recommendation logic
- [ ] Implement data type detection from engine output
- [ ] Add format selection UI with explanations

### Phase 3: Expand Visual Repertoire
- [ ] Add chord_diagram for bilateral flows
- [ ] Add alluvial for temporal category shifts
- [ ] Add ACH matrix for hypothesis analysis
- [ ] Add escalation ladder for trajectory analysis

### Phase 4: Intelligent Defaults
- [ ] Train/configure Engine Curator on intent patterns
- [ ] Train/configure Output Curator on data type patterns
- [ ] Add user preference learning

---

## VIII. The Golden Rules

1. **Engine = WHAT to extract** (analytical operation with defined output schema)
2. **Format = HOW to present** (visualization/articulation of extracted data)
3. **Same data → multiple formats** (user chooses what aspect to highlight)
4. **Curator 1 picks engines** (based on intent + document type)
5. **Curator 2 picks formats** (based on data type + audience)
6. **Both curators explain** (rationale visible to user)
