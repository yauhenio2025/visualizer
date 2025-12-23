# Visual Format Typology for Gemini Image Generation

**What Gemini can produce • When to use each • Prompting patterns**

---

## The Reality

The Analyzer API has ONE visual output mode: `gemini_image`. All differentiation happens through **prompting** - we tell Gemini what type of visualization to create based on the extracted data.

This document defines what visual formats Gemini can reliably produce, what data structures map to each, and what prompting patterns work best.

---

## I. Visual Format Categories

### A. Relational / Network Visualizations

| Format | Data Structure | Use When | Gemini Prompt Pattern |
|--------|---------------|----------|----------------------|
| **Network Graph** | `{nodes[], edges[]}` | Showing relationships between entities (actors, concepts, citations) | "Create a network graph showing [entities] as nodes connected by edges representing [relationship]. Use node size for [importance]. Color-code by [category]." |
| **Chord Diagram** | `{entities[], flows_between[]}` | Showing bilateral relationships with magnitude (trade, communication, co-occurrence) | "Create a chord diagram showing flows between [entities]. Arc thickness represents [magnitude]. Color each entity distinctly." |
| **Hierarchical Tree** | `{nodes[], parent_child[]}` | Showing hierarchies, taxonomies, org charts | "Create a hierarchical tree diagram with [root] at top. Show [levels] of hierarchy. Use consistent spacing." |
| **Radial Tree** | `{center, branches[]}` | Showing influence radiating from center, concept maps | "Create a radial tree with [center_concept] at center. Branches extend outward to [related_concepts]. Size branches by [importance]." |
| **Force-Directed Layout** | `{nodes[], edges[], weights[]}` | Showing organic clustering and community structure | "Create a force-directed network layout where related nodes cluster together. Show communities through spatial grouping." |

### B. Flow / Process Visualizations

| Format | Data Structure | Use When | Gemini Prompt Pattern |
|--------|---------------|----------|----------------------|
| **Sankey Diagram** | `{sources[], targets[], values[]}` | Showing flows with quantities (money, resources, conversions) | "Create a Sankey diagram showing flows from [sources] to [targets]. Width represents [value]. Color flows by [category]." |
| **Alluvial Diagram** | `{stages[], entities[], flows[]}` | Showing how entities change category over time | "Create an alluvial diagram showing how [entities] transition between [categories] across [time_periods]." |
| **Flowchart** | `{steps[], decisions[], paths[]}` | Showing processes, decision trees, algorithms | "Create a flowchart showing the process from [start] to [end]. Diamond shapes for decisions. Arrows show flow direction." |
| **Process Flow** | `{stages[], inputs[], outputs[]}` | Showing transformation processes | "Create a process flow diagram showing [process_name]. Each stage receives [inputs] and produces [outputs]." |
| **Value Stream Map** | `{steps[], cycle_times[], wait_times[]}` | Showing process efficiency and bottlenecks | "Create a value stream map showing [process]. Indicate cycle time and wait time at each step. Highlight bottlenecks." |

### C. Temporal Visualizations

| Format | Data Structure | Use When | Gemini Prompt Pattern |
|--------|---------------|----------|----------------------|
| **Timeline** | `{events[], dates[], categories[]}` | Showing sequence of events over time | "Create a horizontal timeline from [start_date] to [end_date]. Place events at their dates. Color-code by [category]." |
| **Gantt Chart** | `{tasks[], start_dates[], end_dates[], dependencies[]}` | Showing overlapping durations, project schedules | "Create a Gantt chart showing [tasks] with start and end dates. Show dependencies as arrows." |
| **Parallel Timelines** | `{tracks[], events_per_track[]}` | Showing simultaneous developments across domains | "Create parallel timelines for [domains]. Show synchronous events aligned vertically. Each track labeled." |
| **Cycle Diagram** | `{phases[], transitions[], recurrence}` | Showing recurring patterns, boom-bust cycles | "Create a circular cycle diagram showing [phases] in sequence. Indicate typical duration of each phase." |
| **Sparklines** | `{series[], time_points[]}` | Showing trends compactly (inline with data) | "Create small sparkline charts showing trend for each [entity]. Highlight peaks and troughs." |

### D. Comparative Visualizations

| Format | Data Structure | Use When | Gemini Prompt Pattern |
|--------|---------------|----------|----------------------|
| **Matrix / Heatmap** | `{rows[], columns[], values[][]}` | Showing values across two dimensions | "Create a heatmap with [row_entities] as rows and [column_entities] as columns. Color intensity represents [value]. Include legend." |
| **Quadrant Chart** | `{items[], x_scores[], y_scores[]}` | Showing position on two dimensions (power-interest, impact-effort) | "Create a 2x2 quadrant chart with [x_axis_label] horizontal and [y_axis_label] vertical. Plot [items] as labeled points." |
| **Radar / Spider Chart** | `{entities[], dimensions[], scores[]}` | Comparing multiple entities across multiple dimensions | "Create a radar chart comparing [entities] across [dimensions]. Each entity is a different colored polygon." |
| **Bar Chart** | `{categories[], values[]}` | Showing ranked comparisons | "Create a horizontal bar chart showing [categories] ranked by [value]. Include value labels." |
| **Grouped Bar Chart** | `{categories[], groups[], values[][]}` | Comparing groups across categories | "Create a grouped bar chart comparing [groups] across [categories]. Use distinct colors per group." |
| **Dot Plot / Strip Plot** | `{items[], values[], categories[]}` | Showing distribution within categories | "Create a dot plot showing [items] positioned by [value] within each [category] strip." |

### E. Part-of-Whole Visualizations

| Format | Data Structure | Use When | Gemini Prompt Pattern |
|--------|---------------|----------|----------------------|
| **Treemap** | `{categories[], sizes[], hierarchy[]}` | Showing hierarchical composition by size | "Create a treemap showing [hierarchy]. Rectangle size represents [value]. Color-code by [top_level_category]." |
| **Sunburst** | `{levels[], sizes[]}` | Showing hierarchical composition radially | "Create a sunburst diagram with [root] at center. Concentric rings show hierarchy levels. Arc size represents [value]." |
| **Stacked Bar** | `{categories[], components[], values[][]}` | Showing composition across categories | "Create a stacked bar chart showing composition of [components] for each [category]." |
| **Waterfall Chart** | `{start_value, additions[], subtractions[], end_value}` | Showing cumulative effect of sequential changes | "Create a waterfall chart starting at [start]. Show each [change] as addition or subtraction leading to [end]." |
| **Marimekko** | `{rows[], columns[], values[][]}` | Showing two-dimensional part-of-whole | "Create a Marimekko chart where bar width represents [x_total] and segments represent [y_composition]." |

### F. Spatial / Set Visualizations

| Format | Data Structure | Use When | Gemini Prompt Pattern |
|--------|---------------|----------|----------------------|
| **Venn Diagram** | `{sets[], overlaps[]}` | Showing set relationships (2-4 sets max) | "Create a Venn diagram showing overlap between [sets]. Label each region with [contents_or_count]." |
| **Euler Diagram** | `{sets[], overlaps[], exclusions[]}` | Showing set relationships without false implications | "Create an Euler diagram showing [sets] with accurate overlaps. Non-overlapping sets should not touch." |
| **Positioning Map** | `{items[], x_positions[], y_positions[]}` | Showing competitive/conceptual positioning | "Create a positioning map with [x_dimension] and [y_dimension] as axes. Plot [items] at their positions." |
| **Bubble Chart** | `{items[], x[], y[], sizes[]}` | Showing three dimensions (two position + size) | "Create a bubble chart with [x_axis] horizontal, [y_axis] vertical. Bubble size represents [third_dimension]." |

### G. Evidence / Analytical Visualizations

| Format | Data Structure | Use When | Gemini Prompt Pattern |
|--------|---------------|----------|----------------------|
| **ACH Matrix** | `{hypotheses[], evidence[], consistency[][]}` | Showing Analysis of Competing Hypotheses | "Create an ACH matrix with hypotheses as columns and evidence as rows. Color cells: green=consistent, red=inconsistent, gray=neutral." |
| **Confidence Thermometer** | `{findings[], confidence_levels[]}` | Showing confidence levels for key findings | "Create a vertical confidence scale from 'Remote' to 'Almost Certain'. Plot [findings] at their confidence levels." |
| **Evidence Quality Matrix** | `{sources[], reliability[], validity[]}` | Showing source credibility | "Create a matrix with source reliability (A-F) as rows and information validity (1-6) as columns. Plot sources." |
| **Indicator Dashboard** | `{indicators[], status[], trends[]}` | Showing warning indicator status | "Create an indicator dashboard with traffic light status for each [indicator]. Include trend arrows." |
| **Gap Analysis Visual** | `{current[], desired[], gaps[]}` | Showing gap between current and desired state | "Create a gap analysis chart showing [current_state] vs [desired_state] for each [dimension]. Highlight largest gaps." |

### H. Argumentative / Logical Visualizations

| Format | Data Structure | Use When | Gemini Prompt Pattern |
|--------|---------------|----------|----------------------|
| **Argument Tree** | `{claims[], premises[], warrants[]}` | Showing logical structure of arguments | "Create an argument tree with main [claim] at top. Supporting premises below. Show warrant links." |
| **Toulmin Diagram** | `{claim, grounds, warrant, backing, qualifier, rebuttal}` | Showing full argument structure | "Create a Toulmin model diagram for this argument. Show all six components with connecting arrows." |
| **Dialectical Map** | `{thesis, antithesis, synthesis, tensions[]}` | Showing thesis-antithesis-synthesis | "Create a dialectical map showing [thesis] and [antithesis] as opposing positions, with [synthesis] emerging from their tension." |
| **Assumption Web** | `{conclusions[], assumptions[], dependencies[]}` | Showing hidden assumptions underlying conclusions | "Create a web diagram with [conclusions] at top, supported by chains of [assumptions] below. Show dependencies." |
| **Scenario Cone** | `{present, futures[], probabilities[], branches[]}` | Showing multiple possible futures | "Create a cone of plausibility diagram from [present]. Show branching paths to different [futures]. Width indicates probability." |

---

## II. Data Type → Visual Format Mapping

When the Output Curator receives extracted data, it should recommend formats based on the data structure:

| Data Type | Primary Format | Secondary Formats | Avoid |
|-----------|---------------|-------------------|-------|
| `{nodes[], edges[]}` | Network Graph | Chord, Matrix | Pie, Timeline |
| `{flows[], sources[], targets[], values[]}` | Sankey | Chord, Alluvial | Bar, Pie |
| `{events[], dates[]}` | Timeline | Gantt, Table | Network, Pie |
| `{items[], scores_x[], scores_y[]}` | Quadrant | Scatter, Positioning | Pie, Timeline |
| `{entities[], dimensions[], scores[]}` | Radar | Bar, Heatmap | Pie, Network |
| `{categories[], sizes[]}` | Bar Chart | Treemap, Pie | Network, Timeline |
| `{hierarchy[], sizes[]}` | Treemap | Sunburst, Tree | Network, Bar |
| `{hypotheses[], evidence[], consistency[][]}` | ACH Matrix | Heatmap | Network, Flow |
| `{actors[], power[], interest[]}` | Quadrant | Network, Table | Timeline, Flow |
| `{stages[], entities[], transitions[]}` | Alluvial | Sankey, Process | Bar, Pie |
| `{claims[], premises[], logical_structure}` | Argument Tree | Flowchart | Network, Bar |

---

## III. Gemini Prompting Best Practices

### General Principles

1. **Be explicit about visualization type**: "Create a [specific_type] diagram/chart/graph"
2. **Specify data mapping**: "Use [property] for size/color/position"
3. **Request labels**: "Label each [element] clearly"
4. **Request legend**: "Include a legend explaining [color/size] coding"
5. **Specify style**: "Use a clean, professional style suitable for [context]"

### Prompt Template

```
Create a [VISUALIZATION_TYPE] showing [WHAT_TO_SHOW].

DATA:
[Provide the structured data]

REQUIREMENTS:
- [Element] should be represented by [visual property]
- Color-code by [dimension]
- Size indicates [dimension]
- Label [elements] with [labels]
- Include legend for [color/size] coding

STYLE:
- Professional, clean design
- Suitable for [executive presentation / analyst review / publication]
- [Light/Dark] background
- Clear contrast between elements
```

### Format-Specific Prompts

**Network Graph:**
```
Create a network graph visualization.

NODES: [list of entities with optional attributes for size/color]
EDGES: [list of connections with optional weights]

- Node size represents [importance/centrality/mentions]
- Node color indicates [category/type/sentiment]
- Edge thickness shows [strength/frequency]
- Cluster related nodes spatially
- Label significant nodes
```

**Sankey Diagram:**
```
Create a Sankey diagram showing resource flows.

FLOWS:
[source] → [target]: [value]
[source] → [target]: [value]

- Flow width proportional to value
- Color flows by [source category / destination / value range]
- Order sources and targets logically
- Label flows with values
```

**Quadrant Chart:**
```
Create a 2x2 quadrant chart.

X-AXIS: [dimension] (low to high)
Y-AXIS: [dimension] (low to high)

ITEMS:
[item]: x=[score], y=[score]

- Plot items as labeled circles
- Size circles by [optional third dimension]
- Label quadrants: [TL, TR, BL, BR names]
- Show axis labels and values
```

---

## IV. Quality Criteria for Visual Outputs

### Must Have
- [ ] Clear title stating what is shown
- [ ] All elements labeled or in legend
- [ ] Professional color palette
- [ ] Sufficient contrast for readability
- [ ] Logical layout and grouping
- [ ] Scale/axis labels where applicable

### Should Have
- [ ] Visual hierarchy (most important stands out)
- [ ] Consistent styling throughout
- [ ] White space for clarity
- [ ] Color-blind friendly palette
- [ ] Source attribution if applicable

### Avoid
- [ ] 3D effects that distort perception
- [ ] Decorative elements that add no information
- [ ] Too many colors (>7 distinct)
- [ ] Overlapping labels
- [ ] Truncated axes that mislead
- [ ] Pie charts for >5 categories

---

## V. Output Curator Decision Matrix

When the Output Curator recommends visual formats, it should consider:

| Factor | Weight | High Score Formats |
|--------|--------|-------------------|
| Data has relationships | High | Network, Chord, Matrix |
| Data has flows/quantities | High | Sankey, Alluvial |
| Data is temporal | High | Timeline, Gantt |
| Data needs comparison | Medium | Bar, Radar, Quadrant |
| Data is hierarchical | Medium | Tree, Treemap, Sunburst |
| Data shows composition | Medium | Stacked Bar, Pie, Treemap |
| Audience is executive | High | Quadrant, Snapshot visuals |
| Audience is analyst | Medium | Network, Matrix, detailed |
| Space is limited | Low | Sparklines, compact formats |

---

## VI. Implementation Notes

### Current State
- Analyzer API has single `gemini_image` output mode
- Differentiation is purely through prompting
- No structured visual format selection in UI

### Needed
1. **Visual Format Selector**: UI component letting users choose specific format
2. **Data Type Detection**: Analyze engine output to detect data structure
3. **Smart Defaults**: Output Curator suggests format based on data type
4. **Prompt Generator**: Builds optimal Gemini prompt for chosen format + data
5. **Quality Validation**: Check generated image meets quality criteria

### Prompt Engineering Layer

```python
def generate_visual_prompt(
    data: dict,
    format: str,
    audience: str = "analyst"
) -> str:
    """Generate optimized Gemini prompt for visualization."""

    template = PROMPT_TEMPLATES[format]

    # Map data to prompt placeholders
    prompt = template.format(
        data=format_data_for_prompt(data),
        style=AUDIENCE_STYLES[audience],
        **extract_dimensions(data)
    )

    return prompt
```

---

## VII. Format Reference Quick Guide

| If you want to show... | Use... | Example |
|------------------------|--------|---------|
| Who knows whom | Network Graph | Citation networks, actor relationships |
| Where money/resources go | Sankey | Budget flows, supply chains |
| What happened when | Timeline | Event history, project milestones |
| Who has more/less | Bar Chart | Rankings, comparisons |
| Trade-offs between options | Quadrant | Power-interest, impact-effort |
| How things compare on many dimensions | Radar | Entity profiles, capability assessment |
| How categories break down | Treemap / Pie | Market share, budget allocation |
| Logical argument structure | Argument Tree | Thesis defense, claim analysis |
| Possible futures | Scenario Cone | Strategic planning, risk assessment |
| Evidence vs hypotheses | ACH Matrix | Intelligence analysis, investigation |
