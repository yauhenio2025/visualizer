# Session 7: UI Optimization for Scale

## CONTEXT (READ FIRST)

You're working on the **Visualizer** project — a Flask-based document intelligence platform that connects to a backend **Analyzer** with analysis engines.

### Current Scale Problem

The system has grown to:
- **70 engines** across 9 categories
- **18 bundles** (pre-configured engine groups)
- **21 pipelines** (sequential engine chains)

The UI was designed when we had ~30 engines. It's now overwhelmed. Users cannot effectively navigate, discover, or select from 70 options.

### Architecture

```
VISUALIZER (This project)          ANALYZER (Backend)
/home/evgeny/projects/visualizer   /home/evgeny/projects/analyzer
├── app.py (Flask - 8000+ lines)   ├── src/engines/*.py (70 engines)
├── docs/                          ├── src/bundles/*.py (18 bundles)
└── mcp_server/                    └── src/pipelines/*.py (21 pipelines)
```

The visualizer is a single large `app.py` file with embedded HTML/CSS/JS. The analyzer is the backend API that does the actual analysis.

---

## YOUR TASK

### Primary Goals

1. **Information Density** — Rethink how 70 engines are displayed and navigated
2. **Workflow Optimization** — Make the most common paths fast and obvious
3. **Output Mode Coherence** — Ensure engine → output mapping makes sense
4. **Discovery vs. Selection** — Help users find the right tool without being overwhelmed

### What NOT to Focus On

- Keyboard shortcuts (already have sufficient coverage)
- Authentication/security
- Backend analyzer changes (engines work fine)

---

## CURRENT STATE ANALYSIS

### Engine Category Distribution (70 total)

```
epistemology: 12 engines (source credibility, confidence, gaps, calibration)
argument: 10 engines (structure, hypotheses, steelman, stress testing)
temporal: 10 engines (timelines, scenarios, warnings, escalation)
rhetoric: 9 engines (persuasion, manipulation, deepity, attribution)
concepts: 8 engines (frameworks, anomalies, metaphors, evolution)
power: 7 engines (stakeholders, networks, actors, vulnerabilities)
evidence: 6 engines (quality, statistics, gaps)
scholarly: 5 engines (citations, debates, genealogy)
market: 3 engines (deals, regulation, competition)
```

### Output Mode Distribution

```
structured_text_report: 69 engines (almost universal)
table: 49 engines (common)
gemini_network_graph: 18 engines (visual - relationships)
gemini_timeline: 8 engines (visual - time)
comparative_matrix_table: 5 engines
gemini_concept_tree: 3 engines (visual - hierarchy)
gemini_evidence_radar: 3 engines
+ various other specialized modes
```

**Pattern:** Most engines support text reports + tables. Visual modes (Gemini-powered 4K images) are available for ~25 engines.

### Current UI Structure

1. **Mode Tabs:** Engine | Bundle | Pipeline | Intent
2. **Engine Mode:** Category filter → Engine grid → Selection → Output mode → Submit
3. **Bundle Mode:** Bundle cards → Selection → Submit
4. **Pipeline Mode:** Pipeline cards → Selection → Submit
5. **Intent Mode:** Natural language → AI recommends engine → Auto-submit

### Existing Helpers

- **AI Curator:** Analyzes document content, recommends 4-5 best engines with confidence scores
- **Intent Mode:** User describes what they want, AI picks best engine automatically
- **Categories:** 9 categories to filter engines
- **Search:** Text search over engine names/descriptions

---

## PROBLEMS TO SOLVE

### 1. Engine Grid Overwhelm

70 engine cards in a grid is unusable. Even filtered by category, 12 cards (epistemology) is a lot.

**Ideas to explore:**
- Tiered display (show top 5 per category, expand for more)
- "Recommended for this collection" prominently featured
- Recent/favorite engines
- Smart defaults based on document type

### 2. Bundle/Pipeline Discovery

Users don't know which bundle or pipeline to use. 18 bundles + 21 pipelines = 39 more options.

**Ideas to explore:**
- Categorize bundles/pipelines by use case
- Show which engines each contains (already partial)
- "Explain what this does" on hover/click
- Recommend based on document content

### 3. Output Mode Confusion

Users see "visual" vs "textual" but don't understand what they'll get.

**Ideas to explore:**
- Preview/example of each output mode
- Auto-recommend best output mode per engine
- Group engines by what they produce (tables, diagrams, reports)

### 4. Decision Paralysis

Too many equivalent-seeming options. User doesn't know:
- When to use `argument_architecture` vs `dialectical_structure`
- When to use a bundle vs individual engines
- Whether pipeline chains add value for their use case

**Ideas to explore:**
- Decision tree / guided wizard
- "Start here" recommendations
- Use case templates (e.g., "I'm analyzing a policy document")

---

## SPECIFIC UI ELEMENTS TO CONSIDER

### A. Landing State

When user opens the app with documents selected, what should they see first?

Current: Tab bar with Engine/Bundle/Pipeline/Intent modes
Problem: Requires user to know which mode to pick

**Consider:** Single smart entry point that routes to best option

### B. Engine Selection

Current: Grid of 70 cards filterable by category
Problem: Too many options even filtered

**Consider:**
- Collapsed categories that expand
- "Quick picks" section with AI recommendations
- Search-first interface (like Spotlight/Alfred)

### C. Bundle/Pipeline Display

Current: Separate tabs with card grids
Problem: User has to understand difference between bundle and pipeline

**Consider:**
- Unified "analysis packages" view
- Group by outcome ("deep argument analysis", "network mapping", "timeline construction")
- Show expected outputs and time

### D. Output Mode Selection

Current: Dropdown after engine selection
Problem: User doesn't know what "gemini_network_graph" means

**Consider:**
- Thumbnail previews of output types
- Smart default based on engine (some engines only make sense as visualizations)
- "Let AI choose" option

### E. Results Gallery

Current: Shows results one at a time with navigation
Problem: Hard to compare multiple outputs

**Consider:**
- Grid view of thumbnails
- Side-by-side comparison
- Filtering by output type

---

## TECHNICAL CONSTRAINTS

1. **Single File:** All UI is in `app.py` (embedded HTML/CSS/JS)
2. **Flask Templates:** Uses Jinja2 templates inline
3. **No Build System:** Plain HTML/CSS/JS, no React/Vue/etc.
4. **API Calls:** Fetches engine/bundle/pipeline lists from Analyzer API
5. **Job System:** Analysis jobs are async, results come back later

---

## REFERENCE DOCUMENTS

- `/home/evgeny/projects/visualizer/docs/DENNETT_UPGRADE_PROGRESS.md` — What was built
- `/home/evgeny/projects/visualizer/docs/STRATEGIC_AUDIT_CIA_PERSPECTIVE.md` — UI recommendations from CIA analyst perspective (Section V)
- `/home/evgeny/projects/visualizer/app.py` — The main application file

---

## DELIVERABLES

1. **Analysis:** Review current UI structure in app.py
2. **Design:** Propose specific UI changes to handle scale
3. **Implementation:** Make changes to app.py
4. **Verification:** Test that changes work with the full engine set

Focus on **usability at scale**, not feature additions. The goal is to make 70 engines feel like 10 well-organized options.

---

## STARTING COMMAND

```bash
# Check current engine/bundle/pipeline counts
cd /home/evgeny/projects/analyzer && python -c "
from src.engines import EngineRegistry
from src.bundles import BundleRegistry
from src.pipelines import MetaEngineRegistry

print(f'Engines: {len(list(EngineRegistry.list_engines()))}')
print(f'Bundles: {len(list(BundleRegistry.list_bundles()))}')
print(f'Pipelines: {len(list(MetaEngineRegistry.list_pipelines()))}')
"

# Start the visualizer to see current UI
cd /home/evgeny/projects/visualizer && python app.py
```

The app runs on `http://localhost:5005` by default.
