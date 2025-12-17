# Session 8: Engine & Schema Integrity Audit

## Context
The Visualizer/Analyzer system has grown to 70 engines, 18 bundles, and 21 pipelines. During production use, we've encountered warnings like `unknown entity role: Power grid` - indicating mismatches between what engines extract and what schemas expect.

## Your Task
Conduct a comprehensive audit of all engines, bundles, and pipelines to identify and fix schema/type mismatches, validation gaps, and inconsistencies.

## Scope

### 1. Engine Schema Audit
For each engine in `/home/evgeny/projects/analyzer/src/engines/`:
- Compare `get_canonical_schema()` defined types against what extraction prompts actually produce
- Check if extraction prompts reference entity types, roles, or categories not in the schema
- Verify curation prompts consolidate to schema-valid types only
- Flag any hardcoded type strings that should be enums

### 2. Type/Role Consistency
Audit these specific type systems across ALL engines:
- **Entity types:** person, organization, location, product, event - are there others being extracted?
- **Entity subtypes:** Are extraction prompts using subtypes not defined in schemas?
- **Significance levels:** high/medium/low - consistent everywhere?
- **Relationship types:** Check all relationship_type values match defined enums
- **Confidence/strength scores:** Are they all 0-1? Any using percentages?

### 3. Cross-Engine Compatibility
For bundles and pipelines that chain engines:
- Verify output schema of engine N is compatible with input expectations of engine N+1
- Check that canonical field names are consistent (e.g., `source_articles` vs `articles` vs `sources`)
- Identify any engines that produce fields other engines expect but with different names

### 4. Validation Infrastructure
- Check if there's runtime validation of canonical output against schema
- If not, recommend adding Pydantic models or JSON schema validation
- Identify where warnings like "unknown entity role" are generated and ensure ALL type mismatches are caught

## Deliverables

1. **Audit Report** (`/home/evgeny/projects/analyzer/docs/SCHEMA_AUDIT_REPORT.md`):
   - List all discovered mismatches with file:line references
   - Categorize by severity (blocking vs warning)
   - Group by engine/bundle/pipeline

2. **Fixes**: Implement fixes for all discovered issues:
   - Update schemas to include missing valid types
   - Update prompts to use only schema-valid types
   - Add missing type enums where appropriate

3. **Validation Enhancement**: Add runtime schema validation if missing

## Key Files to Audit

```
/home/evgeny/projects/analyzer/src/
├── engines/           # 70 engine files - check each get_canonical_schema()
├── bundles/           # Bundle definitions
├── pipelines/         # Pipeline stage definitions
├── core/schemas.py    # Core Pydantic models
└── renderers/         # Check what types renderers expect
```

## Example Issue Found
```
Engine: entity_extraction
Warning: "unknown entity role: Power grid"
Cause: Extraction prompt allows free-form roles, but renderer expects specific types
Fix: Either constrain extraction OR expand schema to include infrastructure types
```

## Success Criteria
- Zero "unknown type/role" warnings in production
- All engines pass schema validation
- Documented type system across all engines
- Runtime validation catching mismatches before they reach renderers
