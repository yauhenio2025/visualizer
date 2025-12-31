"""
Display Utilities for Visualizer

Provides functions to format data for human-readable display in visualizations.

Key functions:
- format_label: Convert snake_case to Title Case
- sanitize_for_display: Recursively format keys and clean up data
- should_hide_numeric_field: Detect numeric fields that shouldn't be displayed
"""

import re
from typing import Any, Union


def format_label(text: str) -> str:
    """
    Convert snake_case or kebab-case to Title Case for display.

    Examples:
        format_label("technofascism_trajectory") -> "Technofascism Trajectory"
        format_label("rentier_capitalism_framework") -> "Rentier Capitalism Framework"
        format_label("argument-architecture") -> "Argument Architecture"
        format_label("some_long_field_name") -> "Some Long Field Name"

    Args:
        text: The string to format

    Returns:
        Human-readable Title Case string
    """
    if not text or not isinstance(text, str):
        return str(text) if text else ""

    # Replace underscores and hyphens with spaces
    result = text.replace("_", " ").replace("-", " ")

    # Handle camelCase by inserting spaces before capitals
    result = re.sub(r'([a-z])([A-Z])', r'\1 \2', result)

    # Title case each word
    result = result.title()

    # Fix common acronyms that should stay uppercase
    acronyms = ['API', 'URL', 'ID', 'AI', 'ML', 'NLP', 'LLM', 'EU', 'US', 'UK', 'UN']
    for acronym in acronyms:
        # Replace case-insensitive with correct case
        result = re.sub(rf'\b{acronym.title()}\b', acronym, result, flags=re.IGNORECASE)

    return result


# Fields that commonly contain numeric scores that should NOT be displayed on visualizations
HIDDEN_NUMERIC_FIELDS = {
    # Confidence/weight/strength scores
    'confidence', 'confidence_score', 'confidence_level',
    'strength', 'relationship_strength', 'connection_strength',
    'weight', 'edge_weight', 'link_weight',
    'score', 'similarity_score', 'relevance_score',
    # Internal processing fields
    'influence_score', 'power_score', 'interest_score',
    'centrality', 'betweenness', 'pagerank',
    'probability', 'likelihood',
    # Ratios and percentages stored as decimals
    'ratio', 'percentage', 'proportion',
}


def should_hide_numeric_field(key: str) -> bool:
    """
    Check if a field name indicates it contains internal numeric scores
    that should not be displayed directly on visualizations.

    Args:
        key: The field name to check

    Returns:
        True if this field should be hidden/transformed for display
    """
    if not key:
        return False

    key_lower = key.lower()

    # Check exact matches
    if key_lower in HIDDEN_NUMERIC_FIELDS:
        return True

    # Check if key ends with common score suffixes
    score_suffixes = ('_score', '_strength', '_weight', '_confidence', '_probability')
    if any(key_lower.endswith(suffix) for suffix in score_suffixes):
        return True

    return False


def format_numeric_for_display(value: float, field_name: str = "") -> str:
    """
    Format a numeric value for display.

    For confidence scores (0-1), convert to descriptive terms.
    For other numbers, format appropriately.

    Args:
        value: The numeric value
        field_name: Optional field name for context

    Returns:
        Human-readable string representation
    """
    if not isinstance(value, (int, float)):
        return str(value)

    # Check if this looks like a confidence score (0-1 range)
    if 0 <= value <= 1:
        if value >= 0.9:
            return "Very Strong"
        elif value >= 0.75:
            return "Strong"
        elif value >= 0.5:
            return "Moderate"
        elif value >= 0.25:
            return "Weak"
        else:
            return "Very Weak"

    # For larger numbers, format with appropriate precision
    if isinstance(value, float):
        if value == int(value):
            return str(int(value))
        return f"{value:.2f}"

    return str(value)


def looks_like_identifier(text: str) -> bool:
    """
    Check if a string looks like a snake_case or kebab-case identifier.

    Args:
        text: The string to check

    Returns:
        True if it looks like an identifier that should be formatted
    """
    if not text or not isinstance(text, str):
        return False

    # Must contain underscore or hyphen
    if '_' not in text and '-' not in text:
        return False

    # Should be relatively short (identifiers are usually < 100 chars)
    if len(text) > 100:
        return False

    # Should not contain spaces (already formatted)
    if ' ' in text:
        return False

    # Should be mostly lowercase alphanumeric with underscores/hyphens
    cleaned = text.replace('_', '').replace('-', '')
    return cleaned.isalnum()


def sanitize_for_display(
    data: Any,
    format_keys: bool = True,
    convert_scores: bool = True,
    hide_score_fields: bool = True,
    format_string_values: bool = True,
) -> Any:
    """
    Recursively sanitize data structure for display.

    - Converts snake_case keys to Title Case
    - Converts snake_case string values to Title Case (if they look like identifiers)
    - Converts numeric confidence scores to descriptive terms
    - Optionally hides internal score fields

    Args:
        data: The data to sanitize (dict, list, or scalar)
        format_keys: Whether to format dict keys to Title Case
        convert_scores: Whether to convert 0-1 scores to descriptive terms
        hide_score_fields: Whether to remove internal score fields entirely
        format_string_values: Whether to format snake_case string values

    Returns:
        Sanitized data structure
    """
    if data is None:
        return None

    if isinstance(data, dict):
        result = {}
        for key, value in data.items():
            # Optionally skip internal score fields
            if hide_score_fields and should_hide_numeric_field(key):
                continue

            # Format the key
            new_key = format_label(key) if format_keys else key

            # Recursively sanitize the value
            if isinstance(value, (dict, list)):
                result[new_key] = sanitize_for_display(
                    value, format_keys, convert_scores, hide_score_fields, format_string_values
                )
            elif convert_scores and isinstance(value, (int, float)) and should_hide_numeric_field(key):
                # Convert score to descriptive term
                result[new_key] = format_numeric_for_display(value, key)
            elif format_string_values and isinstance(value, str) and looks_like_identifier(value):
                # Format snake_case string values to Title Case
                result[new_key] = format_label(value)
            else:
                result[new_key] = value

        return result

    elif isinstance(data, list):
        return [
            sanitize_for_display(item, format_keys, convert_scores, hide_score_fields, format_string_values)
            for item in data
        ]

    elif format_string_values and isinstance(data, str) and looks_like_identifier(data):
        # Format standalone snake_case strings
        return format_label(data)

    else:
        return data


def get_display_instructions() -> str:
    """
    Return instructions for Gemini about proper data display.

    These instructions should be included in all visualization prompts
    to ensure proper formatting of labels and hiding of internal scores.
    """
    return """
## DISPLAY FORMATTING REQUIREMENTS (CRITICAL)

⚠️ BRANDING & ATTRIBUTION - ABSOLUTELY FORBIDDEN:
- DO NOT include any publication logos (no FT, NYT, WSJ, Bloomberg, etc.)
- DO NOT include journalist names or bylines (no "John Burn-Murdoch", "Amanda Cox", etc.)
- DO NOT include fake credits or attributions
- DO NOT use copyrighted masthead designs or typography
- The visualization must be ORIGINAL work, not attributed to any publication
- Title should describe the CONTENT (e.g., "Stakeholder Power Analysis") NOT mimic a publication
- This is a LEGAL requirement - violating it creates trademark/copyright issues

LABEL FORMATTING:
- Convert ALL snake_case and kebab-case identifiers to Title Case with spaces
- Example: "technofascism_trajectory" → "Technofascism Trajectory"
- Example: "rentier_capitalism_framework" → "Rentier Capitalism Framework"
- Example: "argument_architecture" → "Argument Architecture"
- NEVER display underscores or hyphens in labels visible to users

NUMERIC SCORES - DO NOT DISPLAY (CRITICAL):
- NEVER show raw decimal numbers like 0.85, 0.75, 0.9 on the visualization
- NEVER display "THICKNESS: 0.85" or similar labels on edges - this is FORBIDDEN
- NEVER include "weight:", "strength:", "confidence:", "score:" with numbers
- These are internal processing values and should NOT appear on edges, nodes, or labels
- If edge/relationship strength must be shown, use visual encoding ONLY:
  * Line thickness (thicker = stronger) - but NO numeric thickness labels
  * Line style (solid = strong, dashed = moderate, dotted = weak)
  * Color intensity (darker = stronger)
- If you absolutely must show strength as text, use ONLY descriptive terms: "Strong", "Moderate", "Weak"
- Edge labels should describe the RELATIONSHIP TYPE (e.g., "Influenced", "Derived from") NOT numeric values

FIELD NAME CLEANUP:
- Replace underscores with spaces in all visible text
- Use proper capitalization (Title Case for labels, headings)
- Abbreviations like "ID", "API", "AI" should remain uppercase
"""


# Export for module
__all__ = [
    "format_label",
    "format_numeric_for_display",
    "sanitize_for_display",
    "should_hide_numeric_field",
    "looks_like_identifier",
    "get_display_instructions",
    "HIDDEN_NUMERIC_FIELDS",
]
