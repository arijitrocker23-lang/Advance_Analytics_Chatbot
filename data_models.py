# =============================================================================
# data_models.py
# =============================================================================
# Data-class definitions for structured data used throughout the application.
# This file has NOTHING to do with ML models. It simply defines Python
# dataclasses that hold configuration like custom instructions, few-shot
# examples, and knowledge base entries.
# =============================================================================

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class CustomInstructions:
    """
    Stores all user-provided context and organisation-specific instructions.
    These are injected into the planner and explainer prompts so that
    the model follows domain-specific rules.
    """

    # Free-text description of what the data represents
    data_context: str = ""

    # List of imperative instructions the model must follow
    org_instructions: List[str] = field(default_factory=list)

    # Mapping from user-friendly names to actual DataFrame column names
    column_aliases: Dict[str, str] = field(default_factory=dict)

    # Per-column mapping from user terms to actual cell values
    value_mappings: Dict[str, Dict[str, str]] = field(default_factory=dict)

    # Display / formatting preferences
    formatting_rules: Dict[str, str] = field(default_factory=dict)

    # Domain-specific business rules
    business_rules: List[str] = field(default_factory=list)

    # Glossary of domain terms and their definitions
    terminology: Dict[str, str] = field(default_factory=dict)

    def is_empty(self) -> bool:
        """Return True if no instructions have been configured."""
        return (
            not self.data_context
            and not self.org_instructions
            and not self.column_aliases
            and not self.value_mappings
            and not self.formatting_rules
            and not self.business_rules
            and not self.terminology
        )

    def get_context_for_planner(self) -> str:
        """Build a formatted string for the planner prompt."""
        if self.is_empty():
            return ""

        parts = ["\n\n=== USER-PROVIDED CONTEXT AND INSTRUCTIONS ==="]

        if self.data_context:
            parts.append("\nDATA CONTEXT:\n{}".format(self.data_context))

        if self.org_instructions:
            parts.append("\nORGANISATION INSTRUCTIONS (MUST FOLLOW):")
            for idx, instruction in enumerate(self.org_instructions, 1):
                parts.append("  {}. {}".format(idx, instruction))

        if self.column_aliases:
            parts.append("\nCOLUMN ALIASES (use these mappings):")
            for alias, actual in self.column_aliases.items():
                parts.append("  - '{}' refers to column '{}'".format(alias, actual))

        if self.value_mappings:
            parts.append("\nVALUE MAPPINGS:")
            for column, mappings in self.value_mappings.items():
                parts.append("  Column '{}':".format(column))
                for term, value in mappings.items():
                    parts.append("    - '{}' means '{}'".format(term, value))

        if self.business_rules:
            parts.append("\nBUSINESS RULES (apply when relevant):")
            for idx, rule in enumerate(self.business_rules, 1):
                parts.append("  {}. {}".format(idx, rule))

        if self.terminology:
            parts.append("\nTERMINOLOGY DEFINITIONS:")
            for term, definition in self.terminology.items():
                parts.append("  - {}: {}".format(term, definition))

        if self.formatting_rules:
            parts.append("\nFORMATTING PREFERENCES:")
            for rule_type, rule in self.formatting_rules.items():
                parts.append("  - {}: {}".format(rule_type, rule))

        parts.append("\n=== END OF USER INSTRUCTIONS ===\n")
        return "\n".join(parts)

    def get_context_for_explainer(self) -> str:
        """Build a shorter formatted string for the explainer prompt."""
        if self.is_empty():
            return ""

        parts = ["\n\nUser Context and Instructions to Follow:"]

        if self.data_context:
            parts.append("- Data Context: {}".format(self.data_context))

        if self.org_instructions:
            parts.append("- Organisation Instructions:")
            for instruction in self.org_instructions[:5]:
                parts.append("  * {}".format(instruction))

        if self.formatting_rules:
            parts.append("- Formatting Rules:")
            for rule_type, rule in self.formatting_rules.items():
                parts.append("  * {}: {}".format(rule_type, rule))

        if self.terminology:
            parts.append("- Terminology:")
            for term, definition in list(self.terminology.items())[:5]:
                parts.append("  * {} = {}".format(term, definition))

        return "\n".join(parts)

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to a plain dictionary."""
        return {
            "data_context": self.data_context,
            "org_instructions": self.org_instructions,
            "column_aliases": self.column_aliases,
            "value_mappings": self.value_mappings,
            "formatting_rules": self.formatting_rules,
            "business_rules": self.business_rules,
            "terminology": self.terminology,
        }

    @classmethod
    def from_dict(cls, data):
        """Deserialise from a plain dictionary."""
        return cls(
            data_context=data.get("data_context", ""),
            org_instructions=data.get("org_instructions", []),
            column_aliases=data.get("column_aliases", {}),
            value_mappings=data.get("value_mappings", {}),
            formatting_rules=data.get("formatting_rules", {}),
            business_rules=data.get("business_rules", []),
            terminology=data.get("terminology", {}),
        )


@dataclass
class FewShotExample:
    """
    A single few-shot training example pairing a user question
    with the expected JSON plan.
    """

    question: str = ""
    expected_plan: str = ""
    description: str = ""
    category: str = "general"

    def to_dict(self):
        return {
            "question": self.question,
            "expected_plan": self.expected_plan,
            "description": self.description,
            "category": self.category,
        }

    @classmethod
    def from_dict(cls, data):
        return cls(
            question=data.get("question", ""),
            expected_plan=data.get("expected_plan", ""),
            description=data.get("description", ""),
            category=data.get("category", "general"),
        )


@dataclass
class KnowledgeBaseEntry:
    """
    A single knowledge-base entry storing domain facts or definitions.
    """

    title: str = ""
    content: str = ""
    category: str = "general"
    tags: List[str] = field(default_factory=list)

    def to_dict(self):
        return {
            "title": self.title,
            "content": self.content,
            "category": self.category,
            "tags": self.tags,
        }

    @classmethod
    def from_dict(cls, data):
        return cls(
            title=data.get("title", ""),
            content=data.get("content", ""),
            category=data.get("category", "general"),
            tags=data.get("tags", []),
        )