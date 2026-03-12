# =============================================================================
# knowledge_base.py
# =============================================================================
# In-memory knowledge base that stores domain-specific facts, definitions,
# and reference information.  Entries can be searched by keyword and injected
# into the planner prompt so that the model has access to domain knowledge
# when generating analysis plans.
# =============================================================================

from typing import Any, Dict, List, Optional

from data_models import KnowledgeBaseEntry


class KnowledgeBase:
    """
    Simple in-memory knowledge store.

    Entries are stored in a list and can be searched by keyword against
    their title, content, and tags.
    """

    def __init__(self) -> None:
        self._entries: List[KnowledgeBaseEntry] = []

    # ------------------------------------------------------------------
    # CRUD operations
    # ------------------------------------------------------------------

    def add_entry(self, entry: KnowledgeBaseEntry) -> None:
        """Add a new entry to the knowledge base."""
        self._entries.append(entry)

    def remove_entry(self, index: int) -> None:
        """Remove the entry at *index* (0-based)."""
        if 0 <= index < len(self._entries):
            self._entries.pop(index)

    def get_all_entries(self) -> List[KnowledgeBaseEntry]:
        """Return a copy of the full list of entries."""
        return list(self._entries)

    def clear(self) -> None:
        """Remove all entries."""
        self._entries.clear()

    @property
    def size(self) -> int:
        return len(self._entries)

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        max_results: int = 5,
    ) -> List[KnowledgeBaseEntry]:
        """
        Return entries whose title, content, or tags contain any word
        from *query* (case-insensitive).
        """
        if not query.strip():
            return []

        query_words = set(query.lower().split())
        scored: List[tuple] = []

        for entry in self._entries:
            # Build a combined text blob for matching
            text = (
                f"{entry.title} {entry.content} {' '.join(entry.tags)}"
            ).lower()
            # Count how many query words appear in the text
            score = sum(1 for w in query_words if w in text)
            if score > 0:
                scored.append((score, entry))

        # Sort by descending score and return the top results
        scored.sort(key=lambda x: x[0], reverse=True)
        return [entry for _, entry in scored[:max_results]]

    # ------------------------------------------------------------------
    # Prompt generation
    # ------------------------------------------------------------------

    def get_context_for_prompt(self, question: str) -> str:
        """
        Search the knowledge base for entries relevant to *question*
        and format them as a string suitable for injection into the
        planner prompt.
        """
        relevant = self.search(question, max_results=3)
        if not relevant:
            return ""

        parts = ["\n=== KNOWLEDGE BASE (relevant entries) ==="]
        for idx, entry in enumerate(relevant, 1):
            parts.append(f"\n{idx}. [{entry.category}] {entry.title}")
            parts.append(f"   {entry.content}")
            if entry.tags:
                parts.append(f"   Tags: {', '.join(entry.tags)}")
        parts.append("\n=== END KNOWLEDGE BASE ===\n")
        return "\n".join(parts)

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_list(self) -> List[Dict[str, Any]]:
        return [e.to_dict() for e in self._entries]

    def load_from_list(self, data: List[Dict[str, Any]]) -> None:
        self._entries = [KnowledgeBaseEntry.from_dict(d) for d in data]