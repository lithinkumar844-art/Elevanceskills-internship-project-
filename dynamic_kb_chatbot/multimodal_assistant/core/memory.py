"""
Conversation memory.

Keeps the assistant's short-term context (recent turns, in full) and
long-term context (a running summary of older turns), plus a store
of "evidence" extracted from images so later turns can refer back to
an image without re-uploading or re-analyzing it.

This is what lets the assistant answer "what color is the car in
that photo I sent earlier?" three turns later.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List, Optional


@dataclass
class Evidence:
    """A single piece of grounded information extracted from an image."""
    turn_id: int
    source: str          # e.g. "image:uploaded_1"
    description: str     # structured description produced by the vision step
    raw_findings: List[str] = field(default_factory=list)


@dataclass
class Turn:
    turn_id: int
    role: str             # "user" or "assistant"
    text: str
    had_image: bool = False
    evidence_ids: List[int] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class ConversationMemory:
    def __init__(self, recent_turns_kept: int = 6):
        self.turns: List[Turn] = []
        self.evidence_store: List[Evidence] = []
        self.running_summary: str = ""
        self.recent_turns_kept = recent_turns_kept
        self._next_turn_id = 0

    def _new_turn_id(self) -> int:
        self._next_turn_id += 1
        return self._next_turn_id

    def add_user_turn(self, text: str, had_image: bool = False) -> int:
        turn_id = self._new_turn_id()
        self.turns.append(Turn(turn_id=turn_id, role="user", text=text, had_image=had_image))
        return turn_id

    def add_assistant_turn(self, text: str, evidence_ids: Optional[List[int]] = None) -> int:
        turn_id = self._new_turn_id()
        self.turns.append(
            Turn(turn_id=turn_id, role="assistant", text=text, evidence_ids=evidence_ids or [])
        )
        return turn_id

    def add_evidence(self, turn_id: int, source: str, description: str,
                      raw_findings: Optional[List[str]] = None) -> Evidence:
        ev = Evidence(turn_id=turn_id, source=source, description=description,
                       raw_findings=raw_findings or [])
        self.evidence_store.append(ev)
        return ev

    def all_evidence_text(self) -> str:
        """Flatten all stored evidence into a block the model can ground on."""
        if not self.evidence_store:
            return "(no visual evidence has been extracted yet)"
        lines = []
        for ev in self.evidence_store:
            lines.append(f"[Evidence from {ev.source}, turn {ev.turn_id}]: {ev.description}")
        return "\n".join(lines)

    def recent_dialogue_text(self) -> str:
        recent = self.turns[-self.recent_turns_kept:]
        lines = []
        for t in recent:
            tag = "User" if t.role == "user" else "Assistant"
            marker = " [image attached]" if t.had_image else ""
            lines.append(f"{tag}{marker}: {t.text}")
        return "\n".join(lines)

    def maybe_compress(self, summarizer_fn):
        """
        If the turn history is growing large, fold everything except the
        most recent `recent_turns_kept` turns into `running_summary` using
        the provided summarizer callable (turns_text -> summary_text).
        This keeps prompt size bounded in long-running conversations.
        """
        overflow = len(self.turns) - self.recent_turns_kept
        if overflow <= 0:
            return
        to_fold = self.turns[:overflow]
        fold_text = "\n".join(f"{t.role}: {t.text}" for t in to_fold)
        new_summary = summarizer_fn(self.running_summary, fold_text)
        self.running_summary = new_summary
        self.turns = self.turns[overflow:]

    def context_block(self) -> str:
        parts = []
        if self.running_summary:
            parts.append(f"Summary of earlier conversation:\n{self.running_summary}")
        parts.append(f"Recent dialogue:\n{self.recent_dialogue_text()}")
        parts.append(f"Visual evidence gathered so far:\n{self.all_evidence_text()}")
        return "\n\n".join(parts)
