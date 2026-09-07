"""
Orchestrator.

This is the "intelligent decision-making" layer the task asks for:
instead of piping the user's message straight into one model call
and returning whatever comes back, it runs a small pipeline with
branch points, and returns a trace of what happened at each step so
the reasoning is inspectable (useful for the report/demo, and for
debugging).

Pipeline per turn:
  1. If an image is attached, extract structured evidence from it
     and store it in memory (grounding step).
  2. Check whether the request is answerable given context so far;
     if not, ask a clarifying question instead of guessing
     (ambiguity handling).
  3. Generate a draft, evidence-grounded answer (contextual
     reasoning).
  4. Validate the draft's claims against the evidence/context and
     correct anything unsupported (response validation).
  5. Store the turn and return the final answer + a trace.
"""

from dataclasses import dataclass, field
from typing import List, Optional

from anthropic import Anthropic

from .config import Config
from .memory import ConversationMemory
from .vision import ImageAnalyzer
from .ambiguity import AmbiguityDetector
from .reasoning import ReasoningEngine
from .validator import ResponseValidator


@dataclass
class TurnResult:
    answer: str
    asked_clarification: bool = False
    evidence_extracted: Optional[str] = None
    claims: List[str] = field(default_factory=list)
    unsupported_claims: List[str] = field(default_factory=list)
    trace: List[str] = field(default_factory=list)


class MultimodalAssistant:
    def __init__(self):
        Config.validate()
        client = Anthropic(api_key=Config.ANTHROPIC_API_KEY)
        self.memory = ConversationMemory(recent_turns_kept=Config.RECENT_TURNS_KEPT)
        self.vision = ImageAnalyzer(client)
        self.ambiguity = AmbiguityDetector(client)
        self.reasoning = ReasoningEngine(client)
        self.validator = ResponseValidator(client)
        self._client = client

    def _summarize_for_compression(self, existing_summary: str, fold_text: str) -> str:
        prompt = (
            "Fold the following older dialogue into a concise running summary "
            "that preserves any facts that might matter later "
            "(preferences, decisions, referenced images/objects). "
            "Keep it under 150 words.\n\n"
            f"Existing summary:\n{existing_summary or '(none yet)'}\n\n"
            f"Dialogue to fold in:\n{fold_text}"
        )
        response = self._client.messages.create(
            model=Config.REASONING_MODEL,
            max_tokens=250,
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(b.text for b in response.content if b.type == "text").strip()

    def process(self, user_message: str, image_path: Optional[str] = None) -> TurnResult:
        trace = []
        turn_id = self.memory.add_user_turn(user_message, had_image=bool(image_path))

        evidence_text = None
        # Step 1: ground on any new image before anything else.
        if image_path:
            trace.append("Image detected -> running vision extraction.")
            evidence_text = self.vision.analyze(image_path, user_question=user_message)
            self.memory.add_evidence(turn_id, source=f"image:{image_path}",
                                      description=evidence_text)
            trace.append(f"Evidence extracted: {evidence_text[:200]}...")

        context_block = self.memory.context_block()

        # Step 2: ambiguity check before committing to a full answer.
        verdict = self.ambiguity.check(context_block, user_message)
        if verdict.get("ambiguous"):
            trace.append(f"Ambiguity detected: {verdict.get('reason')}")
            question = verdict.get("clarifying_question") or (
                "Could you clarify what you'd like to know?"
            )
            self.memory.add_assistant_turn(question)
            return TurnResult(
                answer=question,
                asked_clarification=True,
                evidence_extracted=evidence_text,
                trace=trace,
            )
        trace.append("Ambiguity check passed -- sufficient context to answer.")

        # Step 3: contextual, evidence-grounded reasoning.
        draft = self.reasoning.generate(context_block, user_message)
        trace.append(f"Draft answer generated with {len(draft['claims'])} claim(s) to verify.")

        # Step 4: validate the draft's claims before returning it.
        verdict2 = self.validator.validate(context_block, draft["answer"], draft["claims"])
        if not verdict2.get("all_supported", True):
            trace.append(
                f"Validator flagged unsupported claim(s): {verdict2.get('unsupported_claims')}"
                " -- using corrected answer."
            )
        else:
            trace.append("Validator confirmed all claims are supported by evidence/context.")

        final_answer = verdict2.get("corrected_answer") or draft["answer"]
        self.memory.add_assistant_turn(final_answer)

        # Keep prompt size bounded for long conversations.
        self.memory.maybe_compress(self._summarize_for_compression)

        return TurnResult(
            answer=final_answer,
            asked_clarification=False,
            evidence_extracted=evidence_text,
            claims=draft["claims"],
            unsupported_claims=verdict2.get("unsupported_claims", []),
            trace=trace,
        )
