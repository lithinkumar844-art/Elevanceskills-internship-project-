"""
Ambiguity handling.

Before the assistant commits to a full reasoning pass, it checks
whether the user's request is answerable given the current context
(dialogue history + collected evidence). If not, it produces a
clarifying question instead of guessing -- this is the behavior
that distinguishes "intelligent decision-making" from "always emit
some answer."

The check itself is done with a constrained, cheap LLM call that
must return one of two shapes, so it's easy to branch on
programmatically instead of parsing free text.
"""

import json
import re

from anthropic import Anthropic

from .config import Config

AMBIGUITY_SYSTEM_PROMPT = """You are the ambiguity-checking component of an AI \
assistant pipeline. You will be shown the conversation context (including any \
visual evidence already extracted) and the user's newest message.

Decide whether the assistant has enough information to give a grounded, useful \
answer, or whether the request is ambiguous / underspecified / refers to \
something not present in the context (e.g. asks about "the image" when no \
image or evidence exists yet, or uses a pronoun with no clear referent).

Respond with ONLY a JSON object, no other text, in exactly this shape:
{"ambiguous": true or false, "clarifying_question": "..." or null, "reason": "..."}

Only mark something ambiguous if answering would require guessing at missing \
information. Do not mark it ambiguous just because the question is broad or \
open-ended -- broad questions can still be answered from available context."""


class AmbiguityDetector:
    def __init__(self, client: Anthropic = None):
        self.client = client or Anthropic(api_key=Config.ANTHROPIC_API_KEY)

    def check(self, context_block: str, user_message: str) -> dict:
        response = self.client.messages.create(
            model=Config.REASONING_MODEL,
            max_tokens=300,
            system=AMBIGUITY_SYSTEM_PROMPT,
            messages=[
                {
                    "role": "user",
                    "content": (
                        f"Context so far:\n{context_block}\n\n"
                        f"User's newest message: {user_message!r}"
                    ),
                }
            ],
        )
        raw = "".join(b.text for b in response.content if b.type == "text").strip()

        # The model is instructed to return bare JSON, but be defensive
        # in case it wraps it in a code fence or adds stray text.
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not match:
            # Fail safe: if we can't parse the verdict, don't block the
            # user -- treat as unambiguous and let reasoning proceed.
            return {"ambiguous": False, "clarifying_question": None,
                     "reason": "ambiguity check returned unparseable output"}
        try:
            parsed = json.loads(match.group(0))
        except json.JSONDecodeError:
            return {"ambiguous": False, "clarifying_question": None,
                     "reason": "ambiguity check JSON failed to parse"}

        parsed.setdefault("ambiguous", False)
        parsed.setdefault("clarifying_question", None)
        parsed.setdefault("reason", "")
        return parsed
