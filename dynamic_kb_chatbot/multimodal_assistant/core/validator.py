"""
Response validation.

Takes the draft answer + its list of claims from the reasoning step
and checks each claim against the actual stored evidence and
conversation context. This is a second, independent model call --
deliberately separate from the one that generated the answer -- so
the assistant is not just trusting its own first pass.

If unsupported claims are found, the validator returns a corrected
answer with those claims removed or hedged, rather than silently
passing along an unsupported statement.
"""

import json
import re

from anthropic import Anthropic

from .config import Config

VALIDATION_SYSTEM_PROMPT = """You are the validation component of a multi-modal \
AI assistant pipeline. You are given the visual evidence and context that were \
available, a draft answer, and the list of factual claims the draft relies on.

Check each claim against the evidence/context. A claim is SUPPORTED only if it \
is directly stated or clearly implied by the evidence/context. Anything else is \
UNSUPPORTED (including plausible-sounding guesses).

Respond with ONLY a JSON object, no other text, in exactly this shape:
{
  "all_supported": true or false,
  "unsupported_claims": ["...", ...],
  "corrected_answer": "the answer text, with any unsupported claims removed or
     explicitly hedged as uncertain -- identical to the draft if all_supported
     is true"
}"""


class ResponseValidator:
    def __init__(self, client: Anthropic = None):
        self.client = client or Anthropic(api_key=Config.ANTHROPIC_API_KEY)

    def validate(self, context_block: str, draft_answer: str, claims: list) -> dict:
        if not claims:
            # Nothing factual was asserted (e.g. a clarifying remark or
            # opinion) -- nothing to validate against evidence.
            return {"all_supported": True, "unsupported_claims": [],
                    "corrected_answer": draft_answer}

        claims_text = "\n".join(f"- {c}" for c in claims)
        response = self.client.messages.create(
            model=Config.VALIDATION_MODEL,
            max_tokens=Config.MAX_TOKENS_VALIDATION,
            system=VALIDATION_SYSTEM_PROMPT,
            messages=[
                {
                    "role": "user",
                    "content": (
                        f"Evidence/context:\n{context_block}\n\n"
                        f"Draft answer:\n{draft_answer}\n\n"
                        f"Claims to check:\n{claims_text}"
                    ),
                }
            ],
        )
        raw = "".join(b.text for b in response.content if b.type == "text").strip()

        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not match:
            # Fail safe: if validation itself is unparseable, don't
            # block the user -- return the draft, but flag it clearly
            # in the trace so this is visible during evaluation/demo.
            return {"all_supported": True, "unsupported_claims": [],
                    "corrected_answer": draft_answer,
                    "validation_error": "unparseable validator output"}
        try:
            parsed = json.loads(match.group(0))
        except json.JSONDecodeError:
            return {"all_supported": True, "unsupported_claims": [],
                    "corrected_answer": draft_answer,
                    "validation_error": "validator JSON failed to parse"}

        parsed.setdefault("all_supported", True)
        parsed.setdefault("unsupported_claims", [])
        parsed.setdefault("corrected_answer", draft_answer)
        return parsed
