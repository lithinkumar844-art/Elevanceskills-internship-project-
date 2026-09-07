"""
Reasoning module.

Produces the assistant's draft answer, explicitly instructed to
separate claims that are grounded in evidence/context from anything
that would be speculation. The output includes a short list of
"claims" so the validator has discrete units to check rather than
having to re-parse free-flowing prose.
"""

from anthropic import Anthropic

from .config import Config

REASONING_SYSTEM_PROMPT = """You are the reasoning component of a multi-modal AI \
assistant. You are given: (1) conversation context, (2) visual evidence already \
extracted from any images the user has shared, and (3) the user's newest message.

Produce a helpful, evidence-based answer. Ground every factual claim about an \
image in the extracted evidence provided -- never invent visual details that \
aren't in the evidence. If the evidence is insufficient to fully answer, say so \
plainly rather than filling gaps with guesses.

Respond in exactly this format:

ANSWER:
<your natural-language answer to the user>

CLAIMS:
- <short factual claim your answer relies on>
- <short factual claim your answer relies on>
(list every distinct factual claim you made; if none, write "none")
"""


class ReasoningEngine:
    def __init__(self, client: Anthropic = None):
        self.client = client or Anthropic(api_key=Config.ANTHROPIC_API_KEY)

    def generate(self, context_block: str, user_message: str) -> dict:
        response = self.client.messages.create(
            model=Config.REASONING_MODEL,
            max_tokens=Config.MAX_TOKENS_REASONING,
            system=REASONING_SYSTEM_PROMPT,
            messages=[
                {
                    "role": "user",
                    "content": (
                        f"Context:\n{context_block}\n\n"
                        f"User's newest message: {user_message}"
                    ),
                }
            ],
        )
        raw = "".join(b.text for b in response.content if b.type == "text")
        return self._parse(raw)

    @staticmethod
    def _parse(raw: str) -> dict:
        answer, claims = raw, []
        if "CLAIMS:" in raw:
            answer_part, claims_part = raw.split("CLAIMS:", 1)
            answer = answer_part.replace("ANSWER:", "", 1).strip()
            for line in claims_part.strip().splitlines():
                line = line.strip().lstrip("-").strip()
                if line and line.lower() != "none":
                    claims.append(line)
        else:
            answer = raw.replace("ANSWER:", "", 1).strip()
        return {"answer": answer, "claims": claims, "raw": raw}
