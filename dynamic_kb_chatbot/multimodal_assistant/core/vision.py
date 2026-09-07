"""
Vision / image-understanding module.

Responsible for turning a raw image into structured, reusable
"evidence" text: objects present, visible text, notable attributes,
and anything that looks relevant to answering questions about it.

Kept as its own module (rather than inline in the orchestrator) so
the extraction prompt and image-encoding logic can be tested and
tuned independently of the reasoning/validation logic.
"""

import base64
import mimetypes
from pathlib import Path

from anthropic import Anthropic

from .config import Config

EXTRACTION_SYSTEM_PROMPT = """You are the vision component of a larger AI assistant \
pipeline. Your only job is to extract objective, verifiable information from the \
image you are shown. Do not answer any user question and do not speculate about \
things you cannot see.

Return your findings as a short structured report with these sections:
- Objects/Subjects: concrete things visible in the image
- Text: any legible text in the image, transcribed exactly (or "none")
- Attributes: colors, counts, positions, condition, notable visual details
- Uncertain/Ambiguous: anything you cannot determine confidently, and why

Be precise and factual. If something is not visible or not certain, say so \
explicitly rather than guessing -- downstream steps rely on you not inventing \
details."""


class ImageAnalyzer:
    def __init__(self, client: Anthropic = None):
        self.client = client or Anthropic(api_key=Config.ANTHROPIC_API_KEY)

    @staticmethod
    def _encode_image(image_path: str):
        path = Path(image_path)
        media_type, _ = mimetypes.guess_type(str(path))
        if media_type not in ("image/png", "image/jpeg", "image/gif", "image/webp"):
            raise ValueError(
                f"Unsupported or undetected image type for {image_path!r}: {media_type}. "
                "Supported: png, jpeg, gif, webp."
            )
        data = base64.standard_b64encode(path.read_bytes()).decode("utf-8")
        return media_type, data

    def analyze(self, image_path: str, user_question: str = "") -> str:
        """
        Run one vision extraction call and return a structured text
        report describing what is in the image. `user_question` is
        passed only as light context to focus attention -- the model
        is still instructed to report facts, not answer the question.
        """
        media_type, data = self._encode_image(image_path)

        focus_hint = (
            f"\n\nThe user's current question, for context on what to focus on "
            f"(do not answer it, just extract relevant facts): {user_question!r}"
            if user_question else ""
        )

        response = self.client.messages.create(
            model=Config.REASONING_MODEL,
            max_tokens=Config.MAX_TOKENS_VISION,
            system=EXTRACTION_SYSTEM_PROMPT,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": media_type,
                                "data": data,
                            },
                        },
                        {
                            "type": "text",
                            "text": "Extract everything relevant from this image."
                            + focus_hint,
                        },
                    ],
                }
            ],
        )
        return "".join(block.text for block in response.content if block.type == "text")
