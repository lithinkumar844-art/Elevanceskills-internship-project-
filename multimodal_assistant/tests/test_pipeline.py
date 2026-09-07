"""
Offline test suite for the assistant's decision pipeline.

These tests do NOT call the real Anthropic API -- every sub-component
(vision, ambiguity, reasoning, validator) is mocked out at the method
level, so you can verify the orchestrator's branching logic (does it
ask a clarifying question? does it use the corrected answer? does it
store evidence?) without spending API credits or needing network
access. Useful for a quick sanity check before/after making changes,
and as evidence in your report that the pipeline behaves as designed.

Run with:
    python -m unittest tests/test_pipeline.py -v
"""

import os
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Make sure Config.validate() doesn't fail just because no real key is set.
os.environ.setdefault("ANTHROPIC_API_KEY", "test-key-for-offline-tests")

from core.orchestrator import MultimodalAssistant  # noqa: E402


def make_assistant():
    """Build an assistant with a real object graph but a fake Anthropic client
    (no network calls happen because every component method is mocked in
    each test before it's used)."""
    with patch("core.orchestrator.Anthropic") as MockAnthropic:
        MockAnthropic.return_value = object()  # never actually called directly
        assistant = MultimodalAssistant()
    return assistant


class TestAmbiguityBranch(unittest.TestCase):
    def test_ambiguous_request_returns_clarifying_question_without_reasoning(self):
        assistant = make_assistant()

        with patch.object(assistant.ambiguity, "check", return_value={
            "ambiguous": True,
            "clarifying_question": "Which item do you mean?",
            "reason": "no clear referent",
        }) as mock_check, \
             patch.object(assistant.reasoning, "generate") as mock_reasoning:

            result = assistant.process("what about the thing on the left?")

        mock_check.assert_called_once()
        mock_reasoning.assert_not_called()  # must short-circuit, not reason blindly
        self.assertTrue(result.asked_clarification)
        self.assertEqual(result.answer, "Which item do you mean?")


class TestNormalReasoningFlow(unittest.TestCase):
    def test_supported_answer_passes_through_unchanged(self):
        assistant = make_assistant()

        with patch.object(assistant.ambiguity, "check", return_value={"ambiguous": False}), \
             patch.object(assistant.reasoning, "generate", return_value={
                 "answer": "The car in the photo is red.",
                 "claims": ["The car is red."],
                 "raw": "...",
             }), \
             patch.object(assistant.validator, "validate", return_value={
                 "all_supported": True,
                 "unsupported_claims": [],
                 "corrected_answer": "The car in the photo is red.",
             }) as mock_validate:

            result = assistant.process("what color is the car?")

        mock_validate.assert_called_once()
        self.assertFalse(result.asked_clarification)
        self.assertEqual(result.answer, "The car in the photo is red.")
        self.assertEqual(result.unsupported_claims, [])


class TestValidationCorrection(unittest.TestCase):
    def test_unsupported_claim_gets_corrected_answer_used(self):
        assistant = make_assistant()

        with patch.object(assistant.ambiguity, "check", return_value={"ambiguous": False}), \
             patch.object(assistant.reasoning, "generate", return_value={
                 "answer": "The car is red and has a sunroof.",
                 "claims": ["The car is red.", "The car has a sunroof."],
                 "raw": "...",
             }), \
             patch.object(assistant.validator, "validate", return_value={
                 "all_supported": False,
                 "unsupported_claims": ["The car has a sunroof."],
                 "corrected_answer": "The car is red. I can't confirm whether it has a sunroof.",
             }):

            result = assistant.process("tell me about the car")

        # The orchestrator must prefer the validator's corrected answer,
        # not the reasoning engine's original (unverified) draft.
        self.assertIn("can't confirm", result.answer)
        self.assertEqual(result.unsupported_claims, ["The car has a sunroof."])


class TestEvidenceStorage(unittest.TestCase):
    def test_image_evidence_is_extracted_and_stored_in_memory(self):
        assistant = make_assistant()

        with patch.object(assistant.vision, "analyze",
                           return_value="Objects: a red sedan. Text: none.") as mock_vision, \
             patch.object(assistant.ambiguity, "check", return_value={"ambiguous": False}), \
             patch.object(assistant.reasoning, "generate", return_value={
                 "answer": "It's a red sedan.", "claims": ["It is a red sedan."], "raw": "...",
             }), \
             patch.object(assistant.validator, "validate", return_value={
                 "all_supported": True, "unsupported_claims": [],
                 "corrected_answer": "It's a red sedan.",
             }):

            result = assistant.process("what's in this image?", image_path="fake_path.jpg")

        mock_vision.assert_called_once()
        self.assertEqual(len(assistant.memory.evidence_store), 1)
        self.assertIn("red sedan", assistant.memory.evidence_store[0].description)
        self.assertIsNotNone(result.evidence_extracted)

    def test_evidence_persists_across_turns_for_followup_questions(self):
        assistant = make_assistant()

        with patch.object(assistant.vision, "analyze", return_value="Objects: a red sedan."), \
             patch.object(assistant.ambiguity, "check", return_value={"ambiguous": False}), \
             patch.object(assistant.reasoning, "generate", return_value={
                 "answer": "It's a red sedan.", "claims": [], "raw": "...",
             }), \
             patch.object(assistant.validator, "validate", return_value={
                 "all_supported": True, "unsupported_claims": [], "corrected_answer": "It's a red sedan.",
             }):
            assistant.process("what's in this image?", image_path="fake_path.jpg")

        # Second turn: no new image, but evidence from turn 1 should still
        # be visible in the context block built for the next call.
        context = assistant.memory.context_block()
        self.assertIn("red sedan", context)


if __name__ == "__main__":
    unittest.main()
