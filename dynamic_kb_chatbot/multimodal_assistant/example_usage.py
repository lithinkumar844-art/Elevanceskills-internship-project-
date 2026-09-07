"""
Minimal example of using the assistant programmatically (no UI),
useful for demoing the pipeline's decision-making in a terminal or
for including sample output in the report.

Usage:
    python example_usage.py
    python example_usage.py path/to/image.jpg
"""

import sys

from core.orchestrator import MultimodalAssistant


def main():
    assistant = MultimodalAssistant()

    image_path = sys.argv[1] if len(sys.argv) > 1 else None

    turns = [
        ("What can you tell me about this image?", image_path),
        ("Is there any text visible in it?", None),
        ("What about the thing on the left?", None),  # deliberately ambiguous
    ]

    for message, img in turns:
        print(f"\nUSER: {message}" + (f"  [image: {img}]" if img else ""))
        result = assistant.process(message, image_path=img)
        print(f"ASSISTANT: {result.answer}")
        print("--- trace ---")
        for line in result.trace:
            print(f"  * {line}")
        if result.unsupported_claims:
            print(f"  ! corrected unsupported claims: {result.unsupported_claims}")


if __name__ == "__main__":
    main()
