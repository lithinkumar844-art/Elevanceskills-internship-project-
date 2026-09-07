"""
Configuration for the Multi-Modal AI Assistant.

Reads the Anthropic API key from the environment so it is never
hard-coded into source files. Copy `.env.example` to `.env` and fill
in your key, or export ANTHROPIC_API_KEY in your shell.
"""

import os

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # python-dotenv is optional; the app still works if the
    # environment variable is set some other way.
    pass


class Config:
    ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")

    # Model used for the main reasoning + vision calls.
    REASONING_MODEL = os.environ.get("ASSISTANT_MODEL", "claude-sonnet-4-6")

    # A cheaper/faster model could be swapped in here for the
    # validation pass if you want to save cost; using the same
    # model by default keeps validation quality high.
    VALIDATION_MODEL = os.environ.get("ASSISTANT_VALIDATION_MODEL", "claude-sonnet-4-6")

    MAX_TOKENS_REASONING = 1024
    MAX_TOKENS_VALIDATION = 512
    MAX_TOKENS_VISION = 768

    # How many of the most recent raw turns to keep verbatim in the
    # prompt before older turns get folded into a running summary.
    RECENT_TURNS_KEPT = 6

    @classmethod
    def validate(cls):
        if not cls.ANTHROPIC_API_KEY:
            raise EnvironmentError(
                "ANTHROPIC_API_KEY is not set. Create a .env file "
                "(see .env.example) or export it in your shell."
            )
