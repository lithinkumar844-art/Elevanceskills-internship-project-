"""
main.py — Entry point for the Dynamic Knowledge Base Chatbot.

Usage:
    python main.py           # start chatbot + background scheduler
    python main.py --ingest  # run one-shot ingestion and exit
"""

from __future__ import annotations

import argparse
import logging
import sys

# ── Logging setup (before any imports that use it) ────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Dynamic KB Chatbot")
    parser.add_argument(
        "--ingest",
        action="store_true",
        help="Run a one-shot ingestion pass and exit (no chat loop).",
    )
    args = parser.parse_args()

    # ── Initialise vector store ───────────────────────────────────────────────
    from vectordb.store import VectorStore
    store = VectorStore()

    # ── One-shot ingestion mode ───────────────────────────────────────────────
    if args.ingest:
        from ingestion.scheduler import run_update
        run_update(store)
        logger.info("Ingestion complete. Chunks in store: %d", store.count())
        sys.exit(0)

    # ── Start background scheduler ────────────────────────────────────────────
    from ingestion.scheduler import KnowledgeBaseScheduler
    scheduler = KnowledgeBaseScheduler(store)
    scheduler.start(run_now=True)          # first update runs immediately

    # ── Start chatbot ─────────────────────────────────────────────────────────
    from chatbot.rag_chain import RAGChatbot
    bot = RAGChatbot(store)

    print("\n" + "=" * 60)
    print("  Dynamic Knowledge Base Chatbot")
    print("  Knowledge base auto-updates every",
          f"{__import__('config').UPDATE_INTERVAL_HOURS:.0f} hour(s)")
    print("  Type 'quit' to exit, 'clear' to reset conversation history")
    print("=" * 60 + "\n")

    try:
        while True:
            user_input = input("You: ").strip()

            if not user_input:
                continue
            if user_input.lower() in {"quit", "exit", "q"}:
                print("Goodbye!")
                break
            if user_input.lower() == "clear":
                bot.clear_history()
                print("Conversation history cleared.\n")
                continue
            if user_input.lower() == "status":
                print(f"Chunks in knowledge base: {store.count()}\n")
                continue

            result = bot.chat(user_input)
            print(f"\nBot: {result['answer']}")

            if result["sources"]:
                print("\nSources:")
                for src in result["sources"]:
                    print(f"  • {src}")
            print()

    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        scheduler.stop()


if __name__ == "__main__":
    main()
