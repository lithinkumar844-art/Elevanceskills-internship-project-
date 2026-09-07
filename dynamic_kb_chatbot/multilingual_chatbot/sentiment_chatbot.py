"""
Sentiment-Aware Customer Support Chatbot
=========================================
Detects sentiment (positive / negative / neutral) in customer messages and
tailors chatbot responses accordingly. Also flags strongly negative
messages for human escalation and logs interactions for later evaluation
(accuracy, response appropriateness, satisfaction proxy).

Run:
    pip install -r requirements.txt
    python sentiment_chatbot.py

If `vaderSentiment` isn't installed, the script automatically falls back
to a small built-in lexicon so it still runs (with reduced accuracy).
"""

import csv
import os
import random
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

# ---------------------------------------------------------------------------
# 1. Sentiment Analyzer
# ---------------------------------------------------------------------------

_VaderSentimentIntensityAnalyzer: Any = None
try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer as _VaderSentimentIntensityAnalyzer
    _VADER_AVAILABLE = True
except ImportError:
    _VADER_AVAILABLE = False


class FallbackSentimentAnalyzer:
    """Tiny lexicon-based analyzer used only if VADER isn't installed."""

    POSITIVE_WORDS = {
        "great", "good", "excellent", "awesome", "love", "thanks", "thank",
        "happy", "amazing", "perfect", "helpful", "wonderful", "fantastic",
        "nice", "appreciate", "pleased", "satisfied", "best",
    }
    NEGATIVE_WORDS = {
        "bad", "terrible", "awful", "hate", "angry", "worst", "horrible",
        "broken", "disappointed", "frustrated", "annoyed", "useless",
        "slow", "problem", "issue", "refund", "cancel", "furious", "scam",
        "never", "waste",
    }
    NEGATORS = {"not", "no", "never", "n't"}

    def polarity_scores(self, text: str) -> dict:
        words = re.findall(r"[a-zA-Z']+", text.lower())
        score = 0
        for i, w in enumerate(words):
            if w in self.POSITIVE_WORDS:
                score += 1
            elif w in self.NEGATIVE_WORDS:
                score -= 1
        # crude negation handling: flip previous word's contribution
        for i, w in enumerate(words):
            if w in self.NEGATORS and i + 1 < len(words):
                nxt = words[i + 1]
                if nxt in self.POSITIVE_WORDS:
                    score -= 2
                elif nxt in self.NEGATIVE_WORDS:
                    score += 2
        # exclamation marks amplify existing polarity
        if "!" in text:
            score += 1 if score > 0 else (-1 if score < 0 else 0)
        norm = max(-1.0, min(1.0, score / 4.0))
        return {"compound": norm}


# Rule-based override list: dismissive/hostile phrases that a lexicon-based
# sentiment model (VADER included) tends to score as neutral, because they
# score emotional *valence* words, not hostility/rudeness as a dimension.
# "shut up" or "leave me alone" contain no negative-valence words, but no
# support agent would call them neutral. This override forces such turns
# into negative territory before thresholding, and is a common production
# pattern for lexicon/ML sentiment models. Keep this list free of slurs —
# it's meant to catch dismissiveness, not to be a profanity filter.
HOSTILE_PHRASES = {
    "shut up", "shut it", "shut the hell up", "go away", "get lost",
    "leave me alone", "stop bothering me", "screw you", "go to hell",
    "i hate this bot", "i hate you", "you're useless", "you are useless",
    "this is useless", "waste of time", "stupid bot", "dumb bot",
}


def _contains_hostile_phrase(text: str) -> bool:
    lowered = text.lower()
    return any(phrase in lowered for phrase in HOSTILE_PHRASES)


class SentimentAnalyzer:
    """Wraps VADER (preferred) or the fallback analyzer with one interface."""

    POS_THRESHOLD = 0.15
    NEG_THRESHOLD = -0.15
    STRONG_NEG_THRESHOLD = -0.55
    HOSTILE_OVERRIDE_SCORE = -0.7  # forces hostile phrases into strong_negative

    def __init__(self):
        if _VADER_AVAILABLE:
            self.engine = _VaderSentimentIntensityAnalyzer()
            self.backend = "vader"
        else:
            self.engine = FallbackSentimentAnalyzer()
            self.backend = "fallback-lexicon"

    def analyze(self, text: str) -> "SentimentResult":
        scores = self.engine.polarity_scores(text)
        compound = scores["compound"]

        if _contains_hostile_phrase(text):
            compound = min(compound, self.HOSTILE_OVERRIDE_SCORE)

        if compound <= self.STRONG_NEG_THRESHOLD:
            label = "strong_negative"
        elif compound <= self.NEG_THRESHOLD:
            label = "negative"
        elif compound >= self.POS_THRESHOLD:
            label = "positive"
        else:
            label = "neutral"

        return SentimentResult(text=text, compound=compound, label=label)


@dataclass
class SentimentResult:
    text: str
    compound: float
    label: str  # positive | neutral | negative | strong_negative


# ---------------------------------------------------------------------------
# 2. Lightweight intent detection (keeps the bot useful, not just reactive)
# ---------------------------------------------------------------------------

INTENT_PATTERNS = {
    "order_status": r"\b(order|shipment|package|delivery|tracking)\b",
    "refund": r"\b(refund|money back|reimburse)\b",
    "cancel": r"\b(cancel|cancellation)\b",
    "billing": r"\b(bill|charge|invoice|payment|subscription)\b",
    "greeting": r"\b(hi|hello|hey)\b",
    "thanks": r"\b(thanks|thank you|thx)\b",
    "goodbye": r"\b(bye|goodbye|see ya)\b",
}


def detect_intent(text: str) -> str:
    if _contains_hostile_phrase(text):
        return "hostile"
    lowered = text.lower()
    for intent, pattern in INTENT_PATTERNS.items():
        if re.search(pattern, lowered):
            return intent
    return "general"


# ---------------------------------------------------------------------------
# 3. Response templates: sentiment x intent
# ---------------------------------------------------------------------------

RESPONSES = {
    "strong_negative": {
        "default": [
            "I'm really sorry you're dealing with this — that's not the experience we want for you. "
            "I'm connecting you with a human specialist right now who can resolve this quickly.",
        ],
        "hostile": [
            "I hear that you're frustrated, and I'm sorry this conversation hasn't helped. "
            "I'm connecting you with a human agent right now.",
        ],
        "refund": [
            "I completely understand your frustration about the refund. I'm escalating this to a "
            "specialist immediately so it gets resolved without further delay.",
        ],
        "billing": [
            "I'm sorry — billing issues like this are stressful. I'm looping in a billing specialist "
            "right now to fix this for you.",
        ],
    },
    "negative": {
        "default": [
            "I'm sorry to hear that. Let's get this sorted out — can you tell me a bit more about what happened?",
            "That sounds frustrating, and I want to help fix it. Could you share more details?",
        ],
        "order_status": [
            "Sorry for the trouble with your order. Let me look into the tracking details — "
            "could you share your order number?",
        ],
        "cancel": [
            "I'm sorry to hear you want to cancel. I can help with that, or if something specific "
            "went wrong, I'd like to try to fix it first — what happened?",
        ],
    },
    "neutral": {
        "default": [
            "Got it. Could you tell me a bit more so I can help?",
            "Thanks for reaching out — what can I help you with today?",
        ],
        "order_status": [
            "Sure, I can check that. Could you share your order number?",
        ],
        "greeting": [
            "Hi there! How can I help you today?",
        ],
        "billing": [
            "I can help with that. Could you tell me which charge or invoice you're asking about?",
        ],
    },
    "positive": {
        "default": [
            "Glad to hear that! Is there anything else I can help you with?",
            "That's great to hear! Let me know if there's anything else you need.",
        ],
        "thanks": [
            "You're very welcome! Happy to help anytime.",
        ],
        "goodbye": [
            "Glad I could help — have a wonderful day!",
        ],
    },
}


def pick_response(sentiment_label: str, intent: str) -> str:
    bucket = RESPONSES.get(sentiment_label, RESPONSES["neutral"])
    options = bucket.get(intent, bucket["default"])
    return random.choice(options)


# ---------------------------------------------------------------------------
# 4. Chatbot orchestration + logging for evaluation
# ---------------------------------------------------------------------------

@dataclass
class Turn:
    timestamp: str
    user_text: str
    sentiment_label: str
    sentiment_score: float
    intent: str
    bot_response: str
    escalated: bool


class SentimentChatbot:
    def __init__(self, log_path: str = "chat_log.csv"):
        self.analyzer = SentimentAnalyzer()
        self.log_path = log_path
        self.history: list[Turn] = []
        self._init_log()

    def _init_log(self):
        new_file = not os.path.exists(self.log_path)
        if new_file:
            with open(self.log_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(
                    ["timestamp", "user_text", "sentiment_label",
                     "sentiment_score", "intent", "bot_response", "escalated"]
                )

    def _log(self, turn: Turn):
        with open(self.log_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [turn.timestamp, turn.user_text, turn.sentiment_label,
                 f"{turn.sentiment_score:.3f}", turn.intent,
                 turn.bot_response, turn.escalated]
            )

    def respond(self, user_text: str) -> Turn:
        result = self.analyzer.analyze(user_text)
        intent = detect_intent(user_text)
        response = pick_response(result.label, intent)
        escalated = result.label == "strong_negative"

        turn = Turn(
            timestamp=datetime.now().isoformat(timespec="seconds"),
            user_text=user_text,
            sentiment_label=result.label,
            sentiment_score=result.compound,
            intent=intent,
            bot_response=response,
            escalated=escalated,
        )
        self.history.append(turn)
        self._log(turn)
        return turn

    def session_summary(self) -> dict:
        if not self.history:
            return {}
        counts = {"positive": 0, "neutral": 0, "negative": 0, "strong_negative": 0}
        for t in self.history:
            counts[t.sentiment_label] += 1
        avg_score = sum(t.sentiment_score for t in self.history) / len(self.history)
        escalations = sum(1 for t in self.history if t.escalated)
        return {
            "turns": len(self.history),
            "sentiment_breakdown": counts,
            "average_sentiment_score": round(avg_score, 3),
            "escalations": escalations,
        }


# ---------------------------------------------------------------------------
# 5. CLI demo
# ---------------------------------------------------------------------------

def main():
    bot = SentimentChatbot()
    print(f"[sentiment engine: {bot.analyzer.backend}]")
    print("Sentiment-Aware Chatbot — type 'quit' to exit.\n")

    while True:
        try:
            user_text = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not user_text:
            continue
        if user_text.lower() in {"quit", "exit"}:
            break

        turn = bot.respond(user_text)
        tag = f"[{turn.sentiment_label} | score {turn.sentiment_score:+.2f} | intent: {turn.intent}]"
        print(f"Bot {tag}: {turn.bot_response}")
        if turn.escalated:
            print("   >> Flagged for human agent escalation.")

    print("\n--- Session Summary ---")
    for k, v in bot.session_summary().items():
        print(f"{k}: {v}")
    print(f"\nFull interaction log saved to: {bot.log_path}")


if __name__ == "__main__":
    main()
