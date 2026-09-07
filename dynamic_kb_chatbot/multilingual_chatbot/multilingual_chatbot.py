"""
Multilingual extension for the Sentiment-Aware Chatbot
=======================================================
Adds automatic language identification, mixed-language ("code-switched")
input handling, cross-lingual translation, and context retention across
language switches — on top of the existing `sentiment_chatbot.py` engine.

Design (pivot architecture):
    user text (any language)
        -> detect language (per message, and per sentence for mixed input)
        -> translate to English (pivot)
        -> run EXISTING sentiment + intent logic unchanged (English)
        -> resolve ambiguous/contextless queries using conversation memory
        -> pick a response:
             1) a hand-localized native template if one exists (best quality)
             2) else machine-translate the English template into the
                user's language (broad coverage, lower polish)
        -> reply in the language the user is currently using

This keeps sentiment/intent detection accurate (it always runs on English,
where the existing lexicon/VADER was tuned) while letting the bot speak
back naturally in Spanish, French, German, or Hindi — and fall back
gracefully to English for anything else.

Open-source components used (all optional / auto-detected at runtime):
    - langdetect         -> statistical language identification
    - transformers +     -> MarianMT (Helsinki-NLP/opus-mt-*) translation
      sentencepiece
If none of these are installed, the module degrades to:
    - a stopword/character-based heuristic language detector
    - a small built-in phrase dictionary for translation
...so the demo still runs end-to-end with zero downloads.

Run:
    pip install -r requirements.txt      # includes optional NLP deps
    python multilingual_chatbot.py
"""

import os
import re
from dataclasses import dataclass, field
from typing import Any, Optional

from sentiment_chatbot import (
    SentimentAnalyzer,
    SentimentResult,
    detect_intent,
    pick_response as pick_response_en,
    RESPONSES,
)

# ---------------------------------------------------------------------------
# 1. Language identification
# ---------------------------------------------------------------------------

SUPPORTED_LANGUAGES = {
    "en": "English",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "hi": "Hindi",
}

try:
    from langdetect import detect as _langdetect_detect, DetectorFactory
    DetectorFactory.seed = 0  # deterministic results
    _LANGDETECT_AVAILABLE = True
except ImportError:
    _langdetect_detect: Any = None
    _LANGDETECT_AVAILABLE = False


# Heuristic fallback: stopword overlap + script detection.
# Works with zero downloads / no internet, at reduced accuracy — this is
# what the demo below actually exercises in an offline environment.
_STOPWORDS = {
    "en": {"the", "is", "and", "you", "my", "i", "what", "where", "how",
           "please", "thanks", "hello", "order", "status", "not", "can"},
    "es": {"el", "la", "es", "y", "tú", "mi", "yo", "qué", "dónde", "cómo",
           "por", "favor", "gracias", "hola", "pedido", "estado", "no", "puedo"},
    "fr": {"le", "la", "est", "et", "vous", "mon", "je", "quoi", "où",
           "comment", "s'il", "plaît", "merci", "bonjour", "commande",
           "statut", "ne", "pas"},
    "de": {"der", "die", "das", "ist", "und", "du", "mein", "ich", "was",
           "wo", "wie", "bitte", "danke", "hallo", "bestellung", "status",
           "nicht", "kann"},
}

_DEVANAGARI_RE = re.compile(r"[\u0900-\u097F]")


def _heuristic_detect(text: str) -> str:
    if _DEVANAGARI_RE.search(text):
        return "hi"
    words = set(re.findall(r"[a-zA-ZÀ-ÿ']+", text.lower()))
    if not words:
        return "en"
    scores = {lang: len(words & sw) for lang, sw in _STOPWORDS.items()}
    best_lang = max(scores, key=lambda lang: scores[lang])
    return best_lang if scores[best_lang] > 0 else "en"


def detect_language(text: str) -> str:
    """Returns an ISO 639-1 code. Falls back to 'en' if unsure/unsupported."""
    if _LANGDETECT_AVAILABLE:
        try:
            code = _langdetect_detect(text)
            return code if code in SUPPORTED_LANGUAGES else _heuristic_detect(text)
        except Exception:
            pass
    return _heuristic_detect(text)


def split_sentences(text: str) -> list:
    """Naive sentence splitter used for per-sentence language ID on
    mixed-language ("code-switched") messages."""
    parts = re.split(r"(?<=[.!?।])\s+", text.strip())
    return [p for p in parts if p]


def detect_mixed_languages(text: str) -> list:
    """Returns [(sentence, lang_code), ...] — lets the bot notice when a
    single message switches languages mid-sentence-list."""
    sentences = split_sentences(text) or [text]
    return [(s, detect_language(s)) for s in sentences]


# ---------------------------------------------------------------------------
# 2. Translation layer (pivot to/from English)
# ---------------------------------------------------------------------------

try:
    from transformers import MarianMTModel, MarianTokenizer
    _MARIAN_AVAILABLE = True
except ImportError:
    MarianMTModel: Any = None
    MarianTokenizer: Any = None
    _MARIAN_AVAILABLE = False


# Small offline fallback dictionary — word/phrase level, substring-matched.
# This is intentionally limited; it exists purely so the module runs with
# no internet access. Real deployments should rely on MarianMT (or any
# other open-source MT model) installed via requirements.txt — the
# fallback below is deliberately not a substitute for real MT, only a
# way to exercise the full pipeline (detect -> pivot -> analyze -> reply)
# offline. Longer phrases are listed before the single words they contain
# so phrase-level meaning wins when both would match.
_PHRASE_DICT = {
    ("es", "en"): {
        "cuál es el estado de mi pedido": "what is the status of my order",
        "quiero un reembolso": "i want a refund",
        "mi pedido está retrasado": "my order is late",
        "esto es terrible": "this is terrible",
        "estoy muy enojado": "i am very angry furious",
        "muchas gracias": "thank you very much",
        "hola": "hello", "gracias": "thanks", "adiós": "goodbye",
        "pedido": "order", "retrasado": "late", "enojado": "angry",
        "estado": "status", "reembolso": "refund", "terrible": "terrible",
        "cancelar": "cancel", "factura": "billing invoice",
        "por favor": "please", "ayuda": "help", "no puedo": "cannot",
        "problema": "problem",
    },
    ("fr", "en"): {
        "quel est le statut de ma commande": "what is the status of my order",
        "je veux un remboursement": "i want a refund",
        "ma commande est en retard": "my order is late",
        "c'est terrible": "this is terrible",
        "je suis très en colère": "i am very angry furious",
        "bonjour": "hello", "merci": "thanks", "au revoir": "goodbye",
        "commande": "order", "retard": "late", "colère": "angry",
        "statut": "status", "remboursement": "refund", "terrible": "terrible",
        "annuler": "cancel", "facture": "billing invoice",
        "s'il vous plaît": "please", "problème": "problem",
    },
    ("de", "en"): {
        "was ist der status meiner bestellung": "what is the status of my order",
        "ich möchte eine rückerstattung": "i want a refund",
        "meine bestellung ist verspätet": "my order is late",
        "das ist schrecklich": "this is terrible",
        "ich bin sehr wütend": "i am very angry furious",
        "hallo": "hello", "danke": "thanks", "tschüss": "goodbye",
        "bestellung": "order", "verspätet": "late", "wütend": "angry",
        "status": "status", "rückerstattung": "refund", "schrecklich": "terrible",
        "stornieren": "cancel", "rechnung": "billing invoice",
        "bitte": "please", "problem": "problem",
    },
    ("hi", "en"): {
        "मेरे ऑर्डर की स्थिति क्या है": "what is the status of my order",
        "मुझे रिफंड चाहिए": "i want a refund",
        "मेरा ऑर्डर देर से आया है": "my order is late",
        "यह भयानक है": "this is terrible",
        "मैं बहुत गुस्से में हूं": "i am very angry furious",
        "नमस्ते": "hello", "धन्यवाद": "thanks",
        "ऑर्डर": "order", "स्थिति": "status", "रिफंड": "refund",
        "भयानक": "terrible", "मदद": "help",
    },
}
# Reverse dictionaries (en -> other) built from the same phrase pairs —
# used only as a last-resort fallback when no NATIVE_RESPONSES template
# exists for a given (sentiment, intent, language) combination.
_PHRASE_DICT_REV = {
    (src[1], src[0]): {v: k for k, v in table.items()}
    for src, table in _PHRASE_DICT.items()
}


def _dict_translate(text: str, table: dict) -> str:
    """Substring-level replacement, longest keys first, so multi-word
    phrases match before the single words they're built from."""
    result = text.lower()
    for key in sorted(table.keys(), key=len, reverse=True):
        if key in result:
            result = result.replace(key, table[key])
    return result


class Translator:
    """MarianMT-backed translator with a lightweight offline fallback.

    `translate(text, src, tgt)` — src/tgt are ISO 639-1 codes.
    Caches loaded models since each language pair is a separate MarianMT
    checkpoint (e.g. Helsinki-NLP/opus-mt-es-en, opus-mt-en-es, ...).
    """

    def __init__(self):
        self._models = {}
        self.backend = "marianmt" if _MARIAN_AVAILABLE else "phrase-dict-fallback"
        self.allow_model_downloads = os.getenv("CHATBOT_ALLOW_MODEL_DOWNLOADS") == "1"

    def _load(self, src: str, tgt: str):
        key = (src, tgt)
        if key not in self._models:
            name = f"Helsinki-NLP/opus-mt-{src}-{tgt}"
            load_options = {} if self.allow_model_downloads else {"local_files_only": True}
            tok = MarianTokenizer.from_pretrained(name, **load_options)
            model = MarianMTModel.from_pretrained(name, **load_options)
            self._models[key] = (tok, model)
        return self._models[key]

    def translate(self, text: str, src: str, tgt: str) -> str:
        if src == tgt or not text.strip():
            return text

        if _MARIAN_AVAILABLE:
            try:
                tok, model = self._load(src, tgt)
                batch = tok([text], return_tensors="pt", padding=True)
                gen = model.generate(**batch, max_new_tokens=128)
                return tok.decode(gen[0], skip_special_tokens=True)
            except Exception as e:
                # Model pair may not exist on the hub, or no internet to
                # download it — fall through to the phrase-dict fallback.
                pass

        table = _PHRASE_DICT.get((src, tgt)) or _PHRASE_DICT_REV.get((src, tgt))
        if table:
            return _dict_translate(text, table)
        # Last resort: no MT model and no dictionary for this pair —
        # return the original text unchanged rather than fail the turn.
        return text


# ---------------------------------------------------------------------------
# 3. Localized native response templates (best quality, hand-written)
# ---------------------------------------------------------------------------
# Only the highest-traffic sentiment/intent combinations are localized by
# hand; everything else falls back to translating the English template
# from RESPONSES (imported from sentiment_chatbot.py) via the Translator.

NATIVE_RESPONSES = {
    "es": {
        ("strong_negative", "default"): "Lamento mucho lo sucedido. Te estoy conectando de inmediato con un especialista humano.",
        ("strong_negative", "refund"): "Entiendo tu frustración con el reembolso. Escalo esto ahora mismo a un especialista.",
        ("strong_negative", "hostile"): "Entiendo que estás frustrado/a y lamento que esta conversación no haya ayudado. Te conecto con un agente humano ahora mismo.",
        ("negative", "default"): "Lamento escuchar eso. ¿Puedes contarme más para poder ayudarte?",
        ("negative", "order_status"): "Siento el inconveniente con tu pedido. ¿Me compartes tu número de pedido?",
        ("neutral", "default"): "Entendido. ¿En qué puedo ayudarte?",
        ("neutral", "greeting"): "¡Hola! ¿En qué puedo ayudarte hoy?",
        ("neutral", "order_status"): "Claro, puedo revisar eso. ¿Cuál es tu número de pedido?",
        ("positive", "default"): "¡Me alegra escuchar eso! ¿Necesitas algo más?",
        ("positive", "thanks"): "¡De nada! Estoy aquí para ayudarte cuando lo necesites.",
    },
    "fr": {
        ("strong_negative", "default"): "Je suis vraiment désolé pour ce désagrément. Je vous mets en relation avec un spécialiste humain immédiatement.",
        ("strong_negative", "refund"): "Je comprends votre frustration concernant le remboursement. J'escalade cela à un spécialiste tout de suite.",
        ("strong_negative", "hostile"): "Je comprends votre frustration et je suis désolé que cette conversation n'ait pas aidé. Je vous mets en relation avec un agent humain maintenant.",
        ("negative", "default"): "Je suis désolé de l'apprendre. Pouvez-vous m'en dire plus ?",
        ("negative", "order_status"): "Désolé pour ce souci avec votre commande. Pouvez-vous me donner votre numéro de commande ?",
        ("neutral", "default"): "D'accord. Comment puis-je vous aider ?",
        ("neutral", "greeting"): "Bonjour ! Comment puis-je vous aider aujourd'hui ?",
        ("neutral", "order_status"): "Bien sûr, je peux vérifier cela. Quel est votre numéro de commande ?",
        ("positive", "default"): "Ravi de l'entendre ! Puis-je faire autre chose pour vous ?",
        ("positive", "thanks"): "Avec plaisir ! N'hésitez pas si vous avez besoin d'autre chose.",
    },
    "de": {
        ("strong_negative", "default"): "Das tut mir wirklich leid. Ich verbinde Sie sofort mit einem menschlichen Spezialisten.",
        ("strong_negative", "refund"): "Ich verstehe Ihren Frust bezüglich der Rückerstattung. Ich leite dies sofort an einen Spezialisten weiter.",
        ("strong_negative", "hostile"): "Ich verstehe, dass Sie frustriert sind, und es tut mir leid, dass dieses Gespräch nicht geholfen hat. Ich verbinde Sie jetzt mit einem menschlichen Mitarbeiter.",
        ("negative", "default"): "Das tut mir leid. Können Sie mir mehr Details geben?",
        ("negative", "order_status"): "Entschuldigung für das Problem mit Ihrer Bestellung. Wie lautet Ihre Bestellnummer?",
        ("neutral", "default"): "Verstanden. Wie kann ich Ihnen helfen?",
        ("neutral", "greeting"): "Hallo! Wie kann ich Ihnen heute helfen?",
        ("neutral", "order_status"): "Klar, das kann ich prüfen. Wie lautet Ihre Bestellnummer?",
        ("positive", "default"): "Das freut mich zu hören! Kann ich sonst noch etwas für Sie tun?",
        ("positive", "thanks"): "Gern geschehen! Ich helfe jederzeit weiter.",
    },
    "hi": {
        ("strong_negative", "default"): "मुझे बहुत खेद है। मैं आपको तुरंत एक मानव विशेषज्ञ से जोड़ रहा/रही हूं।",
        ("strong_negative", "refund"): "रिफंड को लेकर आपकी परेशानी समझता/समझती हूं। मैं इसे अभी एक विशेषज्ञ के पास भेज रहा/रही हूं।",
        ("strong_negative", "hostile"): "मैं समझता/समझती हूं कि आप निराश हैं, और मुझे खेद है कि यह बातचीत मददगार नहीं रही। मैं अभी आपको एक मानव एजेंट से जोड़ रहा/रही हूं।",
        ("negative", "default"): "यह सुनकर खेद है। क्या आप मुझे और बता सकते हैं?",
        ("negative", "order_status"): "आपके ऑर्डर में हुई परेशानी के लिए क्षमा करें। कृपया अपना ऑर्डर नंबर बताएं।",
        ("neutral", "default"): "ठीक है। मैं आपकी कैसे मदद कर सकता/सकती हूं?",
        ("neutral", "greeting"): "नमस्ते! आज मैं आपकी कैसे मदद कर सकता/सकती हूं?",
        ("neutral", "order_status"): "ज़रूर, मैं जांच सकता/सकती हूं। कृपया अपना ऑर्डर नंबर बताएं।",
        ("positive", "default"): "यह सुनकर अच्छा लगा! क्या मैं और किसी चीज़ में मदद कर सकता/सकती हूं?",
        ("positive", "thanks"): "आपका स्वागत है! जब भी ज़रूरत हो, बताइए।",
    },
}


# ---------------------------------------------------------------------------
# 4. Conversation memory (context retention across language switches)
# ---------------------------------------------------------------------------

@dataclass
class ConversationContext:
    last_intent: Optional[str] = None
    last_task_intent: Optional[str] = None     # sticky topic, survives social turns
    last_entity: Optional[str] = None          # e.g. an order number mentioned earlier
    last_language: str = "en"
    turn_count: int = 0
    language_switches: int = 0
    seen_languages: set = field(default_factory=set)


AMBIGUOUS_INTENTS = {"general"}
SOCIAL_INTENTS = {"greeting", "thanks", "goodbye"}
TASK_INTENTS = {"order_status", "refund", "cancel", "billing"}
ORDER_NUMBER_RE = re.compile(r"\b([A-Z]{0,3}\d{4,})\b")


def resolve_intent(current_intent: str, english_text: str, ctx: ConversationContext) -> str:
    """Cross-lingual ambiguity resolution: if this turn's (pivoted) text
    doesn't clearly state an intent, fall back to the ongoing task topic —
    this is what lets 'what about my order?' -> (thanks, in English) ->
    (later, in French) '¿Et le statut ?' still resolve to order_status,
    without a passing 'thanks'/'hello' in between overwriting the topic."""
    if current_intent not in AMBIGUOUS_INTENTS:
        return current_intent
    if ctx.last_task_intent:
        return ctx.last_task_intent
    return current_intent


# ---------------------------------------------------------------------------
# 5. Multilingual chatbot orchestration
# ---------------------------------------------------------------------------

@dataclass
class MultilingualTurn:
    user_text: str
    detected_language: str
    mixed_segments: list
    pivot_english_text: str
    sentiment_label: str
    sentiment_score: float
    intent: str
    resolved_intent: str
    bot_response: str
    response_language: str
    escalated: bool
    language_switched: bool


class MultilingualChatbot:
    def __init__(self):
        self.analyzer = SentimentAnalyzer()
        self.translator = Translator()
        self.ctx = ConversationContext()
        self.history: list = []

    def _to_english(self, text: str) -> tuple:
        """Translates (possibly mixed-language) input to English for
        analysis. Returns (english_text, primary_lang, mixed_segments)."""
        segments = detect_mixed_languages(text)
        langs_present = {lang for _, lang in segments}
        primary_lang = max(langs_present, key=lambda l: sum(
            len(s) for s, sl in segments if sl == l
        )) if langs_present else "en"

        if len(langs_present) <= 1:
            english = self.translator.translate(text, primary_lang, "en")
        else:
            # Code-switched message: translate each segment individually,
            # then recombine — preserves meaning from every language used.
            translated_parts = [
                self.translator.translate(seg, lang, "en") for seg, lang in segments
            ]
            english = " ".join(translated_parts)

        return english, primary_lang, segments

    def _localize_response(self, sentiment_label: str, intent: str, lang: str) -> str:
        if lang == "en":
            return pick_response_en(sentiment_label, intent)

        native_table = NATIVE_RESPONSES.get(lang, {})
        if (sentiment_label, intent) in native_table:
            return native_table[(sentiment_label, intent)]
        if (sentiment_label, "default") in native_table:
            return native_table[(sentiment_label, "default")]

        # No hand-written template — fall back to MT of the English response.
        english_response = pick_response_en(sentiment_label, intent)
        return self.translator.translate(english_response, "en", lang)

    def respond(self, user_text: str) -> MultilingualTurn:
        english_text, lang, segments = self._to_english(user_text)

        result: SentimentResult = self.analyzer.analyze(english_text)
        raw_intent = detect_intent(english_text)
        resolved_intent = resolve_intent(raw_intent, english_text, self.ctx)

        # Track an order number if one appears, regardless of language,
        # so it stays available as context on later turns.
        entity_match = ORDER_NUMBER_RE.search(user_text)
        if entity_match:
            self.ctx.last_entity = entity_match.group(1)

        language_switched = self.ctx.turn_count > 0 and lang != self.ctx.last_language
        if language_switched:
            self.ctx.language_switches += 1

        response = self._localize_response(result.label, resolved_intent, lang)
        escalated = result.label == "strong_negative"

        turn = MultilingualTurn(
            user_text=user_text,
            detected_language=lang,
            mixed_segments=segments,
            pivot_english_text=english_text,
            sentiment_label=result.label,
            sentiment_score=result.compound,
            intent=raw_intent,
            resolved_intent=resolved_intent,
            bot_response=response,
            response_language=lang,
            escalated=escalated,
            language_switched=language_switched,
        )

        # Update context for the next turn. Social intents (thanks/greeting/
        # goodbye) don't overwrite the ongoing task topic, so it survives
        # small talk in between — see resolve_intent().
        self.ctx.last_intent = resolved_intent
        if resolved_intent in TASK_INTENTS:
            self.ctx.last_task_intent = resolved_intent
        self.ctx.last_language = lang
        self.ctx.seen_languages.add(lang)
        self.ctx.turn_count += 1

        self.history.append(turn)
        return turn

    def session_summary(self) -> dict:
        return {
            "turns": self.ctx.turn_count,
            "languages_used": sorted(self.ctx.seen_languages),
            "language_switches": self.ctx.language_switches,
            "translation_backend": self.translator.backend,
            "escalations": sum(1 for t in self.history if t.escalated),
        }


# ---------------------------------------------------------------------------
# 6. CLI demo
# ---------------------------------------------------------------------------

def main():
    bot = MultilingualChatbot()
    print(f"[translation backend: {bot.translator.backend}] "
          f"[language ID: {'langdetect' if _LANGDETECT_AVAILABLE else 'heuristic fallback'}]")
    print(f"Supported languages: {', '.join(SUPPORTED_LANGUAGES.values())}")
    print("Multilingual Sentiment Chatbot — type 'quit' to exit.\n")

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
        switch_note = " (language switch detected)" if turn.language_switched else ""
        print(
            f"   [{SUPPORTED_LANGUAGES.get(turn.detected_language, turn.detected_language)} | "
            f"{turn.sentiment_label} {turn.sentiment_score:+.2f} | "
            f"intent: {turn.intent} -> resolved: {turn.resolved_intent}]{switch_note}"
        )
        print(f"   pivot (EN): {turn.pivot_english_text}")
        print(f"Bot: {turn.bot_response}")
        if turn.escalated:
            print("   >> Flagged for human agent escalation.")
        print()

    print("--- Session Summary ---")
    for k, v in bot.session_summary().items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
