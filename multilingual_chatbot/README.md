# Sentiment-Aware Chatbot

Detects customer sentiment (positive / neutral / negative / strong-negative)
in real time and adapts chatbot responses accordingly, including
auto-escalation of angry customers to a human agent.

## Run it in VS Code

1. Open this folder in VS Code.
2. Open a terminal (`` Ctrl+` ``) and create a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   venv\Scripts\activate      # Windows
   source venv/bin/activate   # macOS/Linux
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the chatbot:
   ```bash
   python sentiment_chatbot.py
   ```
5. Chat with it, then type `quit` to end the session and see the summary.

> If `vaderSentiment` fails to install (e.g. no internet), the script
> automatically falls back to a built-in lexicon analyzer so it still runs —
> just with lower accuracy. Installing VADER is strongly recommended for
> real use.

## How it works

- **`SentimentAnalyzer`** — wraps VADER's `SentimentIntensityAnalyzer`
  (rule-based, tuned for short/informal text like chat and social media).
  Compound score is bucketed into `positive`, `neutral`, `negative`, and
  `strong_negative` (score ≤ -0.55, e.g. "furious", "worst", "scam").
- **`detect_intent()`** — simple keyword/regex matching for common support
  topics (order status, refund, billing, cancellation, greeting, thanks).
- **`RESPONSES`** — a sentiment × intent lookup table of response templates.
  Strong-negative messages always trigger an escalation flag, regardless
  of intent.
- **`SentimentChatbot`** — orchestrates analysis → intent → response, and
  logs every turn to `chat_log.csv` for offline evaluation.

## Evaluating against the task's criteria

**1. Accuracy of sentiment detection**
- `chat_log.csv` records the sentiment label/score for every message.
- To measure accuracy, label a sample of real transcripts by hand (e.g. in
  a spreadsheet) and compare against the logged labels — compute
  precision/recall per class or overall accuracy.
- Swap in a transformer model (e.g. `distilbert-base-uncased-finetuned-sst-2-english`
  via Hugging Face `transformers`) for higher accuracy on nuanced text if
  VADER's rule-based approach isn't precise enough; the `SentimentAnalyzer`
  class is designed so you can drop in a different backend without touching
  the rest of the bot.

**2. Appropriateness of responses to different sentiments**
- Extend `RESPONSES` with more intents and variants specific to your
  business (this demo covers common support cases as a starting template).
- Strong-negative → escalate; negative → apologize + ask clarifying
  question; neutral → task-focused; positive → warm acknowledgment. Review
  `chat_log.csv` periodically to catch mismatches (e.g. sarcasm scored as
  positive) and refine the lexicon/thresholds or templates.

**3. Impact on customer satisfaction**
- `session_summary()` gives per-session sentiment breakdown, average
  score, and escalation count — track these over time (e.g. append to a
  dashboard) as a proxy for satisfaction trend.
- For a real deployment, pair this with an actual CSAT survey after the
  chat and correlate survey scores against the logged sentiment trajectory
  to see whether sentiment-aware responses measurably improve outcomes
  versus a non-sentiment-aware baseline (A/B test).

## Files

- `sentiment_chatbot.py` — main chatbot logic + CLI demo
- `requirements.txt` — dependencies
- `chat_log.csv` — generated automatically on first run
# Multilingual Extension

`multilingual_chatbot.py` extends the original `sentiment_chatbot.py`
(imported, not duplicated) with automatic language identification,
cross-lingual translation, and context retention across language
switches — supporting **Spanish, French, German, and Hindi** in addition
to English.

## Run it

```bash
pip install -r requirements.txt
python multilingual_chatbot.py
```

If `langdetect` / `transformers` aren't installed (or there's no internet
to download MarianMT checkpoints), it automatically falls back to a
built-in heuristic language detector and a small phrase dictionary, so
the whole pipeline still runs end-to-end offline — with lower accuracy
than the real models. Install the full `requirements.txt` for production
use.

## Architecture: pivot through English

```
user text (any language)
    -> detect language (whole message + per-sentence for mixed input)
    -> translate to English ("pivot")
    -> existing sentiment + intent engine runs UNCHANGED, in English
    -> ambiguous queries resolved using sticky conversation context
    -> response chosen: hand-localized native template first,
       machine-translated English template as fallback
    -> reply sent back in the language the user is currently using
```

Pivoting through English means sentiment/intent accuracy doesn't degrade
per-language — the lexicon/VADER logic only ever sees English text,
regardless of what the customer typed in.

## Capabilities demonstrated

- **Automatic language identification** — `detect_language()`, per
  message. Uses `langdetect` if available; otherwise a stopword/script
  heuristic (e.g. Devanagari script → Hindi).
- **Mixed-language ("code-switched") input** — `detect_mixed_languages()`
  splits a message into sentences, detects each one's language
  separately, and translates each before recombining. Example:
  *"Ich bin sehr wütend. Where is my refund?"* is handled correctly even
  though it starts in German and ends in English.
- **Cross-lingual ambiguity resolution** — `ConversationContext` tracks a
  *sticky* task topic (`last_task_intent`) that survives social turns
  (thanks/greetings) in between. A vague follow-up like *"Et le statut
  maintenant?"* (French, no explicit subject) still resolves to
  `order_status` if that was the last real topic — even if the user said
  "thanks" in English in between.
- **Context retention across switches** — entities (e.g. an order number)
  and the current task topic persist in `ConversationContext` regardless
  of which language they were mentioned in or the language used later.
- **Consistent responses regardless of language** — the same
  sentiment/intent logic drives every language; only the presentation
  layer (`NATIVE_RESPONSES` template lookup, or MT fallback) changes.
  Hand-written native templates are used for common cases (better
  fluency); machine translation covers everything else.

## Extending further

- Add a language: add its stopwords (or rely on `langdetect`), extend
  `NATIVE_RESPONSES` with a template dict, and MarianMT will handle any
  gaps automatically via `Helsinki-NLP/opus-mt-{lang}-en` /
  `opus-mt-en-{lang}` (any pair on the Hugging Face Hub works).
- Swap MarianMT for a stronger open-source MT model (e.g. NLLB-200 for
  200+ languages) by replacing the `Translator` class internals — the
  rest of the pipeline (`detect_language`, sentiment, intent, context)
  is model-agnostic.
- For production-grade language ID on very short/noisy messages, prefer
  `fasttext`'s `lid.176.bin` model over `langdetect` (higher accuracy on
  short text).

## Evaluating cross-lingual quality

- **Language ID accuracy**: hand-label a multilingual transcript sample
  and compare against `detected_language` per turn.
- **Translation/pivot fidelity**: spot-check `pivot_english_text` against
  the original for meaning loss — this is what sentiment/intent accuracy
  actually depends on.
- **Context retention**: construct test conversations that switch
  languages mid-thread (like the German→Hindi→German or Spanish→
  English→French examples above) and verify `resolved_intent` /
  `last_entity` survive correctly — `session_summary()` reports
  `language_switches` for quick sanity checks.
