# Multi-Modal AI Assistant

A text + image assistant built as a small pipeline of specialized steps
(vision extraction, ambiguity checking, evidence-grounded reasoning,
response validation) instead of a single direct model call. See
`report.docx` for the full design write-up.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env   # then fill in your ANTHROPIC_API_KEY
```

## Run the chat UI

```bash
streamlit run app.py
```

## Run the CLI example

```bash
python example_usage.py path/to/some_image.jpg
```

## Run the offline tests (no API key needed)

Every pipeline component is mocked, so this verifies the orchestrator's
decision logic (ambiguity branching, validation correction, evidence
persistence) without calling the real API or spending credits:

```bash
python -m unittest tests/test_pipeline.py -v
```

## Project layout

```
core/
  config.py        # env/model configuration
  memory.py         # conversation history + extracted evidence store
  vision.py          # image -> structured evidence extraction
  ambiguity.py        # decides whether to ask a clarifying question
  reasoning.py         # generates an evidence-grounded draft answer
  validator.py          # checks the draft's claims against evidence
  orchestrator.py        # wires the above into a per-turn pipeline
app.py               # Streamlit chat UI
example_usage.py      # scripted CLI demo of the pipeline
tests/
  test_pipeline.py      # offline, mock-based tests of the decision logic
```

## How a turn is processed

1. **Vision extraction** (if an image is attached): a dedicated call
   extracts only objective, verifiable facts about the image, and
   stores them as reusable "evidence" in memory.
2. **Ambiguity check**: before generating a full answer, a separate
   check decides whether the request is actually answerable from the
   context + evidence gathered so far. If not, the assistant asks a
   clarifying question instead of guessing.
3. **Contextual reasoning**: a draft answer is generated, grounded in
   the evidence and recent/summarized dialogue history, along with an
   explicit list of the factual claims it relies on.
4. **Validation**: a second, independent call checks each claim
   against the evidence/context and rewrites the answer if anything
   is unsupported.

Each step is logged into a `trace` returned alongside the answer, so
the decision-making is inspectable rather than a black box.
