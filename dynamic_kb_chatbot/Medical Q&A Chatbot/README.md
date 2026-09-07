# MedQuAD Medical Q&A Chatbot

A Streamlit-based medical chatbot powered by the [MedQuAD dataset](https://github.com/abachaa/MedQuAD) — 47,457 QA pairs from 12 NIH websites.

---

## Features

- **TF-IDF Retrieval** — finds the most relevant QA pair for any medical question
- **Medical Entity Recognition** — highlights symptoms, diseases, treatments, drugs, body parts, and tests
- **Conversational UI** — Streamlit chat interface with history
- **Adjustable results** — control how many results to retrieve and minimum relevance score
- **Source links** — every answer links back to its NIH source page

---

## Quick Start

### 1. Clone MedQuAD dataset
```bash
git clone --depth=1 --filter=blob:none --sparse https://github.com/abachaa/MedQuAD.git medquad_raw
cd medquad_raw
git sparse-checkout set 1_CancerGov_QA 5_NIDDK_QA 6_NINDS_QA 9_CDC_QA
cd ..
```

> To load ALL 47K QA pairs, add all 12 folders to the sparse-checkout command.

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the app
```bash
streamlit run app.py
```

Open your browser at **http://localhost:8501**

---

## Project Structure

```
medquad_chatbot/
├── app.py                  # Streamlit UI
├── data_loader.py          # XML parser → pandas DataFrame
├── retrieval.py            # TF-IDF search engine (with disk cache)
├── entity_recognition.py   # Rule-based medical NER
├── requirements.txt
├── README.md
└── medquad_raw/            # MedQuAD XML files (cloned separately)
```

---

## How It Works

```
User question
     │
     ▼
Entity Recognition          ← detect symptoms/diseases/drugs in query
     │
     ▼
TF-IDF Retrieval            ← cosine similarity over 2,800+ QA pairs
     │
     ▼
Top-K Results               ← ranked by relevance score
     │
     ▼
Entity Highlighting         ← colour-coded entities in the answer
     │
     ▼
Streamlit UI response
```

### Entity types detected

| Label      | Color   | Examples                          |
|------------|---------|-----------------------------------|
| symptom    | 🔴 Red  | fever, pain, fatigue, cough       |
| disease    | 🟦 Teal | cancer, diabetes, leukemia        |
| treatment  | 🔵 Blue | chemotherapy, surgery, dialysis   |
| drug       | 🟢 Green| aspirin, insulin, penicillin      |
| body_part  | 🟣 Purple| heart, brain, kidney, lung       |
| test       | 🟠 Orange| MRI, biopsy, blood test, CBC     |

---

## Load Full Dataset

To index all 47,457 QA pairs, clone all folders:

```bash
cd medquad_raw
git sparse-checkout set \
  1_CancerGov_QA \
  2_GARD_QA \
  3_GHR_QA \
  4_MPlus_Health_Topics_QA \
  5_NIDDK_QA \
  6_NINDS_QA \
  7_SeniorHealth_QA \
  8_NHLBI_QA_XML \
  9_CDC_QA
```

Note: folders 10, 11, 12 have questions but answers were removed due to copyright.

---

## Disclaimer

This chatbot is for **informational and educational purposes only**.  
It is **not** a substitute for professional medical advice, diagnosis, or treatment.  
Always consult a qualified healthcare professional.
