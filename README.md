# Riddle's Archive

<p align="center">
  <strong>A RAG-based conversational AI application with semantic retrieval, persona prompting, retrieval confidence, guardrails, and live conversation analytics.</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.x-3776AB?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit&logoColor=white" alt="Streamlit">
  <img src="https://img.shields.io/badge/Anthropic-Claude-D97757" alt="Anthropic Claude">
  <img src="https://img.shields.io/badge/RAG-Semantic%20Retrieval-6C63FF" alt="RAG">
</p>

<p align="center">
  <img src="tutorial/images/dialogue_demo.png" alt="Riddle's Archive dialogue interface" width="900">
</p>

## Overview

**Riddle's Archive** explores how a character chatbot can be made more transparent and inspectable than a plain LLM wrapper.

Users select a character, ask questions, and receive persona-consistent responses generated with Anthropic Claude. Before generation, the application retrieves semantically relevant evidence from a curated knowledge base, summarizes retrieval confidence, applies app-level safety checks, and records the interaction for live analytics.

The result is a conversational AI prototype where **generation, grounding, reliability signals, and interaction data are all visible to the user**.

## System Flow

```text
User query
  → input guardrail
  → query embedding
  → semantic retrieval
  → retrieval-confidence label
  → persona prompt + retrieved context
  → Claude response generation
  → reliability note
  → evidence display + conversation analytics
```

## Key Features

### Retrieval-Augmented Generation
Sentence Transformers embed the user query and knowledge-base chunks. Cosine similarity is used to retrieve the most relevant evidence before generation.

### Persona Prompting
Character-specific prompts control voice and behavior while retrieved context provides grounding.

### Retrieval Transparency
The interface exposes retrieved evidence and converts average retrieval similarity into a simple confidence label. This is designed as a transparency signal rather than calibrated probabilistic uncertainty.

### Guardrails
App-level checks block clearly unsafe requests, redirect prompt-injection attempts, and add reliability notes when retrieval confidence is weak.

### Conversation Analytics
The application logs interaction-level data and visualizes sentiment, message length, retrieved topics, average similarity, and low-confidence rate across the session.

## Interface

<table>
<tr>
<td width="50%">
  <img src="tutorial/images/retrieval_demo.png" alt="Retrieval evidence view">
  <br><strong>Retrieval</strong> — inspect source chunks and similarity signals.
</td>
<td width="50%">
  <img src="tutorial/images/analytics_demo.png" alt="Conversation analytics view">
  <br><strong>Analytics</strong> — monitor sentiment, topics, and retrieval confidence.
</td>
</tr>
<tr>
<td width="50%">
  <img src="tutorial/images/guardrails_demo.png" alt="Guardrails view">
  <br><strong>Guardrails</strong> — surface reliability and safety controls.
</td>
<td width="50%">
  <img src="tutorial/images/dialogue_demo.png" alt="Dialogue view">
  <br><strong>Dialogue</strong> — interact with persona-aware, retrieval-grounded responses.
</td>
</tr>
</table>

## Tech Stack

| Layer | Tools |
|---|---|
| Application | Python, Streamlit |
| Generation | Anthropic Claude API |
| Retrieval | Sentence Transformers, scikit-learn cosine similarity |
| Data | pandas |
| Analytics | Plotly, VADER Sentiment |
| Documentation | Quarto |

## Repository Structure

```text
riddles-archive/
├── app.py
├── requirements.txt
├── README.md
├── .streamlit/
│   └── secrets.toml.example
├── data/
│   └── hp_knowledge_base.csv
├── src/
│   ├── analytics.py
│   ├── characters.py
│   ├── claude_client.py
│   ├── rag.py
│   └── styling.py
└── tutorial/
    ├── riddles_archive_tutorial.qmd
    └── images/
```

## Run Locally

### 1. Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
# Windows: .venv\Scripts\activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure the Anthropic API key

```bash
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
```

Then replace the placeholder value in `.streamlit/secrets.toml`:

```toml
ANTHROPIC_API_KEY = "your-anthropic-api-key-here"
```

The real secrets file is excluded from version control.

### 4. Launch

```bash
streamlit run app.py
```

## Data & Copyright Note

The public repository does **not** include full Harry Potter books or long copyrighted excerpts. The included `data/hp_knowledge_base.csv` is curated and paraphrased for demonstration.

Local/private source text can be used during development, but those files are excluded from version control.

## Limitations

- The public knowledge base is intentionally compact and does not cover the full series.
- Persona behavior is created through prompting and retrieval rather than model fine-tuning.
- Retrieval confidence is similarity-based and is not calibrated model uncertainty.
- Conversation memory is session-based rather than persistent.
- This is a portfolio prototype, not a production deployment.

## Documentation

A longer technical walkthrough is available in [`tutorial/riddles_archive_tutorial.qmd`](tutorial/riddles_archive_tutorial.qmd).

---

<sub>Originally developed as a graduate Generative AI project and refined for portfolio presentation.</sub>
