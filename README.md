# Riddle's Archive

**Riddle's Archive** is a RAG-based conversational AI application that combines semantic retrieval, persona prompting, retrieval-confidence signals, guardrails, and conversation analytics in a Streamlit interface.

Users select a character, ask questions, and receive responses generated with Anthropic Claude using retrieved context from a curated, paraphrased knowledge base.

## What the app does

- retrieves semantically relevant context before generation
- generates persona-consistent responses with Anthropic Claude
- maintains session-based conversation history
- displays retrieved evidence and similarity-based confidence labels
- applies input guardrails for unsafe requests and prompt-injection attempts
- logs conversation-level analytics such as sentiment, retrieved topics, message activity, and retrieval confidence
- provides separate views for dialogue, retrieval, analytics, guardrails, and data notes

## How it works

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

The retrieval layer uses Sentence Transformers to embed the user query and knowledge-base chunks, then compares them with cosine similarity. Retrieved chunks are passed to the model as grounding context before response generation.

## Key technical features

### Retrieval-Augmented Generation
A structured knowledge base is embedded and searched before each response so generation is grounded in project-specific context rather than model memory alone.

### Persona prompting
Character-specific prompts control voice and behavior while retrieved evidence provides factual grounding.

### Retrieval confidence
Average retrieval similarity is converted into a simple confidence label. This is a transparency aid, not a probabilistic uncertainty estimate.

### Guardrails
The application includes app-level checks that block clearly unsafe requests, redirect prompt-injection attempts, and add reliability notes when retrieval confidence is weak.

### Conversation analytics
Interaction logs are converted into structured data for live summaries and visualizations, including sentiment, message length, retrieved topics, average similarity, and low-confidence rate.

## Tech stack

- **Python**
- **Streamlit**
- **Anthropic Claude API**
- **Sentence Transformers**
- **scikit-learn cosine similarity**
- **pandas**
- **Plotly**
- **VADER Sentiment**

## Repository structure

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
├── tutorial/
│   └── riddles_archive_tutorial.qmd
├── Background.webp
├── Hogwarts.jpg
└── Map.jpg
```

## Run locally

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

Copy the example secrets file:

```bash
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
```

Then replace the placeholder value in `.streamlit/secrets.toml`:

```toml
ANTHROPIC_API_KEY = "your-anthropic-api-key-here"
```

Do not commit the real secrets file.

### 4. Launch the app

```bash
streamlit run app.py
```

## Data and copyright note

The public repository does **not** include the full Harry Potter books or long copyrighted excerpts. The included knowledge base in `data/hp_knowledge_base.csv` is curated and paraphrased for demonstration purposes.

The application can also work with local source text during development, but local/private source files are excluded from version control.

## Limitations

- The public knowledge base is intentionally compact and does not cover the full series.
- Persona behavior is created through prompting and retrieval rather than model fine-tuning.
- Retrieval confidence is based on similarity scores and should not be interpreted as calibrated model uncertainty.
- Conversation memory is session-based; there is no persistent long-term user storage.
- The application is a portfolio prototype rather than a production deployment.

## Additional documentation

A longer technical walkthrough is available in [`tutorial/riddles_archive_tutorial.qmd`](tutorial/riddles_archive_tutorial.qmd).
