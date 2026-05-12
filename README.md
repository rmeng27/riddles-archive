# Riddle's Archive

**Riddle's Archive** is a Hogwarts-themed character RAG chatbot built for a GenAI final project. Users choose a character, ask questions, and receive persona-consistent responses generated with Anthropic Claude and grounded in a curated knowledge base.

The app includes:

- character selector for Tom Riddle / Lord Voldemort, Hermione Granger, Luna Lovegood, Sirius Black, and Fred & George Weasley
- retrieval-augmented generation using a curated/paraphrased Harry Potter knowledge base
- Claude response generation through the Anthropic API
- session-based conversation memory
- retrieval confidence scoring
- retrieved evidence display
- live conversation analytics: sentiment trend, retrieved topics, message logs
- custom CSS styling for a dark Hogwarts/archive aesthetic

## Project framing

Existing character bots are usually entertainment-first and closed-box: users can chat, but they cannot see the data source, retrieval logic, confidence score, or evaluation process. This project adds a transparent RAG pipeline and a lightweight analytics layer so the chatbot can be discussed as a GenAI/data science system rather than only a roleplay interface.

## Data source policy

The app does **not** include the full Harry Potter books or long copyrighted excerpts. The knowledge base in `data/hp_knowledge_base.csv` is manually curated and paraphrased for educational demonstration. Each row contains:

- `chunk_id`
- `character`
- `topic`
- `source_type`
- `text`

This structure makes the app reproducible and allows retrieval results to be inspected.

## File structure

```text
riddles_archive/
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
    └── final_project_tutorial.qmd
```

## Setup

### 1. Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
# .venv\Scripts\activate   # Windows
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Add your Anthropic API key

For local development, create `.streamlit/secrets.toml`:

```toml
ANTHROPIC_API_KEY = "your-anthropic-api-key-here"
```

Do **not** commit the real `secrets.toml` file to GitHub.

For Streamlit Cloud, add the same key under **App settings → Secrets**.

### 4. Run the app

```bash
streamlit run app.py
```

## Deployment

The easiest deployment path is Streamlit Community Cloud:

1. Push this project folder to a GitHub repository.
2. Go to Streamlit Cloud.
3. Create a new app from the GitHub repo.
4. Set the main file path to `app.py`.
5. Add `ANTHROPIC_API_KEY` in Streamlit secrets.
6. Deploy.

## Notes for grading

The app is designed around four grading-friendly components:

1. **RAG logic**: user questions retrieve semantically relevant context before Claude generates a response.
2. **Prompt engineering**: each character has a specific persona prompt and the user can control persona intensity.
3. **Uncertainty/confidence layer**: retrieval similarity is converted into a confidence label so weakly grounded responses are flagged.
4. **Conversation analytics**: the app logs sentiment, message length, retrieved topics, and confidence across the conversation.

## Limitations

- The knowledge base is intentionally compact and paraphrased, so it does not cover every part of the series.
- Character voice is simulated through prompting and retrieval, not trained as a custom model.
- Confidence is based on retrieval similarity, not a full probabilistic uncertainty model.
- The app uses session memory only; long-term user storage is not enabled.

## Future extensions

- Expand the knowledge base by book, scene, relationship, and theme.
- Add a small hand-labeled retrieval evaluation set and compute Precision@K / MRR.
- Add audio narration or character voice effects.
- Add an optional "Weasleys' Wizard Wheezes interruption" mode.
- Add user controls for canon strictness, tone, and spoiler level.
