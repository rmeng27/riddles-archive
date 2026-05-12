"""Lightweight retrieval utilities for Riddle's Archive.

The retriever prefers sentence-transformers embeddings when available, and falls back
on TF-IDF if deployment environments cannot install/load transformer models.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np
import pandas as pd


@dataclass
class RetrievedChunk:
    chunk_id: str
    character: str
    topic: str
    source_type: str
    text: str
    similarity: float


def load_knowledge_base(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"chunk_id", "character", "topic", "source_type", "text"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Knowledge base is missing required columns: {sorted(missing)}")
    df = df.fillna("")
    return df


class HybridRetriever:
    """A small semantic retriever with an automatic TF-IDF fallback."""

    def __init__(self, df: pd.DataFrame, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.df = df.reset_index(drop=True).copy()
        self.model_name = model_name
        self.backend = "tfidf"
        self.model = None
        self.matrix = None
        self.vectorizer = None
        self._fit()

    def _fit(self) -> None:
        texts = self.df["text"].astype(str).tolist()
        try:
            from sentence_transformers import SentenceTransformer

            self.model = SentenceTransformer(self.model_name)
            embeddings = self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
            self.matrix = np.asarray(embeddings, dtype="float32")
            self.backend = "sentence-transformers"
        except Exception:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.preprocessing import normalize

            self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=1)
            matrix = self.vectorizer.fit_transform(texts)
            self.matrix = normalize(matrix)
            self.backend = "tfidf"

    def _query_vector(self, query: str):
        if self.backend == "sentence-transformers":
            return np.asarray(
                self.model.encode([query], normalize_embeddings=True, show_progress_bar=False),
                dtype="float32",
            )[0]
        return self.vectorizer.transform([query])

    def search(
        self,
        query: str,
        character: Optional[str] = None,
        k: int = 4,
        include_global: bool = True,
    ) -> List[RetrievedChunk]:
        if not query.strip():
            return []

        mask = pd.Series([True] * len(self.df))
        if character:
            char_mask = self.df["character"].eq(character)
            if include_global:
                char_mask = char_mask | self.df["character"].eq("General")
            mask = mask & char_mask

        candidate_idx = np.where(mask.values)[0]
        if len(candidate_idx) == 0:
            candidate_idx = np.arange(len(self.df))

        q = self._query_vector(query)

        if self.backend == "sentence-transformers":
            sims = self.matrix[candidate_idx] @ q
        else:
            sims = (self.matrix[candidate_idx] @ q.T).toarray().ravel()

        order = np.argsort(sims)[::-1][:k]
        results: list[RetrievedChunk] = []
        for pos in order:
            idx = int(candidate_idx[pos])
            row = self.df.iloc[idx]
            results.append(
                RetrievedChunk(
                    chunk_id=str(row["chunk_id"]),
                    character=str(row["character"]),
                    topic=str(row["topic"]),
                    source_type=str(row["source_type"]),
                    text=str(row["text"]),
                    similarity=float(sims[pos]),
                )
            )
        return results


def confidence_label(avg_similarity: float, backend: str) -> tuple[str, str]:
    """Convert similarity into an interpretable confidence label.

    Thresholds differ by retrieval backend because transformer cosine scores and TF-IDF
    cosine scores live on different practical scales.
    """
    if backend == "sentence-transformers":
        high, medium = 0.48, 0.34
    else:
        high, medium = 0.22, 0.10

    if avg_similarity >= high:
        return "High", "Strong match between the user query and retrieved context."
    if avg_similarity >= medium:
        return "Medium", "Some relevant context was retrieved, but the answer may need caution."
    return "Low", "Weak retrieval match. The response may rely more on general generation than grounded context."


def format_context(chunks: Iterable[RetrievedChunk]) -> str:
    lines = []
    for i, chunk in enumerate(chunks, start=1):
        lines.append(
            f"[Chunk {i} | {chunk.character} | {chunk.topic} | similarity={chunk.similarity:.3f}]\n{chunk.text}"
        )
    return "\n\n".join(lines)
