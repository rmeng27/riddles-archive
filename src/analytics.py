"""Conversation analytics utilities."""

from __future__ import annotations

from collections import Counter
from typing import Dict, List

import pandas as pd


def sentiment_score(text: str) -> float:
    """Return a lightweight polarity score in [-1, 1]."""
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

        analyzer = SentimentIntensityAnalyzer()
        return float(analyzer.polarity_scores(text)["compound"])
    except Exception:
        # Very small fallback lexicon to keep app from breaking if VADER is unavailable.
        positive = {"good", "great", "love", "happy", "brave", "hope", "fun", "best"}
        negative = {"bad", "hate", "sad", "angry", "cold", "afraid", "fear", "worst"}
        words = [w.strip(".,!?;:'\"").lower() for w in text.split()]
        if not words:
            return 0.0
        score = sum(w in positive for w in words) - sum(w in negative for w in words)
        return max(-1.0, min(1.0, score / max(5, len(words))))


def log_turn(
    logs: List[Dict],
    *,
    character: str,
    user_message: str,
    assistant_message: str,
    confidence: str,
    avg_similarity: float,
    retrieved_topics: List[str],
    backend: str,
) -> None:
    logs.append(
        {
            "turn": len(logs) + 1,
            "character": character,
            "user_message": user_message,
            "assistant_message": assistant_message,
            "sentiment": sentiment_score(user_message),
            "message_length": len(user_message.split()),
            "confidence": confidence,
            "avg_similarity": avg_similarity,
            "retrieved_topics": ", ".join(retrieved_topics),
            "backend": backend,
        }
    )


def logs_to_frame(logs: List[Dict]) -> pd.DataFrame:
    if not logs:
        return pd.DataFrame(
            columns=[
                "turn",
                "character",
                "user_message",
                "sentiment",
                "message_length",
                "confidence",
                "avg_similarity",
                "retrieved_topics",
                "backend",
            ]
        )
    return pd.DataFrame(logs)


def topic_counts(logs: List[Dict]) -> Counter:
    counter = Counter()
    for row in logs:
        for topic in str(row.get("retrieved_topics", "")).split(","):
            topic = topic.strip()
            if topic:
                counter[topic] += 1
    return counter
