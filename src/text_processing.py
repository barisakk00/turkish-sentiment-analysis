"""Shared text preprocessing for training and inference."""

import re


def clean_text(text: str) -> str:
    """Normalize Turkish text using the same rules as model training."""
    normalized = str(text).translate(str.maketrans({"İ": "i", "I": "ı"})).lower()
    normalized = re.sub(r"http\S+|www\S+", " ", normalized)
    normalized = re.sub(r"[^a-zçğıöşü\s]", " ", normalized)
    return re.sub(r"\s+", " ", normalized).strip()
