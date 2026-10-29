"""Command-line predictions with a trained Turkish sentiment model."""

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import joblib

try:
    from . import __version__
    from .text_processing import clean_text
except ImportError:  # Support direct execution with ``python src/predict.py``.
    from text_processing import clean_text

    __version__ = "2.1.0"


DEFAULT_MODEL_PATH = Path("models/turkish_sentiment_model_30000_features.pkl")


def build_parser() -> argparse.ArgumentParser:
    """Create the command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="Predict sentiment labels for one or more Turkish texts."
    )
    parser.add_argument(
        "texts",
        nargs="*",
        help="Turkish text values to classify. Prompts for input when omitted.",
    )
    parser.add_argument(
        "--file",
        type=Path,
        help="UTF-8 text file containing one input per non-empty line.",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help=f"Model path (default: {DEFAULT_MODEL_PATH}).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Write prediction results as JSON.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    return parser


def collect_texts(
    positional_texts: Sequence[str],
    file_path: Path | None = None,
) -> list[str]:
    """Collect non-empty inputs from command-line values and an optional file."""
    texts = [text.strip() for text in positional_texts if text.strip()]

    if file_path is not None:
        with file_path.open(encoding="utf-8") as input_file:
            texts.extend(line.strip() for line in input_file if line.strip())

    return texts


def load_model(model_path: Path) -> Any:
    """Load a serialized scikit-learn model from disk."""
    if not model_path.is_file():
        raise FileNotFoundError(
            f"Model not found: {model_path}. Run src/train_model.py first "
            "or provide --model."
        )
    return joblib.load(model_path)


def predict_sentiments(model: Any, texts: Sequence[str]) -> list[dict[str, Any]]:
    """Predict labels and confidence scores for Turkish texts."""
    if not texts:
        raise ValueError("At least one non-empty text is required.")

    cleaned_texts = [clean_text(text) for text in texts]
    if any(not text for text in cleaned_texts):
        raise ValueError("An input became empty after text preprocessing.")

    labels = model.predict(cleaned_texts)
    probabilities = (
        model.predict_proba(cleaned_texts)
        if hasattr(model, "predict_proba")
        else None
    )

    results = []
    for index, (original_text, label) in enumerate(zip(texts, labels)):
        confidence = None
        if probabilities is not None:
            confidence = float(max(probabilities[index]))
        results.append(
            {
                "text": original_text,
                "label": str(label),
                "confidence": confidence,
            }
        )
    return results


def print_results(results: Sequence[dict[str, Any]], as_json: bool = False) -> None:
    """Print predictions in human-readable or JSON format."""
    if as_json:
        print(json.dumps(results, ensure_ascii=False, indent=2))
        return

    for index, result in enumerate(results, start=1):
        confidence = result["confidence"]
        confidence_text = (
            f" | Confidence: {confidence:.2%}" if confidence is not None else ""
        )
        print(f"[{index}] {result['text']}")
        print(f"    Label: {result['label']}{confidence_text}")


def main(argv: Sequence[str] | None = None) -> int:
    """Run the prediction command-line interface."""
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        texts = collect_texts(args.texts, args.file)
        if not texts:
            entered_text = input("Enter Turkish text: ").strip()
            texts = collect_texts([entered_text])

        model = load_model(args.model)
        results = predict_sentiments(model, texts)
        print_results(results, args.json)
    except (FileNotFoundError, OSError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
