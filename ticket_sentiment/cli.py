from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from .service import HybridTicketSentimentAnalyzer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ticket-sentiment",
        description="Hybrid ticket sentiment analyzer for positive/neutral/negative ticket descriptions.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    predict_parser = subparsers.add_parser("predict", help="Predict sentiment for one ticket.")
    predict_parser.add_argument("--text", required=True, help="Ticket description")
    predict_parser.add_argument("--json", action="store_true", help="Print JSON output")

    batch_parser = subparsers.add_parser("batch", help="Predict sentiment for a batch file.")
    batch_parser.add_argument("--input", required=True, help="CSV or JSONL file with a text field")
    batch_parser.add_argument("--output", required=True, help="Output JSONL file")

    evaluate_parser = subparsers.add_parser("evaluate", help="Evaluate accuracy on labeled data.")
    evaluate_parser.add_argument("--input", required=True, help="CSV or JSONL file with text and label columns")

    add_example_parser = subparsers.add_parser(
        "add-example",
        help="Append a labeled example to the retrieval knowledge base.",
    )
    add_example_parser.add_argument("--text", required=True, help="Reference ticket text")
    add_example_parser.add_argument(
        "--label",
        required=True,
        choices=["negative", "neutral", "positive"],
        help="Sentiment label",
    )
    add_example_parser.add_argument(
        "--reference-file",
        default=None,
        help="Optional custom JSONL reference file path",
    )

    serve_parser = subparsers.add_parser("serve", help="Run the FastAPI service.")
    serve_parser.add_argument("--host", default="127.0.0.1")
    serve_parser.add_argument("--port", type=int, default=8000)

    return parser


def _load_rows(path: Path) -> list[dict[str, str]]:
    if path.suffix.lower() == ".jsonl":
        rows = []
        with path.open("r", encoding="utf-8") as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
        return rows

    if path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))

    raise ValueError("Only CSV and JSONL are supported.")


def _save_predictions(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _predict_command(args: argparse.Namespace) -> int:
    analyzer = HybridTicketSentimentAnalyzer()
    result = analyzer.predict(args.text)
    if args.json:
        print(json.dumps(result.as_dict(), ensure_ascii=False, indent=2))
    else:
        print(f"label={result.label} confidence={result.confidence:.3f}")
        print(result.rationale)
    return 0


def _batch_command(args: argparse.Namespace) -> int:
    analyzer = HybridTicketSentimentAnalyzer()
    input_path = Path(args.input)
    rows = _load_rows(input_path)
    predictions = []
    for row in rows:
        text = str(row.get("text", "")).strip()
        if not text:
            continue
        result = analyzer.predict(text)
        predictions.append({**row, **result.as_dict()})
    _save_predictions(Path(args.output), predictions)
    print(f"Wrote {len(predictions)} predictions to {args.output}")
    return 0


def _evaluate_command(args: argparse.Namespace) -> int:
    analyzer = HybridTicketSentimentAnalyzer()
    rows = _load_rows(Path(args.input))
    total = 0
    correct = 0
    confusion = {
        "negative": {"negative": 0, "neutral": 0, "positive": 0},
        "neutral": {"negative": 0, "neutral": 0, "positive": 0},
        "positive": {"negative": 0, "neutral": 0, "positive": 0},
    }

    for row in rows:
        text = str(row.get("text", "")).strip()
        label = str(row.get("label", "")).strip().lower()
        if not text or label not in confusion:
            continue
        prediction = analyzer.predict(text)
        confusion[label][prediction.label] += 1
        total += 1
        if prediction.label == label:
            correct += 1

    accuracy = (correct / total) if total else 0.0
    print(
        json.dumps(
            {
                "total": total,
                "correct": correct,
                "accuracy": accuracy,
                "confusion": confusion,
            },
            indent=2,
        )
    )
    return 0


def _add_example_command(args: argparse.Namespace) -> int:
    analyzer = HybridTicketSentimentAnalyzer()
    reference_file = Path(args.reference_file) if args.reference_file else analyzer.settings.reference_data_path
    reference_file.parent.mkdir(parents=True, exist_ok=True)
    payload = {"text": args.text, "label": args.label}
    with reference_file.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
    print(f"Appended labeled example to {reference_file}")
    return 0


def _serve_command(args: argparse.Namespace) -> int:
    try:
        import uvicorn
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("Serving requires 'uvicorn' to be installed.") from exc

    uvicorn.run("ticket_sentiment.api:app", host=args.host, port=args.port, reload=False)
    return 0


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    handlers = {
        "predict": _predict_command,
        "batch": _batch_command,
        "evaluate": _evaluate_command,
        "add-example": _add_example_command,
        "serve": _serve_command,
    }
    return handlers[args.command](args)
