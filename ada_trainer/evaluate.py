from __future__ import annotations

import argparse
import datetime as dt
import json
import re
from pathlib import Path
from typing import Any


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def _infer_direction(text: str) -> str | None:
    s = (text or "").lower()
    if not s:
        return None

    up_hits = 0
    down_hits = 0

    up_patterns = [
        r"\bbullish\b",
        r"\buptrend\b",
        r"\blong\b",
        r"\bbuy\b",
        r"\btang\b",
        r"signal\s*=\s*bullish",
    ]
    down_patterns = [
        r"\bbearish\b",
        r"\bdowntrend\b",
        r"\bshort\b",
        r"\bsell\b",
        r"\bgiam\b",
        r"signal\s*=\s*bearish",
    ]

    for p in up_patterns:
        if re.search(p, s):
            up_hits += 1
    for p in down_patterns:
        if re.search(p, s):
            down_hits += 1

    if up_hits == 0 and down_hits == 0:
        return None
    if up_hits > down_hits:
        return "up"
    if down_hits > up_hits:
        return "down"
    return "flat"


def _text_quality_score(prompt: str, completion: str) -> float:
    comp = (completion or "").strip()
    prm = (prompt or "").strip()
    if not comp:
        return 0.0

    score = 0.0
    length = len(comp)
    if length >= 40:
        score += 0.4
    elif length >= 20:
        score += 0.2

    if any(k in comp.lower() for k in ["risk", "stop", "invalidation", "volatility", "rr", "r/r"]):
        score += 0.2
    if any(k in comp.lower() for k in ["bullish", "bearish", "neutral", "tang", "giam"]):
        score += 0.2
    if any(k in prm for k in ["Symbol:", "Timeframe:", "Regime:"]):
        score += 0.2

    return round(min(score, 1.0), 4)


def _safe_div(n: int, d: int) -> float:
    if d == 0:
        return 0.0
    return round(n / d, 4)


def _evaluate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    labeled = 0
    predicted = 0
    correct = 0
    quality_scores: list[float] = []

    direction_confusion = {
        "up": {"up": 0, "down": 0, "flat": 0, "none": 0},
        "down": {"up": 0, "down": 0, "flat": 0, "none": 0},
        "flat": {"up": 0, "down": 0, "flat": 0, "none": 0},
        "none": {"up": 0, "down": 0, "flat": 0, "none": 0},
    }

    def resolve_gold_direction(meta: dict[str, Any]) -> str | None:
        gold_label = meta.get("outcome_label")
        if gold_label in {"up", "down", "flat"}:
            return gold_label
        pct = meta.get("outcome_pct")
        try:
            v = float(pct)
            if v > 0:
                return "up"
            if v < 0:
                return "down"
            return "flat"
        except Exception:
            return None

    for row in rows:
        prompt = str(row.get("prompt", ""))
        completion = str(row.get("completion", ""))
        metadata = row.get("metadata", {}) if isinstance(row.get("metadata"), dict) else {}

        quality_scores.append(_text_quality_score(prompt, completion))

        gold = resolve_gold_direction(metadata)
        if gold is not None:
            labeled += 1

        pred = _infer_direction(completion)
        if pred is not None:
            predicted += 1

        gk = gold if gold is not None else "none"
        pk = pred if pred is not None else "none"
        direction_confusion[gk][pk] += 1

        if gold is not None and pred is not None and gold == pred:
            correct += 1

    avg_quality = round(sum(quality_scores) / total, 4) if total else 0.0
    coverage = _safe_div(predicted, total)
    accuracy_on_labeled_predicted = _safe_div(correct, sum(direction_confusion[g]["up"] + direction_confusion[g]["down"] + direction_confusion[g]["flat"] for g in ["up", "down", "flat"]))
    accuracy_on_labeled_total = _safe_div(correct, labeled)

    return {
        "counts": {
            "total": total,
            "labeled": labeled,
            "predicted": predicted,
            "correct": correct,
        },
        "metrics": {
            "avg_text_quality": avg_quality,
            "direction_prediction_coverage": coverage,
            "direction_accuracy_on_labeled_predicted": accuracy_on_labeled_predicted,
            "direction_accuracy_on_labeled_total": accuracy_on_labeled_total,
        },
        "confusion": direction_confusion,
    }


def _parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    default_data = root / "ai_path" / "ada_trainer" / "data" / "test_sft.jsonl"
    default_out = root / "ai_path" / "ada_trainer" / "artifacts" / "reports"
    p = argparse.ArgumentParser(description="Evaluate holdout dataset for ada_trainer.")
    p.add_argument("--input", type=Path, default=default_data)
    p.add_argument("--output-dir", type=Path, default=default_out)
    p.add_argument("--tag", default="holdout")
    p.add_argument("--run-id", default="", help="Optional external run id for traceability.")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    if not args.input.exists():
        raise FileNotFoundError(f"Missing evaluation input: {args.input}")

    rows = _read_jsonl(args.input)
    result = _evaluate(rows)
    payload = {
        "eval_ts": _utc_now(),
        "tag": args.tag,
        "run_id": (args.run_id or "").strip() or None,
        "input_path": str(args.input),
        "result": result,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    suffix = (args.run_id or "").strip() or dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = args.output_dir / f"eval_{args.tag}_{suffix}.json"
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(result["counts"], ensure_ascii=False))
    print(json.dumps(result["metrics"], ensure_ascii=False))
    print(f"Report: {out_path}")


if __name__ == "__main__":
    main()
