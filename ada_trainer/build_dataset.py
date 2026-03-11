from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import re
from pathlib import Path
from typing import Any


CONTRACT_VERSION = "1.0.0"
TRANSFORM_VERSION_DEFAULT = "t_v1"
TASK_TYPE = "sft_market_analysis"
MAX_AI_RESPONSE_LEN = 8000
RAW_PREVIEW_MAX_LEN = 500

ALLOWED_TIMEFRAMES = {"1s", "1m", "5m", "15m", "1h", "4h", "1d", "1w", "1M"}
ALLOWED_OUTCOME_LABELS = {"up", "down", "flat"}

SECRET_PATTERNS = [
    re.compile(r"(sk-[A-Za-z0-9_\-]{12,})"),
    re.compile(r"(ghp_[A-Za-z0-9]{20,})"),
    re.compile(r"(AIza[0-9A-Za-z\-_]{20,})"),
    re.compile(r"(?i)(bearer\s+[A-Za-z0-9\-._~+/]+=*)"),
    re.compile(r"(?i)(api[_-]?key\s*[:=]\s*[A-Za-z0-9\-._]{8,})"),
]

SYMBOL_RE = re.compile(r"^[A-Z0-9]{2,20}$")


def _parse_iso8601(value: Any) -> str | None:
    s = str(value or "").strip()
    if not s:
        return None
    try:
        parsed = dt.datetime.fromisoformat(s.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=dt.timezone.utc)
        return parsed.astimezone(dt.timezone.utc).isoformat().replace("+00:00", "Z")
    except Exception:
        return None


def _sanitize_text(value: Any, max_len: int) -> str:
    text = str(value or "").replace("\x00", " ").strip()
    for pat in SECRET_PATTERNS:
        text = pat.sub("[REDACTED]", text)
    if len(text) > max_len:
        text = text[:max_len]
    return text


def _to_float(value: Any) -> float | None:
    try:
        return float(value)
    except Exception:
        return None


def _to_int(value: Any) -> int | None:
    try:
        return int(value)
    except Exception:
        return None


def _normalize_outcome_label(value: Any) -> str | None:
    raw = str(value or "").strip().lower()
    if not raw:
        return None
    if raw in ALLOWED_OUTCOME_LABELS:
        return raw
    if raw in {"bullish"}:
        return "up"
    if raw in {"bearish"}:
        return "down"
    if raw in {"neutral", "sideways"}:
        return "flat"
    return None


def _patterns_summary(patterns: Any) -> str:
    if not isinstance(patterns, list):
        return "none"
    chunks: list[str] = []
    for item in patterns[:20]:
        if not isinstance(item, dict):
            continue
        name = _sanitize_text(item.get("name", ""), 80)
        signal = _sanitize_text(item.get("signal", ""), 24).upper()
        if name and signal:
            chunks.append(f"{name}:{signal}")
        elif name:
            chunks.append(name)
    return "; ".join(chunks) if chunks else "none"


def _make_prompt(
    symbol: str,
    timeframe: str,
    regime: str,
    entry_price: float,
    horizon: int,
    patterns_summary: str,
) -> str:
    return (
        "You are a crypto market analyst.\n"
        "Analyze the market snapshot and provide a concise trading analysis.\n\n"
        f"Symbol: {symbol}\n"
        f"Timeframe: {timeframe}\n"
        f"Regime: {regime}\n"
        f"Entry Price: {entry_price}\n"
        f"Horizon (candles): {horizon}\n"
        f"Patterns: {patterns_summary}\n"
    )


def _make_id(created_at: str, symbol: str, timeframe: str, entry_price: float, ai_response: str) -> str:
    raw = f"{created_at}|{symbol}|{timeframe}|{entry_price:.10f}|{ai_response[:256]}"
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    return f"ada_{digest[:16]}"


def _build_reject(reason: str, source_path: str, source_line_no: int, raw_line: str) -> dict[str, Any]:
    return {
        "reason_code": reason,
        "source_path": source_path,
        "source_line_no": source_line_no,
        "raw_preview": raw_line[:RAW_PREVIEW_MAX_LEN],
    }


def _transform_record(
    rec: dict[str, Any],
    *,
    split: str,
    source_path: str,
    source_line_no: int,
) -> tuple[dict[str, Any] | None, str | None]:
    created_at = _parse_iso8601(rec.get("created_at") or rec.get("ts"))
    if not created_at:
        return None, "invalid_created_at"

    symbol = _sanitize_text(rec.get("symbol"), 24).upper()
    if not SYMBOL_RE.match(symbol):
        return None, "invalid_symbol"

    timeframe = _sanitize_text(rec.get("timeframe"), 10)
    if timeframe not in ALLOWED_TIMEFRAMES:
        return None, "invalid_timeframe"

    entry_price = _to_float(rec.get("entry_price"))
    if entry_price is None or entry_price <= 0:
        return None, "invalid_entry_price"

    horizon = _to_int(rec.get("outcome_horizon_candles"))
    if horizon is None or horizon < 1 or horizon > 500:
        return None, "invalid_horizon"

    ai_response = _sanitize_text(rec.get("ai_response"), MAX_AI_RESPONSE_LEN)
    if not ai_response:
        return None, "empty_ai_response"

    regime = _sanitize_text(rec.get("regime"), 64) or "unknown"
    patterns_summary = _patterns_summary(rec.get("patterns"))

    outcome_label = _normalize_outcome_label(rec.get("outcome_label"))
    outcome_pct = _to_float(rec.get("outcome_pct"))

    prompt = _make_prompt(
        symbol=symbol,
        timeframe=timeframe,
        regime=regime,
        entry_price=entry_price,
        horizon=horizon,
        patterns_summary=patterns_summary,
    )

    rec_id = _make_id(created_at, symbol, timeframe, entry_price, ai_response)

    transformed = {
        "id": rec_id,
        "split": split,
        "task_type": TASK_TYPE,
        "prompt": prompt,
        "completion": ai_response,
        "metadata": {
            "created_at": created_at,
            "symbol": symbol,
            "timeframe": timeframe,
            "regime": regime,
            "entry_price": entry_price,
            "outcome_horizon_candles": horizon,
            "source_path": source_path,
            "source_line_no": source_line_no,
            "patterns_summary": patterns_summary,
            "outcome_label": outcome_label,
            "outcome_pct": outcome_pct,
            "raw_outcome_label": _sanitize_text(rec.get("outcome_label"), 32) or None,
            "ai_provider": _sanitize_text(rec.get("ai_provider"), 32) or None,
            "ai_model": _sanitize_text(rec.get("ai_model"), 120) or None,
            "dataset_version": _sanitize_text(rec.get("dataset_version"), 64) or None,
            "label_policy_version": _sanitize_text(rec.get("label_policy_version"), 64) or None,
        },
    }
    return transformed, None


def _load_and_transform(input_path: Path, split: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    good: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []

    with input_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            raw = line.strip()
            if not raw:
                continue
            try:
                parsed = json.loads(raw)
            except Exception:
                rejected.append(_build_reject("invalid_json", str(input_path), line_no, raw))
                continue

            transformed, reason = _transform_record(
                parsed,
                split=split,
                source_path=str(input_path),
                source_line_no=line_no,
            )
            if reason:
                rejected.append(_build_reject(reason, str(input_path), line_no, raw))
                continue
            good.append(transformed)
    return good, rejected


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 64), b""):
            h.update(chunk)
    return h.hexdigest()


def _ensure_non_overlap(train_rows: list[dict[str, Any]], val_rows: list[dict[str, Any]], test_rows: list[dict[str, Any]]) -> None:
    train_ids = {x["id"] for x in train_rows}
    val_ids = {x["id"] for x in val_rows}
    test_ids = {x["id"] for x in test_rows}

    overlap_tv = train_ids & val_ids
    overlap_tt = train_ids & test_ids
    overlap_vt = val_ids & test_ids
    if overlap_tv or overlap_tt or overlap_vt:
        raise RuntimeError(
            "Split overlap detected by id: "
            f"train-validation={len(overlap_tv)}, "
            f"train-holdout={len(overlap_tt)}, "
            f"validation-holdout={len(overlap_vt)}"
        )


def _parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    default_input_dir = root / "ai-engineer-projects" / "AutomatedDataAnalyst" / "logs" / "dataset_splits"
    default_output_dir = root / "ai_path" / "ada_trainer" / "data"

    p = argparse.ArgumentParser(description="Build training-ready SFT dataset for ada_trainer.")
    p.add_argument("--input-dir", type=Path, default=default_input_dir)
    p.add_argument("--output-dir", type=Path, default=default_output_dir)
    p.add_argument("--transform-version", default=TRANSFORM_VERSION_DEFAULT)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    input_dir: Path = args.input_dir
    output_dir: Path = args.output_dir

    train_in = input_dir / "train.jsonl"
    val_in = input_dir / "validation.jsonl"
    holdout_in = input_dir / "holdout.jsonl"

    for p in [train_in, val_in, holdout_in]:
        if not p.exists():
            raise FileNotFoundError(f"Missing input file: {p}")

    train_rows, train_rej = _load_and_transform(train_in, "train")
    val_rows, val_rej = _load_and_transform(val_in, "validation")
    test_rows, test_rej = _load_and_transform(holdout_in, "holdout")
    rejected = train_rej + val_rej + test_rej

    if not train_rows or not val_rows or not test_rows:
        raise RuntimeError(
            "Dataset-level validation failed: each split must have at least 1 valid row. "
            f"train={len(train_rows)}, validation={len(val_rows)}, holdout={len(test_rows)}"
        )

    _ensure_non_overlap(train_rows, val_rows, test_rows)

    train_out = output_dir / "train_sft.jsonl"
    val_out = output_dir / "val_sft.jsonl"
    test_out = output_dir / "test_sft.jsonl"
    rejected_out = output_dir / "rejected_records.jsonl"
    manifest_out = output_dir / "manifest.json"

    _write_jsonl(train_out, train_rows)
    _write_jsonl(val_out, val_rows)
    _write_jsonl(test_out, test_rows)
    _write_jsonl(rejected_out, rejected)

    manifest = {
        "contract_version": CONTRACT_VERSION,
        "transform_version": str(args.transform_version),
        "build_ts": dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z"),
        "input_paths": [str(train_in), str(val_in), str(holdout_in)],
        "counts": {
            "train": len(train_rows),
            "validation": len(val_rows),
            "holdout": len(test_rows),
            "rejected": len(rejected),
        },
        "hashes": {
            "train_sft.jsonl": _sha256(train_out),
            "val_sft.jsonl": _sha256(val_out),
            "test_sft.jsonl": _sha256(test_out),
            "rejected_records.jsonl": _sha256(rejected_out),
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_out.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(manifest["counts"], ensure_ascii=False))
    print(f"Manifest: {manifest_out}")


if __name__ == "__main__":
    main()
