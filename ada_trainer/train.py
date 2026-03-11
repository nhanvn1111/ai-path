from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ENV_PATH = WORKSPACE_ROOT / "ai-engineer-projects" / "AutomatedDataAnalyst" / ".env"
if PROJECT_ENV_PATH.exists():
    load_dotenv(PROJECT_ENV_PATH)


DEFAULT_TRANSFORM_VERSION = "t_v1"
DEFAULT_MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"


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


def _build_text_sample(row: dict[str, Any]) -> str:
    prompt = str(row.get("prompt", "")).strip()
    completion = str(row.get("completion", "")).strip()
    return (
        "<|user|>\n"
        f"{prompt}\n"
        "<|assistant|>\n"
        f"{completion}"
    )


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _is_truthy(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def _resolve_model_name(model_name: str) -> str:
    if not model_name or "/" not in model_name or model_name.startswith("/"):
        return model_name
    if not (_is_truthy(os.getenv("HF_HUB_OFFLINE")) or _is_truthy(os.getenv("TRANSFORMERS_OFFLINE"))):
        return model_name
    repo_key = model_name.replace("/", "--")
    cache_root = Path.home() / ".cache" / "huggingface" / "hub" / f"models--{repo_key}"
    refs_main = cache_root / "refs" / "main"
    if not refs_main.exists():
        return model_name
    snapshot = refs_main.read_text(encoding="utf-8").strip()
    snapshot_dir = cache_root / "snapshots" / snapshot
    if snapshot and snapshot_dir.exists():
        return str(snapshot_dir)
    return model_name


def _build_run_metadata(
    *,
    run_id: str,
    mode: str,
    status: str,
    model_name: str,
    data_dir: Path,
    output_dir: Path,
    transform_version: str,
    train_count: int,
    val_count: int,
    notes: str = "",
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "run_ts": _utc_now(),
        "mode": mode,
        "status": status,
        "model_name": model_name,
        "data_dir": str(data_dir),
        "output_dir": str(output_dir),
        "transform_version": transform_version,
        "counts": {
            "train": train_count,
            "validation": val_count,
        },
        "notes": notes,
    }


def _parse_args() -> argparse.Namespace:
    default_data_dir = WORKSPACE_ROOT / "ai_path" / "ada_trainer" / "data"
    default_output_dir = WORKSPACE_ROOT / "ai_path" / "ada_trainer" / "artifacts" / "model"

    p = argparse.ArgumentParser(description="Train ada_trainer model.")
    p.add_argument("--mode", choices=["lora"], default="lora")
    p.add_argument("--data-dir", type=Path, default=default_data_dir)
    p.add_argument("--output-dir", type=Path, default=default_output_dir)
    p.add_argument("--model-name", default=os.getenv("ADA_BASE_MODEL", DEFAULT_MODEL_NAME))
    p.add_argument("--transform-version", default=os.getenv("TRANSFORM_VERSION", DEFAULT_TRANSFORM_VERSION))
    p.add_argument("--num-train-epochs", type=float, default=1.0)
    p.add_argument("--learning-rate", type=float, default=2e-4)
    p.add_argument("--per-device-train-batch-size", type=int, default=2)
    p.add_argument("--per-device-eval-batch-size", type=int, default=2)
    p.add_argument("--gradient-accumulation-steps", type=int, default=4)
    p.add_argument("--max-length", type=int, default=1024)
    p.add_argument("--logging-steps", type=int, default=10)
    p.add_argument("--save-steps", type=int, default=5)
    p.add_argument("--max-steps", type=int, default=-1, help="Stop after N steps (useful for partial training).")
    p.add_argument(
        "--resume-from",
        default="",
        help="Checkpoint path or 'latest' to resume from the newest checkpoint in run dir.",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--run-id", default="", help="Optional external run id for traceability.")
    args = p.parse_args()
    args.model_name = _resolve_model_name(args.model_name)
    return args


def _require_paths(data_dir: Path) -> tuple[Path, Path]:
    train_path = data_dir / "train_sft.jsonl"
    val_path = data_dir / "val_sft.jsonl"
    if not train_path.exists():
        raise FileNotFoundError(f"Missing train file: {train_path}")
    if not val_path.exists():
        raise FileNotFoundError(f"Missing validation file: {val_path}")
    return train_path, val_path


def _dataset_gate(data_dir: Path) -> None:
    train_path = data_dir / "train_sft.jsonl"
    val_path = data_dir / "val_sft.jsonl"
    test_path = data_dir / "test_sft.jsonl"
    manifest_path = data_dir / "manifest.json"

    for p in [train_path, val_path, test_path, manifest_path]:
        if not p.exists():
            raise RuntimeError(f"Dataset gate failed: missing file {p}")

    train_rows = _read_jsonl(train_path)
    val_rows = _read_jsonl(val_path)
    test_rows = _read_jsonl(test_path)
    if not train_rows or not val_rows or not test_rows:
        raise RuntimeError(
            "Dataset gate failed: empty split detected. "
            f"train={len(train_rows)}, validation={len(val_rows)}, holdout={len(test_rows)}"
        )

    train_ids = {x.get("id") for x in train_rows}
    val_ids = {x.get("id") for x in val_rows}
    test_ids = {x.get("id") for x in test_rows}
    if (train_ids & val_ids) or (train_ids & test_ids) or (val_ids & test_ids):
        raise RuntimeError("Dataset gate failed: split overlap detected by id.")

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as e:
        raise RuntimeError(f"Dataset gate failed: invalid manifest {manifest_path}") from e

    counts = manifest.get("counts", {})
    expected = {"train": len(train_rows), "validation": len(val_rows), "holdout": len(test_rows)}
    for k, v in expected.items():
        if int(counts.get(k, -1)) != v:
            raise RuntimeError(
                f"Dataset gate failed: manifest count mismatch for {k}. "
                f"manifest={counts.get(k)} actual={v}"
            )


def _run_dry(
    *,
    args: argparse.Namespace,
    run_id: str,
    train_rows: list[dict[str, Any]],
    val_rows: list[dict[str, Any]],
) -> None:
    run_dir = args.output_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    metadata = _build_run_metadata(
        run_id=run_id,
        mode=args.mode,
        status="dry_run",
        model_name=args.model_name,
        data_dir=args.data_dir,
        output_dir=run_dir,
        transform_version=args.transform_version,
        train_count=len(train_rows),
        val_count=len(val_rows),
        notes="Dry run only. No model training executed.",
    )
    _write_json(run_dir / "run_metadata.json", metadata)
    print(json.dumps({"run_id": run_id, "status": "dry_run", "output_dir": str(run_dir)}, ensure_ascii=False))


def _run_lora_train(
    *,
    args: argparse.Namespace,
    run_id: str,
    train_rows: list[dict[str, Any]],
    val_rows: list[dict[str, Any]],
) -> None:
    try:
        import torch
        from datasets import Dataset
        from peft import LoraConfig, TaskType, get_peft_model
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            DataCollatorForLanguageModeling,
            Trainer,
            TrainingArguments,
            TrainerCallback,
            set_seed,
        )
    except Exception as e:
        raise RuntimeError(
            "Missing training dependencies. Install: transformers, datasets, peft, torch."
        ) from e

    set_seed(args.seed)

    run_dir = args.output_dir / run_id
    adapter_dir = run_dir / "adapter"
    run_dir.mkdir(parents=True, exist_ok=True)

    train_texts = [_build_text_sample(x) for x in train_rows]
    val_texts = [_build_text_sample(x) for x in val_rows]

    train_ds = Dataset.from_dict({"text": train_texts})
    val_ds = Dataset.from_dict({"text": val_texts})

    local_files_only = _is_truthy(os.getenv("HF_HUB_OFFLINE")) or _is_truthy(os.getenv("TRANSFORMERS_OFFLINE"))
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        local_files_only=local_files_only,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    def tokenize_fn(batch: dict[str, list[str]]) -> dict[str, Any]:
        tok = tokenizer(
            batch["text"],
            truncation=True,
            max_length=args.max_length,
            padding="max_length",
        )
        tok["labels"] = [ids[:] for ids in tok["input_ids"]]
        return tok

    train_ds = train_ds.map(tokenize_fn, batched=True, remove_columns=["text"])
    val_ds = val_ds.map(tokenize_fn, batched=True, remove_columns=["text"])

    model_kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "local_files_only": local_files_only,
        "dtype": torch.float16 if torch.cuda.is_available() else torch.float32,
    }
    if not torch.cuda.is_available():
        # Avoid meta/offload dispatch on CPU training; it breaks backward for LoRA.
        model_kwargs["low_cpu_mem_usage"] = False
    model = AutoModelForCausalLM.from_pretrained(args.model_name, **model_kwargs)
    model.config.use_cache = False

    lora_cfg = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_cfg)

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    train_args = TrainingArguments(
        output_dir=str(run_dir / "checkpoints"),
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=2,
        eval_strategy="steps",
        eval_steps=max(args.logging_steps, 10),
        max_steps=args.max_steps if args.max_steps and args.max_steps > 0 else -1,
        report_to=[],
        fp16=torch.cuda.is_available(),
        bf16=False,
        remove_unused_columns=False,
        seed=args.seed,
    )

    def _write_progress(payload: dict[str, Any]) -> None:
        path = run_dir / "progress.json"
        try:
            path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        except Exception:
            pass

    class _ProgressCallback(TrainerCallback):
        def __init__(self) -> None:
            super().__init__()
            self.started_at = _utc_now()

        def on_train_begin(self, args, state, control, **kwargs):
            _write_progress({
                "run_id": run_id,
                "phase": "train_begin",
                "started_at": self.started_at,
                "ts": _utc_now(),
                "global_step": int(state.global_step),
                "max_steps": int(state.max_steps),
                "epoch": float(state.epoch or 0.0),
            })

        def on_log(self, args, state, control, logs=None, **kwargs):
            logs = logs or {}
            _write_progress({
                "run_id": run_id,
                "phase": "train",
                "started_at": self.started_at,
                "ts": _utc_now(),
                "global_step": int(state.global_step),
                "max_steps": int(state.max_steps),
                "epoch": float(state.epoch or 0.0),
                "loss": logs.get("loss"),
                "learning_rate": logs.get("learning_rate"),
            })

        def on_evaluate(self, args, state, control, metrics=None, **kwargs):
            metrics = metrics or {}
            _write_progress({
                "run_id": run_id,
                "phase": "eval",
                "started_at": self.started_at,
                "ts": _utc_now(),
                "global_step": int(state.global_step),
                "max_steps": int(state.max_steps),
                "epoch": float(state.epoch or 0.0),
                "eval_loss": metrics.get("eval_loss"),
            })

        def on_train_end(self, args, state, control, **kwargs):
            _write_progress({
                "run_id": run_id,
                "phase": "train_end",
                "started_at": self.started_at,
                "ts": _utc_now(),
                "global_step": int(state.global_step),
                "max_steps": int(state.max_steps),
                "epoch": float(state.epoch or 0.0),
            })

    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        data_collator=data_collator,
        callbacks=[_ProgressCallback()],
    )

    resume_from = None
    resume_arg = (args.resume_from or "").strip()
    if resume_arg:
        if resume_arg.lower() == "latest":
            ckpt_root = run_dir / "checkpoints"
            if ckpt_root.exists():
                candidates = []
                for path in ckpt_root.glob("checkpoint-*"):
                    try:
                        step = int(path.name.split("-")[-1])
                    except Exception:
                        continue
                    candidates.append((step, path))
                if candidates:
                    resume_from = str(sorted(candidates, key=lambda x: x[0])[-1][1])
        else:
            resume_from = resume_arg

    trainer.train(resume_from_checkpoint=resume_from)
    eval_result = trainer.evaluate()

    model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())

    metadata = _build_run_metadata(
        run_id=run_id,
        mode=args.mode,
        status="completed",
        model_name=args.model_name,
        data_dir=args.data_dir,
        output_dir=run_dir,
        transform_version=args.transform_version,
        train_count=len(train_rows),
        val_count=len(val_rows),
        notes="LoRA training completed." + (f" Resumed from {resume_from}." if resume_from else ""),
    )
    metadata["metrics"] = eval_result
    metadata["params"] = {
        "trainable": int(trainable_params),
        "total": int(total_params),
        "trainable_ratio_pct": round((trainable_params / max(total_params, 1)) * 100.0, 4),
    }
    metadata["artifacts"] = {
        "adapter_dir": str(adapter_dir),
        "checkpoints_dir": str(run_dir / "checkpoints"),
    }
    _write_json(run_dir / "run_metadata.json", metadata)
    print(json.dumps({"run_id": run_id, "status": "completed", "output_dir": str(run_dir)}, ensure_ascii=False))


def main() -> None:
    args = _parse_args()
    run_id = (args.run_id or "").strip() or f"run_{dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"

    _dataset_gate(args.data_dir)
    train_path, val_path = _require_paths(args.data_dir)
    train_rows = _read_jsonl(train_path)
    val_rows = _read_jsonl(val_path)

    if not train_rows or not val_rows:
        raise RuntimeError(f"Empty training data: train={len(train_rows)}, validation={len(val_rows)}")

    if args.dry_run:
        _run_dry(args=args, run_id=run_id, train_rows=train_rows, val_rows=val_rows)
        return

    if args.mode == "lora":
        _run_lora_train(args=args, run_id=run_id, train_rows=train_rows, val_rows=val_rows)
        return

    raise ValueError(f"Unsupported mode: {args.mode}")


if __name__ == "__main__":
    main()
