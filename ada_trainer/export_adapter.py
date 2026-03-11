from __future__ import annotations

import argparse
import datetime as dt
import json
import shutil
from pathlib import Path


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def _parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    default_model_root = root / "ai_path" / "ada_trainer" / "artifacts" / "model"
    default_export_root = root / "ai_path" / "ada_trainer" / "artifacts" / "export"
    p = argparse.ArgumentParser(description="Export LoRA adapter to a stable path.")
    p.add_argument("--model-root", type=Path, default=default_model_root)
    p.add_argument("--export-root", type=Path, default=default_export_root)
    p.add_argument("--run-id", default="", help="Specific run id to export. Empty = latest run.")
    p.add_argument("--force", action="store_true", help="Overwrite target if exists.")
    return p.parse_args()


def _pick_run_dir(model_root: Path, run_id: str) -> Path:
    if run_id:
        d = model_root / run_id
        if not d.exists():
            raise FileNotFoundError(f"Run directory not found: {d}")
        return d

    candidates = [p for p in model_root.iterdir() if p.is_dir() and p.name.startswith("run_")]
    if not candidates:
        raise FileNotFoundError(f"No run directories in {model_root}")
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def _find_adapter_dir(run_dir: Path) -> Path:
    adapter = run_dir / "adapter"
    if adapter.exists():
        return adapter
    raise FileNotFoundError(
        f"Adapter folder not found in run {run_dir.name}. "
        "Run non-dry training first to produce adapter artifacts."
    )


def _copy_tree(src: Path, dst: Path, force: bool) -> None:
    if dst.exists():
        if not force:
            raise FileExistsError(f"Target exists: {dst}. Use --force to overwrite.")
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def main() -> None:
    args = _parse_args()
    run_dir = _pick_run_dir(args.model_root, args.run_id)
    adapter_dir = _find_adapter_dir(run_dir)

    export_root = args.export_root
    latest_dir = export_root / "latest_adapter"
    timestamp_dir = export_root / f"adapter_{run_dir.name}"

    export_root.mkdir(parents=True, exist_ok=True)
    _copy_tree(adapter_dir, latest_dir, force=args.force)
    _copy_tree(adapter_dir, timestamp_dir, force=args.force)

    manifest = {
        "export_ts": _utc_now(),
        "source_run_id": run_dir.name,
        "source_adapter_dir": str(adapter_dir),
        "export_latest_dir": str(latest_dir),
        "export_versioned_dir": str(timestamp_dir),
    }
    (export_root / "export_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    main()
