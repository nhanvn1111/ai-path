# ADA Trainer Data Contract v1

## Scope
- Module: `ai_path/ada_trainer`
- Purpose: chuan hoa du lieu tu `ai-engineer-projects/AutomatedDataAnalyst` thanh du lieu train/eval co the tai lap.
- Effective date: `2026-03-06`
- Contract version: `1.0.0`

## Source Of Truth
- Raw/master:
  - `ai-engineer-projects/AutomatedDataAnalyst/logs/master_dataset.jsonl`
- Split inputs:
  - `ai-engineer-projects/AutomatedDataAnalyst/logs/dataset_splits/train.jsonl`
  - `ai-engineer-projects/AutomatedDataAnalyst/logs/dataset_splits/validation.jsonl`
  - `ai-engineer-projects/AutomatedDataAnalyst/logs/dataset_splits/holdout.jsonl`

## Dataset Layers
- `raw`:
  - Nguon goc JSONL tu `AutomatedDataAnalyst`, khong sua noi dung.
- `normalized`:
  - Ban ghi da chuan hoa field name/type, fill default, sanitize text.
- `training-ready`:
  - Ban ghi SFT/eval ma training script doc truc tiep.

## Output Artifacts (Data)
- Training-ready files:
  - `ai_path/ada_trainer/data/train_sft.jsonl`
  - `ai_path/ada_trainer/data/val_sft.jsonl`
  - `ai_path/ada_trainer/data/test_sft.jsonl`
- Metadata/manifests:
  - `ai_path/ada_trainer/data/manifest.json`
  - `ai_path/ada_trainer/data/rejected_records.jsonl`

## Schema - Input Record (Normalized)
Required fields:
- `created_at` (string, ISO8601 UTC)
- `symbol` (string, uppercase pair, example `BTCUSDT`)
- `timeframe` (string enum: `1s|1m|5m|15m|1h|4h|1d|1w|1M`)
- `regime` (string)
- `ai_response` (string, non-empty)
- `entry_price` (number, `> 0`)
- `outcome_horizon_candles` (integer, `1..500`)

Optional fields:
- `patterns` (array<object>)
- `ai_provider` (string)
- `ai_model` (string)
- `outcome_label` (string enum: `up|down|flat|null`)
- `outcome_pct` (number or null)
- `outcome_ts` (string ISO8601 or null)
- `dataset_version` (string)
- `source` (string)
- `label_policy_version` (string)
- `schema_version` (integer)

## Schema - Training Ready Record (SFT)
Required fields:
- `id` (string, deterministic hash from source fields)
- `split` (string enum: `train|validation|holdout`)
- `task_type` (string enum: `sft_market_analysis`)
- `prompt` (string, non-empty)
- `completion` (string, non-empty)
- `metadata` (object)

Required metadata fields:
- `created_at` (string)
- `symbol` (string)
- `timeframe` (string)
- `regime` (string)
- `entry_price` (number)
- `outcome_horizon_candles` (integer)
- `source_path` (string)
- `source_line_no` (integer)

Optional metadata fields:
- `patterns_summary` (string)
- `outcome_label` (string or null)
- `outcome_pct` (number or null)
- `ai_provider` (string)
- `ai_model` (string)
- `dataset_version` (string)
- `label_policy_version` (string)

## Validation Rules
Record-level hard rules (reject record neu fail):
1. `created_at` phai parse duoc ISO time.
2. `symbol` match regex: `^[A-Z0-9]{2,20}$`.
3. `timeframe` thuoc enum hop le.
4. `entry_price > 0`.
5. `outcome_horizon_candles` la integer trong khoang `1..500`.
6. `ai_response` khong rong sau khi trim.
7. `ai_response` sau sanitize khong vuot max length (`8000` chars default).

Soft rules (giu record nhung gan warning):
1. `regime` rong -> map `unknown`.
2. `patterns` khong hop le -> map thanh mang rong.
3. `outcome_label` khong hop le -> map `null`.
4. `outcome_pct` khong parse duoc -> map `null`.

Dataset-level hard rules (fail build neu fail):
1. `train/validation/holdout` khong duoc overlap theo `id`.
2. Tong so record hop le moi split phai lon hon 0.
3. `manifest.json` phai ghi ro counts + rejected_count.

## Sanitization Rules
Apply cho text fields (`ai_response`, `regime`, pattern strings):
1. Mask secret-like substrings:
   - `sk-...`, `ghp_...`, `AIza...`, `Bearer ...`, `api_key=...`
2. Remove null bytes.
3. Trim spaces dau/cuoi.
4. Truncate theo max length policy.

## Transform Rules
### Rule T1 - Normalize input
- Parse JSON line.
- Attach `source_path`, `source_line_no`.
- Coerce types:
  - `entry_price -> float`
  - `outcome_horizon_candles -> int`
  - `outcome_pct -> float|null`
- Standardize enums:
  - `symbol -> uppercase`
  - `timeframe` check enum

### Rule T2 - Patterns summary
- Input `patterns` la array object.
- Build `patterns_summary` dang text ngan:
  - Example: `BOS:BULLISH; Double Top:BEARISH`
- Neu khong co pattern hop le -> `patterns_summary = "none"`.

### Rule T3 - Prompt construction
`prompt` format (v1):
```text
You are a crypto market analyst.
Analyze the market snapshot and provide a concise trading analysis.

Symbol: {symbol}
Timeframe: {timeframe}
Regime: {regime}
Entry Price: {entry_price}
Horizon (candles): {outcome_horizon_candles}
Patterns: {patterns_summary}
```

### Rule T4 - Completion construction
- `completion = ai_response` sau sanitize.
- Khong tu y rewrite noi dung completion trong v1.

### Rule T5 - ID generation
- Deterministic hash on:
  - `created_at|symbol|timeframe|entry_price|ai_response[:256]`
- Muc tieu:
  - de-dup on repeated logs
  - reproducible split audit

## Rejection Policy
Rejected records duoc ghi vao:
- `ai_path/ada_trainer/data/rejected_records.jsonl`

Each rejected row phai co:
- `reason_code`
- `source_path`
- `source_line_no`
- `raw_preview` (max 500 chars)

Reason codes v1:
- `invalid_json`
- `invalid_created_at`
- `invalid_symbol`
- `invalid_timeframe`
- `invalid_entry_price`
- `invalid_horizon`
- `empty_ai_response`

## Manifest Contract
`manifest.json` phai co:
- `contract_version`
- `build_ts`
- `input_paths`
- `counts`:
  - `train`
  - `validation`
  - `holdout`
  - `rejected`
- `hashes`:
  - sha256 cho moi output file
- `transform_version` (default `t_v1`)

## Example Records
### Example Input (normalized view)
```json
{
  "created_at": "2026-03-05T10:22:11Z",
  "symbol": "BTCUSDT",
  "timeframe": "1h",
  "regime": "uptrend",
  "patterns": [
    {"name": "BOS", "signal": "BULLISH"}
  ],
  "ai_response": "Xu huong tang, uu tien buy-the-dip voi stop-loss duoi vung ho tro.",
  "entry_price": 91850.0,
  "outcome_horizon_candles": 8,
  "outcome_label": "up",
  "outcome_pct": 1.8
}
```

### Example Training-ready SFT
```json
{
  "id": "ada_4b6b8f6c...",
  "split": "train",
  "task_type": "sft_market_analysis",
  "prompt": "You are a crypto market analyst.\nAnalyze the market snapshot and provide a concise trading analysis.\n\nSymbol: BTCUSDT\nTimeframe: 1h\nRegime: uptrend\nEntry Price: 91850.0\nHorizon (candles): 8\nPatterns: BOS:BULLISH\n",
  "completion": "Xu huong tang, uu tien buy-the-dip voi stop-loss duoi vung ho tro.",
  "metadata": {
    "created_at": "2026-03-05T10:22:11Z",
    "symbol": "BTCUSDT",
    "timeframe": "1h",
    "regime": "uptrend",
    "entry_price": 91850.0,
    "outcome_horizon_candles": 8,
    "outcome_label": "up",
    "outcome_pct": 1.8,
    "patterns_summary": "BOS:BULLISH",
    "source_path": "ai-engineer-projects/AutomatedDataAnalyst/logs/dataset_splits/train.jsonl",
    "source_line_no": 127
  }
}
```

## Split Policy
- Split input preferred source: `dataset_splits/{train,validation,holdout}.jsonl`.
- Neu chi co `master_dataset.jsonl`, split temporal bat buoc:
  - train: 70%
  - validation: 15%
  - holdout: 15%
- Khong shuffle global theo random trong v1 neu split temporal da co san.

## Change Policy
Versioning:
- `MAJOR`: break schema or transform compatibility.
- `MINOR`: add optional fields/rules khong break.
- `PATCH`: fix validation bug, typo, docs.

Backward compatibility:
- v1 readers phai bo qua unknown optional fields.
- Required fields khong duoc doi ten trong same MAJOR.

Deprecation:
- Muc nao can bo, danh dau `deprecated_in` va `remove_in` trong docs it nhat 1 MINOR truoc khi remove.

## Operational Notes
- Muc tieu cua contract nay la reproducibility:
  - cung input + cung transform_version -> output giong nhau.
- Artifact model (checkpoint/adapter) KHONG thay the dataset contract:
  - dataset contract de biet model hoc tu dau, chat luong ra sao, va co train lai duoc hay khong.
