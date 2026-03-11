# AI From Scratch - Bai Tap

## Cau truc

```
ai_path/
  .venv/                          # Python virtual environment
  01_linear_algebra.py            # Week 1-2: Matrix class (pure Python)
  02_gradient_descent.py          # Week 1-2: Gradient Descent (pure Python)
  03_neural_network.py            # Week 3-4: MLP + Dropout + BatchNorm (numpy)
  04_autograd.py                  # Week 3-4: Autograd system (nhu PyTorch)
  05_tokenizer.py                 # Week 5-7: BPE + WordPiece tokenizer
  06_attention.py                 # Week 5-7: Self/Cross/Multi-Head Attention
  07_gpt.py                       # Week 5-7: Full GPT model
  08_training.py                  # Week 8-9: Training loop + Shakespeare
  09_kv_cache.py                  # Week 10-11: KV-Cache benchmark + Paged Attention
  10_quantization.py              # Week 10-11: INT8/INT4/GPTQ/AWQ quantization
  11_lora.py                      # Week 12: LoRA + QLoRA fine-tuning
  plot_*.png                      # (auto-generated plots)
```

## Yeu cau

- Python 3.8+
- Bai 01, 02: khong can thu vien gi (pure Python)
- Bai 03-07: can numpy, scikit-learn, matplotlib, torch, tokenizers
- Bai 08-11: can numpy, matplotlib, torch

## Setup

```bash
cd ai_path

# Tao venv (chi can lam 1 lan)
python3 -m venv .venv
source .venv/bin/activate
pip install numpy scikit-learn matplotlib torch tokenizers
```

## Cach chay

```bash
cd ai_path
source .venv/bin/activate

# Bai 1: Linear Algebra (khong can venv)
python 01_linear_algebra.py

# Bai 2: Gradient Descent (khong can venv, can venv cho plot)
python 02_gradient_descent.py

# Bai 3: Neural Network (can venv)
python 03_neural_network.py

# Bai 4: Autograd (can venv)
python 04_autograd.py

# Bai 5: Tokenizer
python 05_tokenizer.py

# Bai 6: Attention
python 06_attention.py

# Bai 7: GPT
python 07_gpt.py

# Bai 8: Training Loop (can torch, chay lau ~2-5 phut)
python 08_training.py

# Bai 9: KV-Cache Benchmark
python 09_kv_cache.py

# Bai 10: Quantization
python 10_quantization.py

# Bai 11: LoRA
python 11_lora.py
```

---

## Week 1-2: Nen tang Toan

### 01_linear_algebra.py - Matrix Operations tu Scratch

Class `Matrix` voi cac method (KHONG dung numpy):

| Method | Mo ta |
|--------|-------|
| `dot(other)` | Nhan ma tran (matrix multiplication) |
| `transpose()` | Chuyen vi ma tran |
| `add(other)` | Cong tung phan tu |
| `subtract(other)` | **[Bai tap 1]** Tru tung phan tu |
| `scalar_multiply(s)` | Nhan voi scalar |
| `element_multiply(other)` | Nhan tung phan tu (Hadamard) |
| `apply(func)` | **[Bai tap 2]** Ap dung ham len moi phan tu |

Tests:
- Test co ban: dot, transpose, neural network forward pass
- Bai tap 1: subtract + kiem tra dimension mismatch
- Bai tap 2: apply voi ReLU, Sigmoid, Tanh, Square
- Bai tap 3: Ma tran 100x100 - dot, transpose, subtract, apply, va mini neural network forward pass

### 02_gradient_descent.py - Gradient Descent tu Scratch

3 optimizers implement tu scratch:

| Optimizer | Mo ta |
|-----------|-------|
| `gradient_descent()` | Vanilla GD co ban |
| `sgd_with_momentum()` | GD + momentum (vuot local minima) |
| `adam()` | Adam optimizer (pho bien nhat) |

Bai tap:
1. **Plot loss history** - Ve do thi loss qua cac epoch, luu ra file PNG
2. **So sanh toc do hoi tu** - GD vs Momentum vs Adam tren cung 1 bai toan
3. **Learning rate** - Test voi lr tu 0.001 den 1.1, thay hieu ung diverge
4. **Rosenbrock function** - Ham test kinh dien, narrow valley, kho optimize

### Checklist Week 1-2

- [x] Code duoc matrix multiply khong dung numpy
- [x] Giai thich duoc tai sao matrix multiply la O(n^3) (3 vong for long nhau)
- [x] Implement 3 optimizers: SGD, Momentum, Adam
- [x] Biet learning rate qua lon/nho thi sao (diverge vs hoi tu cham)
- [x] Visualize duoc loss curve (matplotlib plots)

---

## Week 3-4: Neural Network tu Scratch

### 03_neural_network.py - MLP + Dropout + BatchNorm

Cac layer implement tu scratch (chi dung numpy):

| Class | Mo ta |
|-------|-------|
| `Layer` | Dense layer voi forward/backward/update |
| `Dropout` | **[Bai tap 2]** Randomly drop neurons (regularization) |
| `BatchNorm` | **[Bai tap 3]** Normalize input moi layer |
| `NeuralNetwork` | MLP wrapper, ho tro dropout + batchnorm |

Bai tap:
1. **Train MNIST** - 10000 samples tu sklearn, dat ~95-96% accuracy
2. **Dropout** - rate=0.3, giam overfitting
3. **Batch Normalization** - training nhanh hon, accuracy cao hon
4. **So sanh PyTorch** - cung architecture, accuracy gan nhau (~95-96%)
5. **Decision boundary** - visualize tren circular dataset

Ket qua MNIST (30 epochs):

| Model | Test Acc |
|-------|----------|
| Plain MLP | 95.7% |
| + Dropout(0.3) | 96.1% |
| + BatchNorm | 96.7% |
| PyTorch (same arch) | 96.0% |

### 04_autograd.py - Autograd System (nhu PyTorch)

`Tensor` class voi automatic differentiation - computational graph + chain rule:

| Operation | Gradient |
|-----------|----------|
| `+`, `-`, `*`, `**` | Basic arithmetic |
| `@` (matmul) | d(A@B)/dA = grad @ B.T |
| `relu()` | grad * (x > 0) |
| `sigmoid()` | grad * s * (1-s) |
| `tanh()` | **[Bai tap 1]** grad * (1 - tanh^2) |
| `exp()` | **[Bai tap 1]** grad * exp(x) |
| `log()` | **[Bai tap 1]** grad * 1/x |
| `cross_entropy_loss()` | **[Bai tap 2]** softmax(x) - y_true |

Bai tap:
1. **exp, log, tanh** - Them operations + verify gradient
2. **Softmax + Cross Entropy** - Fused cho numerical stability
3. **Full Neural Network** - `AutogradMLP` train tren circular dataset, acc=100%
4. **Verify voi PyTorch** - Moi gradient MATCH chinh xac voi PyTorch

### Checklist Week 3-4

- [x] Build Layer class voi forward/backward (backpropagation)
- [x] Build MLP tu nhieu layers
- [x] Train duoc tren MNIST dat > 95%
- [x] Implement Dropout va BatchNorm
- [x] So sanh voi PyTorch - accuracy tuong duong
- [x] Build autograd system (computational graph + chain rule)
- [x] Train network voi autograd (gradient tu dong)
- [x] Verify tat ca gradients khop voi PyTorch

---

## Week 5-7: Transformer tu Scratch

### 05_tokenizer.py - BPE + WordPiece Tokenizer

| Class | Mo ta |
|-------|-------|
| `BPETokenizer` | Byte Pair Encoding (GPT style) |
| `WordPieceTokenizer` | **[Bai tap 2]** BERT style, ## prefix |

Bai tap:
1. **Dataset lon** - Train tren Shakespeare (27k lines), compression ratio ~2-3x
2. **WordPiece** - Implement BERT-style tokenizer voi ## subword prefix
3. **HuggingFace** - Compare voi `tokenizers` library (Rust backend)
4. **Special tokens** - [CLS], [SEP], [MASK], sentence pair encoding

### 06_attention.py - Attention Mechanism

| Class | Mo ta |
|-------|-------|
| `SelfAttention` | Q, K, V + scaled dot-product |
| `MultiHeadAttention` | Nhieu heads song song |
| `CrossAttention` | **[Bai tap 2]** Encoder-decoder attention |

Bai tap:
1. **Visualize** - Attention heatmap (self + cross attention)
2. **Cross-attention** - Decoder attend to encoder output
3. **Dropout** - Attention dropout during training
4. **Compare PyTorch** - Verify voi `torch.nn.MultiheadAttention`

### 07_gpt.py - Full GPT Model

| Class | Mo ta |
|-------|-------|
| `GPT` | Decoder-only Transformer |
| `TransformerBlock` | MHA + FFN + residual + LayerNorm |
| `PositionalEncoding` | Sinusoidal position embedding |

Bai tap:
1. **Training loop** - Cross-entropy loss, forward pass demo
2. **Shakespeare** - Char-level generation (untrained demo)
3. **KV-Cache** - 1.6x speedup cho generation
4. **Dropout + Weight decay** - Regularization verified
5. **Visualize** - Attention patterns across all layers/heads

### Checklist Week 5-7

- [x] Build BPE tokenizer tu scratch
- [x] Implement WordPiece tokenizer (BERT style)
- [x] Code self-attention, multi-head attention, cross-attention
- [x] Build full GPT architecture (embedding + transformer blocks + output)
- [x] Generate text (random nhung working)
- [x] Implement KV-cache cho faster generation
- [x] Visualize attention patterns

---

## Production: ada_trainer - LoRA Fine-tuning Pipeline

### Muc dich

**ada_trainer** la training pipeline dung trong **AutomatedDataAnalyst** (crypto analysis dashboard). Tac dung chinh: fine-tune small language models (1.5B-7B parameters) bang LoRA (Low-Rank Adaptation) de cai thiep AI quality theo thoi gian.

**Use Case:**
- Input: Labeled trading analysis data (prompt → AI response + human score)
- Output: LoRA adapter tailored cho domain (crypto, candlestick patterns, on-chain metrics)
- Inference: Local deployment (offline, low-latency, chi tieu ~1GB VRAM)

### Cau truc

```
ai_path/ada_trainer/
  ├── build_dataset.py          # Build training dataset (split + transform)
  ├── train.py                  # LoRA fine-tuning (QLoRA, bf16, M1-optimized)
  ├── evaluate.py               # Metrics: MAE, F1, Accuracy
  ├── export_adapter.py         # Export adapter cho production
  │
  ├── data/                      # (Generated - not in git)
  │   ├── train_sft.jsonl       # ~300KB, 308K training samples
  │   ├── val_sft.jsonl         # ~66KB, 66K validation samples
  │   ├── test_sft.jsonl        # ~66KB, 66K test samples
  │   ├── judge_sft.jsonl       # ~12KB, scoring model dataset
  │   ├── manifest.json         # Dataset metadata (schema v7)
  │   └── rejected_records.jsonl # Rows failed validation
  │
  ├── artifacts/                 # (Generated - not in git, ~1.3GB)
  │   └── model/
  │       ├── run_lowmem_20260309T095000Z/    # LoRA adapter checkpoints
  │       ├── mode_inline_20260309T175638885535Z/
  │       └── ... (other runs)
  │
  └── .gitignore               # Exclude data/ + artifacts/
```

### Tính năng chính

#### 1. `build_dataset.py` - Data Pipeline

**Input:** `AutomatedDataAnalyst/logs/master_dataset.jsonl` (AI calls + outcomes)

**Output:** Temporal split (70/15/15) + validation

| Feature | Mo ta |
|---------|-------|
| Temporal split | Oldest 70% train → next 15% val → newest 15% test (khong shuffle) |
| Schema validation | Check required fields (created_at, symbol, ai_response, etc.) |
| Transform | Hash + normalize inputs, reject malformed records |
| Manifest | Track data contract version (v7), row counts, hashes |

**Command:**
```bash
PYTHONPATH=. python ai_path/ada_trainer/build_dataset.py
# Output: data/train_sft.jsonl, val_sft.jsonl, test_sft.jsonl, manifest.json
```

#### 2. `train.py` - LoRA Fine-tuning

**Base Model:** Qwen2.5-1.5B-Instruct (HuggingFace)

**Technique:** QLoRA (Quantized LoRA) - 4-bit quantization

| Parameter | Config |
|-----------|--------|
| Batch size | 1 per device (M1 Pro 16GB limit) |
| Gradient accumulation | 2 steps |
| Max token length | 512 tokens |
| Precision | bfloat16 (bf16) |
| LoRA rank (r) | 8 |
| LoRA alpha | 16 |
| Learning rate | 5e-4 |
| Warmup steps | 100 |
| Total epochs | 3 |

**Output:** LoRA adapter weights (~50MB), training logs, checkpoints

**Command:**
```bash
PYTHONPATH=. python ai_path/ada_trainer/train.py \
  --mode lora \
  --run-id run_lowmem_$(date -u +%Y%m%dT%H%M%SZ) \
  --per-device-train-batch-size 1 \
  --gradient-accumulation-steps 2 \
  --max-length 512
```

#### 3. `evaluate.py` - Metrics Calculation

**Output:** `artifacts/metrics.json` voi:

| Metric | Mo ta |
|--------|-------|
| MAE | Mean Absolute Error (score prediction) |
| F1 (weighted) | Classification: good/ok/bad buckets |
| Accuracy | % predictions dung |
| Loss | Validation cross-entropy |
| Perplexity | Token-level prediction quality |

**Command:**
```bash
PYTHONPATH=. python ai_path/ada_trainer/evaluate.py
```

#### 4. `export_adapter.py` - Production Export

**Converts:** Checkpoint → Inference-ready adapter

**Output:** `artifacts/export/latest_adapter/`
- `adapter_model.bin` (50MB)
- `adapter_config.json`
- Tokenizer config

**Used by:** Dashboard local inference (`LOCAL_ADAPTER_PATH`)

**Command:**
```bash
PYTHONPATH=. python ai_path/ada_trainer/export_adapter.py
```

### Data Contract (bap buoc)

**Input schema** (`master_dataset.jsonl`):
```json
{
  "created_at": "2026-03-10T12:00:00Z",
  "symbol": "BTCUSDT",
  "timeframe": "1h",
  "regime": "trending_up",
  "ai_response": "... full response ...",
  "entry_price": 43200.0,
  "outcome_pct": 2.5,
  "score_total": 7.5
}
```

**Output schema** (`train_sft.jsonl`):
```json
{
  "prompt": "[CONTEXT]\n{symbol} {regime} @ ${entry}...",
  "completion": "{ai_response}",
  "split": "train",
  "score_bucket": "good"
}
```

### Training Loop (End-to-End)

```
1. Run build_dataset.py
   → Validate + split master_dataset.jsonl
   → Create train/val/test splits

2. Run train.py --mode lora
   → Load Qwen2.5-1.5B (4-bit)
   → LoRA adapter on top
   → Train 3 epochs
   → Save checkpoints

3. Run evaluate.py
   → Load best checkpoint
   → Eval on validation set
   → Output metrics.json

4. Run export_adapter.py
   → Merge base model + LoRA adapter (or keep separate)
   → Export to production path
   → Ready for dashboard inference

5. Dashboard loads LOCAL_ADAPTER_PATH
   → Inference: base model + adapter
   → ~1GB VRAM, <500ms latency
```

### Integration voi AutomatedDataAnalyst

**Data flow:**

```
Dashboard AI calls
    ↓
logs/master_dataset.jsonl (log every call + outcome)
    ↓
build_dataset.py (weekly) → validate + split
    ↓
train.py (triggered manually or scheduled)
    ↓
evaluate.py → metrics.json (track improvement)
    ↓
export_adapter.py
    ↓
LOCAL_ADAPTER_PATH (dashboard loads, inference improves)
```

### Cac Gates

Gate system dam bao data sach truoc train:

| Gate | Requirement | Status |
|------|-------------|--------|
| **Gate 0** | Master dataset schema v7 ✅ | PASS |
| **Gate 1** | Observability (Prometheus+Loki) ✅ | PASS |
| **Gate 2** | LoRA training completes (bui chi Qwen1.5B) | IN PROGRESS |
| **Gate 3** | A/B test local vs cloud accuracy | READY |

### Requirements

```bash
pip install \
  transformers \
  peft \
  bitsandbytes \
  torch \
  datasets \
  scikit-learn
```

### Performance (M1 Pro 16GB)

| Config | VRAM | Throughput | Epochs |
|--------|------|-----------|--------|
| Batch=1, grad_accum=2, max_len=512 | ~14GB | 500 sample/min | 3 epochs |
| **Qwen2.5-1.5B (QLoRA, bf16)** | ~13GB | ✅ Stable | ✅ Completes |
| Qwen2.5-3B (QLoRA) | ~14.5GB | ⚠️ Tight | Cham |
| Qwen2.5-7B (QLoRA) | >16GB | ❌ OOM | Khong chay |

---

## Week 8-9: Training loop & Dataset

✅ **DONE** — See `ada_trainer/` above

## Week 10-11: Inference toi uu

> Phan nay thu hiem trong dashboard (KV-cache, int8)

## Week 12: Fine-tuning & LoRA

✅ **DONE** — ada_trainer pipeline fully functional
