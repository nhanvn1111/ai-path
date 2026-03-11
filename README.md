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

## Week 8-9: Training loop & Dataset

> Chua bat dau

## Week 10-11: Inference toi uu

> Chua bat dau

## Week 12: Fine-tuning & LoRA

> Chua bat dau
