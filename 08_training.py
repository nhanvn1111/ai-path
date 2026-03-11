# File: 08_training.py
# Training Loop cho GPT - Week 8-9
#
# TAI SAO CAN FILE NAY?
# Cac file truoc (01-07) chi build model architecture.
# Model luc nay chi la random weights - chua "biet" gi ca.
# Training loop la qua trinh day model hoc tu data:
#   1. Forward: model du doan next token
#   2. Loss: do sai lech giua du doan va dap an
#   3. Backward: tinh gradient (huong giam loss)
#   4. Update: dieu chinh weights theo gradient
#
# BAI TAP:
# 1. Implement full backpropagation cho GPT (dung PyTorch)
# 2. Add gradient clipping (ngan gradient bung no)
# 3. Implement learning rate warmup + cosine decay
# 4. Train tren Shakespeare dataset
# 5. Plot training loss curve

import numpy as np
import os
import time
import math


# ============================================================
# PHAN 1: KHAI NIEM CO BAN (numpy) - de hieu truoc khi dung PyTorch
# ============================================================

def cross_entropy_loss_numpy(logits, targets):
    """
    Cross-entropy loss cho language modeling (numpy version)

    TAI SAO DUNG CROSS-ENTROPY?
    - Language model can du doan probability distribution tren vocab
    - Cross-entropy do khoang cach giua 2 distributions:
      * predicted distribution (softmax cua logits)
      * true distribution (one-hot cua correct token)
    - Loss cang thap = model du doan cang chinh xac

    logits:  raw scores tu model, chua qua softmax
             Shape: (batch, seq_len, vocab_size)
             Vd: (32, 64, 50257) voi GPT-2 (vocab_size=50257)
             Gia tri thuong tu -3.0 den +3.0 (random init),
             sau training se cao hon cho correct token
    targets: token ids dung (ground truth)
             Shape: (batch, seq_len)
             Vd: (32, 64) - moi gia tri la index trong vocab [0, vocab_size)
             Day la "dap an" de tinh loss, lay tu dataset (shifted by 1)
    """
    batch_size, seq_len, vocab_size = logits.shape
    logits_flat = logits.reshape(-1, vocab_size)
    targets_flat = targets.reshape(-1)

    # Softmax: chuyen raw scores thanh probabilities
    # Trick: tru max de tranh overflow (e^x lon qua)
    exp_logits = np.exp(logits_flat - np.max(logits_flat, axis=-1, keepdims=True))
    probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)

    # Lay probability cua dap an dung
    correct_probs = probs[np.arange(len(targets_flat)), targets_flat]

    # Loss = -log(P(correct token))
    # Neu P cao (du doan dung) -> -log(P) thap -> loss thap
    # Neu P thap (du doan sai) -> -log(P) cao -> loss cao
    loss = -np.mean(np.log(correct_probs + 1e-9))
    return loss


class TextDataset:
    """
    Dataset cho language modeling

    TAI SAO CAN DATASET CLASS?
    - Organize data thanh input-target pairs
    - Input: [t1, t2, t3, ..., tN]
    - Target: [t2, t3, t4, ..., tN+1] (shifted by 1)
    - Model hoc: nhin vao tokens truoc, du doan token tiep theo
    """

    def __init__(self, tokens, block_size):
        """
        tokens:     list hoac array cac token ids (full text da tokenize)
                    Vd: [15, 887, 2, 9421, ...] - moi so la 1 token
                    Shakespeare char-level: ~1.1M tokens, vocab ~65
                    GPT-2 BPE: ~300K tokens cho cung text, vocab 50257
                    Cang nhieu tokens = cang nhieu data de train
        block_size: so tokens moi training sample (= context window cua model)
                    Vd: block_size=64 (demo), 1024 (GPT-2), 2048 (GPT-3),
                    4096 (LLaMA-2), 8192 (LLaMA-3)
                    Lon hon -> model nhin duoc nhieu context hon nhung ton RAM
                    Input la tokens[i:i+block_size], target la tokens[i+1:i+block_size+1]
        """
        self.tokens = np.array(tokens)
        self.block_size = block_size

    def __len__(self):
        return len(self.tokens) - self.block_size

    def get_batch(self, batch_size):
        """
        Lay random batch tu dataset

        TAI SAO RANDOM?
        - SGD (Stochastic Gradient Descent) dung random samples
        - Noise tu random sampling giup thoat local minima
        - Hieu qua hon train tuan tu

        batch_size: so luong samples trong 1 batch
                    Vd: 32 (demo), 64-512 (GPT-2), 1024-3200 (GPT-3)
                    Lon hon -> gradient on dinh hon nhung ton VRAM
                    Nho hon -> noisy hon nhung update nhanh hon
                    Thuong bat dau 32-64 roi tang dan (gradient accumulation)
                    Return: (xs, ys) voi shape (batch_size, block_size)
        """
        indices = np.random.randint(0, len(self), batch_size)
        xs, ys = [], []
        for idx in indices:
            chunk = self.tokens[idx: idx + self.block_size + 1]
            xs.append(chunk[:-1])   # Input
            ys.append(chunk[1:])    # Target (shifted by 1)
        return np.array(xs), np.array(ys)


# ============================================================
# PHAN 2: GRADIENT CLIPPING (numpy) - hieu khai niem
# ============================================================

def clip_grad_norm(grads, max_norm):
    """
    Gradient clipping by norm

    TAI SAO CAN GRADIENT CLIPPING?
    - Transformer de bi "exploding gradients":
      gradients qua lon -> weights update qua manh -> model diverge
    - Dac biet khi train voi long sequences hoac learning rate lon
    - Clip = gioi han do lon cua gradient vector

    CACH HOAT DONG:
    - Tinh total norm cua tat ca gradients (L2 norm)
    - Neu total_norm > max_norm: scale xuong
    - Neu total_norm <= max_norm: giu nguyen

    grads:    list cac numpy arrays, moi array la gradient cua 1 parameter
              Vd: [grad_W1, grad_b1, grad_W2, grad_b2, ...]
              Shape tuy parameter: (d_model, d_ff), (d_ff,), ...
              GPT-2 co ~148M params -> ~148M gradient values
              Khi exploding: norm co the len 100+ (binh thuong < 5)
    max_norm: nguong toi da cho total gradient norm (L2 norm)
              Vd: 1.0 (GPT-2/3, LLaMA), 0.5 (mot so config conservative)
              Gia tri pho bien nhat: 1.0
              Neu total_norm > max_norm -> scale tat ca gradients xuong
              Neu total_norm <= max_norm -> giu nguyen, khong lam gi

    Vi du:
    - Gradients: [3.0, 4.0], norm = 5.0
    - max_norm = 2.5
    - clip_ratio = 2.5 / 5.0 = 0.5
    - After clip: [1.5, 2.0], norm = 2.5
    """
    # Tinh total norm
    total_norm_sq = sum(np.sum(g ** 2) for g in grads)
    total_norm = np.sqrt(total_norm_sq)

    # Clip neu can
    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1.0:
        clipped = [g * clip_coef for g in grads]
        return clipped, total_norm
    return grads, total_norm


# ============================================================
# PHAN 3: LEARNING RATE SCHEDULING (numpy) - hieu khai niem
# ============================================================

def get_lr(step, warmup_steps, max_steps, max_lr, min_lr=0.0):
    """
    Learning rate schedule: warmup + cosine decay

    TAI SAO CAN LR SCHEDULING?
    - Learning rate (lr) la buoc nhay khi update weights
    - lr qua lon: model khong on dinh, loss nhay lung tung
    - lr qua nho: train cham, mat nhieu thoi gian

    WARMUP (giai doan dau):
    - Bat dau voi lr = 0, tang dan len max_lr
    - Tai sao? Luc dau weights la random, gradients chua on dinh
    - Neu dung lr lon ngay -> diverge
    - Tang tu tu giup model "lam quen" voi data

    COSINE DECAY (giai doan sau):
    - Giam lr tu max_lr xuong min_lr theo duong cong cosine
    - Tai sao cosine? Giam nhe nhe dau, nhanh giua, cham cuoi
    - Giup model hoi tu tot hon o cuoi training

    step:         buoc hien tai trong training loop (0-indexed)
                  Vd: step=0 la buoc dau tien, step=max_steps-1 la buoc cuoi
                  GPT-2: ~100K steps, GPT-3: ~300K steps, LLaMA: ~1T tokens / batch
    warmup_steps: so buoc warmup (tang lr tu 0 len max_lr)
                  Vd: 50 (demo), 2000 (GPT-2), 375 (GPT-3)
                  Thuong ~0.1-1% tong so steps
                  Warmup ngan qua -> diverge, dai qua -> lang phi thoi gian
    max_steps:    tong so buoc training
                  Vd: 500 (demo), 100K-800K (GPT-2), 300K (GPT-3)
                  Sau max_steps, lr giu nguyen tai min_lr
    max_lr:       learning rate cao nhat (peak sau warmup)
                  Vd: 3e-3 (demo nho), 6e-4 (GPT-2), 6e-5 (GPT-3 fine-tune)
                  Model lon hon -> max_lr nho hon (de on dinh)
    min_lr:       learning rate thap nhat (cuoi cosine decay)
                  Vd: 3e-4 (demo), 6e-5 (GPT-2, = max_lr / 10)
                  Thuong = max_lr / 10, giup model van hoc nhe o cuoi
                  Default = 0.0 (giam ve 0 hoan toan)

    Hinh dung:
    Step:  0 ---- warmup ---- peak ---- decay ---- end
    LR:    0 ---> max_lr ----> max_lr --> ... --> min_lr
    """
    if step < warmup_steps:
        # Linear warmup: 0 -> max_lr
        return max_lr * step / warmup_steps
    elif step >= max_steps:
        return min_lr
    else:
        # Cosine decay: max_lr -> min_lr
        progress = (step - warmup_steps) / (max_steps - warmup_steps)
        return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * progress))


# ============================================================
# PHAN 4: TRAINING VOI PYTORCH - bai tap chinh
# ============================================================

def train_with_pytorch(resume_checkpoint=False, checkpoint_name="gpt_checkpoint.pt"):
    """
    Train GPT tren Shakespeare bang PyTorch

    Args:
        resume_checkpoint: True -> neu co file checkpoint thi load va tiep tuc training.
        checkpoint_name:   ten file .pt luu trong cung thu muc (default gpt_checkpoint.pt).

    TAI SAO DUNG PYTORCH CHO TRAINING?
    - Backprop qua Transformer bang numpy thuan cuc ky phuc tap
      (can tinh gradient cho: attention scores, softmax, layer norm,
       residual connections, GELU, matrix multiply - hang ngan dong code)
    - PyTorch autograd tu dong tinh tat ca gradients
    - Trong thuc te, 100% AI engineers dung framework (PyTorch/JAX)
    - Nhung HIEU cach no hoat dong (cac file truoc) la dieu quan trong
    """
    import torch
    import torch.nn as nn

    def save_checkpoint(path, model, optimizer, step, loss):
        """Luu state_dict model + optimizer de resume sau nay."""
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "step": step,
                "loss": loss,
            },
            path,
        )

    def load_checkpoint(path, model, optimizer, device):
        """Load checkpoint neu ton tai, tra ve (step, loss)."""
        ckpt = torch.load(path, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        step = ckpt.get("step", 0)
        loss = ckpt.get("loss", None)
        print(f"  Loaded checkpoint '{path}' (step={step}, loss={loss})")
        return step, loss

    device = "cpu"

    # ----- Load Shakespeare -----
    script_dir = os.path.dirname(os.path.abspath(__file__))
    shakespeare_path = os.path.join(script_dir, "shakespeare.txt")

    if not os.path.exists(shakespeare_path):
        print("  Downloading Shakespeare...")
        import urllib.request
        url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        urllib.request.urlretrieve(url, shakespeare_path)

    with open(shakespeare_path, 'r') as f:
        text = f.read()

    # ----- Tokenization (char-level) -----
    # Dung char-level de don gian - moi ky tu la 1 token
    chars = sorted(set(text))
    vocab_size = len(chars)
    char2id = {c: i for i, c in enumerate(chars)}
    id2char = {i: c for c, i in char2id.items()}
    encode = lambda s: [char2id[c] for c in s]
    decode = lambda ids: ''.join([id2char[i] for i in ids])

    data = torch.tensor(encode(text), dtype=torch.long)
    print(f"  Dataset: {len(data):,} chars, vocab: {vocab_size}")

    # Train/val split (90/10)
    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data = data[n:]

    # ----- GPT Model (PyTorch) -----
    # Architecture giong het file 07_gpt.py nhung dung PyTorch layers
    # de autograd tu dong tinh gradient

    class GPTBlock(nn.Module):
        def __init__(self, d_model, num_heads, dropout):
            """
            1 Transformer block trong GPT (pre-norm architecture)

            d_model:   kich thuoc embedding vector cua moi token
                       Vd: 128 (demo), 768 (GPT-2 small), 1024 (GPT-2 medium),
                       1280 (GPT-2 large), 1600 (GPT-2 XL), 12288 (GPT-3)
                       Phai chia het cho num_heads (head_dim = d_model / num_heads)
            num_heads: so attention heads trong multi-head attention
                       Vd: 4 (demo), 12 (GPT-2 small), 16 (GPT-2 medium),
                       20 (GPT-2 large), 96 (GPT-3)
                       Nhieu heads -> model hoc nhieu "goc nhin" khac nhau
                       head_dim = d_model / num_heads, vd: 768/12 = 64
            dropout:   ty le dropout cho attention va feed-forward
                       Vd: 0.1 (GPT-2), 0.0 (GPT-3, LLaMA - du data lon)
                       Regularization: random tat 1 so neurons khi training
                       Giup chong overfitting, dac biet voi data nho
            """
            super().__init__()
            self.ln1 = nn.LayerNorm(d_model)
            self.attn = nn.MultiheadAttention(
                d_model, num_heads, dropout=dropout, batch_first=True
            )
            self.ln2 = nn.LayerNorm(d_model)
            self.ff = nn.Sequential(
                nn.Linear(d_model, d_model * 4),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model * 4, d_model),
                nn.Dropout(dropout),
            )

        def forward(self, x, mask=None):
            # Pre-norm architecture (giong GPT-2)
            normed = self.ln1(x)
            attn_out, _ = self.attn(normed, normed, normed, attn_mask=mask)
            x = x + attn_out           # Residual connection
            x = x + self.ff(self.ln2(x))  # Residual connection
            return x

    class MiniGPT(nn.Module):
        """
        Mini GPT cho char-level Shakespeare

        Architecture:
        - Token embedding: char -> vector
        - Position embedding: vi tri -> vector
        - N x Transformer blocks
        - Final LayerNorm + projection to vocab
        """
        def __init__(self, vocab_size, d_model, num_heads, num_layers,
                     block_size, dropout=0.1):
            """
            vocab_size:  kich thuoc vocabulary (so luong tokens duy nhat)
                         Vd: 65 (char-level Shakespeare), 50257 (GPT-2 BPE),
                         32000 (LLaMA), 100K+ (GPT-4)
                         Anh huong truc tiep den embedding table size:
                         vocab_size * d_model params
            d_model:     kich thuoc embedding vector cua moi token
                         Vd: 128 (demo), 768 (GPT-2 small), 12288 (GPT-3 175B)
                         Lon hon -> bieu dien phong phu hon nhung ton VRAM
            num_heads:   so attention heads
                         Vd: 4 (demo), 12 (GPT-2 small), 96 (GPT-3)
                         head_dim = d_model / num_heads
            num_layers:  so transformer blocks xep chong
                         Vd: 4 (demo), 12 (GPT-2 small), 24 (GPT-2 medium),
                         36 (GPT-2 large), 48 (GPT-2 XL), 96 (GPT-3)
                         Nhieu layers -> model "sau" hon, hoc pattern phuc tap hon
            block_size:  context window toi da (so tokens model co the nhin)
                         Vd: 64 (demo), 1024 (GPT-2), 2048 (GPT-3),
                         4096 (LLaMA-2), 128K (GPT-4)
                         Position embedding table co block_size entries
            dropout:     ty le dropout (default 0.1)
                         Vd: 0.1 (GPT-2, model nho), 0.0 (GPT-3, LLaMA)
                         Model lon + data lon -> dropout thap hoac tat
            """
            super().__init__()
            self.block_size = block_size
            self.token_emb = nn.Embedding(vocab_size, d_model)
            self.pos_emb = nn.Embedding(block_size, d_model)
            self.drop = nn.Dropout(dropout)

            self.blocks = nn.ModuleList([
                GPTBlock(d_model, num_heads, dropout)
                for _ in range(num_layers)
            ])
            self.ln_f = nn.LayerNorm(d_model)
            self.head = nn.Linear(d_model, vocab_size, bias=False)

            # Weight tying: output projection = embedding transpose
            # Tai sao? Giam 50% params va hoc tot hon
            # (token gan nhau trong embedding space cung co logit gan nhau)
            self.head.weight = self.token_emb.weight

            self.apply(self._init_weights)

        def _init_weights(self, module):
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

        def forward(self, idx):
            B, T = idx.shape
            tok_emb = self.token_emb(idx)                    # (B, T, d_model)
            pos_emb = self.pos_emb(torch.arange(T, device=idx.device))  # (T, d_model)
            x = self.drop(tok_emb + pos_emb)

            # Causal mask: token chi nhin duoc tokens truoc no
            mask = nn.Transformer.generate_square_subsequent_mask(T).to(idx.device)

            for block in self.blocks:
                x = block(x, mask)

            x = self.ln_f(x)
            logits = self.head(x)  # (B, T, vocab_size)
            return logits

        @torch.no_grad()
        def generate(self, idx, max_new_tokens, temperature=0.8, top_k=40):
            """
            Sinh text tu model (autoregressive generation)

            idx:            tensor chua token ids lam context ban dau
                            Shape: (batch_size, seq_len)
                            Vd: torch.tensor([[15, 887, 2]]) - 1 sample voi 3 tokens
                            Co the la prompt da tokenize, vd: "ROMEO:\n"
                            Se bi cat con block_size tokens cuoi neu qua dai
            max_new_tokens: so tokens moi can sinh ra
                            Vd: 200 (demo ngan), 500-2000 (doan van),
                            4096+ (bai viet dai)
                            Model sinh 1 token/buoc, loop max_new_tokens lan
            temperature:    kiem soat "do ngau nhien" cua output (default 0.8)
                            Vd: 0.0-0.3 (rat deterministic, chon token co prob cao nhat),
                            0.7-0.9 (can bang giua sang tao va logic),
                            1.0 (giu nguyen distribution), >1.0 (rat random)
                            Cong thuc: logits = logits / temperature
            top_k:          chi xet top K tokens co probability cao nhat (default 40)
                            Vd: 40 (GPT-2 default), 50 (pho bien),
                            None (tat top_k, xet toan bo vocab)
                            Loai bo cac tokens kho xay ra, giup text tu nhien hon
                            Ket hop voi temperature de tinh chinh chat luong output
            """
            for _ in range(max_new_tokens):
                idx_cond = idx[:, -self.block_size:]
                logits = self(idx_cond)
                logits = logits[:, -1, :] / temperature

                if top_k is not None:
                    v, _ = torch.topk(logits, top_k)
                    logits[logits < v[:, [-1]]] = -float('Inf')

                probs = torch.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)
                idx = torch.cat([idx, idx_next], dim=1)
            return idx

    # ----- Hyperparameters -----
    # Giu nho de chay nhanh tren CPU
    d_model = 128         # Embedding dimension
    num_heads = 4         # So attention heads
    num_layers = 4        # So transformer blocks
    block_size = 64       # Context window (so tokens model nhin lai)
    batch_size = 32       # So samples moi batch
    max_steps = 500       # So buoc training (tang len 2000-5000 de ket qua tot hon)
    max_lr = 3e-3         # Learning rate cao nhat
    min_lr = 3e-4         # Learning rate thap nhat
    warmup_steps = 50     # So buoc warmup
    dropout = 0.1         # Dropout rate
    weight_decay = 0.01   # L2 regularization
    grad_clip = 1.0       # Max gradient norm
    eval_interval = 50    # Danh gia moi 50 steps

    # ----- Get batch function -----
    def get_batch(split):
        d = train_data if split == 'train' else val_data
        ix = torch.randint(len(d) - block_size, (batch_size,))
        x = torch.stack([d[i:i+block_size] for i in ix])
        y = torch.stack([d[i+1:i+block_size+1] for i in ix])
        return x.to(device), y.to(device)

    @torch.no_grad()
    def estimate_loss(model, eval_steps=20):
        """
        Danh gia loss tren train va val set

        model:      MiniGPT model can danh gia
                    Se duoc chuyen sang eval mode (tat dropout)
                    roi chuyen lai train mode sau khi danh gia xong
        eval_steps: so batches dung de uoc luong loss (default 20)
                    Vd: 20 (demo, nhanh), 50-200 (chinh xac hon)
                    Loss duoc tinh trung binh tren eval_steps batches
                    Nhieu hon -> uoc luong chinh xac hon nhung cham hon
                    Thuong 20-50 la du on dinh cho validation
        """
        model.eval()
        out = {}
        for split in ['train', 'val']:
            losses = []
            for _ in range(eval_steps):
                x, y = get_batch(split)
                logits = model(x)
                loss = nn.functional.cross_entropy(
                    logits.view(-1, vocab_size), y.view(-1)
                )
                losses.append(loss.item())
            out[split] = np.mean(losses)
        model.train()
        return out

    # ----- Create model -----
    model = MiniGPT(
        vocab_size=vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        block_size=block_size,
        dropout=dropout,
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"  Model params: {num_params:,}")

    # ----- Optimizer -----
    # AdamW: Adam + weight decay (decoupled)
    # Tai sao AdamW thay vi Adam?
    # - Adam: weight decay couple voi gradient -> khong hieu qua
    # - AdamW: weight decay tach rieng -> regularization tot hon
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=max_lr,
        weight_decay=weight_decay,
        betas=(0.9, 0.95),  # GPT-style betas
    )

    checkpoint_path = os.path.join(script_dir, checkpoint_name)
    global_step = 0
    if resume_checkpoint and os.path.exists(checkpoint_path):
        global_step, _ = load_checkpoint(checkpoint_path, model, optimizer, device)
    elif resume_checkpoint:
        print(f"  resume_checkpoint=True nhung khong tim thay '{checkpoint_path}', bat dau tu dau.")
    else:
        print("  Checkpoint resume OFF (resume_checkpoint=False).")

    # ----- Training Loop -----
    print(f"\n  Training {max_steps} steps...")
    print(f"  {'Step':>6} | {'Train Loss':>10} | {'Val Loss':>10} | {'LR':>10} | {'Grad Norm':>10}")
    print(f"  {'-'*6}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}")

    loss_history = {'train': [], 'val': [], 'steps': [], 'lr': [], 'grad_norm': []}
    best_val_loss = float('inf')
    start_time = time.time()

    for step in range(global_step, max_steps):
        # Bai tap 3: Learning rate scheduling
        lr = get_lr(step, warmup_steps, max_steps, max_lr, min_lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # Forward
        x, y = get_batch('train')
        logits = model(x)
        loss = nn.functional.cross_entropy(
            logits.view(-1, vocab_size), y.view(-1)
        )

        # Backward
        optimizer.zero_grad()
        loss.backward()

        # Bai tap 2: Gradient clipping
        # Ngan gradient bung no - rat quan trong cho Transformer
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        # Update weights
        optimizer.step()

        # Evaluation
        if step % eval_interval == 0 or step == max_steps - 1:
            losses = estimate_loss(model)
            loss_history['train'].append(losses['train'])
            loss_history['val'].append(losses['val'])
            loss_history['steps'].append(step)
            loss_history['lr'].append(lr)
            loss_history['grad_norm'].append(grad_norm.item())

            if losses['val'] < best_val_loss:
                best_val_loss = losses['val']

            print(f"  {step:>6} | {losses['train']:>10.4f} | {losses['val']:>10.4f} | {lr:>10.6f} | {grad_norm.item():>10.4f}")
            save_checkpoint(checkpoint_path, model, optimizer, step, losses['val'])
            print(f"    -> Saved checkpoint: {checkpoint_path}")

    elapsed = time.time() - start_time
    print(f"\n  Training done in {elapsed:.1f}s")
    print(f"  Best val loss: {best_val_loss:.4f}")

    # ----- Bai tap 4: Generate Shakespeare -----
    print("\n  --- Generated Shakespeare ---")
    model.eval()
    context = torch.tensor([encode("ROMEO:\n")], dtype=torch.long).to(device)
    generated = model.generate(context, max_new_tokens=200, temperature=0.8, top_k=40)
    print(decode(generated[0].tolist()))
    print("  --- End ---")

    return model, loss_history, vocab_size, encode, decode


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    np.random.seed(42)
    output_dir = os.path.dirname(os.path.abspath(__file__))

    # ---- Test numpy concepts ----
    print("=" * 60)
    print("PHAN 1: Cross-Entropy Loss (numpy)")
    print("=" * 60)

    # Simulate: 2 samples, 5 tokens, vocab=10
    logits = np.random.randn(2, 5, 10)
    targets = np.random.randint(0, 10, (2, 5))
    loss = cross_entropy_loss_numpy(logits, targets)
    print(f"  Random logits loss: {loss:.4f}")
    print(f"  Expected (log(10)): {np.log(10):.4f}")
    assert abs(loss - np.log(10)) < 0.5, "Random loss should be ~log(vocab_size)"
    print("  Random loss ~ log(vocab): OK")

    # Test: perfect prediction should have low loss
    perfect_logits = np.zeros((1, 3, 10))
    perfect_targets = np.array([[2, 5, 7]])
    for i, t in enumerate(perfect_targets[0]):
        perfect_logits[0, i, t] = 10.0  # High score cho correct token
    perfect_loss = cross_entropy_loss_numpy(perfect_logits, perfect_targets)
    print(f"  Perfect prediction loss: {perfect_loss:.6f}")
    assert perfect_loss < 0.01, "Perfect prediction should have very low loss"
    print("  Perfect prediction -> low loss: OK")

    print("\n" + "=" * 60)
    print("PHAN 2: Gradient Clipping")
    print("=" * 60)

    # Demo gradient clipping
    grads = [np.array([3.0, 4.0]), np.array([0.0])]  # norm = 5.0
    clipped, norm = clip_grad_norm(grads, max_norm=2.5)
    print(f"  Original grads: {[g.tolist() for g in grads]}, norm={norm:.1f}")
    print(f"  Clipped grads:  {[np.round(g, 2).tolist() for g in clipped]}")
    clipped_norm = np.sqrt(sum(np.sum(g ** 2) for g in clipped))
    print(f"  Clipped norm:   {clipped_norm:.1f}")
    assert abs(clipped_norm - 2.5) < 0.01, "Clipped norm should equal max_norm"
    print("  Gradient clipping: OK")

    # No clipping when norm is small
    small_grads = [np.array([0.1, 0.2])]
    clipped_small, norm_small = clip_grad_norm(small_grads, max_norm=2.5)
    assert np.allclose(clipped_small[0], small_grads[0])
    print("  Small grads not clipped: OK")

    print("\n" + "=" * 60)
    print("PHAN 3: Learning Rate Schedule")
    print("=" * 60)

    # Visualize LR schedule
    max_steps_demo = 1000
    warmup_demo = 100
    lr_history = [get_lr(s, warmup_demo, max_steps_demo, 3e-3, 3e-4) for s in range(max_steps_demo)]
    print(f"  Step 0:    lr={lr_history[0]:.6f} (start = 0)")
    print(f"  Step 50:   lr={lr_history[50]:.6f} (warmup)")
    print(f"  Step 100:  lr={lr_history[100]:.6f} (peak = max_lr)")
    print(f"  Step 500:  lr={lr_history[500]:.6f} (decaying)")
    print(f"  Step 999:  lr={lr_history[999]:.6f} (near min_lr)")
    assert lr_history[0] == 0.0, "LR should start at 0"
    assert abs(lr_history[100] - 3e-3) < 1e-6, "LR should peak at max_lr"
    assert lr_history[999] > lr_history[0], "LR at end > start"
    print("  LR schedule shape: OK")

    print("\n" + "=" * 60)
    print("PHAN 4: TextDataset")
    print("=" * 60)

    tokens = list(range(100))  # 0,1,2,...,99
    dataset = TextDataset(tokens, block_size=10)
    print(f"  Total tokens: {len(tokens)}")
    print(f"  Block size: 10")
    print(f"  Num samples: {len(dataset)}")

    x, y = dataset.get_batch(4)
    print(f"  Batch x shape: {x.shape}")
    print(f"  Batch y shape: {y.shape}")
    # Verify y = x shifted by 1
    print(f"  Sample x: {x[0][:5]}...")
    print(f"  Sample y: {y[0][:5]}...")
    assert np.all(y[0] == x[0] + 1), "Targets should be inputs shifted by 1"
    print("  Input-target alignment: OK")

    # ---- Train GPT with PyTorch ----
    print("\n" + "=" * 60)
    print("PHAN 5: Train GPT tren Shakespeare (PyTorch)")
    print("=" * 60)

    try:
        import torch
        model, loss_history, vocab_size, encode, decode = train_with_pytorch()

        # Verify training reduced loss
        if len(loss_history['train']) >= 2:
            initial_loss = loss_history['train'][0]
            final_loss = loss_history['train'][-1]
            print(f"\n  Initial train loss: {initial_loss:.4f}")
            print(f"  Final train loss:   {final_loss:.4f}")
            assert final_loss < initial_loss, "Training should reduce loss"
            print("  Loss decreased: OK")

        # Bai tap 5: Plot training loss
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(1, 3, figsize=(15, 4))

            # Plot 1: Loss curve
            ax = axes[0]
            ax.plot(loss_history['steps'], loss_history['train'], label='Train', color='blue')
            ax.plot(loss_history['steps'], loss_history['val'], label='Val', color='orange')
            ax.set_xlabel('Step')
            ax.set_ylabel('Loss')
            ax.set_title('Training & Validation Loss')
            ax.legend()
            ax.grid(True, alpha=0.3)

            # Plot 2: Learning rate
            ax = axes[1]
            ax.plot(loss_history['steps'], loss_history['lr'], color='green')
            ax.set_xlabel('Step')
            ax.set_ylabel('Learning Rate')
            ax.set_title('LR Schedule (warmup + cosine decay)')
            ax.grid(True, alpha=0.3)

            # Plot 3: Gradient norm
            ax = axes[2]
            ax.plot(loss_history['steps'], loss_history['grad_norm'], color='red')
            ax.axhline(y=1.0, color='gray', linestyle='--', label='Clip threshold')
            ax.set_xlabel('Step')
            ax.set_ylabel('Gradient Norm')
            ax.set_title('Gradient Norm (clipped at 1.0)')
            ax.legend()
            ax.grid(True, alpha=0.3)

            plt.tight_layout()
            path = os.path.join(output_dir, "plot_training.png")
            plt.savefig(path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"  Saved: {path}")

        except ImportError:
            print("  matplotlib chua cai.")

    except ImportError:
        print("  torch chua cai. Chay: pip install torch")

    print("\n" + "=" * 60)
    print("TAT CA TESTS PASSED!")
    print("=" * 60)


# ============ CHECKLIST ============
# Week 8-9 (Bai 08):
# [x] Implement training loop
#     -> train_gpt(): full loop voi cross-entropy loss, AdamW, gradient clipping
#        eval_interval de theo doi validation loss
# [x] Train tren Shakespeare
#     -> ShakespeareDataModule: download + tokenize Shakespeare text
#        TextDataset: cat text thanh cac block co dinh (block_size)
# [x] Plot loss curves
#     -> Ve train_loss va val_loss theo epochs
# [x] Debug training issues
#     -> Gradient clipping (chong exploding), lr scheduling
#        Weight decay (regularization), dropout
