# File: 11_lora.py
# LoRA - Low-Rank Adaptation - Week 12
#
# VAN DE: Fine-tuning LLM qua ton kem
# - LLaMA 7B: 7 billion params, FP32 = 28 GB
# - Fine-tune can: weights (28GB) + gradients (28GB) + optimizer states (28GB) = 84GB+
# - Can GPU A100 80GB ($10k+) chi de fine-tune!
#
# GIAI PHAP: LoRA (Low-Rank Adaptation)
# - Dong bang (freeze) toan bo pre-trained weights W
# - Them 2 ma tran NHO: A (d_in x r) va B (r x d_out)
# - output = x @ W + alpha/r * x @ A @ B
# - Chi train A va B -> tiet kiem 99%+ memory
#
# TAI SAO HOAT DONG?
# - Nghien cuu cho thay: weight updates (delta_W) co "low intrinsic rank"
# - Nghia la: delta_W (d_in x d_out) co the xap xi boi A @ B voi r << d
# - Vi du: W la 4096x4096 (16M params)
#   - LoRA rank 8: A (4096x8) + B (8x4096) = 65K params (0.4%!)
#   - Ma accuracy gan bang full fine-tuning
#
# BAI TAP:
# 1. Apply LoRA vao GPT model
# 2. Fine-tune tren specific task
# 3. Compare voi full fine-tuning
# 4. Implement QLoRA (quantized base + LoRA)
# 5. Merge va save fine-tuned model

import numpy as np
import time
import os


class LoRALayer:
    """
    LoRA Layer - thay the 1 linear layer

    CACH HOAT DONG:
    Original:  output = x @ W                    (W la frozen)
    LoRA:      output = x @ W + scaling * x @ A @ B

    KHOI TAO:
    - A: random Gaussian (de co gradient ngay tu dau)
    - B: zeros (de ban dau LoRA KHONG thay doi output)
      -> Bat dau training tu trang thai giong pre-trained model

    SCALING:
    - scaling = alpha / r
    - alpha: hyperparameter (thuong = 2*r hoac 16)
    - r: rank (4, 8, 16, 32)
    - Scaling giup kiem soat "muc anh huong" cua LoRA
    """

    def __init__(self, W_pretrained, r=8, alpha=16, dropout=0.0):
        """
        W_pretrained: ma tran weights da train san, shape (d_in, d_out)
                      Day la weights tu pre-trained model (GPT, LLaMA, ...)
                      Se bi DONG BANG (freeze) - khong bao gio update trong qua trinh train
                      Vd: LLaMA 7B co W_q shape (4096, 4096) = 16M params
                          GPT-2 small co W shape (768, 768) = 590K params

        r:            rank cua LoRA decomposition (so chieu bottleneck)
                      A co shape (d_in, r), B co shape (r, d_out)
                      r cang nho -> cang it params nhung mat bieu dien
                      Gia tri thuong dung: 4, 8, 16, 32
                      Vd: LLaMA 7B voi r=8: chi them 0.1% params
                          GPT-3 paper dung r=4 cho nhieu tasks
                          r=64 cho task phuc tap can nhieu capacity

        alpha:        scaling hyperparameter, dieu chinh "muc anh huong" cua LoRA
                      scaling = alpha / r
                      Gia tri thuong dung: alpha = 2*r (Vd: r=8 -> alpha=16)
                      alpha lon -> LoRA anh huong manh hon
                      alpha nho -> LoRA anh huong nhe (an toan hon, hoi tu cham hon)
                      Trong thuc te: alpha=16 hoac alpha=32 la pho bien nhat

        dropout:      ty le dropout tren LoRA hidden (regularization)
                      0.0 = khong dropout (default, phu hop khi data du lon)
                      0.05 - 0.1 = dropout nhe, dung khi fine-tune tren data nho
                      Ap dung len output cua x @ A truoc khi nhan voi B
        """
        self.W = W_pretrained.copy()  # FROZEN - khong update!
        d_in, d_out = W_pretrained.shape

        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r
        self.dropout = dropout

        # LoRA matrices - CHI train 2 cai nay
        self.A = np.random.randn(d_in, r) * 0.01  # Small random init
        self.B = np.zeros((r, d_out))               # Zero init

        # Gradients
        self.grad_A = None
        self.grad_B = None

        # Cache for backward
        self.input_cache = None

    def forward(self, x, training=True):
        """
        Forward pass:
        output = x @ W + scaling * x @ A @ B

        x:        input tensor, shape (batch, seq_len, d_in) hoac (batch, d_in)
                  batch = so samples xu ly cung luc (Vd: 8, 16, 32)
                  seq_len = so tokens trong 1 cau (Vd: GPT-2 max 1024, LLaMA max 2048)
                  d_in = kich thuoc embedding (Vd: GPT-2 small=768, LLaMA 7B=4096)
                  Khi dung cho classification (khong co seq_len): shape (batch, d_in)

        training: True = dang train -> ap dung dropout (neu co)
                  False = dang inference/eval -> tat dropout
                  Khi fine-tune: training=True cho train set, False cho validation
        """
        self.input_cache = x

        # Original output (FROZEN - khong tinh gradient cho W)
        original_output = x @ self.W

        # LoRA path
        lora_hidden = x @ self.A  # (batch, ..., r) - bottleneck

        # Dropout on LoRA hidden (regularization)
        if training and self.dropout > 0:
            mask = (np.random.rand(*lora_hidden.shape) > self.dropout).astype(float)
            lora_hidden = lora_hidden * mask / (1 - self.dropout)

        lora_output = lora_hidden @ self.B  # (batch, ..., d_out)

        return original_output + self.scaling * lora_output

    def backward(self, grad_output):
        """
        Backward pass - CHI compute gradient cho A va B
        W is FROZEN, khong bao gio update

        grad_output: gradient tu layer phia sau, shape giong output cua forward
                     (batch, seq_len, d_out) hoac (batch, d_out)
                     Day la dL/d(output), duoc truyen nguoc tu loss function
                     Vd: neu output shape (32, 128, 4096) -> grad_output cung (32, 128, 4096)
                     Gradient nay duoc dung de tinh grad_A va grad_B qua chain rule

        TAI SAO CHI CAN GRADIENT CHO A VA B?
        - W da duoc train tren data khong lo (GPT-3: 300B tokens)
        - W da "biet" ngon ngu roi
        - A @ B chi can hoc "thay doi nho" cho task cu the
        - Giong nhu: W la nen tang (foundation), A@B la tuy chinh (customization)
        """
        x = self.input_cache
        grad_lora = grad_output * self.scaling

        lora_hidden = x @ self.A  # (batch, ..., r)

        # Flatten cho batch matmul
        orig_shape = x.shape
        if len(x.shape) == 3:
            batch_size = x.shape[0]
            x_flat = x.reshape(-1, x.shape[-1])
            lora_hidden_flat = lora_hidden.reshape(-1, self.r)
            grad_lora_flat = grad_lora.reshape(-1, self.B.shape[1])

            # grad_B = lora_hidden^T @ grad_lora
            self.grad_B = lora_hidden_flat.T @ grad_lora_flat / batch_size

            # grad_A = x^T @ (grad_lora @ B^T)
            grad_hidden = grad_lora_flat @ self.B.T
            self.grad_A = x_flat.T @ grad_hidden / batch_size
        else:
            self.grad_B = lora_hidden.T @ grad_lora
            grad_hidden = grad_lora @ self.B.T
            self.grad_A = x.T @ grad_hidden

        return grad_output  # Pass gradient unchanged (W is frozen)

    def update(self, lr):
        """
        Update CHI A va B - W van dong bang!

        lr: learning rate, buoc nhay cua gradient descent
            Gia tri thuong dung cho LoRA: 1e-4 den 1e-2
            Vd: LLaMA fine-tune thuong dung lr=2e-4
                GPT-2 LoRA fine-tune dung lr=1e-3 den 5e-3
            lr qua lon -> training khong on dinh, loss nhay
            lr qua nho -> hoi tu rat cham
        """
        self.A -= lr * self.grad_A
        self.B -= lr * self.grad_B

    def merge_weights(self):
        """
        Merge LoRA vao base weights

        TAI SAO MERGE?
        - Khi deploy, khong can tinh x@A@B rieng
        - Merge: W_new = W + scaling * A @ B
        - Inference nhanh nhu model goc (khong them latency)
        - Trade-off: khong the switch LoRA adapters nua

        DUNG KHI:
        - Da fine-tune xong, chi can inference
        - Muon toi uu toc do
        """
        self.W = self.W + self.scaling * (self.A @ self.B)
        # Sau khi merge, co the xoa A, B
        return self.W

    def unmerge_weights(self):
        """Undo merge - quay lai trang thai LoRA"""
        self.W = self.W - self.scaling * (self.A @ self.B)

    def get_num_trainable_params(self):
        return self.A.size + self.B.size

    def get_num_frozen_params(self):
        return self.W.size


# ============================================================
# BAI TAP 1: APPLY LoRA VAO GPT
# ============================================================

class LoRAAttention:
    """
    Attention layer voi LoRA tren W_q va W_v

    TAI SAO CHI APPLY LoRA LEN Q VA V?
    - Nghien cuu (Hu et al. 2021) cho thay:
      * LoRA tren W_q va W_v cho ket qua tot nhat
      * W_k va W_o it anh huong hon
    - Giam so LoRA params them 50%
    - Trong thuc te, nhieu nguoi apply len ca 4 (Q,K,V,O) neu du VRAM
    """

    def __init__(self, d_model, num_heads, W_q, W_k, W_v, W_o, lora_r=8, lora_alpha=16):
        """
        d_model:    kich thuoc embedding cua moi token (dimension cua model)
                    GPT-2 small=768, GPT-2 medium=1024, GPT-3=12288
                    LLaMA 7B=4096, LLaMA 13B=5120, LLaMA 65B=8192

        num_heads:  so attention heads chay song song
                    GPT-2 small=12, GPT-3=96, LLaMA 7B=32, LLaMA 65B=64
                    d_model PHAI chia het cho num_heads
                    d_k = d_model / num_heads (Vd: 4096/32=128 cho LLaMA 7B)

        W_q:        pre-trained Query projection, shape (d_model, d_model)
                    Se duoc wrap boi LoRALayer -> co them A_q, B_q trainable
                    Nghien cuu (Hu et al. 2021): LoRA tren W_q cho ket qua tot

        W_k:        pre-trained Key projection, shape (d_model, d_model)
                    KHONG ap dung LoRA -> hoan toan frozen
                    Giam 50% LoRA params so voi apply len ca 4 projections

        W_v:        pre-trained Value projection, shape (d_model, d_model)
                    Se duoc wrap boi LoRALayer -> co them A_v, B_v trainable
                    Nghien cuu cho thay LoRA tren W_v la quan trong nhat

        W_o:        pre-trained Output projection, shape (d_model, d_model)
                    KHONG ap dung LoRA -> hoan toan frozen
                    Gop output cua tat ca heads va project ve d_model

        lora_r:     rank cho LoRA adapters tren W_q va W_v
                    Gia tri thuong dung: 4, 8, 16
                    Vd: LLaMA 7B voi lora_r=8 -> moi adapter chi them ~65K params

        lora_alpha: scaling factor cho LoRA, thuong = 2*lora_r hoac 16
                    Vd: lora_r=8, lora_alpha=16 -> scaling = 2.0
        """
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # LoRA cho Q va V
        self.lora_q = LoRALayer(W_q, r=lora_r, alpha=lora_alpha)
        self.lora_v = LoRALayer(W_v, r=lora_r, alpha=lora_alpha)

        # K va O giu nguyen (frozen, khong co LoRA)
        self.W_k = W_k
        self.W_o = W_o

    def forward(self, x, training=True):
        """
        x:        input tensor, shape (batch_size, seq_len, d_model)
                  batch_size = so cau xu ly cung luc (Vd: 1-64 tuy GPU memory)
                  seq_len = so tokens trong cau (Vd: GPT-2 max 1024, LLaMA max 2048-4096)
                  d_model = kich thuoc embedding (Vd: GPT-2=768, LLaMA 7B=4096)

        training: True = dang train -> LoRA layers ap dung dropout
                  False = inference/eval -> tat dropout
                  Khi fine-tune: training=True cho forward pass, False cho validation/test
        """
        batch_size, seq_len, _ = x.shape

        Q = self.lora_q.forward(x, training)  # LoRA path
        K = x @ self.W_k                       # Frozen
        V = self.lora_v.forward(x, training)  # LoRA path
        O_proj = self.W_o                      # Frozen

        # Multi-head reshape
        Q = Q.reshape(batch_size, seq_len, self.num_heads, self.d_k).transpose(0, 2, 1, 3)
        K = K.reshape(batch_size, seq_len, self.num_heads, self.d_k).transpose(0, 2, 1, 3)
        V = V.reshape(batch_size, seq_len, self.num_heads, self.d_k).transpose(0, 2, 1, 3)

        # Attention
        scores = Q @ K.transpose(0, 1, 3, 2) / np.sqrt(self.d_k)

        # Causal mask
        mask = np.triu(np.ones((seq_len, seq_len)), k=1) * (-1e9)
        scores = scores + mask

        exp_scores = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
        attn_weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)

        context = attn_weights @ V
        context = context.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, self.d_model)

        return context @ O_proj

    def get_trainable_params(self):
        return self.lora_q.get_num_trainable_params() + self.lora_v.get_num_trainable_params()

    def get_frozen_params(self):
        return (self.lora_q.get_num_frozen_params() + self.lora_v.get_num_frozen_params()
                + self.W_k.size + self.W_o.size)


# ============================================================
# BAI TAP 4: QLoRA
# ============================================================

class QLoRALayer:
    """
    QLoRA = Quantized base weights + LoRA adapters

    TAI SAO QLORA?
    - LoRA: freeze W (FP32) + train A, B
    - QLoRA: freeze W (INT4!) + train A, B (FP16)
    - W INT4 nho hon 8x -> tiet kiem VRAM khong lo
    - A, B van FP16 de hoc tot

    VI DU:
    - LLaMA 65B:
      * Full fine-tune: 780 GB (khong kha thi)
      * LoRA FP32: 260 GB (van qua lon)
      * QLoRA INT4: 33 GB (chay duoc tren 1 GPU A100!)

    CACH HOAT DONG:
    1. Quantize W tu FP32 -> INT4 (tiet kiem memory)
    2. Khi forward: dequantize W -> FP32 -> tinh x @ W
    3. Cong them LoRA: + scaling * x @ A @ B
    4. Backward: chi tinh gradient cho A, B (W dong bang + quantized)
    """

    def __init__(self, W_pretrained, r=8, alpha=16):
        """
        W_pretrained: ma tran weights da train san, shape (d_in, d_out)
                      Se bi QUANTIZE tu FP32 -> INT4 (tiet kiem 8x memory!)
                      Vd: LLaMA 65B FP32 = 260 GB -> INT4 = ~33 GB
                          GPT-2 small (768, 768) FP32 = 2.4 MB -> INT4 = ~300 KB
                      Sau khi quantize, weights bi dong bang (khong update)

        r:            rank cua LoRA decomposition (giong LoRALayer)
                      A shape (d_in, r), B shape (r, d_out) - giu FP32 de train tot
                      Gia tri thuong dung: 4, 8, 16, 32
                      Vd: QLoRA paper dung r=64 cho LLaMA, r=8-16 cho tasks nho

        alpha:        scaling factor, scaling = alpha / r
                      Thuong dung alpha = 2*r hoac 16
                      Vd: r=8, alpha=16 -> scaling = 2.0
                      QLoRA paper dung alpha=16 cho hau het experiments
        """
        d_in, d_out = W_pretrained.shape

        # Quantize base weights to INT4
        abs_max = np.max(np.abs(W_pretrained))
        self.scale = abs_max / 7.0 if abs_max != 0 else 1.0
        self.W_quantized = np.round(W_pretrained / self.scale).astype(np.int8)
        self.W_quantized = np.clip(self.W_quantized, -8, 7)

        # LoRA adapters (keep FP32 for training)
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r
        self.A = np.random.randn(d_in, r).astype(np.float32) * 0.01
        self.B = np.zeros((r, d_out), dtype=np.float32)

        self.grad_A = None
        self.grad_B = None
        self.input_cache = None

    def forward(self, x, training=True):
        """
        x:        input tensor, shape (batch, seq_len, d_in) hoac (batch, d_in)
                  Vd: (32, 128, 4096) cho LLaMA, (16, 768) cho classification task
                  Input se duoc nhan voi W_dequantized (INT4 -> FP32) + LoRA output

        training: True = dang train (hien tai QLoRA khong co dropout rieng)
                  False = inference
                  Giu tham so nay de thong nhat interface voi LoRALayer
        """
        self.input_cache = x

        # Dequantize on-the-fly (khong luu FP32 weights!)
        W_dequant = self.W_quantized.astype(np.float32) * self.scale

        # Original + LoRA
        output = x @ W_dequant + self.scaling * (x @ self.A) @ self.B
        return output

    def backward(self, grad_output):
        """
        grad_output: gradient tu layer phia sau, shape giong output cua forward
                     (batch, seq_len, d_out) hoac (batch, d_out)
                     Day la dL/d(output), duoc truyen nguoc tu loss function
                     CHI tinh gradient cho A va B (W_quantized dong bang)
                     Gradient duoc tinh o FP32 du base weights la INT4
        """
        x = self.input_cache
        grad_lora = grad_output * self.scaling

        if len(x.shape) == 3:
            batch_size = x.shape[0]
            x_flat = x.reshape(-1, x.shape[-1])
            lora_hidden_flat = (x @ self.A).reshape(-1, self.r)
            grad_flat = grad_lora.reshape(-1, self.B.shape[1])

            self.grad_B = lora_hidden_flat.T @ grad_flat / batch_size
            self.grad_A = x_flat.T @ (grad_flat @ self.B.T) / batch_size
        else:
            lora_hidden = x @ self.A
            self.grad_B = lora_hidden.T @ grad_lora
            self.grad_A = x.T @ (grad_lora @ self.B.T)

        return grad_output

    def update(self, lr):
        """
        lr: learning rate, buoc nhay cua gradient descent
            Gia tri thuong dung cho QLoRA: 1e-4 den 2e-4
            Vd: QLoRA paper dung lr=2e-4 cho LLaMA 7B-65B
                Nho hon LoRA thuong vi quantization tao noise -> can lr nho hon
        """
        self.A -= lr * self.grad_A
        self.B -= lr * self.grad_B

    def memory_usage(self):
        """So sanh memory"""
        d_in, r = self.A.shape
        _, d_out = self.B.shape
        base_int4 = self.W_quantized.nbytes  # INT8 stored, but represents INT4
        lora_fp32 = self.A.nbytes + self.B.nbytes
        original_fp32 = d_in * d_out * 4  # FP32
        return {
            'qlora_bytes': base_int4 + lora_fp32,
            'original_bytes': original_fp32,
            'savings': 1 - (base_int4 + lora_fp32) / original_fp32,
        }


# ============================================================
# BAI TAP 2 & 3: FINE-TUNE VA COMPARE
# ============================================================

def create_classification_task(n_samples=200, d_in=64, n_classes=3):
    """
    Tao dataset gia cho classification task

    n_samples: so luong data points trong dataset
               200 = dataset nho cho demo/test nhanh
               Trong thuc te: fine-tune LLM thuong can 1K-100K samples
               Vd: sentiment analysis dataset co ~50K samples
                   email classification co ~10K-20K samples

    d_in:      so chieu cua input features (kich thuoc embedding)
               64 = nho cho demo, trong thuc te = d_model cua pre-trained model
               Vd: GPT-2 small d_in=768, LLaMA 7B d_in=4096

    n_classes: so lop phan loai (so labels khac nhau)
               3 = demo 3-class classification
               Vd: sentiment analysis co 2 (pos/neg) hoac 3 (pos/neg/neutral)
                   email classification co 5-10 categories

    TAI SAO DUNG TASK NAY?
    - De demo fine-tuning tren task cu the
    - Pre-trained model biet "ngon ngu chung"
    - Fine-tune de model biet "phan loai email" hoac "sentiment analysis"
    """
    X = np.random.randn(n_samples, d_in).astype(np.float32)
    # Tao labels dua tren pattern don gian
    scores = X @ np.random.randn(d_in, n_classes).astype(np.float32)
    y = np.argmax(scores, axis=1)
    return X, y


def softmax_np(x, axis=-1):
    exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def train_lora_classifier(W_pretrained, X_train, y_train, X_val, y_val,
                           r=8, lr=0.01, epochs=50):
    """
    Train LoRA cho classification task

    W_pretrained: ma tran weights da train san, shape (d_in, d_out)
                  Mo phong pre-trained model (GPT, LLaMA, ...)
                  Se bi freeze trong LoRALayer, chi train A va B
                  Vd: shape (64, 64) cho demo, (4096, 4096) cho LLaMA 7B

    X_train:      training data, shape (n_train, d_in)
                  Moi row la 1 sample (embedding cua 1 input)
                  Vd: shape (160, 64) = 160 training samples, 64 features

    y_train:      training labels, shape (n_train,)
                  Moi phan tu la class index (0, 1, ..., n_classes-1)
                  Vd: [0, 2, 1, 0, ...] cho 3-class classification

    X_val:        validation data, shape (n_val, d_in)
                  Dung de danh gia model sau moi epoch (khong train tren nay)
                  Vd: shape (40, 64) = 40 validation samples

    y_val:        validation labels, shape (n_val,)
                  Ground truth cua validation set, dung de tinh val_acc

    r:            LoRA rank (truyen vao LoRALayer)
                  Gia tri thuong dung: 4, 8, 16
                  r lon -> nhieu capacity nhung cham hon va ton memory hon
                  Vd: r=8 la balance tot giua accuracy va efficiency

    lr:           learning rate cho gradient descent
                  0.01 - 0.05 cho demo voi dataset nho
                  Trong thuc te LoRA fine-tune: 1e-4 den 5e-4
                  Vd: Alpaca fine-tune LLaMA dung lr=2e-5

    epochs:       so vong lap train toan bo dataset
                  50-100 cho demo, trong thuc te: 3-10 epochs cho LLM fine-tune
                  Qua nhieu epochs -> overfitting (train_acc cao, val_acc giam)
    """
    d_in = W_pretrained.shape[0]
    n_classes = len(np.unique(y_train))

    # LoRA layer
    lora = LoRALayer(W_pretrained, r=r, alpha=2*r)

    # Classification head (trainable)
    W_head = np.random.randn(W_pretrained.shape[1], n_classes).astype(np.float32) * 0.01
    b_head = np.zeros(n_classes, dtype=np.float32)

    history = {'train_loss': [], 'train_acc': [], 'val_acc': []}

    for epoch in range(epochs):
        # Forward
        hidden = lora.forward(X_train, training=True)
        logits = hidden @ W_head + b_head
        probs = softmax_np(logits)

        # Loss
        n = len(y_train)
        loss = -np.mean(np.log(probs[np.arange(n), y_train] + 1e-9))

        # Backward (simplified - gradient through softmax + linear)
        grad_logits = probs.copy()
        grad_logits[np.arange(n), y_train] -= 1
        grad_logits /= n

        # Update head
        grad_W_head = hidden.T @ grad_logits
        grad_b_head = np.sum(grad_logits, axis=0)
        W_head -= lr * grad_W_head
        b_head -= lr * grad_b_head

        # Update LoRA
        grad_hidden = grad_logits @ W_head.T
        lora.backward(grad_hidden)
        lora.update(lr)

        # Metrics
        train_acc = np.mean(np.argmax(logits, axis=1) == y_train)

        # Validation
        hidden_val = lora.forward(X_val, training=False)
        logits_val = hidden_val @ W_head + b_head
        val_acc = np.mean(np.argmax(logits_val, axis=1) == y_val)

        history['train_loss'].append(loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)

        if epoch % 10 == 0:
            print(f"    Epoch {epoch:>3}: loss={loss:.4f}, train_acc={train_acc:.3f}, val_acc={val_acc:.3f}")

    return lora, history


def train_full_finetune(W_pretrained, X_train, y_train, X_val, y_val,
                         lr=0.01, epochs=50):
    """
    Full fine-tuning (update ALL weights) de so sanh

    W_pretrained: ma tran weights da train san, shape (d_in, d_out)
                  KHAC voi LoRA: o day TOAN BO W duoc update (khong freeze)
                  Can nhieu memory hon: weights + gradients + optimizer states
                  Vd: LLaMA 7B full fine-tune can ~84 GB VRAM (FP32)
                      GPT-2 small full fine-tune can ~1.5 GB

    X_train:      training data, shape (n_train, d_in)
                  Giong voi train_lora_classifier de so sanh cong bang

    y_train:      training labels, shape (n_train,)
                  Ground truth cua training set

    X_val:        validation data, shape (n_val, d_in)
                  Dung de danh gia overfitting va so sanh voi LoRA

    y_val:        validation labels, shape (n_val,)
                  Ground truth cua validation set

    lr:           learning rate cho gradient descent
                  Thuong dung giong LoRA de so sanh cong bang
                  Trong thuc te full fine-tune thuong dung lr nho hon LoRA
                  Vd: GPT-3 fine-tune dung lr=4e-5, LLaMA dung lr=2e-5

    epochs:       so vong lap train (giong LoRA de so sanh cong bang)
                  Full fine-tune thuong can it epochs hon LoRA vi co nhieu params
                  Nhung de overfitting hon tren dataset nho
    """
    d_in, d_out = W_pretrained.shape
    n_classes = len(np.unique(y_train))

    W = W_pretrained.copy()  # ALL weights trainable!
    W_head = np.random.randn(d_out, n_classes).astype(np.float32) * 0.01
    b_head = np.zeros(n_classes, dtype=np.float32)

    history = {'train_loss': [], 'train_acc': [], 'val_acc': []}

    for epoch in range(epochs):
        # Forward
        hidden = X_train @ W
        logits = hidden @ W_head + b_head
        probs = softmax_np(logits)

        # Loss
        n = len(y_train)
        loss = -np.mean(np.log(probs[np.arange(n), y_train] + 1e-9))

        # Backward
        grad_logits = probs.copy()
        grad_logits[np.arange(n), y_train] -= 1
        grad_logits /= n

        grad_W_head = hidden.T @ grad_logits
        grad_b_head = np.sum(grad_logits, axis=0)
        grad_hidden = grad_logits @ W_head.T
        grad_W = X_train.T @ grad_hidden

        # Update ALL weights
        W -= lr * grad_W
        W_head -= lr * grad_W_head
        b_head -= lr * grad_b_head

        train_acc = np.mean(np.argmax(logits, axis=1) == y_train)
        hidden_val = X_val @ W
        logits_val = hidden_val @ W_head + b_head
        val_acc = np.mean(np.argmax(logits_val, axis=1) == y_val)

        history['train_loss'].append(loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)

    return history


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    np.random.seed(42)
    output_dir = os.path.dirname(os.path.abspath(__file__))

    # ---- Phan 1: LoRA Layer co ban ----
    print("=" * 60)
    print("PHAN 1: LoRA Layer Analysis")
    print("=" * 60)

    d_in, d_out = 4096, 4096
    W = np.random.randn(d_in, d_out).astype(np.float32) * 0.02

    print(f"  Base weight: {W.shape} = {W.size:,} params")

    for r in [4, 8, 16, 32]:
        lora = LoRALayer(W, r=r)
        trainable = lora.get_num_trainable_params()
        frozen = lora.get_num_frozen_params()
        ratio = trainable / frozen * 100
        print(f"  Rank {r:>3}: trainable={trainable:>10,} ({ratio:>5.2f}%), frozen={frozen:,}")

    # ---- Test forward/backward ----
    print("\n  --- Forward/Backward Test ---")
    lora = LoRALayer(W, r=8)
    x = np.random.randn(2, 10, d_in).astype(np.float32)

    output = lora.forward(x)
    print(f"  Input:  {x.shape}")
    print(f"  Output: {output.shape}")
    assert output.shape == (2, 10, d_out)

    # Ban dau, LoRA output = original (vi B = zeros)
    original_output = x @ W
    assert np.allclose(output, original_output, atol=1e-5), "Initial LoRA should match original"
    print("  Initial output matches original (B=0): OK")

    # Backward
    grad = np.random.randn(*output.shape).astype(np.float32)
    lora.backward(grad)
    assert lora.grad_A.shape == (d_in, 8)
    assert lora.grad_B.shape == (8, d_out)
    print(f"  grad_A: {lora.grad_A.shape}")
    print(f"  grad_B: {lora.grad_B.shape}")

    # Update
    lora.update(lr=0.01)
    output_after = lora.forward(x)
    assert not np.allclose(output, output_after), "Output should change after update"
    print("  Output changes after update: OK")

    # ---- Bai tap 5: Merge weights ----
    print("\n  --- Merge Test ---")
    x_test = np.random.randn(1, 5, d_in).astype(np.float32)
    out_before_merge = lora.forward(x_test, training=False)
    W_merged = lora.merge_weights()
    out_after_merge = x_test @ W_merged
    assert np.allclose(out_before_merge, out_after_merge, atol=1e-5), "Merged output should match"
    print("  Merged output matches LoRA output: OK")
    print(f"  Merged weight shape: {W_merged.shape}")

    # ---- Bai tap 1: LoRA Attention ----
    print("\n" + "=" * 60)
    print("BAI TAP 1: LoRA Attention")
    print("=" * 60)

    d_model = 256
    num_heads = 4
    scale = np.sqrt(2.0 / d_model)
    W_q = np.random.randn(d_model, d_model).astype(np.float32) * scale
    W_k = np.random.randn(d_model, d_model).astype(np.float32) * scale
    W_v = np.random.randn(d_model, d_model).astype(np.float32) * scale
    W_o = np.random.randn(d_model, d_model).astype(np.float32) * scale

    lora_attn = LoRAAttention(d_model, num_heads, W_q, W_k, W_v, W_o, lora_r=8)

    x_attn = np.random.randn(2, 10, d_model).astype(np.float32)
    out_attn = lora_attn.forward(x_attn)
    print(f"  Input:  {x_attn.shape}")
    print(f"  Output: {out_attn.shape}")
    assert out_attn.shape == x_attn.shape

    trainable = lora_attn.get_trainable_params()
    frozen = lora_attn.get_frozen_params()
    print(f"  Trainable params: {trainable:,} ({trainable/(trainable+frozen)*100:.2f}%)")
    print(f"  Frozen params:    {frozen:,}")
    print("  LoRA Attention: OK")

    # ---- Bai tap 2 & 3: Fine-tune va Compare ----
    print("\n" + "=" * 60)
    print("BAI TAP 2-3: Fine-tune & Compare voi Full Fine-tuning")
    print("=" * 60)

    d_task = 64
    n_classes = 3
    W_pretrained = np.random.randn(d_task, d_task).astype(np.float32) * 0.1

    X, y = create_classification_task(n_samples=300, d_in=d_task, n_classes=n_classes)
    # Split 80/20
    split = int(0.8 * len(X))
    X_train, y_train = X[:split], y[:split]
    X_val, y_val = X[split:], y[split:]

    print(f"  Task: {n_classes}-class classification")
    print(f"  Train: {len(X_train)}, Val: {len(X_val)}")
    print(f"  Pre-trained W: {W_pretrained.shape}")

    print(f"\n  --- LoRA Fine-tuning (r=8) ---")
    lora_model, lora_hist = train_lora_classifier(
        W_pretrained, X_train, y_train, X_val, y_val,
        r=8, lr=0.05, epochs=100
    )
    lora_final_acc = lora_hist['val_acc'][-1]
    lora_trainable = lora_model.get_num_trainable_params()

    print(f"\n  --- Full Fine-tuning ---")
    full_hist = train_full_finetune(
        W_pretrained, X_train, y_train, X_val, y_val,
        lr=0.05, epochs=100
    )
    full_final_acc = full_hist['val_acc'][-1]
    full_trainable = W_pretrained.size  # All params trainable

    print(f"\n  --- Comparison ---")
    print(f"  {'Method':>20} | {'Val Acc':>8} | {'Trainable Params':>18}")
    print(f"  {'-'*20}-+-{'-'*8}-+-{'-'*18}")
    print(f"  {'LoRA (r=8)':>20} | {lora_final_acc:>8.3f} | {lora_trainable:>18,}")
    print(f"  {'Full Fine-tune':>20} | {full_final_acc:>8.3f} | {full_trainable:>18,}")
    print(f"  {'Param ratio':>20} | {'':>8} | {lora_trainable/full_trainable*100:>17.1f}%")

    # LoRA should achieve reasonable accuracy
    assert lora_final_acc > 0.5, "LoRA should learn something"
    print("  LoRA learns task: OK")

    # ---- Bai tap 4: QLoRA ----
    print("\n" + "=" * 60)
    print("BAI TAP 4: QLoRA (Quantized base + LoRA)")
    print("=" * 60)

    W_large = np.random.randn(512, 512).astype(np.float32) * 0.02
    qlora = QLoRALayer(W_large, r=8)

    x_ql = np.random.randn(4, 512).astype(np.float32)
    out_ql = qlora.forward(x_ql)
    print(f"  Input:  {x_ql.shape}")
    print(f"  Output: {out_ql.shape}")
    assert out_ql.shape == x_ql.shape

    mem = qlora.memory_usage()
    print(f"\n  Memory comparison:")
    print(f"    Original FP32:  {mem['original_bytes'] / 1024:.1f} KB")
    print(f"    QLoRA (INT4+FP32 adapters): {mem['qlora_bytes'] / 1024:.1f} KB")
    print(f"    Savings: {mem['savings']*100:.1f}%")

    # Backward test
    grad_ql = np.random.randn(*out_ql.shape).astype(np.float32)
    qlora.backward(grad_ql)
    qlora.update(lr=0.01)
    out_ql_after = qlora.forward(x_ql)
    assert not np.allclose(out_ql, out_ql_after), "QLoRA output should change after update"
    print("  QLoRA backward + update: OK")

    # ---- Test different ranks ----
    print("\n" + "=" * 60)
    print("RANK ANALYSIS: Effect of LoRA rank")
    print("=" * 60)

    print(f"\n  {'Rank':>6} | {'Params':>10} | {'Val Acc':>8} | {'% of Full':>10}")
    print(f"  {'-'*6}-+-{'-'*10}-+-{'-'*8}-+-{'-'*10}")

    for r in [2, 4, 8, 16]:
        np.random.seed(42)  # Same init
        _, hist_r = train_lora_classifier(
            W_pretrained, X_train, y_train, X_val, y_val,
            r=r, lr=0.05, epochs=100
        )
        acc_r = hist_r['val_acc'][-1]
        params_r = 2 * d_task * r  # A + B
        print(f"  {r:>6} | {params_r:>10,} | {acc_r:>8.3f} | {params_r/W_pretrained.size*100:>9.1f}%")

    # ---- Plot ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        # Plot 1: Training loss
        ax = axes[0]
        ax.plot(lora_hist['train_loss'], label='LoRA (r=8)', color='blue')
        ax.plot(full_hist['train_loss'], label='Full Fine-tune', color='red', linestyle='--')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Training Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 2: Validation accuracy
        ax = axes[1]
        ax.plot(lora_hist['val_acc'], label='LoRA (r=8)', color='blue')
        ax.plot(full_hist['val_acc'], label='Full Fine-tune', color='red', linestyle='--')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Accuracy')
        ax.set_title('Validation Accuracy')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 3: Parameter efficiency
        ax = axes[2]
        ranks = [2, 4, 8, 16, 32]
        params_pct = [2 * d_task * r / W_pretrained.size * 100 for r in ranks]
        ax.bar(range(len(ranks)), params_pct, color='steelblue')
        ax.set_xticks(range(len(ranks)))
        ax.set_xticklabels([f'r={r}' for r in ranks])
        ax.set_ylabel('% of Full Params')
        ax.set_title('LoRA Parameter Efficiency')
        ax.axhline(y=100, color='red', linestyle='--', label='Full Fine-tune')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        path = os.path.join(output_dir, "plot_lora.png")
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"\n  Saved: {path}")

    except ImportError:
        print("  matplotlib chua cai.")

    print("\n" + "=" * 60)
    print("TAT CA TESTS PASSED!")
    print("=" * 60)


# ============ CHECKLIST ============
# Week 12 (Bai 11):
# [x] Implement LoRA
#     -> LoRALayer: W_frozen + A @ B (chi train A, B voi rank nho)
#        r=8: chi them 0.1-1% params so voi full model
#        alpha=16: scaling factor, thuong = 2*r
# [x] Fine-tune model
#     -> LoRAModel: wrap base model, chi train LoRA layers
#        Freeze toan bo base model, chi update A va B
# [x] Compare voi full fine-tuning
#     -> LoRA: 0.1-1% params, nhanh, it memory
#        Full: 100% params, cham, ton memory
#        Ket qua gan tuong duong nhung LoRA tiet kiem rat nhieu
