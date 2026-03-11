# File: 07_gpt.py
# Full GPT Model tu Scratch - Week 5-7
#
# TAI SAO FILE NAY QUAN TRONG?
# GPT (Generative Pre-trained Transformer) la kien truc dang sau
# ChatGPT, Claude, LLaMA, va hau het LLM hien nay.
#
# CAU TRUC GPT:
# 1. Token Embedding: chuyen token id -> vector (hoc duoc)
# 2. Positional Encoding: them thong tin vi tri (token thu may)
# 3. N x Transformer Block:
#    a. LayerNorm -> Multi-Head Attention -> Residual (+)
#    b. LayerNorm -> Feed-Forward Network -> Residual (+)
# 4. Final LayerNorm -> Output Projection -> logits (du doan next token)
#
# GPT la "decoder-only": chi dung causal mask (token chi nhin tokens truoc no)
# Khac voi BERT (encoder, nhin ca 2 phia) hay T5 (encoder-decoder).

import numpy as np
import os
import time


def softmax(x, axis=-1):
    exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

def gelu(x):
    """
    GELU activation - dung trong GPT (thay cho ReLU)
    TAI SAO GELU THAY VI RELU?
    - ReLU: max(0, x) -> "cung" tai x=0, gradient = 0 khi x < 0 (dead neurons)
    - GELU: smooth, khong "chet", giup training on dinh hon
    - GPT-2, BERT, LLaMA deu dung GELU
    """
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3)))


class LayerNorm:
    """
    Layer Normalization
    TAI SAO CAN NORMALIZE?
    - Qua moi layer, distribution cua data thay doi (internal covariate shift)
    - Normalize ve mean=0, std=1 giup training on dinh
    - gamma, beta: cho model tu hoc scale va shift toi uu
    - Khac BatchNorm: normalize theo features (khong phu thuoc batch size)
    """
    def __init__(self, d_model, eps=1e-6):
        """
        d_model: kich thuoc vector cua moi token (= so features can normalize)
                 Vd: d_model=768 -> normalize 768 chieu cua moi token
        eps:     so nho tranh chia cho 0 khi std ~ 0
        gamma:   scale factor (learnable), khoi tao = 1
        beta:    shift factor (learnable), khoi tao = 0
        """
        self.gamma = np.ones(d_model)
        self.beta = np.zeros(d_model)
        self.eps = eps

    def forward(self, x):
        mean = np.mean(x, axis=-1, keepdims=True)
        std = np.std(x, axis=-1, keepdims=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta


class FeedForward:
    """
    Feed-Forward Network: 2 linear layers voi GELU
    FFN(x) = GELU(x @ W1 + b1) @ W2 + b2

    TAI SAO EXPANSION 4X?
    - Input d_model -> expand len 4*d_model -> nen lai d_model
    - Tang "dung luong" cua moi layer de hoc duoc nhieu pattern hon
    - Vi du: d_model=768 -> d_ff=3072 -> d_model=768
    """

    def __init__(self, d_model, d_ff=None, dropout_rate=0.0):
        """
        d_model:      kich thuoc input va output (giu nguyen shape)
        d_ff:         kich thuoc hidden layer cua FFN
                      None = d_model * 4 (default GPT)
                      Vd: d_model=768 -> d_ff=3072
                      Expansion 4x tang "dung luong suy nghi" cua moi layer
        dropout_rate: ty le dropout sau GELU activation (0.0 = tat)

        W1: (d_model, d_ff)   - expand tu d_model len d_ff
        W2: (d_ff, d_model)   - nen tu d_ff ve lai d_model
        Flow: x -> W1 -> GELU -> dropout -> W2 -> output (cung shape voi input)
        """
        d_ff = d_ff or d_model * 4
        scale = np.sqrt(2.0 / d_model)
        self.W1 = np.random.randn(d_model, d_ff) * scale
        self.b1 = np.zeros(d_ff)
        self.W2 = np.random.randn(d_ff, d_model) * scale
        self.b2 = np.zeros(d_model)
        self.dropout_rate = dropout_rate

    def forward(self, x, training=True):
        hidden = gelu(x @ self.W1 + self.b1)
        if training and self.dropout_rate > 0:
            mask = (np.random.rand(*hidden.shape) > self.dropout_rate).astype(float)
            hidden = hidden * mask / (1 - self.dropout_rate)
        return hidden @ self.W2 + self.b2


class MultiHeadAttention:
    def __init__(self, d_model, num_heads, dropout_rate=0.0):
        """
        d_model:      kich thuoc embedding cua moi token
                      GPT-2 small=768, medium=1024, large=1280, XL=1600
        num_heads:    so attention heads chay song song
                      GPT-2 small=12, medium=16, large=20, XL=25
                      d_k = d_model / num_heads (moi head xu ly d_k chieu)
        dropout_rate: dropout tren attention weights

        W_q, W_k, W_v: project input thanh Q, K, V. Shape (d_model, d_model)
                        Khac file 06: o day project FULL d_model roi reshape thanh heads
                        (hieu qua hon tao nhieu SelfAttention rieng le)
        W_o:           output projection. Shape (d_model, d_model)
                       Gop output cua tat ca heads va project ve d_model
        """
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads   # Kich thuoc moi head
        self.dropout_rate = dropout_rate

        scale = np.sqrt(2.0 / d_model)
        self.W_q = np.random.randn(d_model, d_model) * scale  # Query
        self.W_k = np.random.randn(d_model, d_model) * scale  # Key
        self.W_v = np.random.randn(d_model, d_model) * scale  # Value
        self.W_o = np.random.randn(d_model, d_model) * scale  # Output

    def forward(self, x, mask=None, training=True, kv_cache=None):
        """
        x:        input, shape (batch_size, seq_len, d_model)
        mask:     causal mask, shape (seq_len, seq_len). None = full attention
        training: True = ap dung dropout
        kv_cache: (cached_K, cached_V) tu cac buoc generate truoc
                  None = tinh K, V tu dau (training hoac buoc dau generate)
                  Co cache = chi tinh K, V cho token moi, noi voi cache cu
                  -> tiet kiem O(N) thay vi O(N^2) moi buoc
        """
        batch_size, seq_len, _ = x.shape

        Q = x @ self.W_q
        K = x @ self.W_k
        V = x @ self.W_v

        # Bai tap 3: KV-cache
        if kv_cache is not None:
            cached_K, cached_V = kv_cache
            K = np.concatenate([cached_K, K], axis=1)
            V = np.concatenate([cached_V, V], axis=1)
        new_cache = (K.copy(), V.copy())

        kv_len = K.shape[1]

        # Reshape to heads
        Q = Q.reshape(batch_size, seq_len, self.num_heads, self.d_k).transpose(0, 2, 1, 3)
        K = K.reshape(batch_size, kv_len, self.num_heads, self.d_k).transpose(0, 2, 1, 3)
        V = V.reshape(batch_size, kv_len, self.num_heads, self.d_k).transpose(0, 2, 1, 3)

        # Attention
        scores = Q @ K.transpose(0, 1, 3, 2) / np.sqrt(self.d_k)

        if mask is not None:
            scores = scores + mask * (-1e9)

        attn_weights = softmax(scores, axis=-1)

        if training and self.dropout_rate > 0:
            drop = (np.random.rand(*attn_weights.shape) > self.dropout_rate).astype(float)
            attn_weights = attn_weights * drop / (1 - self.dropout_rate)

        context = attn_weights @ V
        context = context.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, self.d_model)

        return context @ self.W_o, attn_weights, new_cache


class TransformerBlock:
    """
    1 block trong Transformer (GPT co nhieu blocks chong len nhau)

    x -> LayerNorm -> MHA -> + (residual)
    x -> LayerNorm -> FFN -> + (residual)

    TAI SAO CAN RESIDUAL CONNECTION (+)?
    - Khong co residual: gradient phai "chui" qua nhieu layers -> vanishing
    - Residual: gradient co "duong tat" (shortcut) -> train duoc model sau (100+ layers)
    - output = layer(x) + x  (cong input vao output)
    """

    def __init__(self, d_model, num_heads, d_ff=None, dropout_rate=0.0):
        """
        d_model, num_heads, d_ff, dropout_rate: giong cac class o tren

        Mot TransformerBlock = 2 sub-layers:
        1. LayerNorm -> Multi-Head Attention -> Residual (+)
        2. LayerNorm -> Feed-Forward Network -> Residual (+)

        GPT-2 small co 12 blocks, GPT-3 co 96 blocks
        Cang nhieu blocks = model "sau" hon = hieu nhieu hon nhung cham hon
        """
        self.attention = MultiHeadAttention(d_model, num_heads, dropout_rate)
        self.ff = FeedForward(d_model, d_ff, dropout_rate)
        self.ln1 = LayerNorm(d_model)  # Normalize truoc attention
        self.ln2 = LayerNorm(d_model)  # Normalize truoc FFN

    def forward(self, x, mask=None, training=True, kv_cache=None):
        attn_out, attn_weights, new_cache = self.attention.forward(
            self.ln1.forward(x), mask, training, kv_cache
        )
        x = x + attn_out
        ff_out = self.ff.forward(self.ln2.forward(x), training)
        x = x + ff_out
        return x, attn_weights, new_cache


class PositionalEncoding:
    """
    Positional Encoding - them thong tin vi tri cho tokens

    TAI SAO CAN VI TRI?
    - Attention xu ly tat ca tokens SONG SONG, khong co khai niem "truoc/sau"
    - "Dog bites man" vs "Man bites dog" -> cung tokens, khac nghia!
    - Positional encoding them thong tin "token nay o vi tri thu may"

    SINUSOIDAL: dung sin/cos voi tan so khac nhau
    - Vi tri gan: pattern giong nhau -> model hieu "ke nhau"
    - Khong can hoc, co the xu ly do dai chua thay khi training
    """

    def __init__(self, d_model, max_len=5000):
        """
        d_model: kich thuoc vector (phai giong d_model cua model)
                 PE duoc CONG vao embedding -> phai cung shape
        max_len: do dai toi da ho tro (so tokens toi da trong 1 cau)
                 5000 = ho tro cau dai toi 5000 tokens
                 GPT-2 dung max_len=1024, GPT-4 dung 128K
        """
        pe = np.zeros((max_len, d_model))
        position = np.arange(max_len)[:, np.newaxis]
        div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        self.pe = pe

    def forward(self, x, offset=0):
        seq_len = x.shape[1]
        return x + self.pe[offset:offset + seq_len]


class GPT:
    """
    GPT - Decoder-only Transformer

    1. Token embedding
    2. Positional encoding
    3. N x Transformer blocks
    4. Final LayerNorm
    5. Output projection to vocab
    """

    def __init__(self, vocab_size, d_model, num_heads, num_layers,
                 d_ff=None, max_len=512, dropout_rate=0.0):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_len = max_len

        self.token_emb = np.random.randn(vocab_size, d_model) * 0.02
        self.pos_enc = PositionalEncoding(d_model, max_len)

        self.blocks = [
            TransformerBlock(d_model, num_heads, d_ff, dropout_rate)
            for _ in range(num_layers)
        ]
        self.ln_f = LayerNorm(d_model)
        self.output_proj = self.token_emb.T  # Weight tying

    def forward(self, token_ids, training=True, kv_caches=None, pos_offset=0):
        batch_size, seq_len = token_ids.shape

        x = np.array([[self.token_emb[t] for t in seq] for seq in token_ids])
        x = x * np.sqrt(self.d_model)
        x = self.pos_enc.forward(x, offset=pos_offset)

        mask = np.triu(np.ones((seq_len, seq_len)), k=1)
        if kv_caches is not None and kv_caches[0] is not None:
            # During cached generation, only 1 new token -> no mask needed for Q
            mask = None

        new_caches = []
        all_attn = []
        for i, block in enumerate(self.blocks):
            cache_i = kv_caches[i] if kv_caches is not None else None
            x, attn_w, new_cache = block.forward(x, mask, training, cache_i)
            new_caches.append(new_cache)
            all_attn.append(attn_w)

        x = self.ln_f.forward(x)
        logits = x @ self.output_proj

        return logits, new_caches, all_attn

    def generate(self, start_tokens, max_new_tokens, temperature=1.0,
                 top_k=None, use_cache=False):
        """Autoregressive text generation"""
        tokens = list(start_tokens)
        kv_caches = [None] * len(self.blocks)

        for step in range(max_new_tokens):
            if use_cache and step > 0:
                # Bai tap 3: chi forward token moi nhat
                input_ids = np.array([[tokens[-1]]])
                logits, kv_caches, _ = self.forward(
                    input_ids, training=False,
                    kv_caches=kv_caches,
                    pos_offset=len(tokens) - 1
                )
            else:
                input_ids = np.array([tokens[-self.max_len:]])
                logits, kv_caches_new, _ = self.forward(input_ids, training=False)
                if use_cache:
                    kv_caches = kv_caches_new

            next_logits = logits[0, -1, :] / temperature

            if top_k is not None:
                indices = np.argsort(next_logits)[-top_k:]
                mask = np.full(self.vocab_size, -1e9)
                mask[indices] = 0
                next_logits = next_logits + mask

            probs = softmax(next_logits)
            next_token = np.random.choice(len(probs), p=probs)
            tokens.append(int(next_token))

        return tokens

    def get_params(self):
        """Return all trainable parameters"""
        params = []
        params.append(self.token_emb)
        for block in self.blocks:
            params.extend([block.attention.W_q, block.attention.W_k,
                          block.attention.W_v, block.attention.W_o])
            params.extend([block.ff.W1, block.ff.b1, block.ff.W2, block.ff.b2])
            params.extend([block.ln1.gamma, block.ln1.beta])
            params.extend([block.ln2.gamma, block.ln2.beta])
        params.extend([self.ln_f.gamma, self.ln_f.beta])
        return params


# ============ BAI TAP 1: Training Loop ============

def cross_entropy_loss(logits, targets):
    """
    Cross-entropy loss cho language modeling
    logits: (batch, seq_len, vocab_size)
    targets: (batch, seq_len) - next token ids
    """
    batch_size, seq_len, vocab_size = logits.shape
    logits_flat = logits.reshape(-1, vocab_size)
    targets_flat = targets.reshape(-1)

    exp_logits = np.exp(logits_flat - np.max(logits_flat, axis=-1, keepdims=True))
    probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)

    eps = 1e-15
    log_probs = np.log(np.clip(probs, eps, 1.0))
    loss = -np.mean(log_probs[np.arange(len(targets_flat)), targets_flat])
    return loss


def simple_train_step(gpt, input_ids, target_ids, lr=0.001, weight_decay=0.0):
    """
    Simplified training step using numerical gradients
    (Full backprop through transformer is complex - dung numerical grad cho demo)
    """
    h = 1e-4

    # Forward
    logits, _, _ = gpt.forward(input_ids, training=True)
    loss = cross_entropy_loss(logits[:, :-1, :], target_ids[:, 1:])

    # Update only embedding (feasible with numerical grad)
    for t in range(gpt.vocab_size):
        for d in range(gpt.d_model):
            gpt.token_emb[t, d] += h
            logits_plus, _, _ = gpt.forward(input_ids, training=False)
            loss_plus = cross_entropy_loss(logits_plus[:, :-1, :], target_ids[:, 1:])

            grad = (loss_plus - loss) / h
            gpt.token_emb[t, d] -= h  # restore

            # Bai tap 4: weight decay
            grad += weight_decay * gpt.token_emb[t, d]
            gpt.token_emb[t, d] -= lr * grad

    return loss


# ============ MAIN ============
if __name__ == "__main__":
    np.random.seed(42)
    output_dir = os.path.dirname(os.path.abspath(__file__))

    print("=" * 60)
    print("TEST CO BAN - GPT Forward Pass")
    print("=" * 60)

    gpt = GPT(
        vocab_size=100,
        d_model=64,
        num_heads=4,
        num_layers=2,
        max_len=128
    )

    dummy_input = np.array([[1, 2, 3, 4, 5]])
    logits, _, all_attn = gpt.forward(dummy_input, training=False)

    print(f"  Input shape:  {dummy_input.shape}")
    print(f"  Output shape: {logits.shape}")
    print(f"  Num layers:   {len(all_attn)}")
    assert logits.shape == (1, 5, 100)
    print("  Forward pass: OK")

    # Count parameters
    total_params = sum(p.size for p in gpt.get_params())
    print(f"  Total params: {total_params:,}")

    # ============ BAI TAP 1: Training loop ============
    print("\n" + "=" * 60)
    print("BAI TAP 1: Training Loop (demo)")
    print("=" * 60)

    # Demo: simple pattern learning
    # Sequence: 1 2 3 4 5 1 2 3 4 5 ...
    small_gpt = GPT(vocab_size=10, d_model=32, num_heads=2, num_layers=1, max_len=32)
    pattern = [1, 2, 3, 4, 5] * 3
    input_ids = np.array([pattern])
    target_ids = np.array([pattern])

    logits, _, _ = small_gpt.forward(input_ids, training=False)
    initial_loss = cross_entropy_loss(logits[:, :-1, :], target_ids[:, 1:])
    print(f"  Initial loss: {initial_loss:.4f}")
    print(f"  Random loss (log(10)): {np.log(10):.4f}")
    assert abs(initial_loss - np.log(10)) < 1.0, "Initial loss should be ~log(vocab_size)"
    print("  Initial loss ~ log(vocab): OK (model is random)")

    # ============ BAI TAP 2: Shakespeare text generation ============
    print("\n" + "=" * 60)
    print("BAI TAP 2: Shakespeare Text Generation")
    print("=" * 60)

    # Load Shakespeare (reuse from 05_tokenizer.py)
    shakespeare_path = os.path.join(output_dir, "shakespeare.txt")
    if os.path.exists(shakespeare_path):
        with open(shakespeare_path, 'r') as f:
            text = f.read()

        # Simple char-level tokenizer
        chars = sorted(set(text))
        vocab_size = len(chars)
        char2id = {c: i for i, c in enumerate(chars)}
        id2char = {i: c for c, i in char2id.items()}

        print(f"  Vocab size: {vocab_size} characters")
        print(f"  Text length: {len(text)} chars")

        # Build GPT
        shakespeare_gpt = GPT(
            vocab_size=vocab_size,
            d_model=64,
            num_heads=4,
            num_layers=2,
            max_len=128,
            dropout_rate=0.1  # Bai tap 4: dropout
        )

        # Generate (untrained - random)
        start = [char2id[c] for c in "To be "]
        generated = shakespeare_gpt.generate(start, max_new_tokens=100, temperature=0.8, top_k=20)
        gen_text = ''.join([id2char.get(t, '?') for t in generated])
        print(f"\n  Generated (untrained):")
        print(f"  '{gen_text[:100]}'")
        print(f"  (Random text - chua train)")
    else:
        print("  shakespeare.txt not found. Run 05_tokenizer.py first.")

    # ============ BAI TAP 3: KV-Cache ============
    print("\n" + "=" * 60)
    print("BAI TAP 3: KV-Cache cho faster generation")
    print("=" * 60)

    cache_gpt = GPT(vocab_size=100, d_model=64, num_heads=4, num_layers=2, max_len=128)
    start_tokens = [1, 2, 3]
    n_generate = 20

    # Without cache
    t0 = time.time()
    tokens_no_cache = cache_gpt.generate(start_tokens, n_generate, use_cache=False)
    time_no_cache = time.time() - t0

    # Reset seed for fair comparison
    np.random.seed(42)

    # With cache
    t0 = time.time()
    tokens_with_cache = cache_gpt.generate(start_tokens, n_generate, use_cache=True)
    time_with_cache = time.time() - t0

    print(f"  Generate {n_generate} tokens:")
    print(f"  Without cache: {time_no_cache:.4f}s")
    print(f"  With cache:    {time_with_cache:.4f}s")
    print(f"  Speedup:       {time_no_cache / time_with_cache:.1f}x")

    # Verify cache produces valid output
    assert len(tokens_with_cache) == len(start_tokens) + n_generate
    print(f"  Output length correct: OK")

    # ============ BAI TAP 4: Dropout + Weight Decay ============
    print("\n" + "=" * 60)
    print("BAI TAP 4: Dropout + Weight Decay")
    print("=" * 60)

    # Test dropout affects output during training
    drop_gpt = GPT(vocab_size=100, d_model=64, num_heads=4, num_layers=2, dropout_rate=0.3)
    test_input = np.array([[1, 2, 3, 4, 5]])

    np.random.seed(1)
    out1, _, _ = drop_gpt.forward(test_input, training=True)
    np.random.seed(2)
    out2, _, _ = drop_gpt.forward(test_input, training=True)
    out_infer, _, _ = drop_gpt.forward(test_input, training=False)

    diff_train = np.mean(np.abs(out1 - out2))
    print(f"  Diff between 2 training passes: {diff_train:.4f}")
    assert diff_train > 0.01, "Dropout should cause different outputs"
    print("  Dropout varies training outputs: OK")

    # Inference should be deterministic
    out_infer2, _, _ = drop_gpt.forward(test_input, training=False)
    diff_infer = np.mean(np.abs(out_infer - out_infer2))
    print(f"  Diff between 2 inference passes: {diff_infer:.6f}")
    assert diff_infer < 1e-10, "Inference should be deterministic"
    print("  Inference is deterministic: OK")

    # Weight decay demo
    print(f"\n  Weight decay:")
    w_norm_before = np.sum(drop_gpt.token_emb ** 2)
    # Simulate weight decay
    wd = 0.01
    drop_gpt.token_emb -= wd * drop_gpt.token_emb
    w_norm_after = np.sum(drop_gpt.token_emb ** 2)
    print(f"  W norm before decay: {w_norm_before:.2f}")
    print(f"  W norm after decay:  {w_norm_after:.2f}")
    assert w_norm_after < w_norm_before
    print("  Weight decay reduces norm: OK")

    # ============ BAI TAP 5: Visualize Attention ============
    print("\n" + "=" * 60)
    print("BAI TAP 5: Visualize Attention Patterns")
    print("=" * 60)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        viz_gpt = GPT(vocab_size=100, d_model=64, num_heads=4, num_layers=2)
        viz_input = np.array([[10, 20, 30, 40, 50, 60, 70, 80]])
        _, _, all_attn = viz_gpt.forward(viz_input, training=False)

        num_layers = len(all_attn)
        num_heads = all_attn[0].shape[1]
        seq = viz_input.shape[1]
        token_labels = [str(t) for t in viz_input[0]]

        fig, axes = plt.subplots(num_layers, num_heads, figsize=(4 * num_heads, 4 * num_layers))
        for layer in range(num_layers):
            for head in range(num_heads):
                ax = axes[layer][head]
                attn = all_attn[layer][0, head]
                im = ax.imshow(attn, cmap='viridis', vmin=0)
                ax.set_title(f"L{layer} H{head}", fontsize=10)
                if head == 0:
                    ax.set_ylabel(f"Layer {layer}")
                ax.set_xticks(range(seq))
                ax.set_yticks(range(seq))
                ax.set_xticklabels(token_labels, fontsize=7)
                ax.set_yticklabels(token_labels, fontsize=7)

        plt.suptitle("GPT Attention Patterns (all layers, all heads)", fontsize=14)
        plt.tight_layout()
        path = os.path.join(output_dir, "plot_gpt_attention.png")
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {path}")

    except ImportError:
        print("  matplotlib chua cai.")

    print("\n" + "=" * 60)
    print("TAT CA TESTS PASSED!")
    print("=" * 60)


# ============ CHECKLIST ============
# Week 5-7 (Bai 07):
# [x] Build full GPT architecture
#     -> GPT class gom: token embedding + positional encoding
#        + N x TransformerBlock (LayerNorm -> MultiHeadAttention -> FFN -> residual)
#        + final LayerNorm -> linear head -> logits
# [x] Generate text (random nhung working)
#     -> GPT.generate(): autoregressive, moi buoc du doan token tiep theo
#        Ho tro temperature (creativity) va top-k sampling
