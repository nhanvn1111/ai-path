# File: 06_attention.py
# Attention Mechanism tu Scratch - Week 5-7
#
# ============ GIAI THICH ATTENTION BANG LOI ============
#
# VAN DE: TAI SAO CAN ATTENTION?
# Truoc attention, model doc text tuan tu (RNN/LSTM):
# "The cat sat on the mat" -> doc tu trai qua phai, quen dau khi doc cuoi
# Cau dai thi mat thong tin dau cau. Giong doc 500 chu ma khong duoc nhin lai.
#
# Attention cho phep moi tu "NHIN LAI" tat ca cac tu khac de quyet dinh
# nen chu y vao tu nao.
#
# VD: "Con meo ngoi tren ban, NO rat luoi"
# Khi model doc "no", no can biet "no" chi ai.
# Attention cho "no" nhin lai toan bo cau va phat hien:
#   - "no" lien quan manh voi "con meo" -> attention score CAO
#   - "no" it lien quan voi "ban"       -> attention score THAP
# => Tu "no" gio mang thong tin cua "con meo" (vi score cao nhat)
#
# CACH HOAT DONG (3 BUOC):
#
# Buoc 1: Tao Q, K, V tu moi tu
#   Moi tu duoc bien thanh 3 vector:
#   - Query (Q) = "Toi dang tim gi?"    (cau hoi)
#   - Key (K)   = "Toi chua thong tin gi?" (nhan)
#   - Value (V) = "Thong tin thuc su"    (noi dung)
#   Giong thu vien: Q = ban hoi "sach AI dau?",
#                   K = nhan tren ke ("AI", "Nau an", "Lich su"),
#                   V = noi dung cuon sach
#
# Buoc 2: Tinh diem attention
#   score = Q @ K.T / sqrt(d_k)
#   Nhan Q voi K de xem muc do lien quan giua 2 tu:
#   - Q cua "no" x K cua "meo" = score CAO  (lien quan manh)
#   - Q cua "no" x K cua "ban" = score THAP  (it lien quan)
#   Chia sqrt(d_k) de score khong qua lon (tranh softmax bi saturate -> gradient chet)
#
# Buoc 3: Lay trung binh co trong so
#   attention = softmax(score) @ V
#   - Softmax bien scores thanh xac suat (tong = 1)
#   - Nhan voi V = lay NHIEU thong tin tu tu lien quan, IT tu tu khong lien quan
#   - Ket qua: tu "no" gio MANG THONG TIN cua "con meo"
#
# MULTI-HEAD ATTENTION:
# 1 head = 1 goc nhin. Nhung ngon ngu phuc tap, can nhieu goc nhin:
#   - Head 1: chu y NGU PHAP ("no" -> "meo" vi cung chu ngu)
#   - Head 2: chu y VI TRI (tu gan nhau lien quan hon)
#   - Head 3: chu y NGU NGHIA ("luoi" -> "meo" vi meo hay luoi)
# GPT-2 dung 12 heads, GPT-3 dung 96 heads.
# MultiHead = Concat(head_1, head_2, ..., head_n) @ W_o
#
# SELF-ATTENTION vs CROSS-ATTENTION:
#   - Self-Attention:  moi tu nhin lai CHINH CAU cua no (GPT, BERT)
#   - Cross-Attention: tu o output nhin sang CAU INPUT (dich thuat, Q&A)
#
# CAUSAL MASK (trong GPT):
# GPT sinh text trai -> phai, nen tu hien tai CHI DUOC nhin cac tu truoc no:
#   "Toi"   nhin duoc: [Toi]
#   "thich"  nhin duoc: [Toi, thich]
#   "meo"   nhin duoc: [Toi, thich, meo]
#
# TOM TAT 1 DONG:
# Attention = moi tu tu hoi "toi nen chu y vao tu nao?" bang cach so sanh
# Query cua minh voi Key cua tat ca tu khac, roi lay thong tin (Value) theo ty le do.
#
# ============================================================
#
# 3 LOAI ATTENTION TRONG FILE NAY:
# 1. Self-Attention: moi token attend to cac token trong cung 1 cau
# 2. Multi-Head: nhieu attention heads song song, moi head focus aspect khac
# 3. Cross-Attention: decoder attend to encoder (dung trong dich thuat)

import numpy as np
import os


def softmax(x, axis=-1):
    exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


class SelfAttention:
    """
    Self-Attention: Moi token "nhin" vao tat ca tokens khac

    Attention(Q, K, V) = softmax(Q @ K.T / sqrt(d_k)) @ V

    - Q (Query): "Toi dang tim gi?"
    - K (Key):   "Toi co gi?"
    - V (Value): "Thong tin thuc su"
    """

    def __init__(self, d_model, d_k=None):
        """
        d_model: kich thuoc embedding cua moi token (dimension cua model)
                 Vd: GPT-2 small = 768, GPT-3 = 12288
                 Moi token duoc bieu dien bang 1 vector co d_model chieu
                 d_model lon -> model "hieu" nhieu hon nhung ton memory/compute

        d_k:     kich thuoc cua Q, K, V sau khi project
                 None = dung d_model (single-head attention)
                 Trong multi-head: d_k = d_model / num_heads
                 Vd: d_model=768, num_heads=12 -> d_k=64

        W_q, W_k, W_v: ma tran project input thanh Q, K, V
                        Shape: (d_model, d_k)
                        Q = input @ W_q, K = input @ W_k, V = input @ W_v
        """
        self.d_model = d_model
        self.d_k = d_k or d_model

        scale = np.sqrt(2.0 / d_model)
        self.W_q = np.random.randn(d_model, self.d_k) * scale  # Query projection
        self.W_k = np.random.randn(d_model, self.d_k) * scale  # Key projection
        self.W_v = np.random.randn(d_model, self.d_k) * scale  # Value projection

    def forward(self, x, mask=None, dropout_rate=0.0, training=True):
        """
        x:            input, shape (batch_size, seq_len, d_model)
                      batch_size = so cau xu ly cung luc
                      seq_len = so tokens trong cau
        mask:         causal mask, shape (seq_len, seq_len)
                      1 = khong duoc nhin (tuong lai), 0 = duoc nhin (qua khu)
                      None = full attention (moi token nhin tat ca)
        dropout_rate: ty le dropout tren attention weights (0.0 = tat)
        training:     True = ap dung dropout, False = khong
        """
        Q = x @ self.W_q
        K = x @ self.W_k
        V = x @ self.W_v

        scores = Q @ np.transpose(K, (0, 2, 1))
        # Chia sqrt(d_k) de on dinh gradient.
        # Neu khong chia, dot product lon -> softmax ra 0/1 (saturate) -> gradient = 0
        scores = scores / np.sqrt(self.d_k)

        if mask is not None:
            # Causal mask: dat -inf cho vi tri tuong lai -> softmax = 0
            # -> token chi "nhin" duoc tokens TRUOC no (autoregressive)
            scores = scores + mask * (-1e9)

        attn_weights = softmax(scores, axis=-1)

        # Bai tap 3: Dropout on attention weights
        # TAI SAO dropout tren attention weights?
        # Ngăn model qua phu thuoc vao 1 token cu the
        # Buoc model phai attend to nhieu tokens khac nhau
        if training and dropout_rate > 0:
            drop_mask = (np.random.rand(*attn_weights.shape) > dropout_rate).astype(float)
            attn_weights = attn_weights * drop_mask / (1 - dropout_rate)

        output = attn_weights @ V
        return output, attn_weights


class MultiHeadAttention:
    """
    Multi-Head Attention: nhieu attention heads song song

    Moi head focus vao aspect khac nhau:
    - Head 1: syntactic (noun-verb)
    - Head 2: semantic (synonyms)
    - Head 3: positional patterns
    """

    def __init__(self, d_model, num_heads):
        """
        d_model:   kich thuoc embedding (giong SelfAttention)
        num_heads: so attention heads chay song song
                   GPT-2 small: 12 heads, GPT-3: 96 heads
                   Nhieu heads = moi head focus 1 aspect khac (syntax, semantics, position)
                   d_model PHAI chia het cho num_heads (d_k = d_model / num_heads)
                   Vd: d_model=768, num_heads=12 -> moi head co d_k=64

        W_o:       output projection, shape (d_model, d_model)
                   Gop output cua tat ca heads lai thanh 1 vector d_model
        """
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # Kich thuoc moi head

        assert d_model % num_heads == 0, "d_model phai chia het cho num_heads"

        self.heads = [SelfAttention(d_model, self.d_k) for _ in range(num_heads)]
        self.W_o = np.random.randn(d_model, d_model) * np.sqrt(2.0 / d_model)

    def forward(self, x, mask=None, dropout_rate=0.0, training=True):
        head_outputs = []
        attention_weights_all = []

        for head in self.heads:
            out, attn_w = head.forward(x, mask, dropout_rate, training)
            head_outputs.append(out)
            attention_weights_all.append(attn_w)

        concat = np.concatenate(head_outputs, axis=-1)
        output = concat @ self.W_o
        return output, attention_weights_all


# Bai tap 2: Cross-Attention (encoder-decoder)
class CrossAttention:
    """
    Cross-Attention: decoder attend to encoder output

    Khac voi Self-Attention:
    - Q tu decoder (dang generate gi)
    - K, V tu encoder (context/source)

    Dung trong:
    - Machine translation: decoder nhin lai source sentence
    - Image captioning: text decoder nhin lai image features
    """

    def __init__(self, d_model, d_k=None):
        """
        d_model: kich thuoc embedding (giong SelfAttention)
        d_k:     kich thuoc Q, K, V projection (giong SelfAttention)

        Khac SelfAttention: Q tu decoder, K va V tu encoder
        -> cho phep decoder "hoi" encoder: "cho context cua input, toi nen generate gi?"
        """
        self.d_model = d_model
        self.d_k = d_k or d_model

        scale = np.sqrt(2.0 / d_model)
        self.W_q = np.random.randn(d_model, self.d_k) * scale  # Q tu decoder
        self.W_k = np.random.randn(d_model, self.d_k) * scale  # K tu encoder
        self.W_v = np.random.randn(d_model, self.d_k) * scale  # V tu encoder

    def forward(self, decoder_input, encoder_output, mask=None):
        """
        decoder_input:  (batch, dec_seq_len, d_model) - nguon cua Q
                        Cau dang generate (decoder side)
        encoder_output: (batch, enc_seq_len, d_model) - nguon cua K, V
                        Context tu input (encoder side)
        mask:           optional mask, thuong None cho cross-attention
        """
        Q = decoder_input @ self.W_q    # (batch, dec_seq, d_k)
        K = encoder_output @ self.W_k   # (batch, enc_seq, d_k)
        V = encoder_output @ self.W_v   # (batch, enc_seq, d_k)

        # scores: (batch, dec_seq, enc_seq)
        # scores[i][j] = how much decoder token i attends to encoder token j
        scores = Q @ np.transpose(K, (0, 2, 1))
        scores = scores / np.sqrt(self.d_k)

        if mask is not None:
            scores = scores + mask * (-1e9)

        attn_weights = softmax(scores, axis=-1)
        output = attn_weights @ V

        return output, attn_weights


def create_causal_mask(seq_len):
    """Causal mask: token chi attend to tokens truoc no"""
    return np.triu(np.ones((seq_len, seq_len)), k=1)


# ============ MAIN ============
if __name__ == "__main__":
    np.random.seed(42)
    output_dir = os.path.dirname(os.path.abspath(__file__))

    batch_size = 2
    seq_len = 5
    d_model = 64
    num_heads = 4

    x = np.random.randn(batch_size, seq_len, d_model)

    print("=" * 60)
    print("TEST CO BAN - Self-Attention")
    print("=" * 60)

    sa = SelfAttention(d_model)
    out, attn_w = sa.forward(x)
    print(f"  Input:    {x.shape}")
    print(f"  Output:   {out.shape}")
    print(f"  Attn:     {attn_w.shape}")
    assert out.shape == (batch_size, seq_len, d_model)
    assert attn_w.shape == (batch_size, seq_len, seq_len)
    # Verify attention weights sum to 1
    row_sums = np.sum(attn_w, axis=-1)
    assert np.allclose(row_sums, 1.0, atol=1e-6), "Attn weights don't sum to 1"
    print("  Weights sum to 1: OK")

    print("\n--- Causal Mask ---")
    mask = create_causal_mask(seq_len)
    out_masked, attn_masked = sa.forward(x, mask)
    print(f"  Causal mask:\n{mask}")
    print(f"  Attn weights (sample 0):\n{np.round(attn_masked[0], 3)}")
    # Verify future tokens have ~0 attention
    for i in range(seq_len):
        for j in range(i + 1, seq_len):
            assert attn_masked[0, i, j] < 1e-6, f"Future leak at [{i},{j}]"
    print("  No future leaking: OK")

    print("\n--- Multi-Head Attention ---")
    mha = MultiHeadAttention(d_model, num_heads)
    out_mha, attn_heads = mha.forward(x, mask)
    print(f"  Output:   {out_mha.shape}")
    print(f"  Heads:    {len(attn_heads)}")
    print(f"  Per head: {attn_heads[0].shape}")
    assert out_mha.shape == (batch_size, seq_len, d_model)
    assert len(attn_heads) == num_heads
    print("  OK")

    # ============ BAI TAP 2: Cross-Attention ============
    print("\n" + "=" * 60)
    print("BAI TAP 2: Cross-Attention")
    print("=" * 60)

    enc_seq_len = 8
    dec_seq_len = 5

    encoder_output = np.random.randn(batch_size, enc_seq_len, d_model)
    decoder_input = np.random.randn(batch_size, dec_seq_len, d_model)

    ca = CrossAttention(d_model)
    cross_out, cross_attn = ca.forward(decoder_input, encoder_output)

    print(f"  Encoder output: {encoder_output.shape}")
    print(f"  Decoder input:  {decoder_input.shape}")
    print(f"  Cross output:   {cross_out.shape}")
    print(f"  Cross attn:     {cross_attn.shape}")
    assert cross_out.shape == (batch_size, dec_seq_len, d_model)
    assert cross_attn.shape == (batch_size, dec_seq_len, enc_seq_len)
    # Each decoder token's attn over encoder sums to 1
    assert np.allclose(np.sum(cross_attn, axis=-1), 1.0, atol=1e-6)
    print("  Cross-attn shape correct: OK")
    print("  Weights sum to 1: OK")

    # ============ BAI TAP 3: Dropout ============
    print("\n" + "=" * 60)
    print("BAI TAP 3: Attention Dropout")
    print("=" * 60)

    sa_drop = SelfAttention(d_model)
    # Copy weights de so sanh cong bang
    sa_nodrop = SelfAttention(d_model)
    sa_nodrop.W_q = sa_drop.W_q.copy()
    sa_nodrop.W_k = sa_drop.W_k.copy()
    sa_nodrop.W_v = sa_drop.W_v.copy()

    out_nodrop, attn_nodrop = sa_nodrop.forward(x, dropout_rate=0.0)
    out_drop, attn_drop = sa_drop.forward(x, dropout_rate=0.3, training=True)
    out_infer, attn_infer = sa_drop.forward(x, dropout_rate=0.3, training=False)

    # During training: some weights should be 0 (dropped)
    zeros_in_drop = np.sum(attn_drop == 0)
    zeros_in_nodrop = np.sum(attn_nodrop == 0)
    print(f"  Zeros without dropout: {zeros_in_nodrop}")
    print(f"  Zeros with dropout:    {zeros_in_drop}")
    assert zeros_in_drop > zeros_in_nodrop, "Dropout should create zeros"
    print("  Dropout creates zeros: OK")

    # During inference: no dropout
    assert np.allclose(attn_infer, attn_nodrop, atol=1e-6), "Inference should not drop"
    print("  No dropout at inference: OK")

    # ============ BAI TAP 4: Compare voi PyTorch ============
    print("\n" + "=" * 60)
    print("BAI TAP 4: Compare voi torch.nn.MultiheadAttention")
    print("=" * 60)

    try:
        import torch
        import torch.nn as nn

        d = 64
        nhead = 4
        seq = 5
        batch = 2

        # Tao input giong nhau
        x_np = np.random.randn(batch, seq, d).astype(np.float32)

        # PyTorch expects (seq, batch, d) for nn.MultiheadAttention
        x_torch = torch.tensor(x_np).transpose(0, 1)  # (seq, batch, d)

        torch_mha = nn.MultiheadAttention(d, nhead, batch_first=False, bias=False)

        # Causal mask
        mask_torch = nn.Transformer.generate_square_subsequent_mask(seq)

        with torch.no_grad():
            out_torch, attn_torch = torch_mha(x_torch, x_torch, x_torch, attn_mask=mask_torch)

        out_torch_np = out_torch.transpose(0, 1).numpy()  # back to (batch, seq, d)
        attn_torch_np = attn_torch.numpy()

        print(f"  PyTorch output shape: {out_torch_np.shape}")
        print(f"  PyTorch attn shape:   {attn_torch_np.shape}")
        assert out_torch_np.shape == (batch, seq, d)
        print("  Shape matches: OK")

        # Verify attn weights sum to 1
        attn_sums = np.sum(attn_torch_np, axis=-1)
        assert np.allclose(attn_sums, 1.0, atol=1e-4)
        print("  PyTorch attn sums to 1: OK")

        # Verify causal masking (future = ~0)
        for i in range(seq):
            for j in range(i + 1, seq):
                assert attn_torch_np[0, i, j] < 1e-3, f"PyTorch future leak [{i},{j}]"
        print("  PyTorch no future leak: OK")

        print(f"\n  -> Cung architecture, cung behavior")
        print(f"  -> PyTorch dung efficient batched implementation")

    except ImportError:
        print("  torch chua cai.")

    # ============ BAI TAP 1: Visualize Attention ============
    print("\n" + "=" * 60)
    print("BAI TAP 1: Visualize Attention Heatmap")
    print("=" * 60)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        tokens = ["The", "cat", "sat", "on", "mat"]

        fig, axes = plt.subplots(1, num_heads, figsize=(4 * num_heads, 4))
        for h in range(num_heads):
            ax = axes[h]
            im = ax.imshow(attn_heads[h][0], cmap='Blues', vmin=0, vmax=1)
            ax.set_xticks(range(seq_len))
            ax.set_yticks(range(seq_len))
            ax.set_xticklabels(tokens, rotation=45, fontsize=8)
            ax.set_yticklabels(tokens, fontsize=8)
            ax.set_title(f"Head {h}")
            plt.colorbar(im, ax=ax, fraction=0.046)

        plt.suptitle("Multi-Head Attention Weights (causal mask)", y=1.02)
        plt.tight_layout()
        path = os.path.join(output_dir, "plot_attention_heatmap.png")
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {path}")

        # Cross-attention heatmap
        enc_tokens = ["I", "love", "cats", "and", "dogs", "very", "much", "."]
        dec_tokens = ["J'", "aime", "les", "chats", "."]

        fig, ax = plt.subplots(figsize=(10, 5))
        im = ax.imshow(cross_attn[0], cmap='Oranges', vmin=0, vmax=0.5)
        ax.set_xticks(range(enc_seq_len))
        ax.set_yticks(range(dec_seq_len))
        ax.set_xticklabels(enc_tokens, rotation=45)
        ax.set_yticklabels(dec_tokens)
        ax.set_xlabel("Encoder (source)")
        ax.set_ylabel("Decoder (target)")
        ax.set_title("Cross-Attention Weights")
        plt.colorbar(im, ax=ax)

        plt.tight_layout()
        path2 = os.path.join(output_dir, "plot_cross_attention.png")
        plt.savefig(path2, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {path2}")

    except ImportError:
        print("  matplotlib chua cai.")

    print("\n" + "=" * 60)
    print("TAT CA TESTS PASSED!")
    print("=" * 60)


# ============ CHECKLIST ============
# Week 5-7 (Bai 06):
# [x] Giai thich attention mechanism bang loi
#     -> Xem phan "GIAI THICH ATTENTION BANG LOI" o dau file (dong 4-67)
#        Tom tat: moi tu hoi "toi nen chu y tu nao?" bang Q@K.T,
#        roi lay thong tin (V) theo ty le do
# [x] Code multi-head attention
#     -> MultiHeadAttention class: chia Q,K,V thanh nhieu heads
#        Moi head hoc 1 pattern khac (ngu phap, vi tri, ngu nghia)
#        Concat tat ca heads roi project qua W_o
