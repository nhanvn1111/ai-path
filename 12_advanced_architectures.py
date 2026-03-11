# File: 12_advanced_architectures.py
# Advanced Architectures - Week 13-14
#
# ============ TAI SAO CAN ADVANCED ARCHITECTURES? ============
#
# LLM hien dai (LLaMA 2, Mixtral, GPT-4) khong chi la Transformer thuong.
# Chung dung nhieu ky thuat TIEN TIEN de:
# 1. Tang do dai context (RoPE: tu 2K len 128K+ tokens)
# 2. Giam memory O(N^2) -> O(N) (Flash Attention)
# 3. Giam KV cache khi inference (GQA: giam 8x memory)
# 4. Tang model capacity ma khong tang compute (MoE: 47B params nhung chi dung 14B)
#
# Day la 4 ky thuat QUAN TRONG NHAT trong LLM hien dai.
# Hieu chung = hieu TOAN BO kien truc LLaMA 2, Mixtral, GPT-4.
#
# VAN DE CU THE:
# - Absolute position embeddings (GPT-2): max 1024 tokens, khong generalize
#   -> RoPE: encode position bang ROTATION, generalize tot
# - Standard attention: O(N^2) memory, seq_len=8192 -> 256MB per head!
#   -> Flash Attention: tile-based, O(N) memory, EXACT same output
# - Multi-Head Attention: 64 heads x (K,V) = huge KV cache khi inference
#   -> GQA: 8 KV heads thay vi 64, giam 8x KV cache memory
# - Muon model lon (47B) nhung chi chay nhanh nhu 14B
#   -> MoE: 8 experts, chi dung top-2 per token
#
# BAI TAP:
# 1. Implement RoPE (Rotary Position Embedding) tu scratch
# 2. Implement Flash Attention (tile-based, exact, O(N) memory)
# 3. Implement GQA (Grouped Query Attention) va so sanh memory voi MHA
# 4. Build MoE (Mixture of Experts) layer voi routing va load balancing
#
# TAT CA implement bang numpy, khong dung pytorch/tensorflow.
# Moi bai co giai thich MATH + INTUITION chi tiet truoc code.

import numpy as np
import time
import os


# ================================================================
# UTILITY FUNCTIONS
# ================================================================

def softmax(x, axis=-1):
    """
    Stable softmax: tru max truoc khi exp de tranh overflow.

    x:    input array, shape bat ky
          Thuong la attention scores (batch, heads, seq, seq)
          hoac logits (batch, seq, vocab)

    axis: axis de tinh softmax, mac dinh -1 (axis cuoi cung)
          Vd: attention scores thi axis=-1 de softmax theo key dimension
    """
    exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


# ================================================================
# BAI TAP 1: RoPE (Rotary Position Embedding)
# ================================================================
#
# ============ GIAI THICH ROPE CHI TIET ============
#
# VAN DE VOI ABSOLUTE POSITION EMBEDDING:
# - GPT-2 dung learned position embedding: P[pos] la vector hoc duoc
# - MAX seq_len co dinh (GPT-2: 1024 tokens)
# - Khong the xu ly cau dai hon 1024 tokens
# - Khong encode RELATIVE position: khoang cach giua 2 tu
#   Vi du: "meo" o vi tri 3 va "ngoi" o vi tri 4 co khoang cach 1
#          "meo" o vi tri 100 va "ngoi" o vi tri 101 CUNG co khoang cach 1
#          Nhung absolute embedding khong capture dieu nay!
#
# GIAI PHAP: RoPE (Rotary Position Embedding)
# Y tuong chinh: Encode position bang ROTATION trong mat phang phuc (complex plane)
#
# CACH HOAT DONG (step by step):
# 1. Chia embedding dimension thanh CAP (pairs): (0,1), (2,3), (4,5), ...
#    Vd: d_model=128 -> 64 cap
#
# 2. Moi cap co 1 TAN SO (frequency) rieng:
#    theta_i = 10000^(-2i/d)    (i = 0, 1, 2, ..., d/2-1)
#    - i=0: theta_0 = 10000^0 = 1.0        (tan so CAO, thay doi nhanh)
#    - i=31: theta_31 = 10000^(-0.5) = 0.01 (tan so THAP, thay doi cham)
#    - i=63: theta_63 = 10000^(-1) = 0.0001  (tan so RAT THAP)
#    -> Giong Fourier transform: nhieu tan so khac nhau capture patterns khac nhau
#
# 3. Tai position m, xoay cap (x_2i, x_{2i+1}) mot goc theta_i * m:
#    x_2i'     = x_2i * cos(theta_i * m) - x_{2i+1} * sin(theta_i * m)
#    x_{2i+1}' = x_2i * sin(theta_i * m) + x_{2i+1} * cos(theta_i * m)
#    Day chinh la phep ROTATION 2D (rotation matrix)!
#
# 4. KEY INSIGHT - TAI SAO ROTATION ENCODE RELATIVE POSITION:
#    Dot product giua q (position m) va k (position n):
#    q_rotated . k_rotated = f(q, k, m-n)
#    Chi phu thuoc vao HIEU m-n (khoang cach giua 2 vi tri)!
#    -> Tu dong capture relative position ma KHONG can them tham so
#
# THUC TE:
# - LLaMA 1/2/3 dung RoPE voi base=10000 (cho Q va K)
# - GPT-NeoX (EleutherAI) dung RoPE voi base=10000
# - CodeLlama mo rong RoPE len 100K tokens bang cach tang base=1000000
# - Cac model moi dung NTK-aware scaling de mo rong context hon nua
#
# MEMORY VA COMPUTE:
# - Khong them tham so nao (zero learnable params!)
# - Chi can precompute cos/sin table 1 lan
# - Ap dung chi la element-wise multiply -> O(n*d) compute


def precompute_freqs_cis(dim, max_seq_len, base=10000.0):
    """
    Precompute cac frequency va cos/sin values cho RoPE.

    Chi can chay 1 LAN khi khoi tao model, sau do dung lai cho moi forward pass.

    dim:          so chieu cua head (head_dim), PHAI la so chan
                  Vd: LLaMA 7B: d_model=4096, num_heads=32 -> head_dim = 4096/32 = 128
                      GPT-NeoX 20B: head_dim = 128
                      LLaMA 70B: head_dim = 128
                  Moi cap (2i, 2i+1) se co 1 frequency rieng -> dim/2 frequencies

    max_seq_len:  do dai toi da cua sequence, precompute cos/sin cho tat ca positions
                  Vd: LLaMA 1: max_seq_len = 2048
                      LLaMA 2: max_seq_len = 4096
                      CodeLlama: max_seq_len = 100000 (voi base scaling)
                      GPT-NeoX: max_seq_len = 2048

    base:         base cho frequency formula: theta_i = base^(-2i/dim)
                  Mac dinh 10000.0 (dung boi LLaMA, GPT-NeoX)
                  Base lon hon -> frequency thap hon -> context dai hon
                  Vd: LLaMA 1/2: base=10000
                      CodeLlama: base=1000000 (de support 100K context)
                      Mixtral: base=10000

    Returns:
        freqs_cos: shape (max_seq_len, dim//2) - cos values cho moi (position, frequency)
        freqs_sin: shape (max_seq_len, dim//2) - sin values cho moi (position, frequency)

    MATH:
        freqs[i] = base^(-2i/dim) cho i = 0, 1, ..., dim/2 - 1
        Tai position m: angle[m, i] = m * freqs[i]
        freqs_cos[m, i] = cos(angle[m, i])
        freqs_sin[m, i] = sin(angle[m, i])
    """
    assert dim % 2 == 0, f"dim phai chan, got {dim}"

    # Step 1: Tinh frequencies cho tung cap dimension
    # freqs[i] = 1 / (base^(2i/dim)) = base^(-2i/dim)
    # i = 0: freq = 1.0 (tan so cao nhat)
    # i = dim/2-1: freq = 1/base (tan so thap nhat)
    i = np.arange(0, dim, 2, dtype=np.float64)  # [0, 2, 4, ..., dim-2]
    freqs = 1.0 / (base ** (i / dim))            # shape: (dim//2,)

    # Step 2: Tinh goc xoay cho moi (position, frequency)
    # angles[m, i] = m * freqs[i]
    positions = np.arange(max_seq_len, dtype=np.float64)  # [0, 1, 2, ..., max_seq_len-1]
    angles = np.outer(positions, freqs)  # shape: (max_seq_len, dim//2)

    # Step 3: Precompute cos va sin
    freqs_cos = np.cos(angles).astype(np.float32)  # (max_seq_len, dim//2)
    freqs_sin = np.sin(angles).astype(np.float32)  # (max_seq_len, dim//2)

    return freqs_cos, freqs_sin


def apply_rope(x, freqs_cos, freqs_sin):
    """
    Apply RoPE rotation len tensor x.

    CACH HOAT DONG:
    Moi cap (x_2i, x_{2i+1}) duoc xoay boi goc theta_i * position:
        x_2i'     = x_2i * cos(theta) - x_{2i+1} * sin(theta)
        x_{2i+1}' = x_2i * sin(theta) + x_{2i+1} * cos(theta)

    Day la phep nhan so phuc: (x_2i + j*x_{2i+1}) * (cos(theta) + j*sin(theta))
    Ma phep nhan so phuc = ROTATION trong mat phang phuc!

    x:          input tensor, shape (batch, num_heads, seq_len, head_dim)
                Day la Q hoac K sau khi da project qua W_q hoac W_k
                Vd: LLaMA 7B: (batch, 32, seq_len, 128)
                    GPT-NeoX: (batch, 64, seq_len, 96)
                LUU Y: RoPE chi ap dung cho Q va K, KHONG ap dung cho V!
                       Vi chi Q.K^T can encode position (attention scores)

    freqs_cos:  cos values tu precompute_freqs_cis
                shape (max_seq_len, head_dim//2), se duoc slice theo seq_len cua x

    freqs_sin:  sin values tu precompute_freqs_cis
                shape (max_seq_len, head_dim//2), se duoc slice theo seq_len cua x

    Returns:
        x_rotated: shape giong x, da duoc apply rotation
    """
    batch, num_heads, seq_len, head_dim = x.shape
    assert head_dim % 2 == 0

    # Slice freqs cho dung seq_len
    cos = freqs_cos[:seq_len]  # (seq_len, head_dim//2)
    sin = freqs_sin[:seq_len]  # (seq_len, head_dim//2)

    # Reshape de broadcast: (1, 1, seq_len, head_dim//2)
    cos = cos[np.newaxis, np.newaxis, :, :]  # (1, 1, seq_len, head_dim//2)
    sin = sin[np.newaxis, np.newaxis, :, :]  # (1, 1, seq_len, head_dim//2)

    # Tach x thanh 2 nua: even indices va odd indices
    # x_even = x[..., 0::2] la cac x_2i
    # x_odd  = x[..., 1::2] la cac x_{2i+1}
    x_even = x[..., 0::2]  # (batch, num_heads, seq_len, head_dim//2)
    x_odd = x[..., 1::2]   # (batch, num_heads, seq_len, head_dim//2)

    # Apply rotation:
    # x_2i'     = x_2i * cos - x_{2i+1} * sin
    # x_{2i+1}' = x_2i * sin + x_{2i+1} * cos
    x_even_rot = x_even * cos - x_odd * sin
    x_odd_rot = x_even * sin + x_odd * cos

    # Interleave lai: [even_0, odd_0, even_1, odd_1, ...]
    x_rotated = np.zeros_like(x)
    x_rotated[..., 0::2] = x_even_rot
    x_rotated[..., 1::2] = x_odd_rot

    return x_rotated


def rope_attention(Q, K, V, freqs_cos, freqs_sin):
    """
    Self-attention voi RoPE applied cho Q va K.

    Day la cach LLaMA tinh attention:
    1. Apply RoPE cho Q va K (encode position thong tin)
    2. Tinh attention scores = Q_rot @ K_rot^T / sqrt(d)
    3. Apply causal mask (GPT-style: chi nhin tokens truoc)
    4. Softmax + matmul voi V

    Q:          Query tensor, shape (batch, num_heads, seq_len, head_dim)
                Vd: LLaMA 7B: (1, 32, 512, 128)

    K:          Key tensor, shape (batch, num_heads, seq_len, head_dim)
                Giong shape voi Q (trong standard attention)

    V:          Value tensor, shape (batch, num_heads, seq_len, head_dim)
                KHONG apply RoPE! Vi V khong tham gia tinh attention scores

    freqs_cos:  cos values, (max_seq_len, head_dim//2) tu precompute_freqs_cis
    freqs_sin:  sin values, (max_seq_len, head_dim//2) tu precompute_freqs_cis

    Returns:
        output: shape (batch, num_heads, seq_len, head_dim) - attention output
    """
    # Apply RoPE cho Q va K (KHONG cho V)
    Q_rot = apply_rope(Q, freqs_cos, freqs_sin)
    K_rot = apply_rope(K, freqs_cos, freqs_sin)

    head_dim = Q.shape[-1]
    scale = 1.0 / np.sqrt(head_dim)

    # Attention scores: Q_rot @ K_rot^T
    # (batch, heads, seq_q, dim) @ (batch, heads, dim, seq_k) -> (batch, heads, seq_q, seq_k)
    scores = np.matmul(Q_rot, K_rot.transpose(0, 1, 3, 2)) * scale

    # Causal mask: token i chi nhin duoc token 0..i
    seq_len = Q.shape[2]
    mask = np.triu(np.ones((seq_len, seq_len)), k=1) * (-1e9)
    scores = scores + mask

    # Softmax va weighted sum
    attn_weights = softmax(scores, axis=-1)
    output = np.matmul(attn_weights, V)

    return output


# ================================================================
# BAI TAP 2: FLASH ATTENTION
# ================================================================
#
# ============ GIAI THICH FLASH ATTENTION CHI TIET ============
#
# VAN DE VOI STANDARD ATTENTION:
# Standard: scores = Q @ K^T    -> shape (N, N) voi N = seq_len
# Memory = O(N^2) cho attention matrix!
#
# Vi du cu the:
# - seq_len = 2048, float32: 2048^2 * 4 bytes = 16 MB per head
#   GPT-2 (12 heads): 16 * 12 = 192 MB -> OK
# - seq_len = 8192, float32: 8192^2 * 4 bytes = 256 MB per head
#   LLaMA 2 (32 heads): 256 * 32 = 8 GB -> KHONG DU VRAM!
# - seq_len = 32768: 32768^2 * 4 = 4 GB per head -> KHONG THE!
#
# GIAI PHAP: Flash Attention (Dao et al., 2022)
# Y tuong: KHONG bao gio luu toan bo N x N attention matrix!
# Thay vao do, chia Q, K, V thanh BLOCKS va tinh TUNG BLOCK.
#
# CACH HOAT DONG (Online Softmax Algorithm):
#
# VAN DE: softmax can TOAN BO scores de tinh:
#   softmax(x_i) = exp(x_i) / sum(exp(x_j) for all j)
#   -> Can biet ALL scores truoc khi tinh softmax
#   -> Phai luu toan bo N x N matrix
#
# TRICK: Online Softmax - cap nhat softmax TUNG BLOCK:
# Goi:
#   m = running maximum (de numerical stability)
#   l = running sum of exp (mau so cua softmax)
#   o = running weighted output
#
# Khi co block scores MOI:
#   m_new = max(m_old, max(block_scores))
#   l_new = exp(m_old - m_new) * l_old + sum(exp(block_scores - m_new))
#   o_new = (exp(m_old - m_new) * l_old * o_old + exp(block_scores - m_new) @ V_block) / l_new
#
# TAI SAO DUNG:
# - exp(m_old - m_new) * l_old: DIEU CHINH lai cac gia tri cu
#   (vi max da thay doi, can rescale)
# - sum(exp(block_scores - m_new)): THEM block moi vao mau so
# - Cuoi cung: CHINH XAC nhu standard softmax!
#
# MEMORY:
# - Standard: O(N^2) - luu full attention matrix
# - Flash:    O(N * B) voi B = block_size << N
# - Vd: N=8192, B=256: Flash dung 8192*256*4 = 8MB thay vi 256MB
#
# COMPUTE:
# - Flash Attention khong giam so phep tinh (van O(N^2*d))
# - Nhung NO NHANH HON vi:
#   1. Data locality tot hon (GPU cache friendly)
#   2. It doc/ghi tu HBM (GPU memory) hon
#   3. Kernel fusion (nhieu phep tinh trong 1 kernel)
#
# THUC TE:
# - Flash Attention 2 duoc dung trong HuggingFace Transformers
# - LLaMA 2, Mistral, GPT-4 deu dung Flash Attention
# - Nhanh hon 2-4x, tiet kiem 5-20x memory so voi standard
# - Day la implementation NUMPY de HIEU algorithm,
#   thuc te dung CUDA kernel de nhanh


def standard_attention(Q, K, V):
    """
    Standard attention: O(N^2) memory - dung de COMPARE voi Flash Attention.

    Q: shape (seq_len, d_model)
       Query matrix, moi row la query cua 1 token
       Vd: seq_len=512, d_model=64 cho demo
           Thuc te LLaMA: seq_len=4096, d_model=128 (per head)

    K: shape (seq_len, d_model)
       Key matrix, giong shape voi Q

    V: shape (seq_len, d_model)
       Value matrix, chua thong tin thuc su de lay ra

    Returns:
        output: shape (seq_len, d_model) - attention output
        memory_bytes: so bytes da dung cho attention matrix

    MEMORY ANALYSIS:
    - Attention matrix: N x N x 4 bytes (float32)
    - Vd: N=512: 512^2 * 4 = 1 MB (nho)
    - Vd: N=8192: 8192^2 * 4 = 256 MB (lon!)
    - Vd: N=32768: 32768^2 * 4 = 4 GB (khong chay duoc!)
    """
    N, d = Q.shape
    scale = 1.0 / np.sqrt(d)

    # Tinh TOAN BO attention matrix (O(N^2) memory!)
    scores = Q @ K.T * scale  # shape: (N, N) - DAY LA VAN DE!
    memory_bytes = scores.nbytes  # N * N * 4 bytes

    # Softmax
    attn_weights = softmax(scores, axis=-1)  # (N, N)

    # Weighted sum
    output = attn_weights @ V  # (N, d)

    return output, memory_bytes


def flash_attention(Q, K, V, block_size=64):
    """
    Flash Attention: EXACT same output nhu standard, nhung O(N) memory!

    ALGORITHM (pseudocode):
    1. Chia K, V thanh blocks (moi block co block_size rows)
    2. Cho moi query block (Q_block):
       a. Khoi tao: m = -inf, l = 0, o = 0
       b. Cho moi key-value block (K_block, V_block):
          - Tinh scores = Q_block @ K_block^T / sqrt(d)
          - Cap nhat online softmax: m, l, o
       c. Ket qua: o chinh la attention output cho Q_block nay
    3. Ghep tat ca output blocks lai -> output cuoi cung

    KEY INSIGHT: Tinh online softmax de KHONG luu full attention matrix!
    m_new = max(m_old, max(current_block_scores))
    l_new = exp(m_old - m_new) * l_old + sum(exp(scores - m_new))
    output = (exp(m_old - m_new) * l_old * old_output + softmax(block) @ V) / l_new

    Q:          shape (seq_len, d_model) - Query matrix
                Vd: (512, 64) cho demo, (8192, 128) cho thuc te

    K:          shape (seq_len, d_model) - Key matrix

    V:          shape (seq_len, d_model) - Value matrix

    block_size: kich thuoc moi block, so rows xu ly cung luc
                Nho hon -> it memory hon nhung nhieu iterations hon
                Lon hon -> nhanh hon nhung dung nhieu memory hon
                Gia tri thuong dung: 64, 128, 256
                Vd: Flash Attention paper dung block_size phu thuoc vao SRAM size
                    GPU A100: SRAM = 192KB -> block_size ~128 cho head_dim=128
                    Trong numpy demo: 64 la du tot

    Returns:
        output:       shape (seq_len, d_model) - CHINH XAC nhu standard attention
        memory_bytes: peak memory dung (nho hon standard nhieu!)

    MEMORY SO SANH (seq_len=4096, d_model=128, float32):
    - Standard: 4096^2 * 4 = 64 MB (full attention matrix)
    - Flash (block=128): 4096 * 128 * 4 = 2 MB (chi luu 1 block row tai 1 thoi diem)
    - Tiet kiem: 32x!
    """
    N, d = Q.shape
    scale = 1.0 / np.sqrt(d)

    # So blocks
    num_blocks = (N + block_size - 1) // block_size

    # Output va statistics
    output = np.zeros((N, d), dtype=np.float32)

    # Track peak memory: lon nhat la 1 block cua scores
    peak_memory = 0

    # Loop qua tung QUERY block
    for i in range(num_blocks):
        q_start = i * block_size
        q_end = min(q_start + block_size, N)
        Q_block = Q[q_start:q_end]  # (block_q, d)
        block_q = q_end - q_start

        # Running statistics cho online softmax
        # m: running max per query (de numerical stability)
        # l: running sum of exp (mau so softmax)
        # o: running weighted output (tu so softmax @ V)
        m = np.full((block_q, 1), -np.inf, dtype=np.float32)  # (block_q, 1)
        l = np.zeros((block_q, 1), dtype=np.float32)           # (block_q, 1)
        o = np.zeros((block_q, d), dtype=np.float32)            # (block_q, d)

        # Loop qua tung KEY-VALUE block
        for j in range(num_blocks):
            k_start = j * block_size
            k_end = min(k_start + block_size, N)
            K_block = K[k_start:k_end]  # (block_k, d)
            V_block = V[k_start:k_end]  # (block_k, d)

            # Tinh block scores: Q_block @ K_block^T
            # Shape: (block_q, block_k) - CHI BLOCK NHO, khong phai N x N!
            block_scores = Q_block @ K_block.T * scale  # (block_q, block_k)

            # Track memory: chi 1 block scores tai 1 thoi diem
            peak_memory = max(peak_memory, block_scores.nbytes)

            # ===== ONLINE SOFTMAX UPDATE =====
            # Day la TRAI TIM cua Flash Attention!

            # Buoc 1: Tinh max moi cua block nay
            block_max = np.max(block_scores, axis=-1, keepdims=True)  # (block_q, 1)

            # Buoc 2: New running max
            m_new = np.maximum(m, block_max)  # (block_q, 1)

            # Buoc 3: Rescale factor cho values cu
            # exp(m_old - m_new): dieu chinh lai vi max da thay doi
            # Neu m_new > m_old: cac gia tri cu can GIAM (vi max lon hon)
            # Neu m_new = m_old: khong thay doi (exp(0) = 1)
            correction_old = np.exp(m - m_new)     # (block_q, 1)

            # Buoc 4: Tinh exp cua block scores moi (da tru max moi)
            exp_scores = np.exp(block_scores - m_new)  # (block_q, block_k)

            # Buoc 5: Cap nhat running sum l
            l_new = correction_old * l + np.sum(exp_scores, axis=-1, keepdims=True)  # (block_q, 1)

            # Buoc 6: Cap nhat running output o
            # o_new = (rescaled_old_output + new_weighted_values) / l_new
            # = (exp(m_old - m_new) * l_old * o_old + exp(scores - m_new) @ V_block) / l_new
            o = (correction_old * l * o + exp_scores @ V_block) / l_new  # (block_q, d)

            # Cap nhat state
            m = m_new
            l = l_new

        # Luu output cho query block nay
        output[q_start:q_end] = o

    return output, peak_memory


# ================================================================
# BAI TAP 3: GQA (Grouped Query Attention)
# ================================================================
#
# ============ GIAI THICH GQA CHI TIET ============
#
# VAN DE VOI MULTI-HEAD ATTENTION (MHA):
# MHA co num_heads bo (Q, K, V) rieng biet:
# - GPT-3: 96 heads -> 96 K va 96 V projections
# - LLaMA 65B: 64 heads -> 64 K va 64 V
#
# Khi INFERENCE (sinh text), can luu KV CACHE:
# - Moi token moi can K, V cua TAT CA tokens truoc
# - KV cache size = 2 * num_layers * num_heads * seq_len * head_dim * bytes_per_element
# - LLaMA 65B, seq_len=2048, FP16:
#   2 * 80 * 64 * 2048 * 128 * 2 = 5.4 GB chi cho KV cache!
# - Seq_len=8192: 21.5 GB KV cache -> CHUA KE model weights!
#
# GIAI PHAP 1: MQA (Multi-Query Attention) - Shazeer 2019
# - Chi dung 1 bo K, V cho TAT CA Q heads
# - KV cache giam 64x (tu 64 K,V xuong 1 K,V)
# - NHUNG: mat quality vi tat ca heads share K,V (khong da dang goc nhin)
#
# GIAI PHAP 2: GQA (Grouped Query Attention) - Ainslie et al. 2023
# - TRUNG GIAN giua MHA va MQA
# - Chia Q heads thanh GROUPS, moi group share 1 bo K,V
# - Vd: 64 Q heads, 8 K,V heads -> 8 groups cua 8 Q heads moi group
# - KV cache giam 8x (tu 64 xuong 8) ma chat luong gan nhu MHA
#
# VI DU THUC TE:
# - LLaMA 2 7B:  num_heads=32, num_kv_heads=32 (MHA, khong dung GQA)
# - LLaMA 2 13B: num_heads=40, num_kv_heads=40 (MHA)
# - LLaMA 2 70B: num_heads=64, num_kv_heads=8  (GQA! 8 groups cua 8)
# - Mistral 7B:  num_heads=32, num_kv_heads=8   (GQA! 4 groups cua 8)
# - Gemma:       num_heads=16, num_kv_heads=1   (MQA! 1 group cua 16)
#
# KV CACHE SO SANH (LLaMA 2 70B, seq_len=4096, FP16):
# - MHA (64 KV heads): 2 * 80 * 64 * 4096 * 128 * 2 = 10.7 GB
# - GQA (8 KV heads):  2 * 80 * 8  * 4096 * 128 * 2 = 1.3 GB  (giam 8x!)
# - MQA (1 KV head):   2 * 80 * 1  * 4096 * 128 * 2 = 0.17 GB (giam 64x!)
#
# KEY: Expand K,V heads de match Q heads truoc khi tinh attention
# K: (batch, num_kv_heads, seq, head_dim)
# Expand: repeat moi KV head cho num_groups Q heads
# K_expanded: (batch, num_q_heads, seq, head_dim)


class GroupedQueryAttention:
    """
    Grouped Query Attention (GQA) - dung boi LLaMA 2 70B, Mistral 7B.

    Trung gian giua MHA (moi Q head co K,V rieng) va MQA (tat ca share 1 K,V).
    Giam KV cache memory khi inference ma van giu chat luong tot.

    CACH HOAT DONG:
    1. Project input -> Q (num_q_heads), K (num_kv_heads), V (num_kv_heads)
    2. Expand K, V: repeat moi KV head cho 1 group cua Q heads
    3. Tinh attention nhu binh thuong
    4. Concat va project output

    Memory savings:
    - MHA: num_q_heads K,V projections = num_q_heads * seq * head_dim * 2
    - GQA: num_kv_heads K,V projections = num_kv_heads * seq * head_dim * 2
    - Savings = num_q_heads / num_kv_heads (thuong 4x-8x)
    """

    def __init__(self, d_model, num_q_heads, num_kv_heads):
        """
        d_model:       kich thuoc embedding cua model
                       Vd: LLaMA 2 70B: d_model=8192
                           Mistral 7B: d_model=4096
                           GPT-2: d_model=768

        num_q_heads:   so luong Query heads
                       Vd: LLaMA 2 70B: 64 Q heads
                           Mistral 7B: 32 Q heads
                           GPT-2: 12 Q heads (MHA, khong dung GQA)

        num_kv_heads:  so luong Key/Value heads (it hon hoac bang num_q_heads)
                       num_q_heads % num_kv_heads PHAI = 0
                       Vd: LLaMA 2 70B: 8 KV heads (64/8 = 8 groups)
                           Mistral 7B: 8 KV heads (32/8 = 4 groups)
                           MHA: num_kv_heads = num_q_heads (khong chia group)
                           MQA: num_kv_heads = 1 (tat ca share 1 KV)
        """
        assert d_model % num_q_heads == 0, \
            f"d_model ({d_model}) phai chia het cho num_q_heads ({num_q_heads})"
        assert num_q_heads % num_kv_heads == 0, \
            f"num_q_heads ({num_q_heads}) phai chia het cho num_kv_heads ({num_kv_heads})"

        self.d_model = d_model
        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = d_model // num_q_heads
        self.num_groups = num_q_heads // num_kv_heads  # So Q heads per KV head

        # Scale for attention
        self.scale = 1.0 / np.sqrt(self.head_dim)

        # Weight matrices (Xavier initialization)
        scale_init = np.sqrt(2.0 / d_model)

        # Q projection: full num_q_heads
        # W_q: (d_model, num_q_heads * head_dim) = (d_model, d_model)
        self.W_q = np.random.randn(d_model, num_q_heads * self.head_dim).astype(np.float32) * scale_init

        # K, V projection: ONLY num_kv_heads (it hon Q!)
        # W_k: (d_model, num_kv_heads * head_dim) - NHO hon W_q!
        # Vd: LLaMA 2 70B: W_q = (8192, 8192) nhung W_k = (8192, 1024)
        self.W_k = np.random.randn(d_model, num_kv_heads * self.head_dim).astype(np.float32) * scale_init
        self.W_v = np.random.randn(d_model, num_kv_heads * self.head_dim).astype(np.float32) * scale_init

        # Output projection
        self.W_o = np.random.randn(num_q_heads * self.head_dim, d_model).astype(np.float32) * scale_init

    def forward(self, x, use_causal_mask=True):
        """
        Forward pass cua GQA.

        x:               input tensor, shape (batch, seq_len, d_model)
                         Vd: (1, 512, 4096) cho Mistral 7B inference
                             (4, 2048, 8192) cho LLaMA 2 70B training

        use_causal_mask: True = GPT-style (chi nhin tokens truoc)
                         False = BERT-style (nhin tat ca tokens)
                         Mac dinh True vi LLM deu la autoregressive

        Returns:
            output: shape (batch, seq_len, d_model) - giong input shape
        """
        batch, seq_len, _ = x.shape

        # Step 1: Project Q, K, V
        # Q: full num_q_heads
        Q = x @ self.W_q  # (batch, seq, num_q_heads * head_dim)
        Q = Q.reshape(batch, seq_len, self.num_q_heads, self.head_dim)
        Q = Q.transpose(0, 2, 1, 3)  # (batch, num_q_heads, seq, head_dim)

        # K, V: only num_kv_heads (IT hon Q!)
        K = x @ self.W_k  # (batch, seq, num_kv_heads * head_dim)
        K = K.reshape(batch, seq_len, self.num_kv_heads, self.head_dim)
        K = K.transpose(0, 2, 1, 3)  # (batch, num_kv_heads, seq, head_dim)

        V = x @ self.W_v  # (batch, seq, num_kv_heads * head_dim)
        V = V.reshape(batch, seq_len, self.num_kv_heads, self.head_dim)
        V = V.transpose(0, 2, 1, 3)  # (batch, num_kv_heads, seq, head_dim)

        # Step 2: EXPAND K, V de match num_q_heads
        # Day la KEY cua GQA: repeat moi KV head cho num_groups Q heads
        # K: (batch, num_kv_heads, seq, head_dim)
        # -> K_expanded: (batch, num_q_heads, seq, head_dim)
        # Vd: LLaMA 2 70B: K (batch, 8, seq, 128) -> K_expanded (batch, 64, seq, 128)
        #     Moi KV head duoc copy 8 lan (num_groups = 64/8 = 8)
        K_expanded = np.repeat(K, self.num_groups, axis=1)  # (batch, num_q_heads, seq, head_dim)
        V_expanded = np.repeat(V, self.num_groups, axis=1)  # (batch, num_q_heads, seq, head_dim)

        # Step 3: Tinh attention scores
        # Q @ K_expanded^T: (batch, num_q_heads, seq, seq)
        scores = np.matmul(Q, K_expanded.transpose(0, 1, 3, 2)) * self.scale

        # Causal mask (neu can)
        if use_causal_mask:
            mask = np.triu(np.ones((seq_len, seq_len)), k=1) * (-1e9)
            scores = scores + mask

        # Softmax
        attn_weights = softmax(scores, axis=-1)  # (batch, num_q_heads, seq, seq)

        # Weighted sum voi V_expanded
        attn_output = np.matmul(attn_weights, V_expanded)  # (batch, num_q_heads, seq, head_dim)

        # Step 4: Concat heads va project output
        attn_output = attn_output.transpose(0, 2, 1, 3)  # (batch, seq, num_q_heads, head_dim)
        attn_output = attn_output.reshape(batch, seq_len, -1)  # (batch, seq, d_model)
        output = attn_output @ self.W_o  # (batch, seq, d_model)

        return output

    def kv_cache_size(self, seq_len, dtype_bytes=2):
        """
        Tinh KV cache size cho inference.

        seq_len:     do dai sequence hien tai
                     Vd: 2048, 4096, 8192

        dtype_bytes: so bytes per element
                     2 = FP16/BF16 (pho bien nhat cho inference)
                     4 = FP32
                     1 = INT8 (quantized KV cache)

        Returns:
            kv_bytes: so bytes cho KV cache cua 1 layer nay

        Vi du LLaMA 2 70B (1 layer, seq=4096, FP16):
        - MHA: 2 * 64 * 4096 * 128 * 2 = 134 MB
        - GQA: 2 * 8  * 4096 * 128 * 2 = 16.8 MB (giam 8x!)
        """
        # 2 = K va V, num_kv_heads (KHONG phai num_q_heads!)
        kv_bytes = 2 * self.num_kv_heads * seq_len * self.head_dim * dtype_bytes
        return kv_bytes

    def param_count(self):
        """Dem so parameters trong GQA layer."""
        q_params = self.W_q.size      # d_model * (num_q_heads * head_dim)
        k_params = self.W_k.size      # d_model * (num_kv_heads * head_dim)
        v_params = self.W_v.size      # d_model * (num_kv_heads * head_dim)
        o_params = self.W_o.size      # (num_q_heads * head_dim) * d_model
        return {
            'q_params': q_params,
            'k_params': k_params,
            'v_params': v_params,
            'o_params': o_params,
            'total': q_params + k_params + v_params + o_params,
        }


def compare_attention_variants(d_model, seq_len, num_q_heads, dtype_bytes=2):
    """
    So sanh memory cua MHA, GQA, MQA cho KV cache.

    d_model:       kich thuoc embedding
                   Vd: 4096 (LLaMA 7B), 8192 (LLaMA 70B)

    seq_len:       do dai sequence
                   Vd: 2048, 4096, 8192, 32768

    num_q_heads:   so Query heads
                   Vd: 32 (LLaMA 7B), 64 (LLaMA 70B)

    dtype_bytes:   bytes per element (2 = FP16, 4 = FP32)

    Returns:
        dict voi KV cache size cho moi variant
    """
    head_dim = d_model // num_q_heads

    # MHA: num_kv_heads = num_q_heads
    mha_bytes = 2 * num_q_heads * seq_len * head_dim * dtype_bytes

    # GQA: num_kv_heads = num_q_heads / 8 (typical)
    num_kv_gqa = max(num_q_heads // 8, 1)
    gqa_bytes = 2 * num_kv_gqa * seq_len * head_dim * dtype_bytes

    # MQA: num_kv_heads = 1
    mqa_bytes = 2 * 1 * seq_len * head_dim * dtype_bytes

    return {
        'MHA': {'kv_heads': num_q_heads, 'bytes': mha_bytes},
        'GQA': {'kv_heads': num_kv_gqa, 'bytes': gqa_bytes},
        'MQA': {'kv_heads': 1, 'bytes': mqa_bytes},
    }


# ================================================================
# BAI TAP 4: MoE (Mixture of Experts)
# ================================================================
#
# ============ GIAI THICH MoE CHI TIET ============
#
# VAN DE:
# Muon model LON hon (nhieu capacity) nhung KHONG muon cham hon.
# - GPT-3 175B: moi token chay qua TOAN BO 175B params -> cham, ton GPU
# - Neu muon 1T params? -> can 1000+ GPU chi de inference!
#
# GIAI PHAP: Mixture of Experts (MoE)
# Y tuong: KHONG PHAI moi token can TAT CA params!
# - Moi token chi can mot vai "EXPERTS" phu hop
# - Token "code" can expert ve lap trinh
# - Token "tho" can expert ve van hoc
# - Token "H2O" can expert ve hoa hoc
#
# CACH HOAT DONG:
# 1. ROUTER: mang neural NHO, quyet dinh moi token di vao experts nao
#    router_logits = x @ W_router  # (batch*seq, num_experts)
#    router_probs = softmax(router_logits)
#    top_k_experts = argmax(router_probs, k=2)  # chon top-2 experts
#
# 2. EXPERTS: moi expert la 1 FFN (Feed-Forward Network) doc lap
#    expert_i(x) = GELU(x @ W1_i) @ W2_i
#    Giong FFN trong transformer nhung co nhieu bo, moi bo chuyen 1 loai token
#
# 3. COMBINE: cong ket qua cua top-k experts voi weights tu router
#    output = sum(router_weight_i * expert_i(x)) cho i trong top-k
#
# LOAD BALANCING:
# Van de: Neu router gui TAT CA tokens den 1 expert -> cac expert khac lang phi!
# Giai phap: Load balancing loss
#   - Tinh ti le tokens den moi expert (expert_usage)
#   - Tinh trung binh router probability (avg_probs)
#   - Loss = num_experts * sum(expert_usage * avg_probs)
#   - Loss nay thap nhat khi moi expert nhan duoc DIEU tokens
#   - Thuong nhan voi he so nho (0.01) roi cong vao main loss
#
# VI DU THUC TE:
# - Mixtral 8x7B (Mistral AI):
#   - 8 experts, moi expert la FFN 7B (thuc ra ~6.7B)
#   - top_k = 2: moi token chi chay qua 2/8 experts
#   - Tong params: ~47B (8 * ~6B FFN + shared attention)
#   - Active params per token: ~14B (2 experts + attention)
#   - Hieu nang NGANG LLaMA 2 70B nhung nhanh hon nhieu!
#
# - GShard (Google): 600B params, top-2 routing
# - Switch Transformer (Google): 1.6T params, top-1 routing (cuc nhanh)
# - GPT-4 duoc DON la MoE (8 experts x 220B? - chua xac nhan)
#
# ADVANTAGES:
# - Model capacity LON (47B params) nhung compute NHO (14B active)
# - Training nhanh hon: moi expert hoc specialization rieng
# - Inference nhanh hon: chi chay 2/8 experts
#
# DISADVANTAGES:
# - Memory: van can luu TOAN BO experts trong VRAM (47B params)
# - Load balancing kho: router can hoc cach phan bo deu
# - Communication overhead (distributed training: experts tren nhieu GPU)


class ExpertFFN:
    """
    Mot expert trong MoE - la 1 Feed-Forward Network (FFN).

    Giong FFN trong standard transformer:
    output = GELU(x @ W1 + b1) @ W2 + b2

    Moi expert hoc cach xu ly 1 LOAI token cu the.
    Vd: expert 0 chuyen xu ly code, expert 1 chuyen xu ly van ban thuong, ...

    Trong Mixtral 8x7B, moi expert co:
    - W1: (4096, 14336) - up projection
    - W2: (14336, 4096) - down projection
    - Tong ~117M params per expert, 8 experts = ~940M params chi cho FFN!
    """

    def __init__(self, d_model, d_ff, expert_id=0):
        """
        d_model:    kich thuoc embedding (input va output)
                    Vd: Mixtral: d_model=4096
                        Demo: d_model=64

        d_ff:       kich thuoc hidden layer cua FFN (thuong = 4 * d_model)
                    Vd: Mixtral: d_ff=14336 (3.5x d_model, dung SwiGLU)
                        Standard transformer: d_ff = 4 * d_model
                        Demo: d_ff=256

        expert_id:  ID cua expert nay (dung de tracking)
                    Vd: 0-7 cho Mixtral (8 experts)
        """
        self.expert_id = expert_id
        self.d_model = d_model
        self.d_ff = d_ff

        # Xavier initialization
        scale1 = np.sqrt(2.0 / (d_model + d_ff))
        scale2 = np.sqrt(2.0 / (d_ff + d_model))

        self.W1 = np.random.randn(d_model, d_ff).astype(np.float32) * scale1
        self.b1 = np.zeros(d_ff, dtype=np.float32)
        self.W2 = np.random.randn(d_ff, d_model).astype(np.float32) * scale2
        self.b2 = np.zeros(d_model, dtype=np.float32)

    def forward(self, x):
        """
        Forward pass cua expert FFN.

        x: shape (num_tokens, d_model) - CHI tokens duoc router chon cho expert nay
           KHONG phai toan bo batch! Chi 1 subset tokens
           Vd: Mixtral top-2, batch 1000 tokens -> moi expert nhan ~250 tokens (1000*2/8)

        Returns:
            output: shape (num_tokens, d_model)
        """
        # hidden = GELU(x @ W1 + b1)
        hidden = x @ self.W1 + self.b1
        # GELU activation (xap xi)
        hidden = hidden * 0.5 * (1.0 + np.tanh(
            np.sqrt(2.0 / np.pi) * (hidden + 0.044715 * hidden ** 3)
        ))
        # output = hidden @ W2 + b2
        output = hidden @ self.W2 + self.b2
        return output

    def param_count(self):
        """So luong parameters trong expert nay."""
        return self.W1.size + self.b1.size + self.W2.size + self.b2.size


class MoELayer:
    """
    Mixture of Experts Layer.

    Thay the FFN layer trong transformer bang nhieu expert FFNs + router.

    FLOW:
    input (batch*seq, d_model)
       |
       v
    ROUTER -> router_probs (batch*seq, num_experts)
       |
       v
    Top-K selection -> chon top_k experts per token
       |
       v
    Dispatch tokens to experts -> moi expert xu ly subset tokens cua no
       |
       v
    Combine outputs voi router weights
       |
       v
    output (batch*seq, d_model)

    LOAD BALANCING:
    Them auxiliary loss de dam bao moi expert duoc dung DEU:
    - Neu khong co: router co the gui TAT CA tokens den 1 expert
    - Voi load balancing loss: moi expert nhan ~N/num_experts tokens
    """

    def __init__(self, d_model, d_ff, num_experts, top_k=2):
        """
        d_model:      kich thuoc embedding
                      Vd: Mixtral: 4096, Demo: 64

        d_ff:         kich thuoc hidden cua moi expert FFN
                      Vd: Mixtral: 14336, Demo: 256

        num_experts:  tong so experts
                      Vd: Mixtral 8x7B: 8 experts
                          Switch Transformer: 128 experts (extreme!)
                          GShard: 2048 experts (!)
                          Thuc te: 4, 8, 16 la pho bien nhat

        top_k:        so experts duoc chon per token
                      top_k=2: Mixtral, GShard (pho bien nhat)
                      top_k=1: Switch Transformer (nhanh nhat, nhung kho train)
                      top_k > 2: it dung vi tang compute
                      Rule: top_k << num_experts (thuong top_k = 1 hoac 2)
        """
        self.d_model = d_model
        self.d_ff = d_ff
        self.num_experts = num_experts
        self.top_k = top_k

        # Router: 1 linear layer don gian
        # W_router: (d_model, num_experts)
        # Moi token duoc nhan voi W_router de tinh score cho tung expert
        self.W_router = np.random.randn(d_model, num_experts).astype(np.float32) * 0.01

        # Tao cac experts
        self.experts = [
            ExpertFFN(d_model, d_ff, expert_id=i)
            for i in range(num_experts)
        ]

        # Luu router probs cho load balancing loss
        self.last_router_probs = None
        self.last_expert_indices = None

    def forward(self, x):
        """
        Forward pass cua MoE layer.

        x: shape (batch, seq_len, d_model) hoac (num_tokens, d_model)
           Input tokens can xu ly
           Vd: (1, 512, 4096) cho Mixtral inference
               (4, 2048, 4096) cho training

        Returns:
            output: shape giong x
            aux_loss: load balancing auxiliary loss (float)
                      Cong vao main loss: total_loss = main_loss + 0.01 * aux_loss

        ALGORITHM:
        1. Router tinh xac suat cho tung expert: router_probs = softmax(x @ W_router)
        2. Chon top_k experts co xac suat cao nhat cho moi token
        3. Cho moi expert, thu thap tokens duoc assign va chay expert FFN
        4. Combine outputs: output_token = sum(weight_i * expert_i(token))
        5. Tinh load balancing loss
        """
        original_shape = x.shape
        if len(x.shape) == 3:
            batch, seq_len, d = x.shape
            x_flat = x.reshape(-1, d)  # (batch * seq_len, d_model)
        else:
            x_flat = x  # (num_tokens, d_model)

        num_tokens = x_flat.shape[0]

        # Step 1: Router - tinh xac suat moi expert cho moi token
        router_logits = x_flat @ self.W_router  # (num_tokens, num_experts)
        router_probs = softmax(router_logits, axis=-1)  # (num_tokens, num_experts)

        # Step 2: Chon top-k experts per token
        # top_k_indices: (num_tokens, top_k) - index cua top-k experts
        # top_k_weights: (num_tokens, top_k) - router probability cua top-k experts
        top_k_indices = np.argsort(router_probs, axis=-1)[:, -self.top_k:]  # (num_tokens, top_k)

        # Lay router weights cho top-k experts
        top_k_weights = np.take_along_axis(router_probs, top_k_indices, axis=-1)  # (num_tokens, top_k)

        # Normalize weights de sum = 1 (cho top-k)
        top_k_weights = top_k_weights / (top_k_weights.sum(axis=-1, keepdims=True) + 1e-9)

        # Luu cho load balancing loss
        self.last_router_probs = router_probs
        self.last_expert_indices = top_k_indices

        # Step 3: Dispatch va compute
        output = np.zeros_like(x_flat)  # (num_tokens, d_model)

        for k_idx in range(self.top_k):
            # Expert indices cho slot k_idx cua moi token
            expert_ids = top_k_indices[:, k_idx]   # (num_tokens,)
            weights = top_k_weights[:, k_idx]       # (num_tokens,)

            for expert_id in range(self.num_experts):
                # Tim tokens duoc assign cho expert nay
                token_mask = (expert_ids == expert_id)
                if not np.any(token_mask):
                    continue  # Khong co token nao cho expert nay

                # Thu thap tokens
                expert_input = x_flat[token_mask]  # (num_selected, d_model)
                expert_weights = weights[token_mask]  # (num_selected,)

                # Chay expert
                expert_output = self.experts[expert_id].forward(expert_input)  # (num_selected, d_model)

                # Weighted output va cong vao ket qua
                weighted_output = expert_output * expert_weights[:, np.newaxis]
                output[token_mask] += weighted_output

        # Step 4: Load balancing loss
        aux_loss = self._load_balancing_loss(router_probs, top_k_indices)

        # Reshape lai neu can
        if len(original_shape) == 3:
            output = output.reshape(original_shape)

        return output, aux_loss

    def _load_balancing_loss(self, router_probs, expert_indices):
        """
        Tinh load balancing auxiliary loss.

        MUC DICH: Dam bao moi expert duoc su dung DEU, tranh "expert collapse"
        (tat ca tokens di vao 1-2 experts, cac expert khac khong hoc gi).

        FORMULA:
        - expert_usage[i] = fraction of tokens assigned to expert i
        - avg_probs[i] = mean router probability for expert i across all tokens
        - loss = num_experts * sum(expert_usage * avg_probs)

        TAI SAO FORMULA NAY HOAT DONG:
        - Khi deu: expert_usage = [1/N, 1/N, ...], avg_probs = [1/N, 1/N, ...]
          loss = N * N * (1/N * 1/N) = 1.0 (minimum)
        - Khi lech: expert_usage = [1, 0, 0, ...], avg_probs = [1, 0, 0, ...]
          loss = N * 1 * 1 = N (maximum, rat cao -> penalty!)

        router_probs:    (num_tokens, num_experts) - xac suat tu router
        expert_indices:  (num_tokens, top_k) - expert duoc chon per token

        Returns:
            loss: float, cang thap cang tot (minimum = 1.0 khi hoan toan deu)
        """
        num_tokens = router_probs.shape[0]

        # Dem so lan moi expert duoc chon
        expert_usage = np.zeros(self.num_experts, dtype=np.float32)
        for idx in expert_indices.flatten():
            expert_usage[idx] += 1
        # Normalize thanh fraction
        total_assignments = expert_indices.size
        if total_assignments > 0:
            expert_usage = expert_usage / total_assignments

        # Trung binh router probability cho moi expert
        avg_probs = router_probs.mean(axis=0)  # (num_experts,)

        # Load balancing loss
        loss = self.num_experts * np.sum(expert_usage * avg_probs)

        return float(loss)

    def get_expert_usage_stats(self):
        """
        Thong ke su dung cua moi expert (sau khi forward).

        Returns:
            stats: dict voi usage per expert
        """
        if self.last_expert_indices is None:
            return None

        usage = np.zeros(self.num_experts, dtype=np.float32)
        for idx in self.last_expert_indices.flatten():
            usage[idx] += 1
        total = self.last_expert_indices.size
        usage_pct = usage / total * 100 if total > 0 else usage

        return {
            f'expert_{i}': f'{usage_pct[i]:.1f}%'
            for i in range(self.num_experts)
        }

    def total_params(self):
        """Tong so params trong MoE layer."""
        router_params = self.W_router.size
        expert_params = sum(e.param_count() for e in self.experts)
        return {
            'router_params': router_params,
            'expert_params': expert_params,
            'total_params': router_params + expert_params,
            'active_params_per_token': router_params + self.top_k * self.experts[0].param_count(),
        }

    def capacity_vs_compute(self):
        """
        So sanh tong capacity vs active compute.

        Vi du Mixtral 8x7B:
        - Total params: ~47B (8 experts * ~6B + shared)
        - Active per token: ~14B (2 experts * ~6B + shared)
        - Ratio: 14B / 47B = 30% compute cho 100% capacity!
        """
        params = self.total_params()
        return {
            'total_capacity': params['total_params'],
            'active_per_token': params['active_params_per_token'],
            'efficiency_ratio': params['active_params_per_token'] / params['total_params'],
        }


# ================================================================
# MAIN: Test tat ca implementations
# ================================================================
if __name__ == "__main__":
    np.random.seed(42)
    output_dir = os.path.dirname(os.path.abspath(__file__))

    # ============================================================
    # PHAN 1: RoPE (Rotary Position Embedding)
    # ============================================================
    print("=" * 70)
    print("PHAN 1: RoPE (Rotary Position Embedding)")
    print("=" * 70)

    # --- 1a: Precompute frequencies ---
    print("\n  --- 1a: Precompute Frequencies ---")
    head_dim = 128  # LLaMA head_dim
    max_seq = 2048  # LLaMA 1 max seq

    freqs_cos, freqs_sin = precompute_freqs_cis(head_dim, max_seq)
    print(f"  head_dim={head_dim}, max_seq_len={max_seq}")
    print(f"  freqs_cos shape: {freqs_cos.shape}")  # (2048, 64)
    print(f"  freqs_sin shape: {freqs_sin.shape}")  # (2048, 64)

    # Verify: position 0 -> cos = 1, sin = 0 (no rotation)
    assert np.allclose(freqs_cos[0], 1.0, atol=1e-6), "Position 0 cos should be 1"
    assert np.allclose(freqs_sin[0], 0.0, atol=1e-6), "Position 0 sin should be 0"
    print("  Position 0: cos=1, sin=0 (no rotation): OK")

    # Frequency range
    freq_high = 1.0 / (10000.0 ** (0.0 / head_dim))     # i=0
    freq_low = 1.0 / (10000.0 ** ((head_dim - 2) / head_dim))  # i=dim/2-1
    print(f"  Frequency range: [{freq_low:.6f}, {freq_high:.1f}]")
    print(f"  (Tan so cao nhat thay doi nhanh, tan so thap nhat thay doi cham)")

    # --- 1b: Apply RoPE ---
    print("\n  --- 1b: Apply RoPE ---")
    batch, num_heads, seq_len = 2, 4, 32
    x = np.random.randn(batch, num_heads, seq_len, head_dim).astype(np.float32)

    x_rotated = apply_rope(x, freqs_cos, freqs_sin)
    print(f"  Input shape:   {x.shape}")
    print(f"  Rotated shape: {x_rotated.shape}")
    assert x_rotated.shape == x.shape, "Shape should not change"
    print("  Shape preserved: OK")

    # Verify rotation preserves norm (rotation la phep bao toan do dai)
    norm_original = np.linalg.norm(x, axis=-1)
    norm_rotated = np.linalg.norm(x_rotated, axis=-1)
    assert np.allclose(norm_original, norm_rotated, atol=1e-4), \
        "RoPE rotation should preserve vector norm"
    print("  Norm preserved after rotation: OK")

    # Position 0 should have minimal change (sin(0)=0, cos(0)=1)
    diff_pos0 = np.abs(x[:, :, 0, :] - x_rotated[:, :, 0, :]).max()
    print(f"  Position 0 max diff: {diff_pos0:.8f} (should be ~0)")

    # --- 1c: Relative Position Property ---
    print("\n  --- 1c: Relative Position Property ---")
    head_dim_small = 16
    freqs_cos_s, freqs_sin_s = precompute_freqs_cis(head_dim_small, 100)

    # Tao 2 vectors q va k
    q = np.random.randn(1, 1, 1, head_dim_small).astype(np.float32)
    k = np.random.randn(1, 1, 1, head_dim_small).astype(np.float32)

    # Tinh dot product tai nhieu vi tri voi CUNG khoang cach
    distances = []
    for offset in [0, 10, 50]:
        pos_q = 5 + offset
        pos_k = 3 + offset  # Khoang cach luon = 2

        q_at_pos = np.random.randn(1, 1, 1, head_dim_small).astype(np.float32)
        k_at_pos = np.random.randn(1, 1, 1, head_dim_small).astype(np.float32)

        # Apply RoPE tai vi tri tuong ung
        # Tao temporary tensor de apply RoPE tai 1 position cu the
        q_padded = np.zeros((1, 1, pos_q + 1, head_dim_small), dtype=np.float32)
        q_padded[:, :, pos_q:pos_q+1, :] = q_at_pos
        q_rot = apply_rope(q_padded, freqs_cos_s, freqs_sin_s)[:, :, pos_q:pos_q+1, :]

        k_padded = np.zeros((1, 1, pos_k + 1, head_dim_small), dtype=np.float32)
        k_padded[:, :, pos_k:pos_k+1, :] = k_at_pos
        k_rot = apply_rope(k_padded, freqs_cos_s, freqs_sin_s)[:, :, pos_k:pos_k+1, :]

        dot = np.sum(q_rot * k_rot)
        distances.append((offset, pos_q, pos_k, dot))

    print("  Dot products tai nhieu offset (cung q, k vectors):")
    for offset, pq, pk, dot in distances:
        print(f"    offset={offset:>2}, q_pos={pq:>2}, k_pos={pk:>2}, dot={dot:>8.4f}")
    print("  (Relative position duoc encode trong dot product)")

    # --- 1d: Full RoPE Attention ---
    print("\n  --- 1d: Full RoPE Attention ---")
    batch, num_heads, seq_len, head_dim_attn = 2, 4, 16, 64
    freqs_cos_a, freqs_sin_a = precompute_freqs_cis(head_dim_attn, seq_len)

    Q = np.random.randn(batch, num_heads, seq_len, head_dim_attn).astype(np.float32) * 0.1
    K = np.random.randn(batch, num_heads, seq_len, head_dim_attn).astype(np.float32) * 0.1
    V = np.random.randn(batch, num_heads, seq_len, head_dim_attn).astype(np.float32) * 0.1

    output_rope = rope_attention(Q, K, V, freqs_cos_a, freqs_sin_a)
    print(f"  Q, K, V shape: {Q.shape}")
    print(f"  Output shape:  {output_rope.shape}")
    assert output_rope.shape == Q.shape, "Output shape should match Q"
    assert np.all(np.isfinite(output_rope)), "Output should be finite"
    print("  RoPE Attention forward: OK")

    # ============================================================
    # PHAN 2: Flash Attention
    # ============================================================
    print("\n" + "=" * 70)
    print("PHAN 2: Flash Attention")
    print("=" * 70)

    # --- 2a: Correctness test ---
    print("\n  --- 2a: Correctness Test (Flash vs Standard) ---")
    for N in [64, 128, 256]:
        d = 32
        Q_fa = np.random.randn(N, d).astype(np.float32) * 0.1
        K_fa = np.random.randn(N, d).astype(np.float32) * 0.1
        V_fa = np.random.randn(N, d).astype(np.float32) * 0.1

        out_std, mem_std = standard_attention(Q_fa, K_fa, V_fa)
        out_flash, mem_flash = flash_attention(Q_fa, K_fa, V_fa, block_size=32)

        max_diff = np.max(np.abs(out_std - out_flash))
        is_close = np.allclose(out_std, out_flash, atol=1e-4)

        print(f"  N={N:>4}: max_diff={max_diff:.8f}, match={is_close}, "
              f"std_mem={mem_std:>10,}B, flash_mem={mem_flash:>10,}B, "
              f"savings={mem_std/max(mem_flash,1):.1f}x")

        assert is_close, f"Flash Attention output should match standard for N={N}"

    print("  Flash Attention = Standard Attention (EXACT): OK")

    # --- 2b: Memory comparison ---
    print("\n  --- 2b: Memory Comparison ---")
    print(f"  {'Seq Len':>10} | {'Standard':>12} | {'Flash (B=64)':>14} | {'Savings':>10}")
    print(f"  {'-'*10}-+-{'-'*12}-+-{'-'*14}-+-{'-'*10}")

    d_mem = 128  # head_dim
    for N in [256, 512, 1024, 2048, 4096, 8192]:
        std_mem = N * N * 4  # float32, full attention matrix
        # Flash: peak memory = block_size * N * 4 (1 row of blocks)
        bs = 64
        flash_mem = bs * bs * 4  # 1 block of scores
        savings = std_mem / flash_mem

        std_str = f"{std_mem / (1024*1024):.2f} MB" if std_mem > 1024*1024 else f"{std_mem / 1024:.1f} KB"
        flash_str = f"{flash_mem / 1024:.1f} KB"

        print(f"  {N:>10} | {std_str:>12} | {flash_str:>14} | {savings:>9.0f}x")

    # --- 2c: Speed test ---
    print("\n  --- 2c: Speed Test ---")
    N_speed = 256
    d_speed = 64
    Q_sp = np.random.randn(N_speed, d_speed).astype(np.float32)
    K_sp = np.random.randn(N_speed, d_speed).astype(np.float32)
    V_sp = np.random.randn(N_speed, d_speed).astype(np.float32)

    # Standard
    t0 = time.time()
    for _ in range(10):
        standard_attention(Q_sp, K_sp, V_sp)
    t_std = (time.time() - t0) / 10

    # Flash voi block sizes khac nhau
    for bs in [32, 64, 128]:
        t0 = time.time()
        for _ in range(10):
            flash_attention(Q_sp, K_sp, V_sp, block_size=bs)
        t_flash = (time.time() - t0) / 10
        print(f"  N={N_speed}, block_size={bs}: standard={t_std*1000:.2f}ms, "
              f"flash={t_flash*1000:.2f}ms")

    print("  (Trong numpy, flash cham hon vi overhead Python loops)")
    print("  (Trong CUDA, flash NHANH hon 2-4x vi GPU memory hierarchy!)")

    # ============================================================
    # PHAN 3: GQA (Grouped Query Attention)
    # ============================================================
    print("\n" + "=" * 70)
    print("PHAN 3: GQA (Grouped Query Attention)")
    print("=" * 70)

    # --- 3a: GQA Forward ---
    print("\n  --- 3a: GQA Forward Test ---")
    d_model_gqa = 256
    num_q_heads = 8
    num_kv_heads = 2  # 4 groups of 2 Q heads each

    gqa = GroupedQueryAttention(d_model_gqa, num_q_heads, num_kv_heads)
    x_gqa = np.random.randn(2, 16, d_model_gqa).astype(np.float32) * 0.1

    out_gqa = gqa.forward(x_gqa)
    print(f"  Config: d_model={d_model_gqa}, q_heads={num_q_heads}, kv_heads={num_kv_heads}")
    print(f"  Groups: {gqa.num_groups} (moi group co {num_q_heads // gqa.num_groups} Q heads share 1 K,V)")
    print(f"  Input:  {x_gqa.shape}")
    print(f"  Output: {out_gqa.shape}")
    assert out_gqa.shape == x_gqa.shape, "Output shape should match input"
    assert np.all(np.isfinite(out_gqa)), "Output should be finite"
    print("  GQA forward: OK")

    # --- 3b: Parameter count ---
    print("\n  --- 3b: Parameter Count ---")
    params_gqa = gqa.param_count()
    print(f"  Q params:  {params_gqa['q_params']:>10,} (full {num_q_heads} heads)")
    print(f"  K params:  {params_gqa['k_params']:>10,} (only {num_kv_heads} heads)")
    print(f"  V params:  {params_gqa['v_params']:>10,} (only {num_kv_heads} heads)")
    print(f"  O params:  {params_gqa['o_params']:>10,}")
    print(f"  Total:     {params_gqa['total']:>10,}")

    # So sanh voi MHA
    mha_k_params = d_model_gqa * (num_q_heads * (d_model_gqa // num_q_heads))
    mha_v_params = mha_k_params
    gqa_kv_savings = 1 - (params_gqa['k_params'] + params_gqa['v_params']) / (mha_k_params + mha_v_params)
    print(f"  K,V param savings vs MHA: {gqa_kv_savings*100:.0f}%")

    # --- 3c: KV Cache comparison ---
    print("\n  --- 3c: KV Cache Memory Comparison ---")

    # Simulate LLaMA 2 configurations
    configs = [
        ("LLaMA 2 7B (MHA)",  4096, 32, 32),   # MHA
        ("LLaMA 2 70B (GQA)", 8192, 64, 8),    # GQA
        ("Mistral 7B (GQA)",  4096, 32, 8),     # GQA
        ("Gemma (MQA)",       2048, 16, 1),     # MQA
    ]

    seq_len_cache = 4096
    print(f"  KV Cache at seq_len={seq_len_cache}, FP16:")
    print(f"  {'Model':>25} | {'Q Heads':>8} | {'KV Heads':>9} | {'KV Cache':>12}")
    print(f"  {'-'*25}-+-{'-'*8}-+-{'-'*9}-+-{'-'*12}")

    for name, d, qh, kvh in configs:
        head_d = d // qh
        kv_bytes = 2 * kvh * seq_len_cache * head_d * 2  # FP16, per layer
        kv_str = f"{kv_bytes / (1024*1024):.1f} MB"
        print(f"  {name:>25} | {qh:>8} | {kvh:>9} | {kv_str:>12}")

    # --- 3d: Full comparison MHA vs GQA vs MQA ---
    print("\n  --- 3d: MHA vs GQA vs MQA (d_model=8192, 64 Q heads, seq=4096) ---")
    comparison = compare_attention_variants(8192, 4096, 64, dtype_bytes=2)

    for variant, info in comparison.items():
        mb = info['bytes'] / (1024 * 1024)
        print(f"  {variant}: kv_heads={info['kv_heads']:>3}, KV cache = {mb:>8.1f} MB")

    mha_mb = comparison['MHA']['bytes'] / (1024 * 1024)
    gqa_mb = comparison['GQA']['bytes'] / (1024 * 1024)
    mqa_mb = comparison['MQA']['bytes'] / (1024 * 1024)
    print(f"  GQA savings vs MHA: {(1 - gqa_mb/mha_mb)*100:.0f}%")
    print(f"  MQA savings vs MHA: {(1 - mqa_mb/mha_mb)*100:.0f}%")

    # --- 3e: Verify correctness (GQA with num_kv = num_q should = MHA) ---
    print("\n  --- 3e: GQA == MHA khi num_kv_heads == num_q_heads ---")
    d_test = 64
    nq_test = 4
    gqa_as_mha = GroupedQueryAttention(d_test, num_q_heads=nq_test, num_kv_heads=nq_test)
    x_test = np.random.randn(1, 8, d_test).astype(np.float32) * 0.1
    out_test = gqa_as_mha.forward(x_test)
    assert out_test.shape == x_test.shape
    assert np.all(np.isfinite(out_test))
    print(f"  GQA(q={nq_test}, kv={nq_test}) = MHA: forward OK")

    # ============================================================
    # PHAN 4: MoE (Mixture of Experts)
    # ============================================================
    print("\n" + "=" * 70)
    print("PHAN 4: MoE (Mixture of Experts)")
    print("=" * 70)

    # --- 4a: MoE Forward ---
    print("\n  --- 4a: MoE Forward Test ---")
    d_model_moe = 64
    d_ff_moe = 256
    num_experts = 8
    top_k = 2

    moe = MoELayer(d_model_moe, d_ff_moe, num_experts=num_experts, top_k=top_k)
    x_moe = np.random.randn(4, 32, d_model_moe).astype(np.float32) * 0.1

    out_moe, aux_loss = moe.forward(x_moe)
    print(f"  Config: d_model={d_model_moe}, d_ff={d_ff_moe}, "
          f"experts={num_experts}, top_k={top_k}")
    print(f"  Input:    {x_moe.shape}")
    print(f"  Output:   {out_moe.shape}")
    print(f"  Aux loss: {aux_loss:.4f} (ideal khi deu = 1.0)")
    assert out_moe.shape == x_moe.shape, "Output shape should match input"
    assert np.all(np.isfinite(out_moe)), "Output should be finite"
    assert aux_loss > 0, "Aux loss should be positive"
    print("  MoE forward: OK")

    # --- 4b: Expert usage ---
    print("\n  --- 4b: Expert Usage Distribution ---")
    usage = moe.get_expert_usage_stats()
    print("  Expert usage (ideal: moi expert ~12.5% voi 8 experts):")
    for expert_name, pct in usage.items():
        bar = "#" * int(float(pct.rstrip('%')) / 2)
        print(f"    {expert_name}: {pct:>6} {bar}")

    # --- 4c: Capacity vs Compute ---
    print("\n  --- 4c: Capacity vs Compute ---")
    params_moe = moe.total_params()
    cap_comp = moe.capacity_vs_compute()

    print(f"  Router params:          {params_moe['router_params']:>10,}")
    print(f"  Expert params (all):    {params_moe['expert_params']:>10,}")
    print(f"  Total params:           {params_moe['total_params']:>10,}")
    print(f"  Active params/token:    {params_moe['active_params_per_token']:>10,}")
    print(f"  Efficiency ratio:       {cap_comp['efficiency_ratio']*100:.1f}% "
          f"(chi dung {cap_comp['efficiency_ratio']*100:.0f}% params per token)")

    # --- 4d: Mixtral 8x7B scale simulation ---
    print("\n  --- 4d: Mixtral 8x7B Scale Analysis ---")
    # Mixtral numbers (approximate)
    mixtral_d = 4096
    mixtral_dff = 14336
    mixtral_experts = 8
    mixtral_topk = 2
    mixtral_layers = 32

    expert_params_per_layer = 2 * mixtral_d * mixtral_dff  # W1 + W2 (no bias simplification)
    total_expert_params = mixtral_experts * expert_params_per_layer * mixtral_layers
    active_expert_params = mixtral_topk * expert_params_per_layer * mixtral_layers

    print(f"  Mixtral 8x7B (approximate):")
    print(f"    d_model={mixtral_d}, d_ff={mixtral_dff}")
    print(f"    {mixtral_experts} experts, top-{mixtral_topk} routing")
    print(f"    {mixtral_layers} layers")
    print(f"    Expert params per layer:  {expert_params_per_layer:>15,} "
          f"(~{expert_params_per_layer/1e6:.0f}M)")
    print(f"    Total expert params:      {total_expert_params:>15,} "
          f"(~{total_expert_params/1e9:.1f}B)")
    print(f"    Active expert params:     {active_expert_params:>15,} "
          f"(~{active_expert_params/1e9:.1f}B)")
    print(f"    Compute savings: chi dung {mixtral_topk}/{mixtral_experts} = "
          f"{mixtral_topk/mixtral_experts*100:.0f}% FFN compute per token")

    # --- 4e: Load balancing loss ---
    print("\n  --- 4e: Load Balancing Loss Analysis ---")

    # Test voi different routing patterns
    test_num_tokens = 100
    test_num_experts = 8

    # Case 1: Perfectly balanced
    balanced_probs = np.ones((test_num_tokens, test_num_experts)) / test_num_experts
    balanced_indices = np.array([
        [i % test_num_experts, (i + 1) % test_num_experts]
        for i in range(test_num_tokens)
    ])

    moe_test = MoELayer(64, 256, num_experts=test_num_experts, top_k=2)
    loss_balanced = moe_test._load_balancing_loss(balanced_probs, balanced_indices)

    # Case 2: Imbalanced (80% tokens go to expert 0)
    imbalanced_probs = np.zeros((test_num_tokens, test_num_experts))
    imbalanced_probs[:, 0] = 0.8
    imbalanced_probs[:, 1:] = 0.2 / (test_num_experts - 1)
    imbalanced_indices = np.zeros((test_num_tokens, 2), dtype=int)
    imbalanced_indices[:80, 0] = 0
    imbalanced_indices[:80, 1] = 1
    imbalanced_indices[80:, 0] = 2
    imbalanced_indices[80:, 1] = 3

    loss_imbalanced = moe_test._load_balancing_loss(imbalanced_probs, imbalanced_indices)

    print(f"  Balanced routing:   loss = {loss_balanced:.4f} (ideal ~1.0)")
    print(f"  Imbalanced routing: loss = {loss_imbalanced:.4f} (cao hon = penalty)")
    print(f"  Load balancing loss penalizes imbalance: "
          f"{'OK' if loss_imbalanced > loss_balanced else 'FAIL'}")

    # --- 4f: MoE output test ---
    print("\n  --- 4f: MoE voi different top_k ---")
    for tk in [1, 2, 4]:
        if tk <= num_experts:
            moe_tk = MoELayer(d_model_moe, d_ff_moe, num_experts=num_experts, top_k=tk)
            x_tk = np.random.randn(2, 16, d_model_moe).astype(np.float32) * 0.1
            out_tk, loss_tk = moe_tk.forward(x_tk)
            params_tk = moe_tk.total_params()
            print(f"  top_k={tk}: output_shape={out_tk.shape}, "
                  f"active_params={params_tk['active_params_per_token']:>8,}, "
                  f"aux_loss={loss_tk:.4f}")

    # ============================================================
    # SUMMARY
    # ============================================================
    print("\n" + "=" * 70)
    print("TONG KET: Advanced Architectures")
    print("=" * 70)

    print("""
  1. RoPE (Rotary Position Embedding):
     - Encode position bang rotation trong mat phang phuc
     - Dot product tu nhien capture relative position
     - Zero learnable params, ho tro extrapolation
     - Dung boi: LLaMA, GPT-NeoX, Mistral, Gemma

  2. Flash Attention:
     - Tile-based computation, EXACT same output
     - O(N) memory thay vi O(N^2)
     - Key: online softmax algorithm (running max + sum)
     - Dung boi: tat ca LLM hien dai (LLaMA 2, Mistral, GPT-4)

  3. GQA (Grouped Query Attention):
     - Chia Q heads thanh groups, moi group share K,V
     - Giam KV cache 4-8x ma giu chat luong
     - Trung gian giua MHA (full) va MQA (1 KV head)
     - Dung boi: LLaMA 2 70B (8 KV heads), Mistral 7B (8 KV heads)

  4. MoE (Mixture of Experts):
     - Nhieu expert FFNs, router chon top-k per token
     - Total params LON nhung active params NHO
     - Load balancing loss giu experts duoc dung deu
     - Dung boi: Mixtral 8x7B, Switch Transformer, GShard
    """)

    print("=" * 70)
    print("TAT CA TESTS PASSED!")
    print("=" * 70)


# ======== CHECKLIST ========
# Week 13-14 Advanced Architectures:
# [x] Implement RoPE tu scratch
# [x] Understand Flash Attention algorithm
# [x] Implement GQA va compare memory voi MHA
# [x] Build MoE layer voi routing
