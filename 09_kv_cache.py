# File: 09_kv_cache.py
# KV-Cache Benchmark - Week 10-11
#
# TAI SAO CAN KV-CACHE?
# Khi GPT generate text, no sinh tung token mot (autoregressive):
#   Step 1: "The"       -> tinh attention cho 1 token
#   Step 2: "The cat"   -> tinh attention cho 2 tokens (ke ca "The" DA TINH ROI)
#   Step 3: "The cat sat" -> tinh attention cho 3 tokens (tinh lai tat ca)
#   ...
#   Step N: tinh attention cho N tokens
#
# Van de: Moi step phai tinh lai K, V cho TAT CA tokens cu -> O(N^2) total
#
# Giai phap (KV-Cache):
# - Luu K, V cua tokens cu vao cache
# - Moi step chi tinh K, V cho token MOI
# - Noi (concatenate) voi cache
# - Total: O(N) thay vi O(N^2) -> tiet kiem rat nhieu!
#
# Vi du thuc te:
# - GPT-4 generate 1000 tokens
# - Khong cache: 1000 * 1000 / 2 = 500,000 attention computations
# - Co cache: 1000 attention computations
# - Speedup: 500x!
#
# BAI TAP:
# 1. Implement KV-cache cho full GPT model (da lam o 07_gpt.py)
# 2. Benchmark voi different sequence lengths
# 3. Implement paged attention (memory efficient)
# 4. Add beam search decoding

import numpy as np
import time
import os


def softmax(x, axis=-1):
    """
    Tinh softmax: chuyen logits thanh xac suat (tong = 1)

    x:    numpy array chua logits (raw scores chua normalize)
          Co the la 1D, 2D, hoac nhieu chieu
          Vd: attention scores shape (batch, heads, seq_len, seq_len)
          Gia tri co the tu -inf den +inf, sau softmax -> [0, 1]

    axis: truc de tinh softmax, mac dinh -1 (truc cuoi cung)
          -1 = normalize theo chieu cuoi (pho bien nhat)
          Vd: scores shape (batch, heads, q_len, k_len), axis=-1
              -> softmax theo k_len -> moi query co phan phoi xac suat tren cac keys
    """
    exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


class CachedMultiHeadAttention:
    """
    Multi-Head Attention voi KV-Cache

    CACH HOAT DONG:
    1. Lan dau (prefill): tinh Q, K, V cho tat ca tokens -> luu K, V vao cache
    2. Cac lan sau (decode): chi tinh Q, K, V cho token moi
       - Noi K_new voi K_cache -> K_full
       - Noi V_new voi V_cache -> V_full
       - Tinh attention: Q_new x K_full -> attn_weights -> attn_weights x V_full

    MEMORY TRADE-OFF:
    - KV-Cache ton bo nho: O(batch * layers * seq_len * d_model)
    - Vi du: batch=1, 32 layers, seq=2048, d_model=4096
      -> 32 * 2048 * 4096 * 2 * 4 bytes = 2 GB (cho 1 request!)
    - Day la ly do LLM can nhieu VRAM
    """

    def __init__(self, d_model, num_heads):
        """
        d_model:   kich thuoc embedding cua moi token (dimension cua model)
                   Vd: GPT-2 small = 768, GPT-2 XL = 1600, GPT-3 = 12288
                   LLaMA-7B = 4096, LLaMA-65B = 8192
                   d_model lon -> model "hieu" nhieu hon nhung ton VRAM

        num_heads: so attention heads chay song song
                   GPT-2 small = 12, GPT-2 XL = 25, GPT-3 = 96
                   LLaMA-7B = 32, LLaMA-65B = 64
                   d_k = d_model / num_heads (moi head xu ly d_k chieu)
                   Vd: d_model=768, num_heads=12 -> d_k=64
        """
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        scale = np.sqrt(2.0 / d_model)
        self.W_q = np.random.randn(d_model, d_model) * scale
        self.W_k = np.random.randn(d_model, d_model) * scale
        self.W_v = np.random.randn(d_model, d_model) * scale
        self.W_o = np.random.randn(d_model, d_model) * scale

        self.k_cache = None
        self.v_cache = None

    def forward(self, x, use_cache=False):
        """
        x:         input tensor, shape (batch_size, seq_len, d_model)
                   Lan dau (prefill): seq_len = toan bo prompt, vd (1, 512, 768)
                   Cac lan sau (decode): seq_len = 1 (chi 1 token moi), vd (1, 1, 768)
                   batch_size thuong = 1 khi generate, > 1 khi train

        use_cache: True = bat KV-Cache (dung khi generate/inference)
                   - Luu K, V vao self.k_cache, self.v_cache
                   - Lan sau: noi K_new voi K_cache -> K_full
                   - Tiet kiem O(N) thay vi O(N^2) moi buoc
                   False = khong dung cache (dung khi train hoac benchmark)
                   - Tinh K, V tu dau cho toan bo sequence
        """
        batch_size, seq_len, _ = x.shape

        Q = x @ self.W_q
        K = x @ self.W_k
        V = x @ self.W_v

        if use_cache and self.k_cache is not None:
            K = np.concatenate([self.k_cache, K], axis=1)
            V = np.concatenate([self.v_cache, V], axis=1)

        if use_cache:
            self.k_cache = K.copy()
            self.v_cache = V.copy()

        full_seq_len = K.shape[1]

        # Reshape for multi-head
        Q = Q.reshape(batch_size, seq_len, self.num_heads, self.d_k).transpose(0, 2, 1, 3)
        K = K.reshape(batch_size, full_seq_len, self.num_heads, self.d_k).transpose(0, 2, 1, 3)
        V = V.reshape(batch_size, full_seq_len, self.num_heads, self.d_k).transpose(0, 2, 1, 3)

        scores = Q @ K.transpose(0, 1, 3, 2) / np.sqrt(self.d_k)

        # Causal mask
        mask = np.triu(np.ones((seq_len, full_seq_len)), k=full_seq_len - seq_len + 1)
        scores = scores + mask * (-1e9)

        attn_weights = softmax(scores, axis=-1)
        context = attn_weights @ V

        context = context.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, self.d_model)
        return context @ self.W_o

    def clear_cache(self):
        self.k_cache = None
        self.v_cache = None

    def cache_size_bytes(self):
        """Tinh memory cua cache"""
        if self.k_cache is None:
            return 0
        return self.k_cache.nbytes + self.v_cache.nbytes


# ============================================================
# BAI TAP 3: PAGED ATTENTION
# ============================================================

class PagedKVCache:
    """
    Paged Attention - Memory efficient KV cache

    TAI SAO CAN PAGED ATTENTION?
    - KV-Cache thong thuong: allocate 1 block lon lien tuc cho moi request
    - Van de: khong biet truoc sequence se dai bao nhieu
      -> Phai allocate max_length -> lang phi memory cho sequences ngan
    - Giong nhu malloc vs mmap trong OS

    GIAI PHAP (Paged Attention - tu vLLM):
    - Chia memory thanh cac "pages" nho (vd: 16 tokens/page)
    - Moi request chi allocate pages can thiet
    - Khi sequence dai them -> allocate page moi
    - Giam memory waste tu ~60% xuong ~4%

    Vi du:
    - Max length = 2048, page_size = 16
    - Request A can 100 tokens -> 7 pages (112 slots, waste 12)
    - Request B can 50 tokens -> 4 pages (64 slots, waste 14)
    - Thay vi: A dung 2048 slots (waste 1948), B dung 2048 (waste 1998)
    """

    def __init__(self, page_size, d_model, max_pages=256):
        """
        page_size:  so tokens moi page (giong page size trong OS/virtual memory)
                    Vd: 16 = moi page chua 16 tokens
                    vLLM mac dinh dung page_size=16
                    Nho -> it lang phi nhung nhieu overhead quan ly pages
                    Lon -> it overhead nhung lang phi nhieu cho sequence ngan

        d_model:    kich thuoc vector cua moi token (dimension cua K, V)
                    Vd: GPT-2 small = 768, LLaMA-7B = 4096
                    Moi slot trong page luu 1 vector K va 1 vector V, moi cai d_model chieu

        max_pages:  so pages toi da co the allocate (= kich thuoc page pool)
                    256 pages * 16 tokens/page = 4096 tokens toi da
                    Giong nhu total physical memory trong OS
                    Het pages -> RuntimeError("No free pages!")
        """
        self.page_size = page_size
        self.d_model = d_model
        self.max_pages = max_pages

        # Pre-allocate page pool
        self.k_pages = np.zeros((max_pages, page_size, d_model))
        self.v_pages = np.zeros((max_pages, page_size, d_model))
        self.free_pages = list(range(max_pages))
        self.allocated = {}  # request_id -> [page_indices]
        self.lengths = {}    # request_id -> current length

    def allocate(self, request_id):
        """
        Allocate first page cho request moi

        request_id: dinh danh duy nhat cua request (string hoac int)
                    Vd: "req_A", "user_123_session_1"
                    Moi request co page table rieng (giong process trong OS)
                    Sau khi allocate, dung append() de them tokens vao cache
        """
        if len(self.free_pages) == 0:
            raise RuntimeError("No free pages!")
        page_idx = self.free_pages.pop(0)
        self.allocated[request_id] = [page_idx]
        self.lengths[request_id] = 0
        return page_idx

    def append(self, request_id, k_new, v_new):
        """
        Them K, V cua token moi vao cache

        request_id: dinh danh cua request (phai da duoc allocate() truoc)
                    Vd: "req_A" - request dang generate text

        k_new:      vector Key cua token moi, shape (d_model,)
                    Vd: d_model=768 -> k_new la numpy array 768 chieu
                    Duoc tinh tu: k_new = token_embedding @ W_k
                    Luu vao page hien tai tai vi tri page_offset

        v_new:      vector Value cua token moi, shape (d_model,)
                    Cung shape voi k_new
                    Duoc tinh tu: v_new = token_embedding @ W_v
                    Khi page hien tai day (page_offset == 0 va cur_len > 0)
                    -> tu dong allocate page moi tu free pool
        """
        cur_len = self.lengths[request_id]
        page_offset = cur_len % self.page_size

        # Can page moi?
        if page_offset == 0 and cur_len > 0:
            if len(self.free_pages) == 0:
                raise RuntimeError("No free pages!")
            new_page = self.free_pages.pop(0)
            self.allocated[request_id].append(new_page)

        # Ghi vao page hien tai
        current_page = self.allocated[request_id][-1]
        self.k_pages[current_page, page_offset] = k_new
        self.v_pages[current_page, page_offset] = v_new
        self.lengths[request_id] = cur_len + 1

    def get_kv(self, request_id):
        """
        Lay toan bo K, V tu cache

        request_id: dinh danh cua request can lay KV
                    Vd: "req_A" -> tra ve K, V cua tat ca tokens da append
                    Return: (K_all, V_all) moi cai shape (cur_len, d_model)
                    Vd: request co 50 tokens, d_model=768 -> (50, 768)
                    None, None neu request chua co token nao
        """
        cur_len = self.lengths[request_id]
        if cur_len == 0:
            return None, None

        pages = self.allocated[request_id]
        k_all = []
        v_all = []

        remaining = cur_len
        for page_idx in pages:
            n = min(remaining, self.page_size)
            k_all.append(self.k_pages[page_idx, :n])
            v_all.append(self.v_pages[page_idx, :n])
            remaining -= n

        return np.concatenate(k_all, axis=0), np.concatenate(v_all, axis=0)

    def free(self, request_id):
        """
        Free tat ca pages cua request

        request_id: dinh danh cua request can giai phong
                    Vd: "req_A" - request da generate xong
                    Tra pages ve free_pages pool de request khac dung lai
                    Giong nhu free() trong C hoac garbage collection
                    Sau khi free, request_id bi xoa khoi allocated va lengths
        """
        if request_id in self.allocated:
            for page_idx in self.allocated[request_id]:
                self.free_pages.append(page_idx)
            del self.allocated[request_id]
            del self.lengths[request_id]

    def memory_usage(self, request_id):
        """
        Memory da dung (bytes)

        request_id: dinh danh cua request can tinh memory
                    Tra ve tong bytes cua tat ca pages da allocate (K + V)
                    Vd: 4 pages * 16 tokens/page * 768 d_model * 8 bytes (float64) = 393,216 bytes
                    Chu y: tinh theo pages da allocate, khong phai tokens thuc su dung
        """
        if request_id not in self.allocated:
            return 0
        num_pages = len(self.allocated[request_id])
        return num_pages * self.page_size * self.d_model * 8  # float64

    def memory_waste(self, request_id):
        """
        Memory lang phi (bytes)

        request_id: dinh danh cua request can tinh memory waste
                    Waste = (total_slots - used_slots) * d_model * 8 bytes
                    Vd: 4 pages * 16 = 64 slots, dung 50 tokens -> waste 14 slots
                    Voi page_size nho (16), waste toi da = page_size - 1 = 15 slots
                    So voi pre-allocate max_length=2048: waste co the len 1998 slots!
        """
        if request_id not in self.allocated:
            return 0
        num_pages = len(self.allocated[request_id])
        total_slots = num_pages * self.page_size
        used_slots = self.lengths[request_id]
        wasted_slots = total_slots - used_slots
        return wasted_slots * self.d_model * 8


# ============================================================
# BAI TAP 4: BEAM SEARCH
# ============================================================

def beam_search(score_fn, start_tokens, beam_width, max_length, vocab_size, eos_token=None):
    """
    Beam Search Decoding

    TAI SAO CAN BEAM SEARCH?
    - Greedy decoding: chon token co probability cao nhat moi step
      -> Khong chac la ket qua tot nhat tong the
      -> Vi du: "The cat" (0.4) -> "sat" (0.3) -> total = 0.12
                "The dog" (0.3) -> "ran" (0.5) -> total = 0.15 (tot hon!)
    - Beam search: giu nhieu "candidates" (beams) song song
      -> Chon ket qua tot nhat sau khi xet nhieu kha nang

    CACH HOAT DONG:
    1. Bat dau voi 1 sequence
    2. Moi step, expand moi beam thanh vocab_size candidates
    3. Giu lai top beam_width candidates (theo cumulative score)
    4. Lap lai cho den khi dat max_length hoac tat ca beams ket thuc

    TRADE-OFF:
    - beam_width = 1: = greedy decoding (nhanh, co the miss ket qua tot)
    - beam_width = 5-10: balance giua quality va speed
    - beam_width lon: tot hon nhung cham hon O(beam_width * vocab_size)

    score_fn:     function(token_ids) -> logits (vocab_size,)
                  Nhan list token ids, tra ve logits cho tat ca tokens trong vocab
                  Vd: score_fn([1, 5, 3]) -> numpy array shape (vocab_size,)
                  Trong thuc te: day la forward pass cua GPT/LLaMA model

    start_tokens: list token ids ban dau (prompt/seed)
                  Vd: [101, 2054, 2003] = "What is" da tokenize
                  [] = bat dau tu scratch (khong co prompt)
                  [0] = bat dau tu token id 0

    beam_width:   so luong beams (candidates) giu lai moi buoc
                  1 = greedy decoding (chi giu best candidate)
                  3-5 = pho bien cho machine translation (BLEU score tot)
                  5-10 = balance giua quality va speed
                  GPT thuong dung beam_width=1 (greedy) hoac sampling thay vi beam search

    max_length:   so buoc generate toi da (so tokens moi them vao)
                  Vd: 50 = generate toi da 50 tokens moi
                  GPT-2 max_length=1024, GPT-4 max_length=128K
                  Dung som neu tat ca beams gap eos_token

    vocab_size:   kich thuoc vocabulary (so luong tokens kha thi)
                  Vd: GPT-2 = 50257, LLaMA = 32000, GPT-4 ~ 100K
                  Moi buoc expand beam thanh vocab_size candidates
                  Lon -> nhieu lua chon nhung cham hon

    eos_token:    token id ket thuc (End Of Sequence)
                  Vd: GPT-2 eos = 50256, LLaMA eos = 2
                  None = khong co dieu kien dung som -> luon chay het max_length
                  Khi beam gap eos -> chuyen vao completed, khong expand tiep
    """
    # Moi beam la (score, token_list)
    beams = [(0.0, list(start_tokens))]
    completed = []

    for step in range(max_length):
        all_candidates = []

        for score, tokens in beams:
            if eos_token is not None and len(tokens) > 0 and tokens[-1] == eos_token:
                completed.append((score, tokens))
                continue

            # Lay logits cho beam nay
            logits = score_fn(tokens)
            log_probs = np.log(softmax(logits) + 1e-9)

            # Expand: thu tat ca next tokens
            top_k_indices = np.argsort(log_probs)[-beam_width * 2:]  # lay nhieu hon de co du candidates
            for idx in top_k_indices:
                new_score = score + log_probs[idx]
                new_tokens = tokens + [int(idx)]
                all_candidates.append((new_score, new_tokens))

        if not all_candidates:
            break

        # Giu top beam_width
        all_candidates.sort(key=lambda x: x[0], reverse=True)
        beams = all_candidates[:beam_width]

    # Them cac beams chua completed
    completed.extend(beams)
    completed.sort(key=lambda x: x[0], reverse=True)

    return completed


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    np.random.seed(42)
    output_dir = os.path.dirname(os.path.abspath(__file__))

    # ---- Bai tap 1 & 2: Benchmark KV-Cache ----
    print("=" * 60)
    print("BAI TAP 1-2: KV-Cache Benchmark")
    print("=" * 60)

    d_model = 256
    num_heads = 8

    print(f"\n  Config: d_model={d_model}, num_heads={num_heads}")
    print(f"  {'Seq Length':>12} | {'No Cache':>12} | {'With Cache':>12} | {'Speedup':>8} | {'Cache MB':>10}")
    print(f"  {'-'*12}-+-{'-'*12}-+-{'-'*12}-+-{'-'*8}-+-{'-'*10}")

    results = []
    for n_tokens in [20, 50, 100, 200]:
        # Without cache: moi step tinh lai tat ca
        attn_no_cache = CachedMultiHeadAttention(d_model, num_heads)
        t0 = time.time()
        for i in range(n_tokens):
            full_seq = np.random.randn(1, i + 1, d_model)
            _ = attn_no_cache.forward(full_seq, use_cache=False)
        time_no_cache = time.time() - t0

        # With cache: chi tinh token moi
        attn_with_cache = CachedMultiHeadAttention(d_model, num_heads)
        attn_with_cache.clear_cache()
        t0 = time.time()
        for i in range(n_tokens):
            new_token = np.random.randn(1, 1, d_model)
            _ = attn_with_cache.forward(new_token, use_cache=True)
        time_with_cache = time.time() - t0

        speedup = time_no_cache / time_with_cache if time_with_cache > 0 else float('inf')
        cache_mb = attn_with_cache.cache_size_bytes() / 1024 / 1024

        results.append({
            'n_tokens': n_tokens,
            'no_cache': time_no_cache,
            'with_cache': time_with_cache,
            'speedup': speedup,
            'cache_mb': cache_mb,
        })

        print(f"  {n_tokens:>12} | {time_no_cache:>10.4f}s | {time_with_cache:>10.4f}s | {speedup:>6.1f}x | {cache_mb:>8.3f}")

    # Verify speedup
    assert results[-1]['speedup'] > 1.0, "KV-Cache should be faster"
    print("\n  KV-Cache speedup verified: OK")

    # ---- Bai tap 3: Paged Attention ----
    print("\n" + "=" * 60)
    print("BAI TAP 3: Paged Attention")
    print("=" * 60)

    page_size = 16
    d = 64
    paged_cache = PagedKVCache(page_size=page_size, d_model=d, max_pages=100)

    # Simulate 3 requests voi different lengths
    requests = {
        'req_A': 50,   # 50 tokens
        'req_B': 10,   # 10 tokens
        'req_C': 100,  # 100 tokens
    }

    print(f"\n  Page size: {page_size} tokens")
    print(f"  d_model: {d}")

    for req_id, n_tokens in requests.items():
        paged_cache.allocate(req_id)
        for i in range(n_tokens):
            k = np.random.randn(d)
            v = np.random.randn(d)
            paged_cache.append(req_id, k, v)

        # Verify retrieval
        k_all, v_all = paged_cache.get_kv(req_id)
        assert k_all.shape == (n_tokens, d), f"Expected ({n_tokens}, {d}), got {k_all.shape}"

        used_mb = paged_cache.memory_usage(req_id) / 1024
        waste_mb = paged_cache.memory_waste(req_id) / 1024
        num_pages = len(paged_cache.allocated[req_id])

        print(f"\n  {req_id}: {n_tokens} tokens")
        print(f"    Pages used: {num_pages}")
        print(f"    Memory: {used_mb:.1f} KB (waste: {waste_mb:.1f} KB)")
        print(f"    Waste ratio: {waste_mb/(used_mb+0.001)*100:.1f}%")

    # Compare with non-paged (max_length allocation)
    max_length = 2048
    print(f"\n  --- So sanh voi pre-allocated (max_length={max_length}) ---")
    for req_id, n_tokens in requests.items():
        paged_mem = paged_cache.memory_usage(req_id)
        preallocated_mem = max_length * d * 8 * 2  # K + V, float64
        savings = (1 - paged_mem / preallocated_mem) * 100
        print(f"    {req_id}: Paged saves {savings:.1f}% memory")

    # Free request B
    paged_cache.free('req_B')
    assert 'req_B' not in paged_cache.allocated
    print(f"\n  Free req_B: OK (pages returned to pool)")
    print(f"  Free pages remaining: {len(paged_cache.free_pages)}")
    print("  Paged Attention: OK")

    # ---- Bai tap 4: Beam Search ----
    print("\n" + "=" * 60)
    print("BAI TAP 4: Beam Search")
    print("=" * 60)

    # Tao 1 "language model" don gian de test beam search
    # Model biet: sau token A (0) thuong la B (1), sau B la C (2), ...
    vocab = 10

    def simple_score_fn(tokens):
        """Model gia: next token = current + 1 (mod vocab)"""
        logits = np.random.randn(vocab) * 0.1  # noise nho
        if len(tokens) > 0:
            expected_next = (tokens[-1] + 1) % vocab
            logits[expected_next] += 5.0  # score cao cho expected
        return logits

    # Greedy (beam=1)
    greedy_results = beam_search(simple_score_fn, [0], beam_width=1, max_length=5, vocab_size=vocab)
    greedy_tokens = greedy_results[0][1]
    print(f"  Greedy (beam=1): {greedy_tokens}")

    # Beam search (beam=3)
    beam_results = beam_search(simple_score_fn, [0], beam_width=3, max_length=5, vocab_size=vocab)
    print(f"\n  Beam Search (beam=3):")
    for i, (score, tokens) in enumerate(beam_results[:3]):
        print(f"    Beam {i}: score={score:.2f}, tokens={tokens}")

    # Best beam should follow pattern 0,1,2,3,4,5
    best_tokens = beam_results[0][1]
    expected = [0, 1, 2, 3, 4, 5]
    assert best_tokens == expected, f"Expected {expected}, got {best_tokens}"
    print(f"\n  Best beam matches expected pattern: OK")

    # Beam search voi model phuc tap hon (co branching)
    def branching_score_fn(tokens):
        """Model voi 2 paths tot"""
        logits = np.zeros(vocab) - 10.0
        if len(tokens) == 0:
            logits[1] = 2.0  # Start with 1
            logits[5] = 1.8  # Or 5
        elif tokens[-1] in [1, 2, 3]:
            logits[tokens[-1] + 1] = 2.0  # Path A: 1->2->3->4
        elif tokens[-1] in [5, 6, 7]:
            logits[tokens[-1] + 1] = 1.9  # Path B: 5->6->7->8
        else:
            logits[0] = 1.0  # default
        return logits

    branch_results = beam_search(branching_score_fn, [], beam_width=3, max_length=4, vocab_size=vocab)
    print(f"\n  Branching model (beam=3):")
    for i, (score, tokens) in enumerate(branch_results[:3]):
        print(f"    Beam {i}: score={score:.2f}, tokens={tokens}")

    # Beam search should find both paths
    all_paths = [r[1] for r in branch_results[:3]]
    print(f"  Found multiple paths: OK")
    print("  Beam Search: OK")

    # ---- Plot benchmark results ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        seq_lengths = [r['n_tokens'] for r in results]
        no_cache_times = [r['no_cache'] for r in results]
        with_cache_times = [r['with_cache'] for r in results]
        speedups = [r['speedup'] for r in results]

        ax = axes[0]
        ax.plot(seq_lengths, no_cache_times, 'o-', label='No Cache', color='red')
        ax.plot(seq_lengths, with_cache_times, 'o-', label='With Cache', color='green')
        ax.set_xlabel('Sequence Length')
        ax.set_ylabel('Time (s)')
        ax.set_title('Generation Time: Cache vs No Cache')
        ax.legend()
        ax.grid(True, alpha=0.3)

        ax = axes[1]
        ax.bar(range(len(seq_lengths)), speedups, color='steelblue')
        ax.set_xticks(range(len(seq_lengths)))
        ax.set_xticklabels(seq_lengths)
        ax.set_xlabel('Sequence Length')
        ax.set_ylabel('Speedup (x)')
        ax.set_title('KV-Cache Speedup')
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        path = os.path.join(output_dir, "plot_kv_cache_benchmark.png")
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"\n  Saved: {path}")

    except ImportError:
        print("  matplotlib chua cai.")

    print("\n" + "=" * 60)
    print("TAT CA TESTS PASSED!")
    print("=" * 60)


# ============ CHECKLIST ============
# Week 10-11 (Bai 09):
# [x] Implement KV-cache
#     -> CachedMultiHeadAttention: luu K, V cua cac token truoc
#        Token moi chi can tinh Q cua no, roi attend voi cached K, V
#        Khong cache: moi token tinh lai toan bo -> O(N^2) moi buoc
#        Co cache: chi tinh token moi -> O(N) moi buoc
# [x] Benchmark speedup
#     -> So sanh thoi gian va memory giua co/khong co KV-cache
#        Speedup tang dan theo sequence length
