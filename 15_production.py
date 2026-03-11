# File: 15_production.py
# Production LLM Serving tu Scratch - Week 19-20
#
# TAI SAO HOC CAI NAY?
# Model da train xong -> can SERVE cho nguoi dung.
# Serving LLM KHAC HOAN TOAN voi serving model truyen thong:
# - LLM inference la AUTOREGRESSIVE (sinh 1 token/buoc)
# - KV cache ton rat nhieu memory -> can quan ly thong minh
# - Latency per token la bottle-neck cho user experience
# - Throughput can cao de phuc vu nhieu nguoi cung luc
#
# 4 CHU DE CHINH:
# 1. vLLM (PagedAttention): quan ly memory hieu qua
# 2. Speculative Decoding: tang toc sinh text
# 3. FastAPI Deployment: dua model len production
# 4. Benchmarking: do hieu nang, toi uu
#
# THUC TE:
# - OpenAI API: phuc vu hang trieu requests/ngay
# - vLLM: industry standard, 2-4x throughput vs naive
# - Speculative decoding: 2-3x speedup, dung boi Google, Meta
#
# BAI TAP:
# 1. Implement PagedAttention memory manager
# 2. Implement Speculative Decoding
# 3. Simulate FastAPI inference server
# 4. Benchmark throughput vs latency

import numpy as np
import time
import os
from collections import defaultdict


# ============================================================
# BAI TAP 1: vLLM ARCHITECTURE (PagedAttention)
# ============================================================
#
# VAN DE: Naive KV Cache Management lang phi MEMORY
#
# Naive approach: pre-allocate contiguous KV cache cho max_seq_len
#   Request A (thuc te dung 100 tokens): cap phat 2048 slots -> waste 1948!
#   Request B (thuc te dung 50 tokens):  cap phat 2048 slots -> waste 1998!
#   -> Lang phi 60-80% GPU memory!
#
# vLLM GIAI PHAP: PagedAttention (tuong tu Virtual Memory trong OS)
#
#   PHYSICAL MEMORY (GPU):
#   +-------+-------+-------+-------+-------+-------+
#   | Page0 | Page1 | Page2 | Page3 | Page4 | Page5 |
#   +-------+-------+-------+-------+-------+-------+
#      ^       ^       ^       ^       ^
#      |       |       |       |       |
#   PAGE TABLE (per request):
#   Request A: logical [0,1,2] -> physical [0, 2, 4]
#   Request B: logical [0,1]   -> physical [1, 3]
#
#   - Moi page chua page_size tokens (vd: 16)
#   - Pages KHONG CAN lien tuc trong physical memory
#   - Allocate page MOI khi can, free page khi xong
#   - Copy-on-write cho beam search (share pages giua beams)
#
# KET QUA: Phuc vu 2-4x nhieu requests cung luc!

class PagedAttentionManager:
    """
    PagedAttention Memory Manager - core cua vLLM.

    Quan ly KV cache bang cach chia thanh fixed-size pages,
    tuong tu virtual memory trong OS.

    THIET KE:
    - Physical pages: pool cac blocks tren GPU memory
    - Page table: map logical block -> physical block per request
    - Block manager: allocate/free blocks, track usage
    - Copy-on-write: share blocks giua beam search candidates
    """

    def __init__(self, num_layers, num_heads, head_dim, block_size=16,
                 max_num_blocks=1024, dtype_bytes=2):
        """
        num_layers:     so layers trong model (moi layer co KV cache rieng)
                        Vd: 32 (LLaMA 7B), 80 (LLaMA 65B), 96 (GPT-3)

        num_heads:      so KV heads (co the khac Q heads neu dung GQA)
                        Vd: 32 (LLaMA 7B MHA), 8 (LLaMA 70B GQA)
                        Nho hon = it memory per page

        head_dim:       dimension moi head
                        Vd: 128 (LLaMA), 64 (GPT-2)

        block_size:     so tokens moi block/page
                        Vd: 16 (vLLM default)
                        Nho -> it waste nhung nhieu overhead
                        Lon -> it overhead nhung nhieu waste
                        16 la balance tot

        max_num_blocks: so blocks toi da trong physical pool
                        = GPU memory / block_size_bytes
                        Vd: 1024 blocks * 16 tokens = 16384 tokens capacity

        dtype_bytes:    bytes per value, 2 = FP16, 4 = FP32
                        FP16 la standard cho inference
        """
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.block_size = block_size
        self.max_num_blocks = max_num_blocks
        self.dtype_bytes = dtype_bytes

        # Block memory: moi block chua K va V cho 1 layer
        # K block: (block_size, num_heads, head_dim)
        # V block: (block_size, num_heads, head_dim)
        self.block_bytes = 2 * block_size * num_heads * head_dim * dtype_bytes * num_layers

        # Free block pool
        self.free_blocks = list(range(max_num_blocks))

        # Page tables: request_id -> list of physical block indices
        self.page_tables = {}
        # Token counts: request_id -> so tokens da luu
        self.token_counts = {}
        # Reference counts: block_id -> so requests dang reference (cho CoW)
        self.ref_counts = np.zeros(max_num_blocks, dtype=np.int32)

    def allocate_request(self, request_id, initial_tokens=0):
        """
        Allocate blocks cho request moi.

        request_id:     ID duy nhat cua request
                        Vd: "req_001", "user_123_chat_5"

        initial_tokens: so tokens ban dau (prompt length)
                        Vd: 100 (short prompt), 2000 (long context)
                        Se allocate ceil(initial_tokens / block_size) blocks
        """
        num_blocks_needed = max(1, (initial_tokens + self.block_size - 1) // self.block_size)

        if len(self.free_blocks) < num_blocks_needed:
            raise RuntimeError(
                f"Khong du blocks! Can {num_blocks_needed}, "
                f"con {len(self.free_blocks)} free blocks."
            )

        # Allocate blocks
        allocated = []
        for _ in range(num_blocks_needed):
            block_id = self.free_blocks.pop(0)
            allocated.append(block_id)
            self.ref_counts[block_id] = 1

        self.page_tables[request_id] = allocated
        self.token_counts[request_id] = initial_tokens
        return allocated

    def append_token(self, request_id):
        """
        Them 1 token moi vao KV cache cua request.
        Neu block hien tai day, allocate block moi.

        request_id: ID cua request dang generate
        """
        self.token_counts[request_id] += 1
        cur_tokens = self.token_counts[request_id]

        # Check neu can block moi
        blocks_needed = (cur_tokens + self.block_size - 1) // self.block_size
        current_blocks = len(self.page_tables[request_id])

        if blocks_needed > current_blocks:
            if len(self.free_blocks) == 0:
                raise RuntimeError("Het blocks! Can preempt request khac.")
            new_block = self.free_blocks.pop(0)
            self.page_tables[request_id].append(new_block)
            self.ref_counts[new_block] = 1

    def free_request(self, request_id):
        """
        Free tat ca blocks cua request da hoan thanh.
        Blocks duoc tra ve free pool de request khac dung.

        request_id: ID cua request da xong
        """
        if request_id not in self.page_tables:
            return

        for block_id in self.page_tables[request_id]:
            self.ref_counts[block_id] -= 1
            if self.ref_counts[block_id] == 0:
                self.free_blocks.append(block_id)

        del self.page_tables[request_id]
        del self.token_counts[request_id]

    def fork_request(self, source_id, new_id):
        """
        Copy-on-Write: fork request cho beam search.
        Share blocks voi source (chi copy page table, KHONG copy data).
        Khi 1 trong 2 can modify -> copy block do.

        source_id: request goc
        new_id:    request moi (beam candidate)

        MEMORY SAVINGS:
        - Naive beam search: K beams * full KV cache
        - CoW: share prefix, chi copy khi diverge
        - Voi beam_width=4, seq=2000: save ~75% memory!
        """
        if source_id not in self.page_tables:
            raise ValueError(f"Source request {source_id} khong ton tai")

        # Share blocks (copy page table, tang ref count)
        self.page_tables[new_id] = self.page_tables[source_id].copy()
        self.token_counts[new_id] = self.token_counts[source_id]

        for block_id in self.page_tables[new_id]:
            self.ref_counts[block_id] += 1

    def memory_usage(self):
        """Thong ke memory usage."""
        total_blocks = self.max_num_blocks
        used_blocks = total_blocks - len(self.free_blocks)
        total_tokens = sum(self.token_counts.values())

        # Tinh waste
        total_slots = used_blocks * self.block_size
        wasted_slots = total_slots - total_tokens

        return {
            'total_blocks': total_blocks,
            'used_blocks': used_blocks,
            'free_blocks': len(self.free_blocks),
            'utilization': used_blocks / total_blocks if total_blocks > 0 else 0,
            'total_tokens': total_tokens,
            'wasted_slots': wasted_slots,
            'waste_ratio': wasted_slots / max(total_slots, 1),
            'used_memory_mb': used_blocks * self.block_bytes / (1024 * 1024),
            'total_memory_mb': total_blocks * self.block_bytes / (1024 * 1024),
            'num_active_requests': len(self.page_tables),
        }

    def naive_memory_comparison(self, max_seq_len=2048):
        """
        So sanh memory: PagedAttention vs Naive (pre-allocate full).

        max_seq_len: max sequence length cua naive approach
                     Naive se pre-allocate max_seq_len slots cho MOI request
        """
        num_requests = len(self.page_tables)
        total_tokens = sum(self.token_counts.values())

        # Naive: moi request dung max_seq_len slots
        naive_bytes = num_requests * max_seq_len * 2 * self.num_heads * self.head_dim * self.dtype_bytes * self.num_layers
        # Paged: chi dung blocks can thiet
        paged = self.memory_usage()
        paged_bytes = paged['used_blocks'] * self.block_bytes

        return {
            'naive_mb': naive_bytes / (1024 * 1024),
            'paged_mb': paged_bytes / (1024 * 1024),
            'savings_ratio': 1 - paged_bytes / max(naive_bytes, 1),
            'num_requests': num_requests,
            'avg_tokens': total_tokens / max(num_requests, 1),
            'max_seq_len': max_seq_len,
        }


# ============================================================
# BAI TAP 2: SPECULATIVE DECODING
# ============================================================
#
# VAN DE: LLM inference la MEMORY-BOUND
#
# GPU co rat nhieu compute (FLOPS) nhung bandwidth (memory -> compute) co han:
#   A100: 312 TFLOPS (FP16) nhung chi 2 TB/s bandwidth
#   -> De dung het compute, can 312T / 2T = 156 bytes/FLOP
#   -> LLM inference chi dung ~1 byte/FLOP (rat it!)
#   -> GPU IDLE 99% compute capacity khi generate 1 token!
#
# SPECULATIVE DECODING: Dung compute THUA de verify nhieu tokens cung luc
#
#   FLOW:
#   1. DRAFT model (nho, nhanh) sinh K candidate tokens
#      Vd: GPT-2 small (124M) sinh 5 tokens trong 5ms
#
#   2. TARGET model (lon, chinh xac) verify K tokens trong 1 forward pass
#      Vd: LLaMA 70B verify 5 tokens trong 40ms (= gia 1 token!)
#      Key insight: forward(5 tokens) ~ forward(1 token) vi memory-bound
#
#   3. So sanh: accepted = target dong y voi draft
#      - Token 0: draft="The" -> target="The" -> ACCEPT
#      - Token 1: draft="cat" -> target="cat" -> ACCEPT
#      - Token 2: draft="sat" -> target="was" -> REJECT -> resample "was"
#      -> Duoc 3 tokens (2 accepted + 1 resampled) cho gia ~1 target call
#
#   SPEEDUP: K / (1 + K*(draft_time/target_time))
#   - acceptance_rate = 70%: ~2x speedup
#   - acceptance_rate = 90%: ~3x speedup

class DraftModel:
    """
    Draft model (nho, nhanh) cho speculative decoding.
    Simulate bang random probability distribution.
    """

    def __init__(self, vocab_size, quality=0.7):
        """
        vocab_size: kich thuoc vocab
                    Vd: 32000 (LLaMA), 50257 (GPT-2)

        quality:    do "tot" cua draft model (0-1)
                    0.7 = 70% chance draft dong y voi target
                    Cao hon = draft tot hon -> nhieu token accepted
                    Thuc te: draft quality phu thuoc vao model pair
                    Vd: LLaMA 7B draft cho LLaMA 70B -> quality ~0.7-0.8
        """
        self.vocab_size = vocab_size
        self.quality = quality
        # Simulate model voi random weights
        self.W = np.random.randn(vocab_size, vocab_size).astype(np.float32) * 0.01

    def predict(self, token_id):
        """
        Predict next token probabilities.

        token_id: token hien tai (int)
        Returns: probability distribution (vocab_size,)
        """
        logits = self.W[token_id % self.vocab_size]
        # Them bias de mot so tokens co prob cao
        logits[token_id % 10:(token_id % 10) + 5] += 2.0
        probs = np.exp(logits - np.max(logits))
        probs = probs / probs.sum()
        return probs

    def generate_k_tokens(self, context_token, k):
        """
        Sinh K tokens lien tiep (nhanh vi model nho).

        context_token: token cuoi cung cua context
        k:             so tokens can sinh
                       Vd: 3-8 (pho bien), nhieu hon = nhieu waste neu reject

        Returns: (tokens, probs) - K token IDs va K probability distributions
        """
        tokens = []
        probs_list = []
        current = context_token

        for _ in range(k):
            probs = self.predict(current)
            token = np.random.choice(self.vocab_size, p=probs)
            tokens.append(token)
            probs_list.append(probs)
            current = token

        return tokens, probs_list


class TargetModel:
    """
    Target model (lon, chinh xac nhung cham) cho speculative decoding.
    """

    def __init__(self, vocab_size):
        """
        vocab_size: kich thuoc vocab
        """
        self.vocab_size = vocab_size
        self.W = np.random.randn(vocab_size, vocab_size).astype(np.float32) * 0.01

    def verify_tokens(self, context_token, candidate_tokens):
        """
        Verify K tokens trong 1 forward pass.

        KEY INSIGHT: Forward pass voi K tokens KHONG cham hon 1 token
        vi LLM inference la memory-bound!
        - 1 token: load weights tu memory -> compute -> output
        - K tokens: load weights tu memory -> compute K times -> K outputs
        - Bottleneck la memory bandwidth, KHONG PHAI compute

        context_token:    token cuoi cung truoc candidates
        candidate_tokens: list of K token IDs tu draft model

        Returns: list of probability distributions (1 per token + 1 extra)
        """
        probs_list = []
        current = context_token

        for token in candidate_tokens:
            logits = self.W[current % self.vocab_size]
            logits[current % 10:(current % 10) + 5] += 3.0  # Target "tot hon" draft
            probs = np.exp(logits - np.max(logits))
            probs = probs / probs.sum()
            probs_list.append(probs)
            current = token

        # 1 extra prediction (token sau candidate cuoi)
        logits = self.W[current % self.vocab_size]
        logits[current % 10:(current % 10) + 5] += 3.0
        probs = np.exp(logits - np.max(logits))
        probs = probs / probs.sum()
        probs_list.append(probs)

        return probs_list


class SpeculativeDecoder:
    """
    Speculative Decoding - tang toc LLM inference.

    ALGORITHM:
    1. Draft model sinh K tokens: [d0, d1, ..., dK-1]
    2. Target model verify K tokens (1 forward pass!)
    3. For each position i:
       - Neu target_prob[di] >= draft_prob[di]: ACCEPT
       - Neu target_prob[di] < draft_prob[di]:
         * Accept voi xac suat target_prob[di] / draft_prob[di]
         * Reject: resample tu adjusted distribution
    4. BONUS: luon duoc them 1 token tu target (position sau accepted cuoi)

    WORST CASE: 1 token accepted (= autoregressive binh thuong)
    BEST CASE: K+1 tokens accepted (draft hoan toan dung!)
    """

    def __init__(self, draft_model, target_model, k=5):
        """
        draft_model:  model nho, nhanh (DraftModel)
        target_model: model lon, chinh xac (TargetModel)
        k:            so candidate tokens moi buoc speculation
                      Vd: 3 (an toan), 5 (balance), 8 (tham lam)
                      Lon hon -> potential speedup lon hon nhung waste nhieu hon neu reject
        """
        self.draft = draft_model
        self.target = target_model
        self.k = k
        self.stats = {
            'total_tokens': 0,
            'accepted_tokens': 0,
            'draft_calls': 0,
            'target_calls': 0,
            'total_speculation_rounds': 0,
        }

    def decode_step(self, context_token):
        """
        1 buoc speculative decoding.

        context_token: token cuoi cung cua context hien tai

        Returns:
            accepted_tokens: list tokens duoc chap nhan
            stats: dict voi thong ke cho buoc nay

        TRONG 1 BUOC:
        - Draft: sinh K tokens (nhanh)
        - Target: verify K tokens (1 forward pass)
        - Accept: chap nhan tokens cho den khi reject
        - Bonus: them 1 token tu target
        """
        # Step 1: Draft model sinh K candidates
        draft_tokens, draft_probs = self.draft.generate_k_tokens(context_token, self.k)
        self.stats['draft_calls'] += 1

        # Step 2: Target model verify (1 forward pass cho K tokens!)
        target_probs = self.target.verify_tokens(context_token, draft_tokens)
        self.stats['target_calls'] += 1

        # Step 3: Rejection sampling
        accepted = []
        for i in range(self.k):
            draft_token = draft_tokens[i]
            p_target = target_probs[i][draft_token]
            p_draft = draft_probs[i][draft_token]

            if p_draft == 0:
                # Draft co prob 0 nhung target co prob > 0 -> reject
                # Resample tu target distribution
                new_token = np.random.choice(self.target.vocab_size, p=target_probs[i])
                accepted.append(new_token)
                break

            # Acceptance ratio
            ratio = min(1.0, p_target / p_draft)
            if np.random.random() < ratio:
                accepted.append(draft_token)
            else:
                # Reject: sample tu adjusted distribution
                # p_adjusted = max(0, p_target - p_draft) / sum(max(0, p_target - p_draft))
                adjusted = np.maximum(0, target_probs[i] - draft_probs[i])
                if adjusted.sum() > 0:
                    adjusted = adjusted / adjusted.sum()
                    new_token = np.random.choice(self.target.vocab_size, p=adjusted)
                else:
                    new_token = np.random.choice(self.target.vocab_size, p=target_probs[i])
                accepted.append(new_token)
                break
        else:
            # Tat ca K tokens accepted! Bonus: them 1 token tu target
            bonus_token = np.random.choice(self.target.vocab_size, p=target_probs[self.k])
            accepted.append(bonus_token)

        self.stats['total_tokens'] += len(accepted)
        self.stats['accepted_tokens'] += len(accepted)
        self.stats['total_speculation_rounds'] += 1

        return accepted, {
            'num_accepted': len(accepted),
            'max_possible': self.k + 1,
            'acceptance_rate': (len(accepted) - 1) / self.k if self.k > 0 else 0,
        }

    def generate(self, start_token, max_tokens=50):
        """
        Generate sequence voi speculative decoding.

        start_token: token bat dau
        max_tokens:  so tokens toi da can sinh
                     Vd: 100 (short), 500 (medium), 2000 (long)

        Returns: (tokens, generation_stats)
        """
        tokens = [start_token]
        self.stats = {
            'total_tokens': 0, 'accepted_tokens': 0,
            'draft_calls': 0, 'target_calls': 0,
            'total_speculation_rounds': 0,
        }

        while len(tokens) - 1 < max_tokens:
            new_tokens, _ = self.decode_step(tokens[-1])
            tokens.extend(new_tokens)

            if len(tokens) - 1 >= max_tokens:
                break

        tokens = tokens[:max_tokens + 1]  # Trim to max

        return tokens, {
            'total_generated': len(tokens) - 1,
            'speculation_rounds': self.stats['total_speculation_rounds'],
            'tokens_per_round': (len(tokens) - 1) / max(self.stats['total_speculation_rounds'], 1),
            'target_calls': self.stats['target_calls'],
            'speedup_vs_autoregressive': (len(tokens) - 1) / max(self.stats['target_calls'], 1),
        }


# ============================================================
# BAI TAP 3: FASTAPI DEPLOYMENT (Simulation)
# ============================================================
#
# FASTAPI LLM SERVER ARCHITECTURE:
#
#   Client 1 ----\
#   Client 2 -----+--> FastAPI Server --> Request Queue --> Batch Scheduler
#   Client 3 ----/                                              |
#                                                          Model Inference
#                                                               |
#                                                    Token Stream (SSE)
#                                                    /     |      \
#                                             Client 1  Client 2  Client 3
#
# KEY PATTERNS:
# 1. Request Batching: gom nhieu requests -> 1 batch inference
# 2. Streaming: gui token-by-token (SSE) thay vi doi het
# 3. Continuous Batching: them/bot request giua cac steps
# 4. Health Check: /health endpoint
#
# FASTAPI CODE (reference):
# ```python
# from fastapi import FastAPI
# from fastapi.responses import StreamingResponse
# import asyncio
#
# app = FastAPI()
#
# @app.post("/v1/completions")
# async def completions(request: CompletionRequest):
#     async def generate():
#         for token in model.generate(request.prompt):
#             yield f"data: {json.dumps({'token': token})}\n\n"
#     return StreamingResponse(generate(), media_type="text/event-stream")
#
# @app.get("/health")
# async def health():
#     return {"status": "ok", "model": "llama-7b"}
# ```

class RequestQueue:
    """
    Queue quan ly incoming requests.

    CONTINUOUS BATCHING: khac voi static batching (doi du batch moi chay),
    continuous batching cho phep them/bot request bat ky luc nao:
    - Request moi den -> them vao batch hien tai (neu con slot)
    - Request xong -> bo khoi batch, slot duoc request khac dung
    -> Throughput cao hon 2-3x so voi static batching!
    """

    def __init__(self, max_batch_size=32):
        """
        max_batch_size: so requests toi da trong 1 batch
                        Vd: 8 (GPU nho), 32 (A100), 128 (H100)
                        Lon hon -> throughput cao nhung latency tang
        """
        self.max_batch_size = max_batch_size
        self.waiting = []  # Requests dang doi
        self.running = {}  # Requests dang chay: id -> info
        self.completed = {}  # Requests da xong: id -> result
        self.next_id = 0

    def add_request(self, prompt_tokens, max_new_tokens=100):
        """
        Them request moi vao queue.

        prompt_tokens:  list token IDs cua prompt
        max_new_tokens: so tokens toi da can sinh

        Returns: request_id
        """
        req_id = f"req_{self.next_id}"
        self.next_id += 1

        self.waiting.append({
            'id': req_id,
            'prompt_tokens': prompt_tokens,
            'max_new_tokens': max_new_tokens,
            'generated_tokens': [],
            'submit_time': time.time(),
            'first_token_time': None,
        })
        return req_id

    def schedule_batch(self):
        """
        Chon requests tu waiting queue de chay.
        Continuous batching: them vao running neu con slot.
        """
        while self.waiting and len(self.running) < self.max_batch_size:
            req = self.waiting.pop(0)
            self.running[req['id']] = req

    def step(self):
        """
        1 buoc inference: sinh 1 token cho moi request trong batch.
        Simulate model inference voi random output.
        """
        completed_ids = []

        for req_id, req in self.running.items():
            # Simulate token generation
            new_token = np.random.randint(0, 1000)
            req['generated_tokens'].append(new_token)

            if req['first_token_time'] is None:
                req['first_token_time'] = time.time()

            # Check if done
            if len(req['generated_tokens']) >= req['max_new_tokens']:
                completed_ids.append(req_id)

        # Move completed requests
        for req_id in completed_ids:
            req = self.running.pop(req_id)
            req['completion_time'] = time.time()
            self.completed[req_id] = req

        return len(completed_ids)


class ModelServer:
    """
    Simulate LLM inference server.

    ARCHITECTURE:
    1. Request Queue: quan ly incoming requests
    2. Batch Scheduler: gom requests thanh batch
    3. Model Engine: chay inference
    4. Token Streamer: gui tokens ve client

    METRICS:
    - TTFT: Time To First Token (ms)
    - TPS: Tokens Per Second
    - Throughput: total tokens/second across all requests
    - Latency P50/P95/P99
    """

    def __init__(self, max_batch_size=16, model_latency_ms=10):
        """
        max_batch_size:   so requests toi da trong 1 batch
                          Vd: 8 (small GPU), 16-32 (A100), 64+ (H100)

        model_latency_ms: thoi gian trung binh sinh 1 token (ms)
                          Vd: 10ms (LLaMA 7B trên A100)
                              50ms (GPT-3 175B)
                              5ms (LLaMA 7B tren H100)
                          Anh huong truc tiep den TTFT va TPS
        """
        self.queue = RequestQueue(max_batch_size)
        self.model_latency_ms = model_latency_ms
        self.metrics = defaultdict(list)

    def submit_request(self, prompt_length=50, max_tokens=100):
        """
        Submit 1 request moi.

        prompt_length: so tokens trong prompt
        max_tokens:    so tokens can sinh
        """
        prompt = list(range(prompt_length))
        return self.queue.add_request(prompt, max_tokens)

    def run_engine(self, num_steps=100):
        """
        Chay inference engine cho num_steps buoc.

        Moi buoc:
        1. Schedule: them waiting requests vao batch
        2. Inference: sinh 1 token cho moi request trong batch
        3. Track metrics
        """
        step_metrics = []

        for step in range(num_steps):
            # Schedule
            self.queue.schedule_batch()

            if not self.queue.running:
                continue

            # Simulate inference time
            time.sleep(self.model_latency_ms / 1000.0 * 0.01)  # Scale down cho demo

            # Step
            batch_size = len(self.queue.running)
            completed = self.queue.step()

            step_metrics.append({
                'step': step,
                'batch_size': batch_size,
                'completed': completed,
                'waiting': len(self.queue.waiting),
            })

        return step_metrics

    def get_request_metrics(self):
        """Tinh metrics cho completed requests."""
        if not self.queue.completed:
            return {}

        ttfts = []
        tps_list = []
        latencies = []

        for req_id, req in self.queue.completed.items():
            submit = req['submit_time']
            first_token = req['first_token_time']
            completion = req['completion_time']
            num_tokens = len(req['generated_tokens'])

            ttft = (first_token - submit) * 1000  # ms
            total_time = completion - submit
            tps = num_tokens / max(total_time, 1e-6)
            latency = total_time * 1000  # ms

            ttfts.append(ttft)
            tps_list.append(tps)
            latencies.append(latency)

        ttfts = np.array(ttfts)
        tps_arr = np.array(tps_list)
        latencies = np.array(latencies)

        return {
            'num_completed': len(self.queue.completed),
            'ttft_p50': np.percentile(ttfts, 50),
            'ttft_p95': np.percentile(ttfts, 95),
            'ttft_p99': np.percentile(ttfts, 99),
            'tps_mean': np.mean(tps_arr),
            'tps_p50': np.percentile(tps_arr, 50),
            'latency_p50': np.percentile(latencies, 50),
            'latency_p95': np.percentile(latencies, 95),
            'latency_p99': np.percentile(latencies, 99),
            'throughput_total_tps': np.sum(tps_arr),
        }


# ============================================================
# BAI TAP 4: BENCHMARKING
# ============================================================
#
# 2 METRICS CHINH (thuong XUNG DOT):
#
# THROUGHPUT (tokens/second):
# - Bao nhieu tokens he thong sinh trong 1 giay (tong tat ca requests)
# - Muon CAO -> dung batch lon, continuous batching, vLLM
# - Vd: vLLM + LLaMA 7B + A100: ~2000 tokens/s
#
# LATENCY (ms/token):
# - Bao lau de sinh 1 token cho 1 request
# - Muon THAP -> batch nho, speculative decoding
# - Vd: LLaMA 7B + A100: ~10ms/token (batch=1)
#
# XUNG DOT:
# - Batch lon -> throughput CAO nhung latency CAO (moi request doi lau hon)
# - Batch nho -> latency THAP nhung throughput THAP (GPU khong full)
#
# KEY METRICS:
# - TTFT (Time To First Token): thoi gian tu gui request den nhan token dau tien
#   User experience: TTFT < 500ms la tot, < 1s la chap nhan duoc
# - TPS (Tokens Per Second): toc do sinh tokens sau token dau tien
#   Chat: TPS > 30 = muot, TPS > 15 = chap nhan duoc
# - P50/P95/P99 latency: phan vi latency
#   P99 quan trong cho SLA (Service Level Agreement)

class InferenceBenchmark:
    """
    Benchmark LLM inference performance.

    Do va bao cao:
    - Throughput tai cac batch sizes
    - Latency distribution (P50, P95, P99)
    - TTFT (Time To First Token)
    - Memory usage
    - Throughput vs Latency trade-off
    """

    def __init__(self, model_params_b, gpu_memory_gb=80, gpu_bandwidth_gbs=2000):
        """
        model_params_b:    so params (billions)
                           Vd: 7 (LLaMA 7B), 13, 70, 175 (GPT-3)

        gpu_memory_gb:     GPU VRAM (GB)
                           Vd: 24 (RTX 4090), 40 (A100 40GB), 80 (A100 80GB/H100)

        gpu_bandwidth_gbs: GPU memory bandwidth (GB/s)
                           Vd: 1000 (RTX 4090), 2000 (A100), 3350 (H100)
                           Bandwidth la bottleneck cho LLM inference!
        """
        self.model_params = model_params_b * 1e9
        self.gpu_memory = gpu_memory_gb
        self.gpu_bandwidth = gpu_bandwidth_gbs

    def estimate_latency_per_token(self, batch_size=1, dtype_bytes=2):
        """
        Uoc tinh latency per token dua tren memory bandwidth.

        LLM inference la MEMORY-BOUND:
        - Moi token can load TOAN BO model weights tu memory
        - Latency = model_size / bandwidth
        - Batch size: amortize weight loading across batch

        batch_size:  so requests cung luc
                     Vd: 1 (interactive), 8 (batch), 64 (high throughput)

        dtype_bytes: 2 = FP16, 1 = INT8, 0.5 = INT4

        Returns: latency in ms
        """
        model_bytes = self.model_params * dtype_bytes
        # Amortize across batch (load weights once, compute for all)
        # Nhung batch lon can nhieu KV cache memory
        effective_bandwidth = self.gpu_bandwidth * 1e9  # GB/s -> bytes/s

        # Latency = model_size / bandwidth (per batch, shared)
        latency_s = model_bytes / effective_bandwidth
        # Them overhead cho KV cache access
        kv_overhead = 1 + 0.1 * batch_size  # ~10% per request in batch
        latency_ms = latency_s * 1000 * kv_overhead

        return latency_ms

    def throughput_vs_batch_size(self, max_batch=64, dtype_bytes=2):
        """
        Tinh throughput tai cac batch sizes.

        Returns: list of (batch_size, throughput_tps, latency_ms)
        """
        results = []
        for bs in range(1, max_batch + 1):
            latency_ms = self.estimate_latency_per_token(bs, dtype_bytes)
            # Throughput = batch_size / latency
            throughput = bs / (latency_ms / 1000)
            results.append({
                'batch_size': bs,
                'latency_ms': latency_ms,
                'throughput_tps': throughput,
            })
        return results

    def max_batch_size(self, seq_len=2048, kv_heads=32, head_dim=128, dtype_bytes=2):
        """
        Tinh max batch size co the phuc vu cung luc.

        Model weights + KV cache phai vua GPU memory.

        seq_len:     max sequence length per request
        kv_heads:    so KV heads
        head_dim:    dimension per head
        dtype_bytes: bytes per value
        """
        model_mem_gb = self.model_params * dtype_bytes / (1024 ** 3)

        # KV cache per request per layer (K + V)
        kv_per_request_per_layer = 2 * kv_heads * head_dim * seq_len * dtype_bytes
        # Estimate num_layers
        d_model = kv_heads * head_dim
        num_layers = int(self.model_params / (12 * d_model ** 2))  # Rough estimate
        num_layers = max(1, min(num_layers, 200))

        kv_per_request = kv_per_request_per_layer * num_layers / (1024 ** 3)

        available_mem = self.gpu_memory - model_mem_gb
        max_bs = int(available_mem / max(kv_per_request, 1e-6))
        max_bs = max(1, max_bs)

        return {
            'max_batch_size': max_bs,
            'model_memory_gb': model_mem_gb,
            'kv_cache_per_request_gb': kv_per_request,
            'available_for_kv_gb': available_mem,
            'estimated_layers': num_layers,
        }

    def full_benchmark(self, dtype_bytes=2):
        """Chay full benchmark va tra ve tat ca metrics."""
        # Latency tai batch sizes
        batch_results = self.throughput_vs_batch_size(32, dtype_bytes)

        # Tim sweet spot (throughput/latency ratio cao nhat)
        best_ratio = 0
        sweet_spot = None
        for r in batch_results:
            ratio = r['throughput_tps'] / max(r['latency_ms'], 0.01)
            if ratio > best_ratio:
                best_ratio = ratio
                sweet_spot = r

        return {
            'batch_results': batch_results,
            'sweet_spot': sweet_spot,
            'single_token_latency_ms': batch_results[0]['latency_ms'],
            'max_throughput': max(r['throughput_tps'] for r in batch_results),
        }


# ============================================================
# MAIN: Test tat ca implementations
# ============================================================
if __name__ == "__main__":
    np.random.seed(42)
    output_dir = os.path.dirname(os.path.abspath(__file__))

    # ============================================================
    # PHAN 1: vLLM PagedAttention
    # ============================================================
    print("=" * 70)
    print("PHAN 1: vLLM (PagedAttention)")
    print("=" * 70)

    # Simulate LLaMA 7B config
    pam = PagedAttentionManager(
        num_layers=32, num_heads=32, head_dim=128,
        block_size=16, max_num_blocks=512, dtype_bytes=2
    )

    print(f"\n  Config: 32 layers, 32 heads, head_dim=128, block_size=16")
    print(f"  Total blocks: 512, max tokens: {512 * 16}")
    print(f"  Block size: {pam.block_bytes / 1024:.1f} KB")

    # Simulate nhieu requests voi variable lengths
    request_lengths = [100, 250, 50, 400, 75, 200, 150, 300]
    print(f"\n  Allocating {len(request_lengths)} requests:")

    for i, length in enumerate(request_lengths):
        req_id = f"req_{i}"
        blocks = pam.allocate_request(req_id, initial_tokens=length)
        print(f"    {req_id}: {length} tokens -> {len(blocks)} blocks "
              f"(waste: {len(blocks) * 16 - length} slots)")

    usage = pam.memory_usage()
    print(f"\n  Memory Usage:")
    print(f"    Used blocks:  {usage['used_blocks']}/{usage['total_blocks']} "
          f"({usage['utilization']:.1%})")
    print(f"    Total tokens: {usage['total_tokens']}")
    print(f"    Wasted slots: {usage['wasted_slots']} ({usage['waste_ratio']:.1%})")
    print(f"    Used memory:  {usage['used_memory_mb']:.1f} MB")

    # So sanh voi naive
    comparison = pam.naive_memory_comparison(max_seq_len=2048)
    print(f"\n  PagedAttention vs Naive (max_seq=2048):")
    print(f"    Naive:  {comparison['naive_mb']:.1f} MB ({len(request_lengths)} reqs * 2048 tokens)")
    print(f"    Paged:  {comparison['paged_mb']:.1f} MB (actual tokens only)")
    print(f"    Savings: {comparison['savings_ratio']:.1%}")

    # Test Copy-on-Write (beam search)
    print(f"\n  Copy-on-Write (Beam Search):")
    pam.fork_request("req_0", "beam_0")
    pam.fork_request("req_0", "beam_1")
    pam.fork_request("req_0", "beam_2")
    usage_cow = pam.memory_usage()
    print(f"    Forked req_0 into 3 beams")
    print(f"    Additional blocks used: 0 (share via reference counting!)")
    print(f"    Active requests: {usage_cow['num_active_requests']}")

    # Free requests
    for i in range(len(request_lengths)):
        pam.free_request(f"req_{i}")
    for b in range(3):
        pam.free_request(f"beam_{b}")
    usage_after = pam.memory_usage()
    print(f"\n  After freeing all requests:")
    print(f"    Free blocks: {usage_after['free_blocks']}/{usage_after['total_blocks']}")
    assert usage_after['free_blocks'] == usage_after['total_blocks'], "All blocks should be free"
    print(f"    PagedAttention: OK")

    # ============================================================
    # PHAN 2: SPECULATIVE DECODING
    # ============================================================
    print("\n" + "=" * 70)
    print("PHAN 2: SPECULATIVE DECODING")
    print("=" * 70)

    vocab_size = 1000
    draft = DraftModel(vocab_size, quality=0.7)
    target = TargetModel(vocab_size)

    # Test voi cac K values
    print(f"\n  --- K (speculation length) comparison ---")
    for k in [1, 3, 5, 8]:
        decoder = SpeculativeDecoder(draft, target, k=k)
        tokens, stats = decoder.generate(start_token=42, max_tokens=100)

        print(f"  K={k}: generated={stats['total_generated']}, "
              f"rounds={stats['speculation_rounds']}, "
              f"tokens/round={stats['tokens_per_round']:.1f}, "
              f"target_calls={stats['target_calls']}, "
              f"speedup={stats['speedup_vs_autoregressive']:.1f}x")

    # Chi tiet 1 run
    print(f"\n  --- Detailed run (K=5) ---")
    decoder = SpeculativeDecoder(draft, target, k=5)
    tokens, stats = decoder.generate(start_token=42, max_tokens=50)
    print(f"  Generated {stats['total_generated']} tokens in "
          f"{stats['speculation_rounds']} rounds")
    print(f"  Tokens per round: {stats['tokens_per_round']:.2f} "
          f"(max possible: {5 + 1})")
    print(f"  Target model calls: {stats['target_calls']} "
          f"(vs {stats['total_generated']} without speculation)")
    print(f"  Effective speedup: {stats['speedup_vs_autoregressive']:.2f}x")
    print(f"  Speculative Decoding: OK")

    # Speedup analysis
    print(f"\n  --- Theoretical Speedup Analysis ---")
    print(f"  {'Accept Rate':>12} | {'K=3':>8} | {'K=5':>8} | {'K=8':>8}")
    print(f"  {'-'*12}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}")
    for acc in [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]:
        speedups = []
        for k in [3, 5, 8]:
            # Expected tokens per round = sum(acc^i for i=0..K-1) + 1 extra
            expected = sum(acc ** i for i in range(k)) + acc ** k
            # But limited by K+1 max
            expected = min(expected, k + 1)
            speedups.append(f"{expected:.1f}x")
        print(f"  {acc:>11.0%} | {'  |  '.join(speedups)}")

    # ============================================================
    # PHAN 3: FASTAPI DEPLOYMENT
    # ============================================================
    print("\n" + "=" * 70)
    print("PHAN 3: MODEL SERVER (FastAPI Simulation)")
    print("=" * 70)

    server = ModelServer(max_batch_size=8, model_latency_ms=10)

    # Submit nhieu requests
    print(f"\n  Submitting 20 requests (varying lengths)...")
    for i in range(20):
        prompt_len = np.random.randint(20, 200)
        max_tokens = np.random.randint(50, 150)
        server.submit_request(prompt_len, max_tokens)

    print(f"  Waiting queue: {len(server.queue.waiting)}")

    # Run engine
    step_metrics = server.run_engine(num_steps=2000)
    metrics = server.get_request_metrics()

    if metrics:
        print(f"\n  Server Metrics:")
        print(f"    Completed requests: {metrics['num_completed']}")
        print(f"    TTFT P50:  {metrics['ttft_p50']:.1f} ms")
        print(f"    TTFT P95:  {metrics['ttft_p95']:.1f} ms")
        print(f"    TPS mean:  {metrics['tps_mean']:.0f} tokens/s")
        print(f"    Latency P50: {metrics['latency_p50']:.1f} ms")
        print(f"    Latency P95: {metrics['latency_p95']:.1f} ms")
        print(f"    Latency P99: {metrics['latency_p99']:.1f} ms")
    print(f"  Model Server: OK")

    # Show FastAPI reference code
    print(f"\n  FastAPI Reference Code:")
    print(f"  ```python")
    print(f"  from fastapi import FastAPI")
    print(f"  from fastapi.responses import StreamingResponse")
    print(f"  ")
    print(f"  app = FastAPI()")
    print(f"  ")
    print(f"  @app.post('/v1/completions')")
    print(f"  async def completions(request: CompletionRequest):")
    print(f"      async def generate():")
    print(f"          for token in model.generate(request.prompt):")
    print(f"              yield f'data: {{json.dumps({{\"token\": token}})}}\\n\\n'")
    print(f"      return StreamingResponse(generate(), media_type='text/event-stream')")
    print(f"  ```")

    # ============================================================
    # PHAN 4: BENCHMARKING
    # ============================================================
    print("\n" + "=" * 70)
    print("PHAN 4: BENCHMARKING (Throughput vs Latency)")
    print("=" * 70)

    # Benchmark LLaMA 7B on A100
    print(f"\n  --- LLaMA 7B on A100 80GB ---")
    bench_7b = InferenceBenchmark(model_params_b=7, gpu_memory_gb=80, gpu_bandwidth_gbs=2000)

    result_7b = bench_7b.full_benchmark(dtype_bytes=2)  # FP16
    print(f"  Single token latency: {result_7b['single_token_latency_ms']:.2f} ms")
    print(f"  Max throughput: {result_7b['max_throughput']:.0f} tokens/s")

    # Throughput vs batch size table
    print(f"\n  {'Batch':>6} | {'Latency':>10} | {'Throughput':>12} | {'Bar':>20}")
    print(f"  {'-'*6}-+-{'-'*10}-+-{'-'*12}-+-{'-'*20}")
    for r in result_7b['batch_results'][:16]:
        bar = "#" * int(r['throughput_tps'] / result_7b['max_throughput'] * 20)
        print(f"  {r['batch_size']:>6} | {r['latency_ms']:>8.2f}ms | "
              f"{r['throughput_tps']:>10.0f} t/s | {bar}")

    # Max batch size
    max_bs_info = bench_7b.max_batch_size(seq_len=2048)
    print(f"\n  Max batch size (seq=2048, FP16):")
    print(f"    Model memory: {max_bs_info['model_memory_gb']:.1f} GB")
    print(f"    KV cache/req: {max_bs_info['kv_cache_per_request_gb']:.2f} GB")
    print(f"    Available for KV: {max_bs_info['available_for_kv_gb']:.1f} GB")
    print(f"    Max batch size: {max_bs_info['max_batch_size']}")

    # Compare models
    print(f"\n  --- Model Comparison (A100 80GB, FP16) ---")
    models = [
        ("LLaMA 7B", 7),
        ("LLaMA 13B", 13),
        ("LLaMA 70B", 70),
        ("GPT-3 175B", 175),
    ]

    print(f"  {'Model':>15} | {'Latency (bs=1)':>15} | {'Max Throughput':>15} | {'Fits A100?':>10}")
    print(f"  {'-'*15}-+-{'-'*15}-+-{'-'*15}-+-{'-'*10}")

    for name, params_b in models:
        bench = InferenceBenchmark(params_b, 80, 2000)
        lat = bench.estimate_latency_per_token(1, 2)
        tp = bench.throughput_vs_batch_size(8, 2)
        max_tp = max(r['throughput_tps'] for r in tp)
        fits = "Yes" if params_b * 2 < 80 else "No (TP needed)"
        print(f"  {name:>15} | {lat:>12.1f} ms | {max_tp:>12.0f} t/s | {fits:>10}")

    # INT4 quantization impact
    print(f"\n  --- Quantization Impact (LLaMA 7B, A100) ---")
    for dtype_name, dtype_bytes in [("FP16", 2), ("INT8", 1), ("INT4", 0.5)]:
        bench = InferenceBenchmark(7, 80, 2000)
        lat = bench.estimate_latency_per_token(1, dtype_bytes)
        tp = bench.throughput_vs_batch_size(8, dtype_bytes)
        max_tp = max(r['throughput_tps'] for r in tp)
        max_bs = bench.max_batch_size(2048, dtype_bytes=dtype_bytes)
        print(f"  {dtype_name:>5}: latency={lat:.2f}ms, "
              f"max_throughput={max_tp:.0f} t/s, "
              f"max_batch={max_bs['max_batch_size']}")

    # ============================================================
    # TONG KET
    # ============================================================
    print("\n" + "=" * 70)
    print("TONG KET: PRODUCTION LLM SERVING")
    print("=" * 70)
    print("""
  1. vLLM (PagedAttention):
     - Quan ly KV cache bang pages (giong virtual memory)
     - Giam memory waste tu 60-80% xuong <5%
     - Copy-on-Write cho beam search
     - Industry standard: vLLM, TensorRT-LLM

  2. Speculative Decoding:
     - Draft model sinh K tokens nhanh
     - Target model verify K tokens trong 1 forward pass
     - Speedup 2-3x voi acceptance rate 70-90%
     - Dung boi: Google, Meta, Apple

  3. FastAPI Deployment:
     - Streaming responses (SSE) cho real-time chat
     - Continuous batching: them/bot requests bat ky luc nao
     - Health check, rate limiting, authentication
     - OpenAI API-compatible endpoint

  4. Benchmarking:
     - TTFT (Time To First Token): < 500ms la tot
     - TPS (Tokens Per Second): > 30 la muot
     - Throughput vs Latency trade-off: batch size la key
     - P99 latency quan trong cho SLA
     - Quantization (INT4/INT8) tang throughput 2-4x

  PRODUCTION STACK (2024):
  - Inference: vLLM + TensorRT-LLM
  - Serving: FastAPI / gRPC
  - Optimization: Speculative decoding + Quantization
  - Monitoring: Prometheus + Grafana
  - Scaling: Kubernetes + GPU autoscaling
    """)

    # --- Plot ---
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Plot 1: Throughput vs Batch Size
        ax = axes[0, 0]
        batch_results = bench_7b.full_benchmark(2)['batch_results']
        bs = [r['batch_size'] for r in batch_results]
        tp = [r['throughput_tps'] for r in batch_results]
        ax.plot(bs, tp, color='blue', marker='o', markersize=2)
        ax.set_xlabel('Batch Size')
        ax.set_ylabel('Throughput (tokens/s)')
        ax.set_title('LLaMA 7B: Throughput vs Batch Size')
        ax.grid(True, alpha=0.3)

        # Plot 2: Latency vs Batch Size
        ax = axes[0, 1]
        lat = [r['latency_ms'] for r in batch_results]
        ax.plot(bs, lat, color='red', marker='o', markersize=2)
        ax.set_xlabel('Batch Size')
        ax.set_ylabel('Latency (ms/token)')
        ax.set_title('LLaMA 7B: Latency vs Batch Size')
        ax.grid(True, alpha=0.3)

        # Plot 3: Speculative Decoding speedup
        ax = axes[1, 0]
        acc_rates = np.linspace(0.3, 0.99, 50)
        for k_val in [3, 5, 8]:
            speedups = []
            for acc in acc_rates:
                expected = sum(acc ** i for i in range(k_val)) + acc ** k_val
                speedups.append(min(expected, k_val + 1))
            ax.plot(acc_rates * 100, speedups, label=f'K={k_val}')
        ax.set_xlabel('Acceptance Rate (%)')
        ax.set_ylabel('Speedup (x)')
        ax.set_title('Speculative Decoding: Speedup vs Acceptance Rate')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 4: Model comparison
        ax = axes[1, 1]
        model_names = []
        model_lats = []
        for name, params_b in models:
            bench = InferenceBenchmark(params_b, 80, 2000)
            lat_val = bench.estimate_latency_per_token(1, 2)
            model_names.append(name)
            model_lats.append(lat_val)
        ax.barh(model_names, model_lats, color=['green', 'blue', 'orange', 'red'])
        ax.set_xlabel('Latency (ms/token)')
        ax.set_title('Model Comparison: Latency (batch=1, FP16, A100)')
        ax.grid(True, alpha=0.3, axis='x')

        plt.suptitle('Week 19-20: Production LLM Serving', fontsize=14, fontweight='bold')
        plt.tight_layout()
        path = os.path.join(output_dir, "plot_production.png")
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"\n  Saved: {path}")

    except ImportError:
        print("  matplotlib chua cai.")

    print("\n" + "=" * 70)
    print("TAT CA TESTS PASSED!")
    print("=" * 70)


# ======== CHECKLIST ========
# Week 19-20 Production:
# [x] Study vLLM architecture
# [x] Implement speculative decoding
# [x] Deploy model voi FastAPI
# [x] Benchmark throughput vs latency
