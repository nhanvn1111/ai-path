# File: 14_distributed.py
# Distributed Training tu Scratch - Week 17-18
#
# TAI SAO HOC CAI NAY?
# Model lon nhu GPT-3 (175B params = 700GB FP32) KHONG THE train tren 1 GPU.
# GPU manh nhat (A100 80GB, H100 80GB) cung chi chua duoc ~20B params FP32.
# Can chia viec ra NHIEU GPU -> Distributed Training.
#
# 3 CHIEN LUOC CHINH:
# 1. Data Parallelism: copy model -> moi GPU xu ly 1 phan DATA
# 2. Tensor Parallelism: chia TUNG LAYER ra nhieu GPU
# 3. Pipeline Parallelism: chia NHOM LAYERS ra nhieu GPU
#
# NGOAI RA:
# - ZeRO: tiet kiem memory bang cach chia optimizer states
# - Gradient Checkpointing: doi memory lay compute
#
# THUC TE:
# - GPT-3 175B: 1024 A100 GPUs, train 34 ngay
# - LLaMA 65B: 2048 A100 GPUs, train ~21 ngay
# - PaLM 540B: 6144 TPU v4 chips
#
# BAI TAP:
# 1. Simulate Data Parallelism voi All-Reduce
# 2. Simulate Tensor Parallelism (Column + Row parallel)
# 3. Simulate Pipeline Parallelism voi micro-batching
# 4. Simulate ZeRO optimizer stages
# 5. Implement Gradient Checkpointing

import numpy as np
import time
import os


def softmax(x, axis=-1):
    exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


# ============================================================
# BAI TAP 1: DATA PARALLELISM
# ============================================================
#
# DATA PARALLELISM - chien luoc don gian nhat:
#
#   GPU 0: Model copy 0 + Data batch 0 -> Grad 0 --|
#   GPU 1: Model copy 1 + Data batch 1 -> Grad 1 --|-- All-Reduce --> Avg Grad
#   GPU 2: Model copy 2 + Data batch 2 -> Grad 2 --|
#   GPU 3: Model copy 3 + Data batch 3 -> Grad 3 --|
#                                                        |
#                                           All GPUs update voi CUNG gradient
#
# ALL-REDUCE: tong hop gradients tu tat ca GPUs
# - Ring All-Reduce: moi GPU gui/nhan voi 2 lang gieng -> 2*(N-1) steps
# - Tree All-Reduce: dang cay nhi phan -> 2*log(N) steps
# - NCCL (NVIDIA): tu dong chon algorithm toi uu
#
# COMMUNICATION OVERHEAD:
# - Moi step: gui toan bo gradients qua network
# - GPT-3: 175B params * 4 bytes = 700GB gradients moi step!
# - Can NVLink (600 GB/s) hoac InfiniBand (200 Gb/s)

class SimpleLinear:
    """
    Linear layer don gian de demo distributed training.

    Tuong tu nn.Linear trong PyTorch nhung chi dung numpy.
    """

    def __init__(self, d_in, d_out):
        """
        d_in:  so chieu input (so features dau vao)
               Vd: 768 (GPT-2 small), 4096 (LLaMA 7B)

        d_out: so chieu output (so features dau ra)
               Vd: 3072 (GPT-2 FFN), 11008 (LLaMA 7B FFN)

        Weights khoi tao bang He initialization: scale = sqrt(2/d_in)
        de dam bao variance on dinh qua nhieu layers.
        """
        scale = np.sqrt(2.0 / d_in)
        self.W = np.random.randn(d_in, d_out).astype(np.float32) * scale
        self.b = np.zeros(d_out, dtype=np.float32)
        self.grad_W = None
        self.grad_b = None
        self.input_cache = None

    def forward(self, x):
        """
        x: input, shape (batch_size, d_in)
           Vd: (32, 768) cho GPT-2 small, (8, 4096) cho LLaMA 7B
        """
        self.input_cache = x
        return x @ self.W + self.b

    def backward(self, grad_output):
        """
        grad_output: gradient tu layer sau, shape (batch_size, d_out)

        Tinh gradient cho W va b:
        - grad_W = x.T @ grad_output  (chain rule)
        - grad_b = sum(grad_output)
        """
        x = self.input_cache
        self.grad_W = x.T @ grad_output / x.shape[0]
        self.grad_b = np.mean(grad_output, axis=0)
        return grad_output @ self.W.T

    def param_count(self):
        return self.W.size + self.b.size


class DataParallelSimulator:
    """
    Simulate Data Parallelism tren nhieu "GPU ao".

    CACH HOAT DONG:
    1. Copy model len tat ca GPUs
    2. Chia data batch thanh N phan (N = so GPUs)
    3. Moi GPU tinh forward + backward tren phan data cua no
    4. All-Reduce: tinh trung binh gradients tu tat ca GPUs
    5. Moi GPU update weights voi CUNG gradient -> models DONG BO

    THUC TE (PyTorch DDP):
    ```python
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP

    dist.init_process_group("nccl")  # NVIDIA Collective Communications Library
    rank = dist.get_rank()
    model = GPT(...).to(rank)
    model = DDP(model, device_ids=[rank])

    for batch in dataloader:
        loss = model(batch)
        loss.backward()  # All-reduce gradients TU DONG o day
        optimizer.step()
    ```
    """

    def __init__(self, model_fn, num_gpus=4):
        """
        model_fn: function() -> model, ham tao model moi
                  Moi GPU se co 1 ban sao cua model
                  Vd: lambda: SimpleLinear(768, 3072)

        num_gpus: so GPU ao de simulate
                  Vd: 4 (demo), 8 (pho bien), 64 (GPT-3 data parallel)
                  Nhieu GPU -> batch size lon hon -> gradient on dinh hon
                  Nhung communication overhead cung tang
        """
        self.num_gpus = num_gpus
        # Tao model tren moi "GPU"
        self.models = [model_fn() for _ in range(num_gpus)]

        # Dong bo weights ban dau (tat ca GPUs phai bat dau GIONG NHAU)
        base_model = self.models[0]
        for i in range(1, num_gpus):
            self.models[i].W = base_model.W.copy()
            self.models[i].b = base_model.b.copy()

    def _all_reduce_avg(self, gradients_list):
        """
        All-Reduce: tinh trung binh gradients tu tat ca GPUs.

        gradients_list: list of gradients, moi phan tu tu 1 GPU
                        Vd: [grad_gpu0, grad_gpu1, grad_gpu2, grad_gpu3]
                        Moi grad co shape giong nhau (d_in, d_out)

        TRONG THUC TE:
        - Ring All-Reduce: moi GPU gui 1 phan cho GPU ke -> 2*(N-1) steps
          Bandwidth-optimal, duoc NCCL su dung cho message lon
        - Tree All-Reduce: cay nhi phan -> 2*log(N) steps
          Latency-optimal, duoc dung cho message nho

        O day simulate bang cach cong roi chia -> tuong duong ket qua.
        """
        avg = np.zeros_like(gradients_list[0])
        for g in gradients_list:
            avg += g
        avg /= len(gradients_list)
        return avg

    def train_step(self, X, y, lr=0.01):
        """
        1 buoc training voi Data Parallelism.

        X:  input data, shape (total_batch, d_in)
            Se duoc chia thanh num_gpus phan bang nhau
            Vd: total_batch=128, num_gpus=4 -> moi GPU xu ly 32 samples

        y:  labels, shape (total_batch, d_out) hoac (total_batch,)

        lr: learning rate

        Returns: dict voi loss, grad_norm, communication stats
        """
        batch_size = X.shape[0] // self.num_gpus
        total_loss = 0
        all_grad_W = []
        all_grad_b = []

        # Step 1: Forward + Backward tren moi GPU (SONG SONG trong thuc te)
        for gpu_id in range(self.num_gpus):
            start = gpu_id * batch_size
            end = start + batch_size
            X_local = X[start:end]
            y_local = y[start:end]

            # Forward
            output = self.models[gpu_id].forward(X_local)
            # Simple MSE loss
            loss = np.mean((output - y_local) ** 2)
            total_loss += loss

            # Backward
            grad_output = 2 * (output - y_local) / y_local.shape[0]
            self.models[gpu_id].backward(grad_output)

            all_grad_W.append(self.models[gpu_id].grad_W.copy())
            all_grad_b.append(self.models[gpu_id].grad_b.copy())

        # Step 2: All-Reduce - tinh trung binh gradients
        avg_grad_W = self._all_reduce_avg(all_grad_W)
        avg_grad_b = self._all_reduce_avg(all_grad_b)

        # Step 3: Update weights tren tat ca GPUs (GIONG NHAU)
        for gpu_id in range(self.num_gpus):
            self.models[gpu_id].W -= lr * avg_grad_W
            self.models[gpu_id].b -= lr * avg_grad_b

        # Verify: tat ca GPUs co CUNG weights
        for i in range(1, self.num_gpus):
            assert np.allclose(self.models[0].W, self.models[i].W), \
                f"GPU 0 va GPU {i} weights khong dong bo!"

        # Communication stats
        grad_bytes = (avg_grad_W.nbytes + avg_grad_b.nbytes) * 2  # send + receive
        comm_overhead_pct = grad_bytes / (self.models[0].W.nbytes + self.models[0].b.nbytes) * 100

        return {
            'loss': total_loss / self.num_gpus,
            'grad_norm': np.linalg.norm(avg_grad_W),
            'communication_bytes': grad_bytes,
            'comm_overhead_pct': comm_overhead_pct,
        }


# ============================================================
# BAI TAP 2: TENSOR PARALLELISM
# ============================================================
#
# TENSOR PARALLELISM (Megatron-LM style):
#
# COLUMN PARALLEL (split output dim):
#   W shape (d, 4d) -> chia thanh 2 GPU:
#   GPU 0: W[:, :2d]  -> output_0 shape (batch, 2d)
#   GPU 1: W[:, 2d:]  -> output_1 shape (batch, 2d)
#   -> Concat: output = [output_0 | output_1] shape (batch, 4d)
#
# ROW PARALLEL (split input dim):
#   W shape (4d, d) -> chia thanh 2 GPU:
#   GPU 0: W[:2d, :]  x input[:, :2d]  -> partial_0 shape (batch, d)
#   GPU 1: W[2d:, :]  x input[:, 2d:]  -> partial_1 shape (batch, d)
#   -> All-Reduce: output = partial_0 + partial_1 shape (batch, d)
#
# FFN TRONG TRANSFORMER:
#   FFN(x) = GELU(x @ W1) @ W2
#   W1: column parallel (chia output dim) -> KHONG CAN communication
#   W2: row parallel (chia input dim) -> All-Reduce sau W2
#   -> Chi can 1 All-Reduce cho toan bo FFN!

class TensorParallelLinear:
    """
    Tensor Parallel Linear layer - chia 1 layer ra nhieu GPUs.

    COLUMN PARALLEL:
    - Split W theo cot (output dimension)
    - Moi GPU tinh 1 phan output
    - Cat ket qua lai (hoac giu rieng neu layer tiep theo la row parallel)

    ROW PARALLEL:
    - Split W theo hang (input dimension)
    - Moi GPU nhan 1 phan input, nhan voi phan W cua no
    - All-Reduce (sum) de co output day du

    TAI SAO KET HOP COLUMN + ROW?
    - FFN: W1 (column) -> activation -> W2 (row)
    - Giua W1 va W2 KHONG CAN communication (output W1 da tren dung GPU)
    - Chi can 1 All-Reduce sau W2 -> TIET KIEM bandwidth!
    - Day la thiet ke cua Megatron-LM (NVIDIA)
    """

    def __init__(self, d_in, d_out, num_gpus=2, parallel_mode='column'):
        """
        d_in:          so chieu input
                       Vd: 4096 (LLaMA 7B), 12288 (GPT-3)

        d_out:         so chieu output
                       Vd: 11008 (LLaMA 7B FFN), 49152 (GPT-3 FFN = 4 * 12288)

        num_gpus:      so GPU de chia layer
                       Vd: 2 (pho bien cho 7B), 4-8 (cho 70B+)
                       d_out (column) hoac d_in (row) PHAI chia het cho num_gpus

        parallel_mode: 'column' = chia output dim (W[:, chunk_i])
                       'row' = chia input dim (W[chunk_i, :])
                       Column cho W1 (FFN), Row cho W2 (FFN)
                       Attention: column cho W_q/W_k/W_v, row cho W_o
        """
        self.d_in = d_in
        self.d_out = d_out
        self.num_gpus = num_gpus
        self.parallel_mode = parallel_mode

        scale = np.sqrt(2.0 / d_in)
        full_W = np.random.randn(d_in, d_out).astype(np.float32) * scale

        # Split W theo mode
        if parallel_mode == 'column':
            # Chia cot: moi GPU co W shape (d_in, d_out/num_gpus)
            chunk_size = d_out // num_gpus
            self.W_shards = [
                full_W[:, i * chunk_size:(i + 1) * chunk_size].copy()
                for i in range(num_gpus)
            ]
        else:  # row
            # Chia hang: moi GPU co W shape (d_in/num_gpus, d_out)
            chunk_size = d_in // num_gpus
            self.W_shards = [
                full_W[i * chunk_size:(i + 1) * chunk_size, :].copy()
                for i in range(num_gpus)
            ]

    def forward(self, x):
        """
        x: input, shape (batch_size, d_in)

        Column parallel:
          - Moi GPU: output_i = x @ W_shard_i, shape (batch, d_out/N)
          - Concat: output = [output_0 | output_1 | ...], shape (batch, d_out)

        Row parallel:
          - Split x: x_i = x[:, chunk_i], shape (batch, d_in/N)
          - Moi GPU: partial_i = x_i @ W_shard_i, shape (batch, d_out)
          - All-Reduce (sum): output = sum(partial_i), shape (batch, d_out)
        """
        if self.parallel_mode == 'column':
            # Moi GPU tinh 1 phan output
            outputs = [x @ shard for shard in self.W_shards]  # SONG SONG
            # Concat (trong thuc te: giu rieng neu W2 la row parallel)
            return np.concatenate(outputs, axis=-1)
        else:  # row
            chunk_size = self.d_in // self.num_gpus
            partials = []
            for i, shard in enumerate(self.W_shards):
                x_chunk = x[:, i * chunk_size:(i + 1) * chunk_size]
                partials.append(x_chunk @ shard)  # SONG SONG
            # All-Reduce (sum)
            output = np.sum(partials, axis=0)
            return output

    def memory_per_gpu(self):
        """Memory moi GPU phai giu (bytes, FP32)."""
        return self.W_shards[0].nbytes

    def total_memory(self):
        """Tong memory neu khong chia (bytes, FP32)."""
        return self.d_in * self.d_out * 4  # FP32


class TensorParallelFFN:
    """
    FFN voi Tensor Parallelism (Megatron-LM style).

    FFN(x) = GELU(x @ W1) @ W2

    W1: COLUMN parallel -> moi GPU co W1[:, chunk]
    W2: ROW parallel -> moi GPU co W2[chunk, :]

    Communication:
    - Giua W1 va W2: KHONG CAN (output W1 da tren dung GPU!)
    - Sau W2: 1 All-Reduce (sum)
    -> Chi 1 communication step cho toan bo FFN
    """

    def __init__(self, d_model, d_ff, num_gpus=2):
        """
        d_model: kich thuoc model
                 Vd: 4096 (LLaMA 7B), 12288 (GPT-3)

        d_ff:    kich thuoc FFN hidden (thuong = 4 * d_model hoac custom)
                 Vd: 11008 (LLaMA 7B, SwiGLU), 49152 (GPT-3)
                 PHAI chia het cho num_gpus

        num_gpus: so GPU
                  Vd: 2, 4, 8
        """
        self.num_gpus = num_gpus
        self.d_model = d_model
        self.d_ff = d_ff

        # W1: column parallel (d_model, d_ff) -> moi GPU co (d_model, d_ff/N)
        self.W1 = TensorParallelLinear(d_model, d_ff, num_gpus, 'column')

        # W2: row parallel (d_ff, d_model) -> moi GPU co (d_ff/N, d_model)
        self.W2 = TensorParallelLinear(d_ff, d_model, num_gpus, 'row')

    def forward(self, x):
        """
        x: shape (batch_size, d_model)

        Step 1: h = GELU(x @ W1)  <- column parallel, KHONG CAN communication
        Step 2: output = h @ W2   <- row parallel, can All-Reduce (sum)

        Trong thuc te, step 1 va 2 overlap voi communication.
        """
        # Column parallel: moi GPU co 1 phan cua hidden
        hidden_parts = [x @ shard for shard in self.W1.W_shards]

        # GELU activation (tren moi GPU doc lap, KHONG CAN sync)
        hidden_parts = [np.maximum(0, h) * (1 + np.tanh(np.sqrt(2 / np.pi) * (h + 0.044715 * h ** 3)))
                        if False else np.maximum(h, 0)  # ReLU cho don gian
                        for h in hidden_parts]

        # Row parallel: moi GPU nhan hidden_part voi W2_shard
        partials = [hidden_parts[i] @ self.W2.W_shards[i]
                    for i in range(self.num_gpus)]

        # All-Reduce (sum) - diem communication DUY NHAT!
        output = np.sum(partials, axis=0)
        return output


# ============================================================
# BAI TAP 3: PIPELINE PARALLELISM
# ============================================================
#
# PIPELINE PARALLELISM:
#
# Chia MODEL thanh STAGES, moi stage tren 1 GPU:
#   GPU 0: Layers 0-5    (Stage 0)
#   GPU 1: Layers 6-11   (Stage 1)
#   GPU 2: Layers 12-17  (Stage 2)
#   GPU 3: Layers 18-23  (Stage 3)
#
# VAN DE: PIPELINE BUBBLE
# Naive (1 batch):
#   GPU 0: [===F===][          ][===B===]
#   GPU 1: [       ][===F===][       ][===B===]
#   GPU 2: [       ][       ][===F===][       ][===B===]
#   GPU 3: [       ][       ][       ][===F===][===B===]
#   -> 75% thoi gian GPU IDLE (bubble!)
#
# GIAI PHAP: Micro-batching (GPipe)
# Chia batch thanh M micro-batches:
#   GPU 0: [F0][F1][F2][F3][B3][B2][B1][B0]
#   GPU 1:    [F0][F1][F2][F3][B3][B2][B1][B0]
#   GPU 2:       [F0][F1][F2][F3][B3][B2][B1][B0]
#   GPU 3:          [F0][F1][F2][F3][B3][B2][B1][B0]
#   -> Bubble ratio = (N-1) / (M + N - 1)
#      N=4 stages, M=8 microbatches -> bubble = 3/11 = 27%
#      N=4 stages, M=32 microbatches -> bubble = 3/35 = 8.6%

class PipelineStage:
    """
    1 stage trong Pipeline Parallelism.
    Moi stage chua 1 hoac nhieu layers tren 1 GPU.
    """

    def __init__(self, d_in, d_out, stage_id):
        """
        d_in:     so chieu input cua stage
        d_out:    so chieu output cua stage
        stage_id: index cua stage (0, 1, 2, ...)
                  Dung de tracking va debug
        """
        self.stage_id = stage_id
        self.layer = SimpleLinear(d_in, d_out)
        self.compute_time = 0.0

    def forward(self, x):
        """Forward qua stage, do thoi gian compute."""
        t0 = time.time()
        output = self.layer.forward(x)
        output = np.maximum(output, 0)  # ReLU
        self.compute_time = time.time() - t0
        return output

    def backward(self, grad):
        """Backward qua stage."""
        t0 = time.time()
        grad_input = self.layer.backward(grad)
        self.compute_time += time.time() - t0
        return grad_input


class PipelineParallelSimulator:
    """
    Simulate Pipeline Parallelism voi micro-batching (GPipe style).

    CACH HOAT DONG:
    1. Chia batch thanh M micro-batches
    2. Pipeline: micro-batch 0 qua stage 0, roi stage 1, ...
       Trong khi do, micro-batch 1 bat dau vao stage 0
    3. Sau tat ca forward: backward tu stage cuoi ve stage dau
    4. Accumulate gradients tu tat ca micro-batches, roi update 1 lan

    BUBBLE TIME:
    - Naive (M=1): bubble = (N-1) / (2*N - 1) ~ 50% cho N lon
    - GPipe (M micro-batches): bubble = (N-1) / (M + N - 1)
    - 1F1B (PipeDream): bubble = (N-1) / (2*M + N - 1) (tot hon)
    """

    def __init__(self, stage_configs, num_micro_batches=4):
        """
        stage_configs:      list of (d_in, d_out) tuples, moi tuple la 1 stage
                            Vd: [(768, 512), (512, 512), (512, 256), (256, 10)]
                            Stage 0 tren GPU 0, Stage 1 tren GPU 1, ...

        num_micro_batches:  so micro-batches de chia batch
                            Vd: 4 (demo), 8-32 (thuc te)
                            Nhieu hon -> it bubble nhung tang memory (luu activations)
                            GPipe paper dung M = 32 cho pipeline 4 stages
        """
        self.stages = []
        for i, (d_in, d_out) in enumerate(stage_configs):
            self.stages.append(PipelineStage(d_in, d_out, stage_id=i))
        self.num_stages = len(self.stages)
        self.num_micro_batches = num_micro_batches

    def forward_pass(self, X):
        """
        Pipeline forward pass voi micro-batching.

        X: input, shape (total_batch, d_in)
           Se duoc chia thanh num_micro_batches phan

        Returns: list of outputs (1 per micro-batch)
        """
        micro_batch_size = X.shape[0] // self.num_micro_batches
        activations = []  # Luu activations cho backward

        outputs = []
        for mb in range(self.num_micro_batches):
            start = mb * micro_batch_size
            end = start + micro_batch_size
            x = X[start:end]

            mb_activations = [x]
            for stage in self.stages:
                x = stage.forward(x)
                mb_activations.append(x)

            activations.append(mb_activations)
            outputs.append(x)

        self._activations = activations
        return outputs

    def compute_bubble_time(self):
        """
        Tinh bubble time ratio.

        Bubble ratio = (num_stages - 1) / (num_micro_batches + num_stages - 1)

        Vi du:
        - 4 stages, 4 micro-batches: bubble = 3/7 = 42.9%
        - 4 stages, 8 micro-batches: bubble = 3/11 = 27.3%
        - 4 stages, 32 micro-batches: bubble = 3/35 = 8.6%
        - 8 stages, 64 micro-batches: bubble = 7/71 = 9.9%
        """
        N = self.num_stages
        M = self.num_micro_batches
        bubble_ratio = (N - 1) / (M + N - 1)
        return {
            'num_stages': N,
            'num_micro_batches': M,
            'bubble_ratio': bubble_ratio,
            'efficiency': 1 - bubble_ratio,
            'bubble_slots': N - 1,
            'total_slots': M + N - 1,
        }

    def visualize_pipeline(self):
        """
        In ASCII art cua pipeline schedule.

        F = Forward, B = Backward, . = Idle (bubble)
        """
        N = self.num_stages
        M = self.num_micro_batches
        total_time = 2 * M + 2 * (N - 1)

        lines = []
        for stage in range(N):
            line = f"  GPU {stage}: "
            slots = []

            # Forward phase: stage `stage` bat dau forward tu time=stage
            for t in range(total_time):
                # Forward slots
                mb_fwd = t - stage  # micro-batch index for forward
                mb_bwd = t - (M + N - 1) - (N - 1 - stage)  # backward
                # Simplified: chinh xac hon
                fwd_start = stage
                fwd_end = stage + M
                bwd_start = M + N - 1 + (N - 1 - stage)
                bwd_end = bwd_start + M

                if fwd_start <= t < fwd_end:
                    mb_id = t - fwd_start
                    slots.append(f"F{mb_id}")
                elif bwd_start <= t < bwd_end:
                    mb_id = M - 1 - (t - bwd_start)
                    slots.append(f"B{mb_id}")
                else:
                    slots.append(" .")

            line += "|".join(f"{s:>2}" for s in slots[:min(len(slots), 20)])
            if len(slots) > 20:
                line += "|..."
            lines.append(line)

        return "\n".join(lines)


# ============================================================
# BAI TAP 4: ZeRO OPTIMIZER
# ============================================================
#
# ZeRO (Zero Redundancy Optimizer) - Microsoft DeepSpeed
#
# VAN DE: Data Parallelism lang phi memory
# Moi GPU giu: model params + gradients + optimizer states
# Voi Adam optimizer, FP16 mixed precision, model 7.5B params:
#   - Params (FP16):          15 GB
#   - Gradients (FP16):       15 GB
#   - Optimizer (FP32):       90 GB (params copy + momentum + variance)
#   - TONG:                  120 GB per GPU (!!!)
# Va TAT CA GPU giu ban sao GIONG NHAU!
#
# ZeRO STAGES:
# Stage 1: Partition OPTIMIZER STATES
#   - Moi GPU chi giu 1/N optimizer states
#   - Khi can update: gather tu GPU khac
#   - Memory: params(15) + grads(15) + opt_states(90/N)
#   - N=64: 15 + 15 + 1.4 = 31.4 GB
#
# Stage 2: + Partition GRADIENTS
#   - Moi GPU chi giu 1/N gradients
#   - Memory: params(15) + grads(15/N) + opt_states(90/N)
#   - N=64: 15 + 0.23 + 1.4 = 16.6 GB
#
# Stage 3: + Partition PARAMETERS
#   - Moi GPU chi giu 1/N params!
#   - Khi can: gather params tu GPU khac (all-gather)
#   - Memory: params(15/N) + grads(15/N) + opt_states(90/N)
#   - N=64: 0.23 + 0.23 + 1.4 = 1.9 GB
#   -> TU 120 GB xuong 1.9 GB!

class ZeROSimulator:
    """
    Simulate ZeRO optimizer stages.

    ZeRO giam memory KICH THUOC cho Data Parallelism
    bang cach PARTITION (chia) optimizer states, gradients, va params
    ra nhieu GPUs thay vi replicate.
    """

    def __init__(self, param_count, num_gpus=8, dtype_bytes=2):
        """
        param_count: so parameters cua model
                     Vd: 7_500_000_000 (7.5B), 175_000_000_000 (175B)
                     Dung de tinh memory usage

        num_gpus:    so GPUs trong cluster
                     Vd: 8 (1 node), 64 (8 nodes), 1024 (GPT-3 scale)
                     Nhieu GPU -> memory per GPU giam (ZeRO stage 3)

        dtype_bytes: bytes per parameter
                     2 = FP16/BF16 (pho bien nhat cho training)
                     4 = FP32
                     Mixed precision: params FP16, optimizer FP32
        """
        self.param_count = param_count
        self.num_gpus = num_gpus
        self.dtype_bytes = dtype_bytes

    def memory_no_zero(self):
        """
        Memory khi KHONG dung ZeRO (Data Parallel binh thuong).
        Moi GPU giu FULL: params + grads + optimizer states.

        Optimizer states (Adam, FP32):
        - FP32 copy of params: param_count * 4 bytes
        - Momentum (FP32): param_count * 4 bytes
        - Variance (FP32): param_count * 4 bytes
        -> 12 bytes per param cho optimizer
        """
        param_mem = self.param_count * self.dtype_bytes
        grad_mem = self.param_count * self.dtype_bytes
        # Adam optimizer states: fp32 copy + momentum + variance
        opt_mem = self.param_count * 12  # 3 * 4 bytes per param

        return {
            'params_bytes': param_mem,
            'gradients_bytes': grad_mem,
            'optimizer_bytes': opt_mem,
            'total_bytes': param_mem + grad_mem + opt_mem,
            'total_gb': (param_mem + grad_mem + opt_mem) / (1024 ** 3),
            'per_gpu_gb': (param_mem + grad_mem + opt_mem) / (1024 ** 3),
        }

    def memory_stage1(self):
        """
        ZeRO Stage 1: Partition optimizer states.
        Moi GPU giu: FULL params + FULL grads + 1/N optimizer states.
        """
        param_mem = self.param_count * self.dtype_bytes
        grad_mem = self.param_count * self.dtype_bytes
        opt_mem = self.param_count * 12 / self.num_gpus  # PARTITIONED

        return {
            'params_bytes': param_mem,
            'gradients_bytes': grad_mem,
            'optimizer_bytes': opt_mem,
            'total_bytes': param_mem + grad_mem + opt_mem,
            'per_gpu_gb': (param_mem + grad_mem + opt_mem) / (1024 ** 3),
        }

    def memory_stage2(self):
        """
        ZeRO Stage 2: Partition optimizer states + gradients.
        Moi GPU giu: FULL params + 1/N grads + 1/N optimizer states.
        """
        param_mem = self.param_count * self.dtype_bytes
        grad_mem = self.param_count * self.dtype_bytes / self.num_gpus  # PARTITIONED
        opt_mem = self.param_count * 12 / self.num_gpus

        return {
            'params_bytes': param_mem,
            'gradients_bytes': grad_mem,
            'optimizer_bytes': opt_mem,
            'total_bytes': param_mem + grad_mem + opt_mem,
            'per_gpu_gb': (param_mem + grad_mem + opt_mem) / (1024 ** 3),
        }

    def memory_stage3(self):
        """
        ZeRO Stage 3: Partition EVERYTHING (params + grads + optimizer).
        Moi GPU giu: 1/N params + 1/N grads + 1/N optimizer states.
        """
        param_mem = self.param_count * self.dtype_bytes / self.num_gpus  # PARTITIONED
        grad_mem = self.param_count * self.dtype_bytes / self.num_gpus
        opt_mem = self.param_count * 12 / self.num_gpus

        return {
            'params_bytes': param_mem,
            'gradients_bytes': grad_mem,
            'optimizer_bytes': opt_mem,
            'total_bytes': param_mem + grad_mem + opt_mem,
            'per_gpu_gb': (param_mem + grad_mem + opt_mem) / (1024 ** 3),
        }

    def communication_cost(self, stage):
        """
        Communication cost per training step.

        stage: 1, 2, hoac 3

        Stage 1: 1 All-Reduce cho gradients = 2 * param_count * dtype_bytes
        Stage 2: 1 Reduce-Scatter + 1 All-Gather = 2 * param_count * dtype_bytes
        Stage 3: + All-Gather params truoc forward + Reduce-Scatter grads
                 = ~3 * param_count * dtype_bytes (nhieu hon)
        """
        base_comm = 2 * self.param_count * self.dtype_bytes
        if stage == 1:
            return {'bytes': base_comm, 'description': 'All-Reduce gradients'}
        elif stage == 2:
            return {'bytes': base_comm, 'description': 'Reduce-Scatter + All-Gather'}
        else:  # stage 3
            return {'bytes': int(base_comm * 1.5),
                    'description': 'All-Gather params + Reduce-Scatter grads'}

    def summary_table(self):
        """In bang so sanh memory cua cac stages."""
        no_zero = self.memory_no_zero()
        s1 = self.memory_stage1()
        s2 = self.memory_stage2()
        s3 = self.memory_stage3()

        return {
            'No ZeRO': no_zero['per_gpu_gb'],
            'Stage 1': s1['per_gpu_gb'],
            'Stage 2': s2['per_gpu_gb'],
            'Stage 3': s3['per_gpu_gb'],
            'savings_s1': 1 - s1['per_gpu_gb'] / no_zero['per_gpu_gb'],
            'savings_s2': 1 - s2['per_gpu_gb'] / no_zero['per_gpu_gb'],
            'savings_s3': 1 - s3['per_gpu_gb'] / no_zero['per_gpu_gb'],
        }


# ============================================================
# BAI TAP 5: GRADIENT CHECKPOINTING
# ============================================================
#
# GRADIENT CHECKPOINTING (Activation Checkpointing):
#
# VAN DE: Forward pass luu TAT CA activations cho backward.
#
# GPT-3 175B, batch_size=1, seq_len=2048:
# - Moi layer: activations ~ batch * seq * d_model * 4 bytes
# - = 1 * 2048 * 12288 * 4 = ~100 MB per layer
# - 96 layers: ~9.6 GB chi cho activations!
# - Voi batch_size=8: ~77 GB -> KHONG DU VRAM
#
# GIAI PHAP:
# - Chi luu activations tai 1 so "checkpoints" (vd: moi 3 layers)
# - Khi backward can activation cua layer khong luu:
#   -> RE-COMPUTE tu checkpoint gan nhat
#
# TRADE-OFF:
# - Memory: tu O(N) xuong O(sqrt(N)) (N = so layers)
# - Compute: tang ~33% (re-compute 1 lan moi segment)
#
# Optimal: dat checkpoint moi sqrt(N) layers
# - 96 layers: checkpoint moi 10 layers -> 10 checkpoints
# - Memory: 10 * 100MB = 1GB thay vi 9.6GB (giam 9.6x!)
# - Compute: tang ~33% (acceptable)

class GradientCheckpointing:
    """
    Simulate va tinh toan memory savings cua gradient checkpointing.

    CACH HOAT DONG:
    1. Forward: chi luu activations tai checkpoint layers
    2. Backward: khi can activation cua layer khong luu:
       a. Load activation tu checkpoint gan nhat
       b. Re-compute forward tu checkpoint den layer can
       c. Dung activation vua tinh de backward
       d. Free activation sau khi dung xong

    OPTIMAL STRATEGY:
    - Dat checkpoint moi sqrt(N) layers
    - Memory: O(sqrt(N)) thay vi O(N)
    - Compute overhead: ~33% (re-compute 1 lan)
    """

    def __init__(self, num_layers, d_model, batch_size, seq_len, checkpoint_every=None):
        """
        num_layers:       so layers trong model
                          Vd: 12 (GPT-2 small), 96 (GPT-3), 80 (LLaMA 65B)

        d_model:          kich thuoc model dimension
                          Vd: 768 (GPT-2), 12288 (GPT-3), 8192 (LLaMA 65B)

        batch_size:       so samples moi batch
                          Vd: 1-8 cho inference, 32-512 cho training

        seq_len:          do dai sequence
                          Vd: 1024 (GPT-2), 2048 (GPT-3), 4096 (LLaMA 2)

        checkpoint_every: dat checkpoint moi N layers
                          None = tu dong chon sqrt(num_layers) (optimal)
                          Vd: 3 (moi 3 layers), 10 (moi 10 layers)
                          Nho hon -> it memory nhung nhieu re-compute
                          Lon hon -> nhieu memory nhung it re-compute
        """
        self.num_layers = num_layers
        self.d_model = d_model
        self.batch_size = batch_size
        self.seq_len = seq_len

        if checkpoint_every is None:
            self.checkpoint_every = max(1, int(np.sqrt(num_layers)))
        else:
            self.checkpoint_every = checkpoint_every

        # Tinh activation size per layer
        # Moi layer luu: input activation shape (batch, seq_len, d_model)
        self.activation_bytes_per_layer = batch_size * seq_len * d_model * 4  # FP32

    def memory_no_checkpointing(self):
        """Memory khi luu TAT CA activations (binh thuong)."""
        total = self.num_layers * self.activation_bytes_per_layer
        return {
            'total_bytes': total,
            'total_gb': total / (1024 ** 3),
            'num_saved_activations': self.num_layers,
        }

    def memory_with_checkpointing(self):
        """Memory khi dung gradient checkpointing."""
        # So checkpoint layers
        num_checkpoints = self.num_layers // self.checkpoint_every
        if self.num_layers % self.checkpoint_every != 0:
            num_checkpoints += 1

        # Memory cho checkpointed activations
        checkpoint_mem = num_checkpoints * self.activation_bytes_per_layer

        # Memory cho re-computed activations (1 segment tai 1 thoi diem)
        segment_mem = self.checkpoint_every * self.activation_bytes_per_layer

        # Peak memory = checkpoints + 1 segment
        peak_mem = checkpoint_mem + segment_mem

        return {
            'checkpoint_bytes': checkpoint_mem,
            'segment_bytes': segment_mem,
            'peak_bytes': peak_mem,
            'peak_gb': peak_mem / (1024 ** 3),
            'num_checkpoints': num_checkpoints,
            'segment_size': self.checkpoint_every,
        }

    def compute_overhead(self):
        """
        Tinh compute overhead cua gradient checkpointing.

        Moi segment giua 2 checkpoints phai re-compute 1 lan trong backward.
        Overhead = (so layers re-compute) / (so layers forward)
        """
        # Moi segment co checkpoint_every layers
        # Moi segment re-compute 1 lan (tru segment cuoi)
        num_segments = self.num_layers // self.checkpoint_every
        recompute_layers = num_segments * self.checkpoint_every
        overhead = recompute_layers / self.num_layers

        return {
            'overhead_ratio': overhead,
            'overhead_pct': overhead * 100,
            'recompute_layers': recompute_layers,
            'total_forward_layers': self.num_layers + recompute_layers,
        }

    def savings_summary(self):
        """Tong ket memory savings va compute overhead."""
        no_ckpt = self.memory_no_checkpointing()
        with_ckpt = self.memory_with_checkpointing()
        overhead = self.compute_overhead()

        savings = 1 - with_ckpt['peak_gb'] / no_ckpt['total_gb']

        return {
            'no_checkpoint_gb': no_ckpt['total_gb'],
            'with_checkpoint_gb': with_ckpt['peak_gb'],
            'memory_savings_pct': savings * 100,
            'compute_overhead_pct': overhead['overhead_pct'],
            'num_checkpoints': with_ckpt['num_checkpoints'],
            'checkpoint_every': self.checkpoint_every,
        }


# ============================================================
# HELPER: 3D Parallelism Calculator
# ============================================================

def calculate_3d_parallelism(model_params, num_gpus, dp=None, tp=None, pp=None):
    """
    Tinh memory va communication cho 3D parallelism.

    Trong thuc te, LLM training dung KET HOP ca 3 loai:
    - Data Parallel (DP): replicate model, split data
    - Tensor Parallel (TP): split layers, heavy communication
    - Pipeline Parallel (PP): split layer groups, micro-batching

    Vd: GPT-3 175B tren 1024 GPUs:
    - TP = 8 (trong 1 node, dung NVLink nhanh)
    - PP = 16 (giua cac nodes)
    - DP = 8 (1024 / 8 / 16 = 8)
    -> 8 * 16 * 8 = 1024 GPUs

    model_params: so params cua model (vd: 175_000_000_000)
    num_gpus:     tong so GPUs (vd: 1024)
    dp, tp, pp:   so GPUs cho moi chieu parallelism
                  dp * tp * pp = num_gpus
    """
    if dp is None or tp is None or pp is None:
        # Auto-configure
        tp = min(8, num_gpus)  # Max 8 (1 node)
        remaining = num_gpus // tp
        pp = min(remaining, max(1, model_params // (20_000_000_000 * tp)))
        dp = num_gpus // (tp * pp)

    assert dp * tp * pp == num_gpus, f"dp*tp*pp={dp*tp*pp} != num_gpus={num_gpus}"

    # Memory per GPU
    params_per_tp_pp = model_params / (tp * pp)  # Chia theo TP va PP
    param_mem = params_per_tp_pp * 2  # FP16
    grad_mem = params_per_tp_pp * 2
    opt_mem = params_per_tp_pp * 12  # Adam FP32

    return {
        'dp': dp, 'tp': tp, 'pp': pp,
        'total_gpus': dp * tp * pp,
        'params_per_gpu': params_per_tp_pp,
        'memory_per_gpu_gb': (param_mem + grad_mem + opt_mem) / (1024 ** 3),
        'effective_batch_multiplier': dp,
    }


# ============================================================
# MAIN: Test tat ca implementations
# ============================================================
if __name__ == "__main__":
    np.random.seed(42)
    output_dir = os.path.dirname(os.path.abspath(__file__))

    # ============================================================
    # PHAN 1: DATA PARALLELISM
    # ============================================================
    print("=" * 70)
    print("PHAN 1: DATA PARALLELISM")
    print("=" * 70)

    d_in, d_out = 128, 32
    num_gpus = 4
    total_batch = 128  # 128 / 4 = 32 per GPU

    dp = DataParallelSimulator(
        model_fn=lambda: SimpleLinear(d_in, d_out),
        num_gpus=num_gpus
    )

    X = np.random.randn(total_batch, d_in).astype(np.float32)
    y = np.random.randn(total_batch, d_out).astype(np.float32)

    print(f"\n  Config: d_in={d_in}, d_out={d_out}, num_gpus={num_gpus}")
    print(f"  Total batch: {total_batch}, per GPU: {total_batch // num_gpus}")

    dp_history = []
    for step in range(50):
        metrics = dp.train_step(X, y, lr=0.01)
        dp_history.append(metrics)
        if step % 10 == 0:
            print(f"    Step {step:>3}: loss={metrics['loss']:.6f}, "
                  f"grad_norm={metrics['grad_norm']:.4f}")

    print(f"  Final loss: {dp_history[-1]['loss']:.6f}")
    print(f"  Communication per step: {dp_history[0]['communication_bytes']:,} bytes")

    # Verify weights dong bo
    for i in range(1, num_gpus):
        assert np.allclose(dp.models[0].W, dp.models[i].W), f"GPU {i} out of sync!"
    print(f"  All {num_gpus} GPUs synchronized: OK")

    # ============================================================
    # PHAN 2: TENSOR PARALLELISM
    # ============================================================
    print("\n" + "=" * 70)
    print("PHAN 2: TENSOR PARALLELISM")
    print("=" * 70)

    d_model = 256
    d_ff = 1024  # 4 * d_model
    tp_gpus = 4

    # --- Column Parallel ---
    print(f"\n  --- Column Parallel (W1: split output dim) ---")
    tp_col = TensorParallelLinear(d_model, d_ff, tp_gpus, 'column')
    x_tp = np.random.randn(8, d_model).astype(np.float32)
    out_col = tp_col.forward(x_tp)

    print(f"  W shape: ({d_model}, {d_ff}) -> {tp_gpus} shards of ({d_model}, {d_ff // tp_gpus})")
    print(f"  Input:  {x_tp.shape}")
    print(f"  Output: {out_col.shape}")
    print(f"  Memory per GPU: {tp_col.memory_per_gpu() / 1024:.1f} KB "
          f"(vs {tp_col.total_memory() / 1024:.1f} KB total)")

    # --- Row Parallel ---
    print(f"\n  --- Row Parallel (W2: split input dim) ---")
    tp_row = TensorParallelLinear(d_ff, d_model, tp_gpus, 'row')
    x_row = np.random.randn(8, d_ff).astype(np.float32)
    out_row = tp_row.forward(x_row)

    print(f"  W shape: ({d_ff}, {d_model}) -> {tp_gpus} shards of ({d_ff // tp_gpus}, {d_model})")
    print(f"  Input:  {x_row.shape}")
    print(f"  Output: {out_row.shape}")

    # --- Full FFN ---
    print(f"\n  --- Tensor Parallel FFN (Column W1 + Row W2) ---")
    tp_ffn = TensorParallelFFN(d_model, d_ff, tp_gpus)
    x_ffn = np.random.randn(8, d_model).astype(np.float32)
    out_ffn = tp_ffn.forward(x_ffn)

    print(f"  FFN: ({d_model}) -> ({d_ff}) -> ({d_model})")
    print(f"  Input:  {x_ffn.shape}")
    print(f"  Output: {out_ffn.shape}")
    print(f"  Communication: 1 All-Reduce sau W2 (chi 1 lan cho toan bo FFN!)")
    assert out_ffn.shape == x_ffn.shape, "FFN output shape should match input"
    print(f"  Tensor Parallel FFN: OK")

    # --- Real-world scale ---
    print(f"\n  --- Real-world Memory Savings ---")
    configs = [
        ("LLaMA 7B FFN", 4096, 11008),
        ("LLaMA 13B FFN", 5120, 13824),
        ("LLaMA 65B FFN", 8192, 22016),
        ("GPT-3 FFN", 12288, 49152),
    ]
    for name, dm, df in configs:
        total = dm * df * 4 / (1024 ** 2)  # FP32 MB
        per_gpu_2 = total / 2
        per_gpu_4 = total / 4
        per_gpu_8 = total / 8
        print(f"  {name:>20}: total={total:>8.1f} MB | "
              f"2 GPU={per_gpu_2:>7.1f} MB | "
              f"4 GPU={per_gpu_4:>7.1f} MB | "
              f"8 GPU={per_gpu_8:>7.1f} MB")

    # ============================================================
    # PHAN 3: PIPELINE PARALLELISM
    # ============================================================
    print("\n" + "=" * 70)
    print("PHAN 3: PIPELINE PARALLELISM")
    print("=" * 70)

    stages = [(128, 64), (64, 64), (64, 32), (32, 16)]
    num_microbatches = 8

    pp = PipelineParallelSimulator(stages, num_micro_batches=num_microbatches)

    print(f"\n  Config: {len(stages)} stages, {num_microbatches} micro-batches")
    for i, (di, do) in enumerate(stages):
        print(f"    Stage {i} (GPU {i}): ({di}) -> ({do})")

    # Forward pass
    X_pp = np.random.randn(num_microbatches * 16, stages[0][0]).astype(np.float32)
    outputs = pp.forward_pass(X_pp)
    print(f"\n  Input: {X_pp.shape}")
    print(f"  Output per micro-batch: {outputs[0].shape}")

    # Bubble time
    bubble = pp.compute_bubble_time()
    print(f"\n  Pipeline Bubble Analysis:")
    print(f"    Stages: {bubble['num_stages']}")
    print(f"    Micro-batches: {bubble['num_micro_batches']}")
    print(f"    Bubble ratio: {bubble['bubble_ratio']:.1%}")
    print(f"    Efficiency: {bubble['efficiency']:.1%}")

    # So sanh bubble voi so micro-batches khac nhau
    print(f"\n  Bubble ratio vs num_micro_batches (4 stages):")
    for M in [1, 2, 4, 8, 16, 32, 64]:
        ratio = (4 - 1) / (M + 4 - 1)
        bar = "#" * int(ratio * 40)
        print(f"    M={M:>3}: bubble={ratio:.1%} {bar}")

    # Pipeline visualization
    print(f"\n  Pipeline Schedule (simplified):")
    print(pp.visualize_pipeline())

    # ============================================================
    # PHAN 4: ZeRO OPTIMIZER
    # ============================================================
    print("\n" + "=" * 70)
    print("PHAN 4: ZeRO OPTIMIZER")
    print("=" * 70)

    # --- LLaMA 7B ---
    print(f"\n  --- LLaMA 7B (7B params, 64 GPUs, FP16) ---")
    zero_7b = ZeROSimulator(param_count=7_000_000_000, num_gpus=64, dtype_bytes=2)
    summary_7b = zero_7b.summary_table()

    print(f"  {'Stage':<12} | {'Memory/GPU':>12} | {'Savings':>10}")
    print(f"  {'-'*12}-+-{'-'*12}-+-{'-'*10}")
    for stage in ['No ZeRO', 'Stage 1', 'Stage 2', 'Stage 3']:
        mem = summary_7b[stage]
        savings_key = f"savings_{stage.lower().replace(' ', '')}" if stage != 'No ZeRO' else None
        savings = f"{summary_7b[savings_key]*100:.0f}%" if savings_key and savings_key in summary_7b else "-"
        print(f"  {stage:<12} | {mem:>9.1f} GB | {savings:>10}")

    # --- GPT-3 175B ---
    print(f"\n  --- GPT-3 175B (175B params, 1024 GPUs, FP16) ---")
    zero_175b = ZeROSimulator(param_count=175_000_000_000, num_gpus=1024, dtype_bytes=2)
    summary_175b = zero_175b.summary_table()

    print(f"  {'Stage':<12} | {'Memory/GPU':>12} | {'Savings':>10}")
    print(f"  {'-'*12}-+-{'-'*12}-+-{'-'*10}")
    for stage in ['No ZeRO', 'Stage 1', 'Stage 2', 'Stage 3']:
        mem = summary_175b[stage]
        savings_key = f"savings_{stage.lower().replace(' ', '')}" if stage != 'No ZeRO' else None
        savings = f"{summary_175b[savings_key]*100:.0f}%" if savings_key and savings_key in summary_175b else "-"
        print(f"  {stage:<12} | {mem:>9.1f} GB | {savings:>10}")

    # Communication cost
    print(f"\n  Communication Cost (GPT-3 175B):")
    for stage in [1, 2, 3]:
        comm = zero_175b.communication_cost(stage)
        gb = comm['bytes'] / (1024 ** 3)
        print(f"    Stage {stage}: {gb:.1f} GB/step ({comm['description']})")

    # ============================================================
    # PHAN 5: GRADIENT CHECKPOINTING
    # ============================================================
    print("\n" + "=" * 70)
    print("PHAN 5: GRADIENT CHECKPOINTING")
    print("=" * 70)

    # GPT-2 Small
    print(f"\n  --- GPT-2 Small (12 layers, d=768, batch=32, seq=1024) ---")
    gc_gpt2 = GradientCheckpointing(
        num_layers=12, d_model=768, batch_size=32, seq_len=1024
    )
    summary_gpt2 = gc_gpt2.savings_summary()
    print(f"  No checkpoint:   {summary_gpt2['no_checkpoint_gb']:.2f} GB")
    print(f"  With checkpoint: {summary_gpt2['with_checkpoint_gb']:.2f} GB")
    print(f"  Memory savings:  {summary_gpt2['memory_savings_pct']:.0f}%")
    print(f"  Compute overhead: +{summary_gpt2['compute_overhead_pct']:.0f}%")
    print(f"  Checkpoints: every {summary_gpt2['checkpoint_every']} layers "
          f"({summary_gpt2['num_checkpoints']} checkpoints)")

    # GPT-3
    print(f"\n  --- GPT-3 175B (96 layers, d=12288, batch=1, seq=2048) ---")
    gc_gpt3 = GradientCheckpointing(
        num_layers=96, d_model=12288, batch_size=1, seq_len=2048
    )
    summary_gpt3 = gc_gpt3.savings_summary()
    print(f"  No checkpoint:   {summary_gpt3['no_checkpoint_gb']:.2f} GB")
    print(f"  With checkpoint: {summary_gpt3['with_checkpoint_gb']:.2f} GB")
    print(f"  Memory savings:  {summary_gpt3['memory_savings_pct']:.0f}%")
    print(f"  Compute overhead: +{summary_gpt3['compute_overhead_pct']:.0f}%")
    print(f"  Checkpoints: every {summary_gpt3['checkpoint_every']} layers "
          f"({summary_gpt3['num_checkpoints']} checkpoints)")

    # LLaMA 65B
    print(f"\n  --- LLaMA 65B (80 layers, d=8192, batch=8, seq=4096) ---")
    gc_llama = GradientCheckpointing(
        num_layers=80, d_model=8192, batch_size=8, seq_len=4096
    )
    summary_llama = gc_llama.savings_summary()
    print(f"  No checkpoint:   {summary_llama['no_checkpoint_gb']:.2f} GB")
    print(f"  With checkpoint: {summary_llama['with_checkpoint_gb']:.2f} GB")
    print(f"  Memory savings:  {summary_llama['memory_savings_pct']:.0f}%")
    print(f"  Compute overhead: +{summary_llama['compute_overhead_pct']:.0f}%")
    print(f"  Checkpoints: every {summary_llama['checkpoint_every']} layers "
          f"({summary_llama['num_checkpoints']} checkpoints)")

    # So sanh checkpoint_every values
    print(f"\n  --- Checkpoint Frequency vs Savings (96 layers, GPT-3 config) ---")
    print(f"  {'Every N':>8} | {'Checkpoints':>12} | {'Memory GB':>10} | {'Savings':>8} | {'Overhead':>10}")
    print(f"  {'-'*8}-+-{'-'*12}-+-{'-'*10}-+-{'-'*8}-+-{'-'*10}")
    for every in [1, 2, 4, 8, 10, 16, 32, 96]:
        gc_test = GradientCheckpointing(96, 12288, 1, 2048, checkpoint_every=every)
        s = gc_test.savings_summary()
        print(f"  {every:>8} | {s['num_checkpoints']:>12} | {s['with_checkpoint_gb']:>9.2f} | "
              f"{s['memory_savings_pct']:>7.0f}% | +{s['compute_overhead_pct']:>8.0f}%")

    # ============================================================
    # PHAN 6: 3D PARALLELISM
    # ============================================================
    print("\n" + "=" * 70)
    print("PHAN 6: 3D PARALLELISM (KET HOP TAT CA)")
    print("=" * 70)

    configs_3d = [
        ("LLaMA 7B", 7_000_000_000, 8, 1, 1, 8),
        ("LLaMA 7B", 7_000_000_000, 64, 2, 4, 8),
        ("LLaMA 65B", 65_000_000_000, 64, 8, 8, 1),
        ("GPT-3 175B", 175_000_000_000, 1024, 8, 16, 8),
        ("PaLM 540B", 540_000_000_000, 6144, 8, 12, 64),
    ]

    print(f"\n  {'Model':>15} | {'GPUs':>5} | {'TP':>3} | {'PP':>3} | {'DP':>3} | {'Mem/GPU':>10}")
    print(f"  {'-'*15}-+-{'-'*5}-+-{'-'*3}-+-{'-'*3}-+-{'-'*3}-+-{'-'*10}")

    for name, params, gpus, tp, pp_val, dp in configs_3d:
        result = calculate_3d_parallelism(params, gpus, dp, tp, pp_val)
        print(f"  {name:>15} | {gpus:>5} | {tp:>3} | {pp_val:>3} | {dp:>3} | "
              f"{result['memory_per_gpu_gb']:>8.1f} GB")

    # ============================================================
    # TONG KET
    # ============================================================
    print("\n" + "=" * 70)
    print("TONG KET: DISTRIBUTED TRAINING")
    print("=" * 70)
    print("""
  1. DATA PARALLELISM (DP):
     - Don gian nhat: copy model, chia data, All-Reduce gradients
     - Communication: 2 * model_size per step
     - Dung khi: model vua 1 GPU, muon tang batch size
     - PyTorch: DistributedDataParallel (DDP)

  2. TENSOR PARALLELISM (TP):
     - Chia TUNG LAYER ra nhieu GPUs
     - Column parallel (W1) + Row parallel (W2) -> 1 All-Reduce/FFN
     - Dung khi: model TOO BIG cho 1 GPU
     - Can NVLink (nhanh) -> thuong trong 1 node (8 GPUs)
     - NVIDIA Megatron-LM

  3. PIPELINE PARALLELISM (PP):
     - Chia NHOM LAYERS ra nhieu GPUs
     - Micro-batching giam bubble time
     - Dung khi: nhieu nodes, model rat lon
     - GPipe, PipeDream, DeepSpeed

  4. ZeRO OPTIMIZER:
     - Partition optimizer states (S1), gradients (S2), params (S3)
     - S3: LINEAR memory scaling voi so GPUs!
     - DeepSpeed ZeRO, PyTorch FSDP

  5. GRADIENT CHECKPOINTING:
     - Trade compute (33%) cho memory (sqrt(N) reduction)
     - Dung boi TAT CA LLM training
     - Optimal: checkpoint moi sqrt(N) layers

  THUC TE (GPT-3 175B, 1024 A100 GPUs):
  - TP = 8 (trong 1 node, NVLink)
  - PP = 16 (giua cac nodes)
  - DP = 8 (1024 / 8 / 16)
  - ZeRO Stage 1 + Gradient Checkpointing
    """)

    # --- Plot ---
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Plot 1: Data Parallel training loss
        ax = axes[0, 0]
        ax.plot([m['loss'] for m in dp_history], color='blue')
        ax.set_xlabel('Step')
        ax.set_ylabel('Loss')
        ax.set_title('Data Parallelism: Training Loss')
        ax.grid(True, alpha=0.3)

        # Plot 2: Pipeline bubble vs micro-batches
        ax = axes[0, 1]
        ms = list(range(1, 65))
        bubbles = [(4 - 1) / (m + 4 - 1) for m in ms]
        ax.plot(ms, bubbles, color='red')
        ax.set_xlabel('Micro-batches')
        ax.set_ylabel('Bubble Ratio')
        ax.set_title('Pipeline Parallelism: Bubble vs Micro-batches')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0.1, color='green', linestyle='--', alpha=0.5, label='10% target')
        ax.legend()

        # Plot 3: ZeRO memory comparison
        ax = axes[1, 0]
        gpus_list = [8, 16, 32, 64, 128, 256, 512, 1024]
        for stage, color, label in [
            ('No ZeRO', 'red', 'No ZeRO'),
            ('Stage 1', 'orange', 'Stage 1'),
            ('Stage 2', 'blue', 'Stage 2'),
            ('Stage 3', 'green', 'Stage 3'),
        ]:
            mems = []
            for ng in gpus_list:
                z = ZeROSimulator(7_000_000_000, ng, 2)
                s = z.summary_table()
                mems.append(s[stage])
            ax.plot(gpus_list, mems, color=color, label=label, marker='o', markersize=3)
        ax.set_xlabel('Num GPUs')
        ax.set_ylabel('Memory per GPU (GB)')
        ax.set_title('ZeRO: Memory vs GPUs (7B model)')
        ax.set_xscale('log', base=2)
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 4: Gradient checkpointing savings
        ax = axes[1, 1]
        every_vals = list(range(1, 97))
        savings_vals = []
        overhead_vals = []
        for ev in every_vals:
            gc = GradientCheckpointing(96, 12288, 1, 2048, checkpoint_every=ev)
            s = gc.savings_summary()
            savings_vals.append(s['memory_savings_pct'])
            overhead_vals.append(s['compute_overhead_pct'])
        ax.plot(every_vals, savings_vals, color='blue', label='Memory Savings %')
        ax.plot(every_vals, overhead_vals, color='red', label='Compute Overhead %')
        ax.set_xlabel('Checkpoint Every N Layers')
        ax.set_ylabel('Percentage')
        ax.set_title('Gradient Checkpointing: Savings vs Overhead')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.suptitle('Week 17-18: Distributed Training', fontsize=14, fontweight='bold')
        plt.tight_layout()
        path = os.path.join(output_dir, "plot_distributed.png")
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"\n  Saved: {path}")

    except ImportError:
        print("  matplotlib chua cai.")

    print("\n" + "=" * 70)
    print("TAT CA TESTS PASSED!")
    print("=" * 70)


# ======== CHECKLIST ========
# Week 17-18 Distributed Training:
# [x] Understand Data/Tensor/Pipeline parallel
# [x] Study ZeRO optimizer
# [x] Implement gradient checkpointing
# [x] Run distributed training (neu co multi-GPU)
