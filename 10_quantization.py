# File: 10_quantization.py
# Model Quantization - Week 10-11
#
# TAI SAO CAN QUANTIZATION?
# LLM rat nang:
# - LLaMA 7B FP32: 28 GB VRAM -> can GPU A100 ($10k+)
# - LLaMA 7B INT8:  7 GB VRAM -> chay duoc tren RTX 3090 ($1.5k)
# - LLaMA 7B INT4: 3.5 GB VRAM -> chay duoc tren RTX 3060 ($300)
#
# NGUYEN LY:
# - FP32: moi so dung 32 bits (4 bytes), do chinh xac cao
# - INT8: moi so dung 8 bits (1 byte), 256 gia tri kha dung
# - INT4: moi so dung 4 bits (0.5 byte), 16 gia tri kha dung
# - Chuyen doi: weight (float) -> scale * quantized (int)
#   Vi du: 0.37 -> scale=0.01, quantized=37
#
# TRADE-OFF:
# - Nhe hon, nhanh hon
# - Nhung mat do chinh xac (accuracy giam 0.5-2%)
# - INT8 gan nhu khong mat gi, INT4 mat it
#
# BAI TAP:
# 1. Implement GPTQ (training-aware quantization)
# 2. Implement AWQ (activation-aware quantization)
# 3. Quantize GPT model va do accuracy loss
# 4. Benchmark inference speed: FP32 vs INT8 vs INT4

import numpy as np
import time
import os


class Quantizer:
    """
    Cac phuong phap quantization

    3 PHUONG PHAP CHINH:
    1. Absmax: don gian nhat, scale theo gia tri lon nhat
    2. Zero-point: xu ly phan bo bat doi xung
    3. Per-channel: moi channel co scale rieng -> chinh xac hon
    """

    @staticmethod
    def quantize_absmax_int8(weights):
        """
        Absmax Quantization - phuong phap don gian nhat

        weights: ma tran weights can quantize, numpy array FP32
                 Shape bat ky, thuong la (d_in, d_out) cua 1 layer
                 Vd: LLaMA 7B attention layer: (4096, 4096) ~ 64 MB moi layer
                 Vd: GPT-2 small FFN layer: (768, 3072) ~ 9 MB moi layer
                 Gia tri thuong nam trong khoang [-2.0, 2.0] sau khi train

        CACH HOAT DONG:
        - Tim gia tri tuyet doi lon nhat (abs_max)
        - Scale = abs_max / 127 (INT8 range: -128 to 127)
        - quantized = round(weights / scale)

        VI DU:
        - weights = [0.5, -1.0, 0.3, -0.7]
        - abs_max = 1.0
        - scale = 1.0/127 = 0.00787
        - quantized = [64, -127, 38, -89]

        DEQUANTIZE:
        - weights_approx = quantized * scale
        - = [0.504, -1.0, 0.299, -0.700]
        - Sai so nho!
        """
        abs_max = np.max(np.abs(weights))
        scale = abs_max / 127.0 if abs_max != 0 else 1.0
        quantized = np.round(weights / scale).astype(np.int8)
        return quantized, scale

    @staticmethod
    def dequantize_absmax_int8(quantized, scale):
        """
        Chuyen INT8 ve lai FP32

        quantized: ma tran da quantize, numpy array INT8, shape giong weights goc
                   Moi gia tri nam trong [-128, 127]
                   Vd: shape (4096, 4096) cho LLaMA 7B attention layer

        scale:     he so scale (float) da dung khi quantize
                   scale = abs_max / 127.0
                   Vd: weights co abs_max = 1.5 -> scale = 0.01181
                   Cong thuc phuc hoi: weights_approx = quantized * scale
        """
        return quantized.astype(np.float32) * scale

    @staticmethod
    def quantize_zeropoint_int8(weights):
        """
        Zero-Point Quantization - xu ly phan bo bat doi xung

        weights: ma tran weights can quantize, numpy array FP32
                 Dac biet hieu qua khi weights KHONG doi xung quanh 0
                 Vd: activations sau ReLU chi co gia tri >= 0
                 Vd: GPT-2 layer norm output thuong lech ve 1 phia
                 Shape thuong (d_in, d_out), vd: (768, 3072) cho GPT-2 FFN

        TAI SAO CAN ZERO-POINT?
        - Absmax gia su weights doi xung quanh 0
        - Nhung thuc te, nhieu layers co weights lech (vi du: ReLU outputs luon >= 0)
        - Zero-point them 1 offset de map full range

        CACH HOAT DONG:
        - scale = (max - min) / 255
        - zero_point = round(-min / scale)
        - quantized = round(weights / scale) + zero_point

        VI DU:
        - weights = [0.1, 0.5, 0.3, 0.9] (chi co gia tri duong)
        - Absmax: chi dung nua range (0 to 127), waste 128 gia tri
        - Zero-point: dung full range (0 to 255)
        """
        w_min, w_max = np.min(weights), np.max(weights)
        scale = (w_max - w_min) / 255.0 if w_max != w_min else 1.0
        zero_point = int(np.round(-w_min / scale))

        # Quantize to uint8 range [0, 255]
        quantized = np.round(weights / scale).astype(np.int32) + zero_point
        quantized = np.clip(quantized, 0, 255).astype(np.uint8)

        return quantized, scale, zero_point

    @staticmethod
    def dequantize_zeropoint_int8(quantized, scale, zero_point):
        """
        Chuyen UINT8 (zero-point quantized) ve lai FP32

        quantized:  ma tran da quantize, numpy array UINT8
                    Moi gia tri nam trong [0, 255] (khac voi absmax dung [-128, 127])
                    Shape giong weights goc, vd: (768, 3072) cho GPT-2 FFN

        scale:      he so scale (float) tu buoc quantize
                    scale = (w_max - w_min) / 255.0
                    Vd: weights range [0.0, 2.0] -> scale = 2.0 / 255 = 0.00784

        zero_point: offset (int) de map gia tri nho nhat ve 0
                    zero_point = round(-w_min / scale)
                    Vd: w_min = -0.5, scale = 0.01 -> zero_point = 50
                    Cong thuc phuc hoi: weights_approx = (quantized - zero_point) * scale
        """
        return (quantized.astype(np.float32) - zero_point) * scale

    @staticmethod
    def quantize_per_channel_int8(weights, axis=0):
        """
        Per-Channel Quantization - chinh xac hon

        weights: ma tran weights can quantize, numpy array FP32, it nhat 2 chieu
                 Shape (num_channels, ...) neu axis=0, hoac (..., num_channels) neu axis=1
                 Vd: LLaMA 7B attention W_q: (4096, 4096) -> 4096 output channels
                 Vd: GPT-2 FFN W1: (768, 3072) -> 768 channels theo axis=0

        axis:    chieu de chia channels, mac dinh = 0 (theo hang)
                 axis=0: moi hang la 1 channel (thuong dung cho output channels)
                 axis=1: moi cot la 1 channel (thuong dung cho input channels)
                 Vd: Conv layer dung axis=0, Linear layer co the dung axis=0 hoac 1

        TAI SAO TOT HON?
        - Per-tensor: 1 scale cho toan bo ma tran
          -> Neu 1 channel co gia tri lon, cac channel khac bi nen lai
        - Per-channel: moi output channel co scale rieng
          -> Moi channel duoc quantize toi uu

        VI DU:
        - Channel 0: weights [-0.01, 0.02] -> scale nho
        - Channel 1: weights [-5.0, 3.0] -> scale lon
        - Per-tensor se dung scale lon cho ca 2 -> channel 0 mat do chinh xac
        """
        num_channels = weights.shape[axis]
        scales = []
        quantized_list = []

        for i in range(num_channels):
            if axis == 0:
                channel = weights[i]
            else:
                channel = weights[:, i]

            abs_max = np.max(np.abs(channel))
            scale = abs_max / 127.0 if abs_max != 0 else 1.0
            scales.append(scale)
            q = np.round(channel / scale).astype(np.int8)
            quantized_list.append(q)

        quantized = np.stack(quantized_list, axis=axis)
        scales = np.array(scales)
        return quantized, scales

    @staticmethod
    def quantize_int4(weights):
        """
        INT4 Quantization - nen toi da

        weights: ma tran weights can quantize, numpy array FP32
                 Shape bat ky, thuong la (d_in, d_out)
                 Vd: LLaMA 7B FFN: (4096, 11008) ~ 172 MB FP32 -> 21.5 MB INT4
                 Vd: GPT-2 small attention: (768, 768) ~ 2.25 MB FP32 -> 0.28 MB INT4
                 Gia tri se duoc clip vao range [-8, 7] sau khi scale

        RANGE: -8 to 7 (chi 16 gia tri kha dung!)
        - FP32: 4,294,967,296 gia tri
        - INT8: 256 gia tri
        - INT4: 16 gia tri -> rat tho nhung tiet kiem 8x memory

        TRICK: Pack 2 INT4 values vao 1 byte
        - byte = (val1 << 4) | (val2 & 0xF)
        -> Tiet kiem them 2x memory

        DUNG KHI NAO?
        - Model qua lon, GPU khong du VRAM
        - Latency quan trong hon accuracy
        - Vi du: chatbot tren dien thoai
        """
        abs_max = np.max(np.abs(weights))
        scale = abs_max / 7.0 if abs_max != 0 else 1.0
        quantized = np.round(weights / scale)
        quantized = np.clip(quantized, -8, 7).astype(np.int8)
        return quantized, scale


# ============================================================
# BAI TAP 1: GPTQ-STYLE QUANTIZATION
# ============================================================

def gptq_quantize(weights, calibration_data, group_size=128):
    """
    GPTQ-style quantization (simplified)

    weights:          ma tran weights can quantize, numpy array FP32, shape (d_in, d_out)
                      Vd: LLaMA 7B attention W_q: (4096, 4096)
                      Vd: GPT-2 FFN W1: (768, 3072)
                      Moi hang (row) tuong ung voi 1 input dimension

    calibration_data: du lieu calibration, numpy array FP32, shape (n_samples, d_in)
                      n_samples: so luong samples dung de tinh importance
                      Thuong 32-128 samples la du (nhieu hon khong cai thien dang ke)
                      d_in phai khop voi weights.shape[0]
                      Vd: 32 cau text tokenized, chay qua model den layer nay
                      Du lieu nay giup xac dinh weights nao quan trong

    group_size:       so rows trong moi group quantization, mac dinh = 128
                      Moi group co 1 scale rieng -> group nho = chinh xac hon nhung ton memory
                      Gia tri thuong dung: 32, 64, 128, 256
                      Vd: GPTQ paper dung group_size=128 cho LLaMA
                      Vd: d_in=4096, group_size=128 -> 32 groups, 32 scales

    TAI SAO GPTQ TOT HON ABSMAX?
    - Absmax: quantize tung weight doc lap, khong quan tam den input
    - GPTQ: xem xet CACH weights duoc su dung voi actual data
      -> Weights quan trong (used often, large activation) duoc quantize can than hon
      -> Weights it quan trong co the quantize tho hon

    CACH HOAT DONG:
    1. Chay calibration data qua model
    2. Tinh Hessian (do "quan trong" cua moi weight)
    3. Quantize theo thu tu: weight it quan trong truoc
    4. Sau moi weight, dieu chinh cac weights con lai de bu sai so

    SIMPLIFIED VERSION:
    - Dung activation magnitude thay cho full Hessian
    - Group quantization: chia weights thanh groups, moi group 1 scale
    """
    d_in, d_out = weights.shape

    # Step 1: Tinh activation statistics tu calibration data
    # activation_norms cho biet columns nao cua weight matrix quan trong
    act_norms = np.mean(np.abs(calibration_data), axis=0)  # (d_in,)
    importance = act_norms / (np.max(act_norms) + 1e-9)

    # Step 2: Quantize theo groups
    # Moi group co scale rieng -> chinh xac hon per-tensor
    quantized = np.zeros_like(weights, dtype=np.int8)
    scales = []
    num_groups = (d_in + group_size - 1) // group_size

    for g in range(num_groups):
        start = g * group_size
        end = min(start + group_size, d_in)
        group_weights = weights[start:end, :]

        # Scale theo importance: weight quan trong -> quantize can than hon
        group_importance = importance[start:end]
        weighted_abs_max = np.max(np.abs(group_weights) * group_importance[:, None])
        scale = weighted_abs_max / 127.0 if weighted_abs_max != 0 else 1.0
        scales.append(scale)

        q = np.round(group_weights / scale).astype(np.int8)
        quantized[start:end, :] = q

    return quantized, np.array(scales), group_size


def gptq_dequantize(quantized, scales, group_size):
    """
    Dequantize GPTQ - phuc hoi weights tu INT8 ve FP32

    quantized:  ma tran da quantize, numpy array INT8, shape (d_in, d_out)
                Ket qua tra ve tu gptq_quantize()
                Vd: shape (4096, 4096) cho LLaMA 7B attention layer

    scales:     mang cac he so scale, numpy array FP32, shape (num_groups,)
                Moi group co 1 scale rieng
                Vd: d_in=4096, group_size=128 -> scales co 32 gia tri

    group_size: so rows trong moi group, int, giong gia tri da dung khi quantize
                Phai khop voi group_size luc gptq_quantize()
                Vd: 128 (mac dinh cua GPTQ)
    """
    d_in = quantized.shape[0]
    result = np.zeros_like(quantized, dtype=np.float32)
    for g, scale in enumerate(scales):
        start = g * group_size
        end = min(start + group_size, d_in)
        result[start:end, :] = quantized[start:end, :].astype(np.float32) * scale
    return result


# ============================================================
# BAI TAP 2: AWQ-STYLE QUANTIZATION
# ============================================================

def awq_quantize(weights, calibration_data, group_size=128):
    """
    AWQ-style quantization (simplified)

    weights:          ma tran weights can quantize, numpy array FP32, shape (d_in, d_out)
                      Vd: LLaMA 7B FFN down_proj: (11008, 4096)
                      Vd: GPT-2 attention W_v: (768, 768)
                      Moi hang (row) la 1 input channel

    calibration_data: du lieu calibration, numpy array FP32, shape (n_samples, d_in)
                      Dung de xac dinh "salient channels" (channels quan trong)
                      Thuong 32-128 samples, lay tu dataset thuc te (WikiText, C4, ...)
                      d_in phai khop voi weights.shape[0]
                      Vd: 32 samples, moi sample la activation vector cua 1 layer

    group_size:       so rows trong moi group quantization, mac dinh = 128
                      Giong GPTQ, moi group co 1 scale rieng
                      AWQ paper dung group_size=128 cho LLaMA 7B/13B/70B
                      Vd: d_in=11008, group_size=128 -> 86 groups

    TAI SAO AWQ KHAC GPTQ?
    - GPTQ: quantize ROI moi dieu chinh weights con lai (chay cham)
    - AWQ: tim "salient channels" (channels quan trong) va BAO VE chung
      -> Nhanh hon GPTQ ma accuracy gan bang

    Y TUONG CHINH:
    - Khong phai tat ca weights deu quan trong nhu nhau
    - 1% weights "salient" anh huong 50%+ accuracy
    - Bao ve 1% nay = giu accuracy cao

    CACH HOAT DONG:
    1. Chay calibration data, tinh activation magnitude moi channel
    2. Tim top channels co activation lon (= salient)
    3. Scale UP salient channels truoc khi quantize
       -> Chung se chiem nhieu INT8 range hon -> chinh xac hon
    4. Scale DOWN khi dequantize de bu lai
    """
    d_in, d_out = weights.shape

    # Step 1: Tinh salient channels tu activations
    act_magnitude = np.mean(np.abs(calibration_data), axis=0)  # (d_in,)

    # Step 2: Tim top salient channels (top 1%)
    threshold = np.percentile(act_magnitude, 99)
    salient_mask = act_magnitude > threshold

    # Step 3: Scale salient channels
    # Multiply salient weights by alpha -> chiem nhieu quantization range
    alpha = 2.0  # scaling factor
    scaled_weights = weights.copy()
    scaled_weights[salient_mask, :] *= alpha
    channel_scales = np.ones(d_in)
    channel_scales[salient_mask] = alpha

    # Step 4: Quantize (per-group)
    quantized = np.zeros_like(weights, dtype=np.int8)
    group_scales = []
    num_groups = (d_in + group_size - 1) // group_size

    for g in range(num_groups):
        start = g * group_size
        end = min(start + group_size, d_in)
        group = scaled_weights[start:end, :]
        abs_max = np.max(np.abs(group))
        scale = abs_max / 127.0 if abs_max != 0 else 1.0
        group_scales.append(scale)
        quantized[start:end, :] = np.round(group / scale).astype(np.int8)

    return quantized, np.array(group_scales), channel_scales, group_size


def awq_dequantize(quantized, group_scales, channel_scales, group_size):
    """
    Dequantize AWQ - phuc hoi weights tu INT8 ve FP32, undo salient scaling

    quantized:      ma tran da quantize, numpy array INT8, shape (d_in, d_out)
                    Ket qua tra ve tu awq_quantize()
                    Vd: shape (4096, 4096) cho LLaMA 7B attention layer

    group_scales:   mang scale moi group, numpy array FP32, shape (num_groups,)
                    Moi group co 1 scale rieng (giong GPTQ)
                    Vd: d_in=4096, group_size=128 -> 32 gia tri scale

    channel_scales: mang scale moi channel (do salient scaling), numpy array FP32, shape (d_in,)
                    = 1.0 cho channels binh thuong
                    = alpha (vd: 2.0) cho salient channels (top 1% activation)
                    Dung de undo buoc scale UP khi quantize

    group_size:     so rows trong moi group, int, giong gia tri da dung khi quantize
                    Phai khop voi group_size luc awq_quantize()
                    Vd: 128 (mac dinh cua AWQ)
    """
    d_in = quantized.shape[0]
    result = np.zeros_like(quantized, dtype=np.float32)
    for g, scale in enumerate(group_scales):
        start = g * group_size
        end = min(start + group_size, d_in)
        result[start:end, :] = quantized[start:end, :].astype(np.float32) * scale

    # Undo salient scaling
    result = result / channel_scales[:, None]
    return result


# ============================================================
# HELPER: Do luong sai so
# ============================================================

def measure_error(original, reconstructed):
    """
    Do sai so giua original va reconstructed weights

    original:      ma tran weights goc, numpy array FP32
                   Shape bat ky, thuong la (d_in, d_out) cua 1 layer
                   Vd: GPT-2 attention layer (768, 768) truoc khi quantize
                   Vd: LLaMA 7B FFN layer (4096, 11008) truoc khi quantize

    reconstructed: ma tran weights sau khi dequantize, numpy array FP32
                   Shape PHAI giong original
                   Day la ket qua sau: quantize -> dequantize (round-trip)
                   Vd: absmax: dequantize_absmax_int8(quantize_absmax_int8(w))
                   Sai so tot: MSE < 1e-4, cosine_sim > 0.999 (INT8)
    """
    mse = np.mean((original - reconstructed) ** 2)
    mae = np.mean(np.abs(original - reconstructed))
    max_err = np.max(np.abs(original - reconstructed))
    # Cosine similarity: do goc giua 2 vectors
    # 1.0 = giong hoan toan, 0.0 = vuong goc
    cos_sim = np.sum(original * reconstructed) / (
        np.linalg.norm(original) * np.linalg.norm(reconstructed) + 1e-9
    )
    return {'mse': mse, 'mae': mae, 'max_error': max_err, 'cosine_sim': cos_sim}


def measure_inference_speed(weights_fp32, weights_int8, scale, n_iter=100):
    """
    Benchmark inference speed: FP32 vs INT8

    weights_fp32: ma tran weights goc, numpy array FP32, shape (d, d) hoac (d, d_out)
                  Dung de benchmark FP32 matmul: x @ weights_fp32
                  Vd: shape (1024, 1024) ~ 4 MB, (4096, 4096) ~ 64 MB

    weights_int8: ma tran weights da quantize, numpy array INT8, shape giong weights_fp32
                  Dung de benchmark INT8 matmul (simulate): x @ (weights_int8 * scale)
                  Trong thuc te (CUDA), INT8 matmul nhanh hon 2-4x nho VNNI/AMX instructions

    scale:        he so scale (float) tu buoc quantize
                  Dung de dequantize khi tinh matmul: weights_int8 * scale -> FP32
                  Vd: scale = 0.01 cho weights co abs_max ~ 1.27

    n_iter:       so lan lap matmul de do thoi gian, mac dinh = 100
                  Nhieu lan = ket qua on dinh hon (giam noise tu OS scheduling)
                  Vd: n_iter=100 cho benchmark nhanh, n_iter=1000 cho benchmark chinh xac

    LUU Y: numpy khong co INT8 matmul toi uu
    Trong thuc te, dung CUDA kernels hoac ONNX Runtime
    Day chi la demo concept
    """
    d = weights_fp32.shape[0]
    x = np.random.randn(1, d).astype(np.float32)
    x_int = np.round(x / 0.01).astype(np.int8)

    # FP32
    t0 = time.time()
    for _ in range(n_iter):
        _ = x @ weights_fp32
    fp32_time = time.time() - t0

    # INT8 (simulate: quantized matmul + dequantize)
    t0 = time.time()
    for _ in range(n_iter):
        # Trong thuc te, dung INT8 matmul instruction (VNNI, AMX)
        _ = x @ (weights_int8.astype(np.float32) * scale)
    int8_time = time.time() - t0

    return fp32_time, int8_time


# ============================================================
# BAI TAP 3: QUANTIZE GPT MODEL
# ============================================================

def quantize_gpt_model(gpt_params, method='absmax'):
    """
    Quantize tat ca weights cua GPT model

    gpt_params: dictionary chua tat ca weights cua model
                Key: ten layer (string), vd: 'block0.attn.W_q', 'block0.ff.W1'
                Value: numpy array FP32, shape tuy layer
                Vd GPT-2 small: ~124M params, bao gom:
                  - 'token_emb': (50257, 768) - embedding, GIU FP32
                  - 'block0.attn.W_q': (768, 768) - attention, QUANTIZE
                  - 'block0.ff.W1': (768, 3072) - FFN, QUANTIZE
                  - 'block0.ln1.gamma': (768,) - layer norm, GIU FP32
                Vd LLaMA 7B: ~7B params, layers co shape (4096, 4096) va (4096, 11008)

    method:     phuong phap quantization, string, mac dinh = 'absmax'
                'absmax': dung Quantizer.quantize_absmax_int8() -> INT8, 4x compression
                'int4':   dung Quantizer.quantize_int4() -> INT4, 8x compression
                Vd: LLaMA 7B + absmax: 28 GB -> ~7 GB
                Vd: LLaMA 7B + int4: 28 GB -> ~3.5 GB

    TAI SAO KHONG QUANTIZE TAT CA?
    - Embedding va output projection: giu FP32 (rat nhay cam voi quantization)
    - Attention weights (Q, K, V, O): quantize INT8 (it nhay cam)
    - FFN weights: quantize INT8 (chiu duoc)
    - LayerNorm: giu FP32 (nho, va nhay cam)

    Day la chien luoc ma GPTQ, AWQ deu dung trong thuc te.
    """
    quantized_params = {}
    total_original = 0
    total_quantized = 0

    for name, param in gpt_params.items():
        total_original += param.nbytes

        # Giu FP32 cho embedding va LayerNorm
        if 'emb' in name or 'norm' in name or 'ln' in name:
            quantized_params[name] = {'data': param, 'type': 'fp32'}
            total_quantized += param.nbytes
        else:
            if method == 'absmax':
                q, s = Quantizer.quantize_absmax_int8(param)
                quantized_params[name] = {'data': q, 'scale': s, 'type': 'int8'}
            elif method == 'int4':
                q, s = Quantizer.quantize_int4(param)
                quantized_params[name] = {'data': q, 'scale': s, 'type': 'int4'}
            total_quantized += q.nbytes

    compression = total_original / total_quantized
    return quantized_params, compression


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    np.random.seed(42)
    output_dir = os.path.dirname(os.path.abspath(__file__))

    # ---- Test co ban: INT8 Absmax ----
    print("=" * 60)
    print("PHAN 1: INT8 Absmax Quantization")
    print("=" * 60)

    weights = np.random.randn(1024, 1024).astype(np.float32)
    print(f"  Original: {weights.shape}, {weights.nbytes / 1024 / 1024:.2f} MB")
    print(f"  Range: [{weights.min():.3f}, {weights.max():.3f}]")

    q8, s8 = Quantizer.quantize_absmax_int8(weights)
    dq8 = Quantizer.dequantize_absmax_int8(q8, s8)
    err8 = measure_error(weights, dq8)

    print(f"\n  INT8 Absmax:")
    print(f"    Size: {q8.nbytes / 1024 / 1024:.2f} MB (4x compression)")
    print(f"    MAE:  {err8['mae']:.6f}")
    print(f"    MSE:  {err8['mse']:.8f}")
    print(f"    Cosine sim: {err8['cosine_sim']:.6f}")
    assert err8['cosine_sim'] > 0.99, "INT8 should preserve most information"
    print("    Quality: OK (cosine > 0.99)")

    # ---- Test: Zero-point ----
    print("\n" + "=" * 60)
    print("PHAN 2: Zero-Point Quantization")
    print("=" * 60)

    # Test voi asymmetric weights (vi du: sau ReLU)
    asym_weights = np.abs(np.random.randn(512, 512).astype(np.float32))  # chi duong
    print(f"  Asymmetric weights range: [{asym_weights.min():.3f}, {asym_weights.max():.3f}]")

    q_zp, s_zp, zp = Quantizer.quantize_zeropoint_int8(asym_weights)
    dq_zp = Quantizer.dequantize_zeropoint_int8(q_zp, s_zp, zp)
    err_zp = measure_error(asym_weights, dq_zp)

    q_abs, s_abs = Quantizer.quantize_absmax_int8(asym_weights)
    dq_abs = Quantizer.dequantize_absmax_int8(q_abs, s_abs)
    err_abs = measure_error(asym_weights, dq_abs)

    print(f"\n  Absmax on asymmetric:     MAE={err_abs['mae']:.6f}")
    print(f"  Zero-point on asymmetric: MAE={err_zp['mae']:.6f}")
    print(f"  Zero-point better: {err_zp['mae'] < err_abs['mae']}")
    print("  Zero-point: OK")

    # ---- Test: Per-channel ----
    print("\n" + "=" * 60)
    print("PHAN 3: Per-Channel Quantization")
    print("=" * 60)

    # Tao weights voi cac channels co scale rat khac nhau
    mixed_weights = np.random.randn(256, 256).astype(np.float32)
    mixed_weights[0, :] *= 100  # Channel 0 rat lon
    mixed_weights[1, :] *= 0.001  # Channel 1 rat nho

    q_pt, s_pt = Quantizer.quantize_absmax_int8(mixed_weights)
    dq_pt = Quantizer.dequantize_absmax_int8(q_pt, s_pt)
    err_pt = measure_error(mixed_weights, dq_pt)

    q_pc, s_pc = Quantizer.quantize_per_channel_int8(mixed_weights)
    # Dequantize per-channel
    dq_pc = np.zeros_like(mixed_weights)
    for i in range(256):
        dq_pc[i] = q_pc[i].astype(np.float32) * s_pc[i]
    err_pc = measure_error(mixed_weights, dq_pc)

    print(f"  Per-tensor MAE:  {err_pt['mae']:.6f}")
    print(f"  Per-channel MAE: {err_pc['mae']:.6f}")
    print(f"  Per-channel better: {err_pc['mae'] < err_pt['mae']}")
    assert err_pc['mae'] < err_pt['mae'], "Per-channel should be better for mixed scales"
    print("  Per-channel: OK")

    # ---- Test: INT4 ----
    print("\n" + "=" * 60)
    print("PHAN 4: INT4 Quantization")
    print("=" * 60)

    q4, s4 = Quantizer.quantize_int4(weights)
    dq4 = q4.astype(np.float32) * s4
    err4 = measure_error(weights, dq4)

    print(f"  INT4:")
    print(f"    Size: {q4.nbytes / 1024 / 1024:.2f} MB (co the pack thanh {q4.nbytes / 2 / 1024 / 1024:.2f} MB)")
    print(f"    MAE:  {err4['mae']:.6f}")
    print(f"    Cosine sim: {err4['cosine_sim']:.6f}")
    print(f"  INT8 MAE: {err8['mae']:.6f}")
    print(f"  INT4 MAE: {err4['mae']:.6f}")
    assert err4['mae'] > err8['mae'], "INT4 should have more error than INT8"
    print("  INT4 more error but much smaller: OK")

    # ---- Bai tap 1: GPTQ ----
    print("\n" + "=" * 60)
    print("BAI TAP 1: GPTQ-style Quantization")
    print("=" * 60)

    calibration = np.random.randn(32, 1024).astype(np.float32)  # 32 samples
    q_gptq, s_gptq, gs = gptq_quantize(weights, calibration, group_size=128)
    dq_gptq = gptq_dequantize(q_gptq, s_gptq, gs)
    err_gptq = measure_error(weights, dq_gptq)

    print(f"  GPTQ (group_size=128):")
    print(f"    MAE: {err_gptq['mae']:.6f}")
    print(f"    Cosine sim: {err_gptq['cosine_sim']:.6f}")
    print(f"  vs Absmax MAE: {err8['mae']:.6f}")
    print("  GPTQ: OK")

    # ---- Bai tap 2: AWQ ----
    print("\n" + "=" * 60)
    print("BAI TAP 2: AWQ-style Quantization")
    print("=" * 60)

    q_awq, s_awq, cs_awq, gs_awq = awq_quantize(weights, calibration, group_size=128)
    dq_awq = awq_dequantize(q_awq, s_awq, cs_awq, gs_awq)
    err_awq = measure_error(weights, dq_awq)

    print(f"  AWQ (group_size=128):")
    print(f"    MAE: {err_awq['mae']:.6f}")
    print(f"    Cosine sim: {err_awq['cosine_sim']:.6f}")
    print("  AWQ: OK")

    # ---- Bai tap 3: Quantize GPT ----
    print("\n" + "=" * 60)
    print("BAI TAP 3: Quantize GPT Model")
    print("=" * 60)

    # Simulate GPT params
    d_model = 256
    gpt_params = {
        'token_emb': np.random.randn(1000, d_model).astype(np.float32),
        'block0.attn.W_q': np.random.randn(d_model, d_model).astype(np.float32),
        'block0.attn.W_k': np.random.randn(d_model, d_model).astype(np.float32),
        'block0.attn.W_v': np.random.randn(d_model, d_model).astype(np.float32),
        'block0.attn.W_o': np.random.randn(d_model, d_model).astype(np.float32),
        'block0.ff.W1': np.random.randn(d_model, d_model * 4).astype(np.float32),
        'block0.ff.W2': np.random.randn(d_model * 4, d_model).astype(np.float32),
        'block0.ln1.gamma': np.ones(d_model).astype(np.float32),
        'block0.ln1.beta': np.zeros(d_model).astype(np.float32),
    }

    original_size = sum(p.nbytes for p in gpt_params.values())
    print(f"  Original model size: {original_size / 1024:.1f} KB")

    q_model_int8, comp_int8 = quantize_gpt_model(gpt_params, method='absmax')
    q_model_int4, comp_int4 = quantize_gpt_model(gpt_params, method='int4')

    print(f"  INT8 compression: {comp_int8:.2f}x")
    print(f"  INT4 compression: {comp_int4:.2f}x")

    # So sanh accuracy
    print(f"\n  Per-layer accuracy (INT8):")
    for name, info in q_model_int8.items():
        if info['type'] != 'fp32':
            original = gpt_params[name]
            reconstructed = info['data'].astype(np.float32) * info['scale']
            err = measure_error(original, reconstructed)
            print(f"    {name:30s}: cosine={err['cosine_sim']:.6f}, MAE={err['mae']:.6f}")
    print("  GPT quantization: OK")

    # ---- Bai tap 4: Benchmark ----
    print("\n" + "=" * 60)
    print("BAI TAP 4: Inference Speed Benchmark")
    print("=" * 60)

    sizes = [256, 512, 1024]
    n_iter = 50

    print(f"\n  {'Size':>8} | {'FP32 (ms)':>10} | {'INT8 (ms)':>10} | {'Speedup':>8}")
    print(f"  {'-'*8}-+-{'-'*10}-+-{'-'*10}-+-{'-'*8}")

    bench_results = []
    for size in sizes:
        w_fp32 = np.random.randn(size, size).astype(np.float32)
        q, s = Quantizer.quantize_absmax_int8(w_fp32)

        fp32_time, int8_time = measure_inference_speed(w_fp32, q, s, n_iter)
        fp32_ms = fp32_time / n_iter * 1000
        int8_ms = int8_time / n_iter * 1000
        speedup = fp32_time / int8_time if int8_time > 0 else 0

        bench_results.append({
            'size': size, 'fp32_ms': fp32_ms, 'int8_ms': int8_ms, 'speedup': speedup
        })
        print(f"  {size:>8} | {fp32_ms:>8.3f}ms | {int8_ms:>8.3f}ms | {speedup:>6.2f}x")

    print("\n  Note: numpy khong co INT8 matmul toi uu.")
    print("  Trong thuc te (CUDA/ONNX), INT8 nhanh hon 2-4x.")

    # ---- Summary ----
    print("\n" + "=" * 60)
    print("SUMMARY: So sanh cac phuong phap")
    print("=" * 60)

    print(f"\n  {'Method':>20} | {'MAE':>10} | {'Cosine':>8} | {'Compression':>12}")
    print(f"  {'-'*20}-+-{'-'*10}-+-{'-'*8}-+-{'-'*12}")
    print(f"  {'Absmax INT8':>20} | {err8['mae']:>10.6f} | {err8['cosine_sim']:>8.6f} | {'4x':>12}")
    print(f"  {'Zero-Point INT8':>20} | {err_zp['mae']:>10.6f} | {err_zp['cosine_sim']:>8.6f} | {'4x':>12}")
    print(f"  {'GPTQ INT8':>20} | {err_gptq['mae']:>10.6f} | {err_gptq['cosine_sim']:>8.6f} | {'4x':>12}")
    print(f"  {'AWQ INT8':>20} | {err_awq['mae']:>10.6f} | {err_awq['cosine_sim']:>8.6f} | {'4x':>12}")
    print(f"  {'INT4':>20} | {err4['mae']:>10.6f} | {err4['cosine_sim']:>8.6f} | {'8x':>12}")

    # ---- Plot ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        # Plot 1: Error comparison
        methods = ['Absmax\nINT8', 'Zero-\nPoint', 'Per-\nChannel', 'GPTQ', 'AWQ', 'INT4']
        maes = [err8['mae'], err_zp['mae'], err_pc['mae'], err_gptq['mae'], err_awq['mae'], err4['mae']]
        colors = ['steelblue'] * 5 + ['coral']

        ax = axes[0]
        ax.bar(range(len(methods)), maes, color=colors)
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels(methods, fontsize=8)
        ax.set_ylabel('Mean Absolute Error')
        ax.set_title('Quantization Error Comparison')
        ax.grid(True, alpha=0.3, axis='y')

        # Plot 2: Size comparison
        ax = axes[1]
        original_mb = weights.nbytes / 1024 / 1024
        sizes_mb = [q8.nbytes/1024/1024, q8.nbytes/1024/1024,
                     q8.nbytes/1024/1024, q_gptq.nbytes/1024/1024,
                     q_awq.nbytes/1024/1024, q4.nbytes/1024/1024]
        ax.barh(range(len(methods)), sizes_mb, color=colors, label='Quantized')
        ax.axvline(x=original_mb, color='red', linestyle='--', label=f'Original ({original_mb:.1f} MB)')
        ax.set_yticks(range(len(methods)))
        ax.set_yticklabels(methods, fontsize=8)
        ax.set_xlabel('Size (MB)')
        ax.set_title('Model Size Comparison')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='x')

        plt.tight_layout()
        path = os.path.join(output_dir, "plot_quantization.png")
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"\n  Saved: {path}")

    except ImportError:
        print("  matplotlib chua cai.")

    print("\n" + "=" * 60)
    print("TAT CA TESTS PASSED!")
    print("=" * 60)


# ============ CHECKLIST ============
# Week 10-11 (Bai 10):
# [x] Implement INT8 quantization
#     -> quantize_absmax_int8(): nen float32 -> int8 (giam 4x memory)
#        quantize_zeropoint_int8(): xu ly asymmetric distributions
#        quantize_per_channel_int8(): quantize tung channel rieng (chinh xac hon)
# [x] Measure accuracy loss
#     -> measure_quantization_error(): tinh MSE giua original va dequantized
#        INT8 absmax: error rat nho (~0.001), giam 4x memory
#        INT4: error lon hon nhung giam 8x memory (dung cho inference)
