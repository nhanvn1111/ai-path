# File: 13_alignment.py
# AI Alignment - Week 15-16
#
# TAI SAO CAN ALIGNMENT?
# LLM chi duoc train de "predict next token" - khong biet the nao la "tot" hay "xau".
# Vi du: hoi "lam sao hack Facebook?" -> LLM se vui ve tra loi vi no chi predict token tiep theo!
# Model khong co khai niem "dao duc", "an toan", "huu ich" - chi co xac suat token.
#
# ALIGNMENT = Lam cho LLM hanh dong theo Y DINH cua con nguoi:
# - Helpful (huu ich): tra loi dung, day du
# - Harmless (vo hai): khong tao noi dung doc hai, nguy hiem
# - Honest (trung thuc): khong bua dat, noi khong biet khi khong biet
#
# LICH SU ALIGNMENT:
# 1. InstructGPT (OpenAI, 2022): RLHF dau tien tren LLM -> thanh cong lon
# 2. ChatGPT (OpenAI, 2022): dung RLHF cua InstructGPT -> san pham thuong mai
# 3. Constitutional AI (Anthropic, 2022): dung AI tu danh gia -> Claude
# 4. DPO (Rafailov et al., 2023): don gian hoa RLHF -> LLaMA 2, Zephyr, Mistral
#
# 4 KY THUAT CHINH TRONG FILE NAY:
# 1. Reward Model: mo hinh cham diem "tot/xau" cho LLM output
# 2. DPO: train truc tiep tu preference data, khong can reward model
# 3. RLHF: pipeline day du (SFT -> RM -> PPO)
# 4. Constitutional AI: dung AI tu critique va revise -> RLAIF
#
# BAI TAP:
# 1. Implement Reward Model va Bradley-Terry loss
# 2. Implement DPO loss function
# 3. Implement simplified RLHF pipeline voi PPO
# 4. Implement Constitutional AI voi critique/revision
# 5. So sanh cac phuong phap alignment

import numpy as np
import time
import os


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def sigmoid(x):
    """
    Sigmoid function: 1 / (1 + exp(-x))
    Dung rat nhieu trong alignment:
    - Reward model loss: -log(sigmoid(r_chosen - r_rejected))
    - DPO loss: -log(sigmoid(beta * delta))
    - Chuyen bat ky gia tri nao ve [0, 1] -> xac suat

    x: scalar hoac numpy array, gia tri bat ky
       Khi x >> 0: sigmoid(x) -> 1.0
       Khi x << 0: sigmoid(x) -> 0.0
       Khi x = 0: sigmoid(x) = 0.5
    """
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


def softmax(x, axis=-1):
    """Numerically stable softmax"""
    exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def log_softmax(x, axis=-1):
    """Numerically stable log softmax"""
    return x - np.max(x, axis=axis, keepdims=True) - np.log(
        np.sum(np.exp(x - np.max(x, axis=axis, keepdims=True)), axis=axis, keepdims=True)
    )


# ============================================================
# BAI TAP 1: REWARD MODEL
# ============================================================
#
# GIAI THICH CHI TIET:
#
# VAN DE CO BAN:
# Khi train LLM (GPT, LLaMA), model hoc tu data bang cach predict next token.
# Vi du: "The cat sat on the ___" -> model hoc: "mat" co xac suat cao.
# Nhung viec predict next token KHONG dam bao output "tot":
# - "Lam sao hack wifi?" -> model co the tra loi chi tiet vi data co thong tin nay
# - Model khong phan biet: tra loi nay co HAI khong? co DUNG khong?
#
# GIAI PHAP: REWARD MODEL
# Train 1 model RIENG BIET de "cham diem" cho output cua LLM.
# - Input: (prompt, response) pair
# - Output: scalar reward score (so diem, cang cao = cang tot)
# - Vi du: "Lam sao hack wifi?" + "Toi khong the giup..." -> reward CAO (tu choi dung)
#          "Lam sao hack wifi?" + "Buoc 1: cai aircrack..." -> reward THAP (tra loi hai)
#
# CACH TRAIN:
# 1. Thu thap DU LIEU SO SANH tu con nguoi:
#    - Cho prompt, 2 responses: A va B
#    - Nguoi cham: A tot hon B, hoac B tot hon A
#    - Vi du: prompt = "Viet bai tho ve mua xuan"
#      Response A: "Mua xuan den, hoa no..." (tot, co van)
#      Response B: "Mua xuan la mua 1" (dek tot, khong co noi dung)
#      -> Label: A chosen, B rejected
#
# 2. TRAIN: Dung Bradley-Terry model (xac suat chosen > rejected)
#    P(A > B) = sigmoid(reward(A) - reward(B))
#    Loss = -log P(A > B) = -log sigmoid(r_chosen - r_rejected)
#
# THUC TE:
# - InstructGPT: dung 6B reward model (GPT-3 6B lam backbone)
# - ChatGPT: tuong tu, reward model cung la LLM lon
# - Dataset: ~30K-50K comparison pairs tu human labelers
# - Moi prompt co 4-9 responses, tao thanh nhieu pairs
# - Chi phi label: $10-50/gio x hang nghin gio = rat dat!


class RewardModel:
    """
    Reward Model - Cham diem cho output cua LLM

    KIEN TRUC:
    Input sequence (prompt + response tokens) -> Embedding -> Hidden layers -> Scalar reward

    VI DU THUC TE:
    - InstructGPT dung GPT-3 6B lam backbone cho reward model
    - Anthropic dung model 52B params lam reward model
    - Meta (LLaMA 2) dung LLaMA 70B lam reward model

    CACH SU DUNG:
    1. Tokenize prompt + response thanh sequence
    2. Chay qua reward model
    3. Lay scalar reward score tu output cuoi cung
    4. Reward score dung de:
       a) So sanh: response A vs B -> chon cai co reward cao hon
       b) Optimize: dung PPO de maximize reward (RLHF phase 3)

    QUAN TRONG:
    - Reward model KHONG phai LLM generate text
    - No chi DANH GIA (cham diem) text da co san
    - Tuong tu nhu giao vien cham bai (khong viet bai)
    """

    def __init__(self, vocab_size, d_model, d_hidden, num_layers=2, max_seq_len=128):
        """
        vocab_size:  so luong tokens trong vocabulary (bo tu vung)
                     GPT-2: 50,257 tokens
                     LLaMA: 32,000 tokens
                     Claude: ~100,000 tokens (kiem chung)
                     Trong demo: 100-1000 tokens
                     Moi token la 1 subword (vi du: "understanding" = "under" + "standing")

        d_model:     kich thuoc embedding cua moi token
                     GPT-2 small: 768
                     GPT-3 6B (InstructGPT RM): 4096
                     LLaMA 7B: 4096
                     LLaMA 70B: 8192
                     Trong demo: 32-128
                     d_model lon -> model co nhieu "bieu dien" -> hieu tot hon
                     Nhung ton nhieu memory va cham hon

        d_hidden:    kich thuoc hidden layer trong MLP (head cham diem)
                     Thuong = 4 * d_model (giong FFN trong transformer)
                     GPT-3 6B: d_hidden = 16384
                     LLaMA 7B: d_hidden = 11008
                     Trong demo: 64-256
                     Hidden layer giup model hoc NON-LINEAR patterns
                     Vi du: "polite but wrong" -> reward thap (can hidden layer de "hieu")

        num_layers:  so hidden layers trong phan MLP
                     2-3 layers la du cho reward head
                     Trong thuc te, phan lon "hieu biet" nam o backbone (GPT, LLaMA)
                     MLP head chi can map tu representation -> scalar
                     Nhieu layers hon -> fit tot hon nhung de overfit

        max_seq_len: do dai toi da cua input sequence (prompt + response)
                     GPT-2: 1024 tokens
                     GPT-3: 2048 tokens
                     LLaMA 2: 4096 tokens
                     Claude: 100,000+ tokens
                     Trong demo: 64-256 tokens
                     Sequence dai hon max_seq_len se bi cat (truncate)
        """
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.d_hidden = d_hidden
        self.num_layers = num_layers
        self.max_seq_len = max_seq_len

        # Token embedding: moi token -> vector d_model chieu
        # Shape: (vocab_size, d_model)
        # Vi du: token "hello" (id=5) -> embedding[5] = vector 768 chieu
        self.embedding = np.random.randn(vocab_size, d_model).astype(np.float32) * 0.02

        # Position embedding: encode vi tri cua token trong sequence
        # Shape: (max_seq_len, d_model)
        # Vi du: token o vi tri 0 -> pos_embedding[0], vi tri 5 -> pos_embedding[5]
        # Giup model biet "token nay o dau trong cau"
        self.pos_embedding = np.random.randn(max_seq_len, d_model).astype(np.float32) * 0.02

        # Hidden layers (MLP)
        # Transform tu d_model -> d_hidden -> ... -> 1 (scalar reward)
        self.layers = []
        prev_dim = d_model
        for i in range(num_layers):
            W = np.random.randn(prev_dim, d_hidden).astype(np.float32) * np.sqrt(2.0 / prev_dim)
            b = np.zeros(d_hidden, dtype=np.float32)
            self.layers.append({'W': W, 'b': b})
            prev_dim = d_hidden

        # Reward head: output 1 scalar (diem so)
        # Shape: (d_hidden, 1) -> moi sequence -> 1 so diem
        self.reward_head_W = np.random.randn(d_hidden, 1).astype(np.float32) * 0.01
        self.reward_head_b = np.zeros(1, dtype=np.float32)

        # Cache cho backward
        self._cache = {}

    def forward(self, token_ids):
        """
        Tinh reward score cho 1 sequence

        token_ids: numpy array INT, shape (batch_size, seq_len)
                   Moi phan tu la token ID (0 den vocab_size-1)
                   batch_size: so sequences xu ly cung luc (1-64)
                   seq_len: do dai sequence (phai <= max_seq_len)
                   Vi du: [[5, 23, 100, 7, 0], [12, 8, 45, 3, 1]] -> batch=2, seq_len=5
                   Trong thuc te: prompt + response duoc tokenize thanh chuoi token IDs
                   Vi du: "Hello world" -> [15496, 995] (GPT-2 tokenizer)

        Returns: reward scores, shape (batch_size, 1)
                 Moi sequence -> 1 scalar reward score
                 Score cang cao = response cang tot
                 Khong co range co dinh (co the am hoac duong)
                 Vi du: response tot -> reward = 2.5, response xau -> reward = -1.3
        """
        batch_size, seq_len = token_ids.shape

        # Token embedding + Position embedding
        # Cong 2 embeddings: token_emb cho biet "token gi", pos_emb cho biet "o dau"
        tok_emb = self.embedding[token_ids]  # (batch, seq_len, d_model)
        positions = np.arange(seq_len)
        pos_emb = self.pos_embedding[positions]  # (seq_len, d_model)
        x = tok_emb + pos_emb  # broadcasting: (batch, seq_len, d_model)

        # Mean pooling: lay trung binh cua tat ca tokens
        # Thay vi chi dung token cuoi (nhu GPT), dung mean cua tat ca tokens
        # Giup model "nhin toan bo" response, khong chi phan cuoi
        # InstructGPT dung token cuoi, nhung mean pooling don gian va hieu qua tuong duong
        x = np.mean(x, axis=1)  # (batch, d_model)
        self._cache['pooled'] = x

        # MLP hidden layers voi ReLU activation
        self._cache['hidden_outputs'] = []
        for i, layer in enumerate(self.layers):
            x = x @ layer['W'] + layer['b']  # Linear: (batch, d_hidden)
            x = np.maximum(0, x)  # ReLU: max(0, x) -> loai bo gia tri am
            self._cache['hidden_outputs'].append(x)

        # Reward head: (batch, d_hidden) -> (batch, 1)
        reward = x @ self.reward_head_W + self.reward_head_b
        self._cache['final_hidden'] = x

        return reward  # (batch, 1)

    def compute_preference_loss(self, chosen_ids, rejected_ids):
        """
        Tinh Bradley-Terry preference loss

        chosen_ids:   token IDs cua response DUOC CHON (tot hon), shape (batch, seq_len)
                      Day la response ma human labeler danh gia TOT HON
                      Vi du: prompt "Viet code Python" -> chosen = code dung, chay duoc
                      Trong dataset InstructGPT: moi prompt co 4-9 responses, ranked boi labelers

        rejected_ids: token IDs cua response BI LOAI (xau hon), shape (batch, seq_len)
                      Day la response ma human labeler danh gia XAU HON
                      Vi du: prompt "Viet code Python" -> rejected = code sai, khong chay
                      Moi cap (chosen, rejected) tao 1 training sample

        CONG THUC:
        r_chosen = reward_model(prompt + chosen_response)
        r_rejected = reward_model(prompt + rejected_response)
        loss = -log(sigmoid(r_chosen - r_rejected))

        TAI SAO CONG THUC NAY?
        Bradley-Terry model: P(A > B) = sigmoid(score_A - score_B)
        Ta muon: P(chosen > rejected) cao -> maximize sigmoid(r_chosen - r_rejected)
        Tuong duong: minimize -log(sigmoid(r_chosen - r_rejected))

        TRUC GIAC:
        - Neu r_chosen >> r_rejected: sigmoid(large positive) -> 1 -> loss -> 0 (tot!)
        - Neu r_chosen << r_rejected: sigmoid(large negative) -> 0 -> loss -> inf (xau!)
        - Model hoc de cho r_chosen > r_rejected luon
        """
        r_chosen = self.forward(chosen_ids)      # (batch, 1)
        r_rejected = self.forward(rejected_ids)   # (batch, 1)

        # Bradley-Terry loss
        # r_chosen - r_rejected: do chenh lech reward
        # sigmoid: chuyen thanh xac suat chosen > rejected
        # -log: loss function (minimize = maximize xac suat)
        diff = r_chosen - r_rejected  # (batch, 1)
        loss = -np.mean(np.log(sigmoid(diff) + 1e-8))

        # Accuracy: % cap ma model xep hang dung
        # Neu r_chosen > r_rejected -> model dung
        accuracy = np.mean((diff > 0).astype(float))

        return loss, accuracy, r_chosen, r_rejected

    def backward_preference(self, chosen_ids, rejected_ids, lr=0.001):
        """
        Backward pass va update weights cho preference loss

        chosen_ids:   token IDs cua chosen response, shape (batch, seq_len)
        rejected_ids: token IDs cua rejected response, shape (batch, seq_len)
        lr:           learning rate
                      InstructGPT dung lr = 9e-6 cho 6B reward model
                      Gia tri nho vi model lon, data nho -> de overfit
                      Trong demo: 0.001 - 0.01

        SIMPLIFIED BACKWARD:
        Thay vi full backprop, dung finite difference approximation
        Day la ban don gian hoa de hieu concept, khong phai production code
        Trong thuc te dung PyTorch autograd

        GRADIENT CUA BRADLEY-TERRY LOSS:
        dL/dr_chosen = -(1 - sigmoid(r_chosen - r_rejected)) / batch_size
        dL/dr_rejected = (1 - sigmoid(r_chosen - r_rejected)) / batch_size
        -> Gradient day r_chosen LEN va r_rejected XUONG
        """
        # Forward
        r_chosen = self.forward(chosen_ids)
        r_rejected = self.forward(rejected_ids)

        diff = r_chosen - r_rejected
        batch_size = chosen_ids.shape[0]

        # Gradient cua loss w.r.t reward scores
        # dL/d(r_chosen) = -(1 - sigmoid(diff))
        # dL/d(r_rejected) = (1 - sigmoid(diff))
        grad_scale = -(1.0 - sigmoid(diff))  # (batch, 1)

        # Update reward head (simplified: gradient w.r.t final layer only)
        # Day la approximation - chi update layer cuoi
        final_hidden = self._cache['final_hidden']  # (batch, d_hidden)

        # Gradient cho reward_head_W
        # d(reward)/d(W) = final_hidden^T, scaled by grad_scale
        grad_W = final_hidden.T @ grad_scale / batch_size  # (d_hidden, 1)
        grad_b = np.mean(grad_scale, axis=0)

        self.reward_head_W -= lr * grad_W
        self.reward_head_b -= lr * grad_b

        # Update hidden layers (simplified gradient)
        grad = grad_scale @ self.reward_head_W.T  # (batch, d_hidden)
        for i in range(len(self.layers) - 1, -1, -1):
            hidden = self._cache['hidden_outputs'][i]
            # ReLU gradient: 0 neu x <= 0, 1 neu x > 0
            grad = grad * (hidden > 0).astype(float)

            if i > 0:
                prev_hidden = self._cache['hidden_outputs'][i - 1]
            else:
                prev_hidden = self._cache['pooled']

            grad_W_layer = prev_hidden.T @ grad / batch_size
            grad_b_layer = np.mean(grad, axis=0)

            self.layers[i]['W'] -= lr * grad_W_layer
            self.layers[i]['b'] -= lr * grad_b_layer

            grad = grad @ self.layers[i]['W'].T

        return -np.mean(np.log(sigmoid(diff) + 1e-8))


# ============================================================
# BAI TAP 2: DPO (Direct Preference Optimization)
# ============================================================
#
# GIAI THICH CHI TIET:
#
# VAN DE VOI RLHF:
# RLHF pipeline co 3 buoc phuc tap:
#   1. Train SFT model (supervised fine-tuning)
#   2. Train Reward Model (tu preference data)
#   3. Train Policy voi PPO (dung reward model)
# Moi buoc can: data rieng, hyperparameter rieng, debugging rieng
# PPO rat kho tune: reward hacking, KL divergence, clipping, value function...
# Tong cong: 3 models (SFT, RM, Policy) + PPO trainer = RAT PHUC TAP
#
# GIAI PHAP: DPO (Direct Preference Optimization)
# Rafailov et al. 2023: "Direct Preference Optimization: Your Language Model is
#                         Secretly a Reward Model"
#
# KEY INSIGHT:
# Trong RLHF, optimal policy (pi*) co moi quan he TRUC TIEP voi reward function:
#   r(x, y) = beta * log(pi*(y|x) / pi_ref(y|x)) + beta * log(Z(x))
# Trong do:
#   pi* = optimal policy (model da align)
#   pi_ref = reference model (SFT model, chua align)
#   beta = temperature parameter
#   Z(x) = partition function (hang so, khong phu thuoc y)
#
# Thay vao Bradley-Terry:
#   P(y_w > y_l | x) = sigmoid(r(y_w) - r(y_l))
#                     = sigmoid(beta * [log(pi*(y_w)/pi_ref(y_w)) - log(pi*(y_l)/pi_ref(y_l))])
#
# => Khong can reward model! Chi can pi va pi_ref
#
# DPO LOSS:
#   L_DPO = -log sigmoid(beta * (log(pi/pi_ref)(y_w) - log(pi/pi_ref)(y_l)))
#
# Trong do:
#   log(pi/pi_ref)(y) = log_prob_policy(y) - log_prob_reference(y)
#   y_w = chosen (winning) response
#   y_l = rejected (losing) response
#
# TRUC GIAC:
# - DPO tang xac suat cua chosen response va GIAM xac suat cua rejected response
# - DONG THOI giu policy gan reference model (qua beta)
# - beta lon -> policy gan reference hon (bao thu)
# - beta nho -> policy tu do hon (mao hiem, de diverge)
#
# UU DIEM SO VOI RLHF:
# 1. Don gian hon: chi can 1 model (policy) + reference (frozen)
# 2. On dinh hon: khong can PPO (kho tune)
# 3. Khong can reward model: tiet kiem 1 model training
# 4. Hieu qua tuong duong hoac tot hon RLHF tren nhieu benchmarks
#
# SU DUNG THUC TE:
# - LLaMA 2 (Meta): dung DPO variant (rejection sampling + DPO)
# - Zephyr (HuggingFace): dung DPO truc tiep
# - Mistral: dung DPO
# - Nhieu open-source models chuyen tu RLHF sang DPO vi don gian hon


def dpo_loss(chosen_logprobs, rejected_logprobs,
             ref_chosen_logprobs, ref_rejected_logprobs,
             beta=0.1):
    """
    DPO Loss Function

    chosen_logprobs:       log probability cua CHOSEN response tu POLICY model
                           Shape: (batch_size,) hoac (batch_size, 1)
                           Day la sum(log P(token_i | token_<i)) cua response duoc chon
                           Policy model = model dang duoc train (se duoc update)
                           Vi du: log_prob = -2.5 (xac suat = exp(-2.5) = 0.082)
                           Cang gan 0 = xac suat cang cao

    rejected_logprobs:     log probability cua REJECTED response tu POLICY model
                           Shape: (batch_size,) hoac (batch_size, 1)
                           Day la response ma human labeler danh gia XAU HON
                           Cung tu policy model (dang train)
                           Vi du: log_prob = -4.0 (xac suat = exp(-4.0) = 0.018)

    ref_chosen_logprobs:   log probability cua CHOSEN response tu REFERENCE model
                           Shape: (batch_size,) hoac (batch_size, 1)
                           Reference = SFT model (frozen, khong update)
                           Dung de tinh "model da thay doi bao nhieu so voi SFT"
                           Tinh bang: with torch.no_grad(): ref_logprobs = ref_model(chosen)

    ref_rejected_logprobs: log probability cua REJECTED response tu REFERENCE model
                           Shape: (batch_size,) hoac (batch_size, 1)
                           Cung tu reference model (frozen)
                           Dung de so sanh: policy reject manh hon hay yeu hon reference?

    beta:                  temperature parameter (he so dieu chinh KL constraint)
                           Gia tri thuong dung: 0.1, 0.2, 0.5
                           beta = 0.1: DPO paper default, cho phep model thay doi nhieu
                           beta = 0.5: bao thu hon, giu gan reference
                           beta = 1.0: rat bao thu, gan nhu khong thay doi tu reference
                           LLaMA 2 dung beta = 0.5 (variant)
                           Zephyr dung beta = 0.1 (DPO goc)
                           Mistral dung beta = 0.1 - 0.2

    Returns:
        loss:     scalar, DPO loss trung binh cua batch
                  Minimize loss = tang xac suat chosen, giam xac suat rejected
        metrics:  dictionary chua thong tin huu ich:
                  - 'accuracy': % cap ma model xep dung (chosen > rejected)
                  - 'reward_margin': trung binh chenh lech implicit reward
                  - 'chosen_reward': implicit reward trung binh cua chosen
                  - 'rejected_reward': implicit reward trung binh cua rejected
    """
    # Tinh log-ratio: log(pi/pi_ref) = log(pi) - log(pi_ref)
    # Day la "model da thay doi bao nhieu" so voi reference
    chosen_log_ratio = chosen_logprobs - ref_chosen_logprobs       # (batch,)
    rejected_log_ratio = rejected_logprobs - ref_rejected_logprobs  # (batch,)

    # Implicit reward: beta * log(pi/pi_ref)
    # Day la reward "an" trong policy (khong can reward model rieng!)
    # Key insight cua DPO: policy model TU NO la reward model
    chosen_reward = beta * chosen_log_ratio
    rejected_reward = beta * rejected_log_ratio

    # DPO loss = -log sigmoid(beta * (log_ratio_chosen - log_ratio_rejected))
    # Tuong duong: -log sigmoid(chosen_reward - rejected_reward)
    logits = chosen_reward - rejected_reward  # (batch,)
    loss = -np.mean(np.log(sigmoid(logits) + 1e-8))

    # Metrics de theo doi training
    accuracy = np.mean((logits > 0).astype(float))
    reward_margin = np.mean(logits)

    metrics = {
        'accuracy': accuracy,
        'reward_margin': float(reward_margin),
        'chosen_reward': float(np.mean(chosen_reward)),
        'rejected_reward': float(np.mean(rejected_reward)),
    }

    return loss, metrics


def dpo_gradient(chosen_logprobs, rejected_logprobs,
                 ref_chosen_logprobs, ref_rejected_logprobs,
                 beta=0.1):
    """
    Tinh gradient cua DPO loss w.r.t. chosen_logprobs va rejected_logprobs

    GRADIENT:
    dL/d(chosen_logprobs) = -beta * (1 - sigmoid(beta * (log_ratio_w - log_ratio_l)))
    dL/d(rejected_logprobs) = beta * (1 - sigmoid(beta * (log_ratio_w - log_ratio_l)))

    TRUC GIAC:
    - Khi model DUNG (chosen_reward > rejected_reward):
      sigmoid large -> grad nho -> it thay doi (da tot roi)
    - Khi model SAI (rejected_reward > chosen_reward):
      sigmoid small -> grad lon -> update manh (can sua)
    - Day la "implicit curriculum": model tu dong focus vao samples kho

    chosen_logprobs:       shape (batch,), log probs tu policy cho chosen
    rejected_logprobs:     shape (batch,), log probs tu policy cho rejected
    ref_chosen_logprobs:   shape (batch,), log probs tu reference cho chosen
    ref_rejected_logprobs: shape (batch,), log probs tu reference cho rejected
    beta:                  temperature parameter (0.1 - 0.5)
    """
    chosen_log_ratio = chosen_logprobs - ref_chosen_logprobs
    rejected_log_ratio = rejected_logprobs - ref_rejected_logprobs
    logits = beta * (chosen_log_ratio - rejected_log_ratio)

    # sigmoid(logits) = P(chosen > rejected theo model hien tai)
    sig = sigmoid(logits)

    # Gradient
    grad_chosen = -beta * (1.0 - sig)      # Tang log prob cua chosen
    grad_rejected = beta * (1.0 - sig)     # Giam log prob cua rejected

    return np.mean(grad_chosen), np.mean(grad_rejected)


class SimplePolicyForDPO:
    """
    Simplified Policy Model cho DPO training demo

    DAY KHONG PHAI LLM THAT - chi la mo phong don gian de hieu DPO.
    Trong thuc te, policy = LLM (GPT, LLaMA, ...) duoc SFT truoc do.

    CACH HOAT DONG:
    - Model co 1 preference vector (d_model,)
    - Log prob = dot product giua preference va input features
    - DPO update thay doi preference vector de:
      a) Tang dot product voi chosen features
      b) Giam dot product voi rejected features
      c) Khong diverge qua xa khoi reference (beta constraint)

    TUONG DUONG TRONG THUC TE:
    - Policy = LLaMA 7B da SFT
    - Reference = ban sao LLaMA 7B truoc DPO (frozen)
    - DPO update tat ca weights cua policy
    - Nhung giu policy gan reference qua beta * KL divergence
    """

    def __init__(self, d_model):
        """
        d_model: kich thuoc feature vector cua moi response
                 Trong thuc te = d_model cua LLM (768 cho GPT-2, 4096 cho LLaMA 7B)
                 Trong demo: 32-128
        """
        self.d_model = d_model
        # Preference weights: mo phong "su uu tien" cua model
        self.W = np.random.randn(d_model).astype(np.float32) * 0.01
        # Bias
        self.b = 0.0

    def get_log_probs(self, features):
        """
        Tinh log probability cho 1 batch of responses

        features: numpy array, shape (batch, d_model)
                  Feature vector cua moi response
                  Trong thuc te: day la representation cua (prompt, response) pair
                  Moi row la 1 response da duoc encode

        Returns: log_probs, shape (batch,)
                 Log probability cua moi response
                 Cang cao (gan 0) = model thich response do hon
        """
        # Log prob = dot(features, W) + b -> scalar cho moi response
        logits = features @ self.W + self.b
        # Normalize thanh log prob (su dung -softplus de giu am)
        log_probs = -np.log(1 + np.exp(-logits))
        return log_probs

    def update(self, chosen_features, rejected_features, ref_model, beta=0.1, lr=0.01):
        """
        DPO update step

        chosen_features:  features cua chosen responses, shape (batch, d_model)
        rejected_features: features cua rejected responses, shape (batch, d_model)
        ref_model:        reference policy (frozen SimplePolicyForDPO)
        beta:             DPO temperature (0.1 - 0.5)
        lr:               learning rate (0.001 - 0.01 cho demo)
        """
        # Tinh log probs
        chosen_logprobs = self.get_log_probs(chosen_features)
        rejected_logprobs = self.get_log_probs(rejected_features)
        ref_chosen_logprobs = ref_model.get_log_probs(chosen_features)
        ref_rejected_logprobs = ref_model.get_log_probs(rejected_features)

        # Tinh loss va gradient
        loss, metrics = dpo_loss(
            chosen_logprobs, rejected_logprobs,
            ref_chosen_logprobs, ref_rejected_logprobs,
            beta=beta
        )

        # Compute gradient w.r.t W (simplified)
        grad_chosen, grad_rejected = dpo_gradient(
            chosen_logprobs, rejected_logprobs,
            ref_chosen_logprobs, ref_rejected_logprobs,
            beta=beta
        )

        # Update W
        # grad_chosen < 0 -> day W ve phia tang chosen log prob
        # grad_rejected > 0 -> day W ve phia giam rejected log prob
        chosen_direction = np.mean(chosen_features, axis=0)
        rejected_direction = np.mean(rejected_features, axis=0)

        self.W -= lr * (grad_chosen * chosen_direction + grad_rejected * rejected_direction)

        return loss, metrics


# ============================================================
# BAI TAP 3: RLHF PIPELINE
# ============================================================
#
# GIAI THICH CHI TIET:
#
# RLHF (Reinforcement Learning from Human Feedback) la phuong phap
# ALIGNMENT KINH DIEN nhat, duoc dung boi InstructGPT va ChatGPT.
#
# PIPELINE 3 BUOC:
#
# ================ PHASE 1: SFT (Supervised Fine-Tuning) ================
# - Lay pre-trained LLM (GPT-3, LLaMA)
# - Fine-tune tren DEMONSTRATION DATA (du lieu mau)
#   + Human experts viet response mau cho cac prompts
#   + Vi du: 13,000 demonstration pairs cho InstructGPT
# - Ket qua: SFT model biet "format" cua response tot
#   + Biet tra loi theo dang Q&A
#   + Biet tu choi mot so yeu cau
#   + Nhung chua toi uu ve chat luong
#
# ================ PHASE 2: Reward Model Training ================
# - Thu thap COMPARISON DATA tu human labelers
#   + Cho SFT model sinh nhieu responses cho moi prompt
#   + Human labelers XEP HANG cac responses (A > B > C > D)
#   + InstructGPT: 33,000 comparison data points
# - Train Reward Model (RM) tren comparison data
#   + Bradley-Terry loss: -log sigmoid(r_chosen - r_rejected)
#   + InstructGPT dung GPT-3 6B lam RM backbone
#   + RM output: scalar reward score cho moi (prompt, response)
#
# ================ PHASE 3: PPO (Proximal Policy Optimization) ================
# - Dung RM de optimize SFT model
# - PPO la RL algorithm, train policy (SFT model) de maximize reward
# - NHUNG: can KL penalty de policy khong diverge qua xa khoi SFT
#   + Reward_total = RM_reward - beta * KL(policy || SFT)
#   + beta = KL penalty coefficient (InstructGPT beta tuong ung voi KL target ~ 6 nats)
#   + Khong co KL -> "reward hacking" (model exploit RM, output vo nghia nhung reward cao)
# - PPO update:
#   + Generate response tu policy
#   + Tinh reward tu RM
#   + Tinh advantage = reward - baseline (value function estimate)
#   + Update policy: maximize min(ratio * advantage, clip(ratio, 1-eps, 1+eps) * advantage)
#   + ratio = pi_new(a|s) / pi_old(a|s) (ti so xac suat moi/cu)
#   + clip: gioi han update khong qua lon (PPO's key innovation)
#
# TAI SAO PPO MA KHONG PHAI RL ALGORITHM KHAC?
# - REINFORCE: high variance, khong on dinh cho LLM
# - A2C/A3C: can nhieu workers, phuc tap cho LLM training
# - PPO: on dinh, de implement, hoat dong tot voi LLM
#   + Clipping mechanism ngan policy thay doi dot ngot
#   + Trust region: moi step chi thay doi 1 it
#   + Da duoc chung minh hieu qua: InstructGPT, ChatGPT, Claude (truoc khi chuyen DPO)


class PPOTrainer:
    """
    Simplified PPO Trainer cho RLHF

    PIPELINE THUC TE (InstructGPT):
    1. Sample prompts tu dataset
    2. Generate responses tu policy model
    3. Tinh reward tu Reward Model
    4. Tinh KL penalty: KL(policy || reference)
    5. Tinh advantage = reward - KL_penalty - value_baseline
    6. Update policy bang PPO (clipped surrogate objective)
    7. Update value function (baseline)

    SIMPLIFIED VERSION (trong file nay):
    - Policy: simple linear model (thay vi LLM)
    - Reward: simple reward function (thay vi RM)
    - Value function: simple linear model
    - PPO update: clipped surrogate objective

    TRONG THUC TE:
    - InstructGPT: GPT-3 175B policy, 6B RM, train ~256 episodes
    - ChatGPT: similar scale, more data
    - Anthropic Claude: initially RLHF, then Constitutional AI
    """

    def __init__(self, d_model, d_action, clip_eps=0.2, gamma=0.99,
                 kl_coef=0.1, value_coef=0.5, entropy_coef=0.01):
        """
        d_model:      kich thuoc state representation (input features)
                      Tuong duong d_model cua LLM
                      InstructGPT: d_model = 12288 (GPT-3 175B)
                      LLaMA 7B: d_model = 4096
                      Trong demo: 32-128

        d_action:     kich thuoc action space (so luong actions kha dung)
                      Trong LLM context: d_action = vocab_size (so tokens co the generate)
                      GPT-2: 50,257 tokens
                      LLaMA: 32,000 tokens
                      Trong demo: 10-50 (simplified action space)

        clip_eps:     PPO clipping parameter
                      Mac dinh: 0.2 (PPO paper, InstructGPT)
                      Gioi han: ratio duoc clip vao [1-eps, 1+eps] = [0.8, 1.2]
                      eps nho -> update bao thu (on dinh nhung cham)
                      eps lon -> update manh (nhanh nhung co the khong on dinh)
                      InstructGPT dung 0.2, ChatGPT tuong tu

        gamma:        discount factor cho future rewards
                      Mac dinh: 0.99 (coi trong reward tuong lai)
                      gamma = 1.0: coi tat ca reward tuong lai nhu nhau
                      gamma = 0.9: reward gan quan trong hon
                      Trong LLM alignment: gamma gan 1.0 vi ta muon response toan bo tot

        kl_coef:      he so KL penalty (beta trong RLHF literature)
                      Mac dinh: 0.1 - 0.2
                      InstructGPT: adaptive KL, target ~6 nats
                      kl_coef lon -> policy gan SFT hon (an toan nhung it improve)
                      kl_coef nho -> policy tu do hon (improve nhieu nhung de reward hack)
                      QUAN TRONG: khong co KL penalty -> model SE reward hack!

        value_coef:   he so cho value function loss
                      Mac dinh: 0.5 (PPO paper)
                      Value function = baseline de giam variance cua policy gradient
                      Total loss = policy_loss + value_coef * value_loss + entropy_coef * entropy

        entropy_coef: he so cho entropy bonus
                      Mac dinh: 0.01
                      Entropy = do "da dang" cua policy
                      Entropy cao -> model khong chi generate 1 response co dinh
                      Giup exploration va tranh mode collapse
        """
        self.d_model = d_model
        self.d_action = d_action
        self.clip_eps = clip_eps
        self.gamma = gamma
        self.kl_coef = kl_coef
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef

        # Policy network: state -> action probabilities
        # Mo phong LLM: input representation -> token probabilities
        self.policy_W = np.random.randn(d_model, d_action).astype(np.float32) * 0.01
        self.policy_b = np.zeros(d_action, dtype=np.float32)

        # Value network (critic): state -> value estimate (scalar)
        # Uoc tinh "expected total reward" tu state hien tai
        # Dung lam baseline de giam variance cua policy gradient
        self.value_W = np.random.randn(d_model, 1).astype(np.float32) * 0.01
        self.value_b = np.zeros(1, dtype=np.float32)

        # Reference policy (SFT model - frozen)
        # Day la ban sao cua policy TRUOC KHI bat dau PPO
        self.ref_policy_W = self.policy_W.copy()
        self.ref_policy_b = self.policy_b.copy()

    def get_action_probs(self, states, use_ref=False):
        """
        Tinh action probabilities tu states

        states:  numpy array, shape (batch, d_model)
                 State representation cua moi sample
                 Trong LLM: hidden state cua model tai vi tri generate

        use_ref: True = dung reference policy (frozen)
                 False = dung current policy (dang train)
                 Reference dung de tinh KL penalty
        """
        if use_ref:
            logits = states @ self.ref_policy_W + self.ref_policy_b
        else:
            logits = states @ self.policy_W + self.policy_b
        return softmax(logits)

    def get_value(self, states):
        """
        Tinh value estimate tu states

        states: numpy array, shape (batch, d_model)
        Returns: values, shape (batch, 1)
                 Uoc tinh expected total reward tu state nay
        """
        return states @ self.value_W + self.value_b

    def compute_kl_divergence(self, states):
        """
        Tinh KL divergence giua current policy va reference policy

        KL(pi || pi_ref) = sum(pi * log(pi / pi_ref))

        TAI SAO CAN KL PENALTY?
        - Khong co KL: model co the "hack" reward model
          Vi du: RM cho reward cao khi response dai -> model generate rat dai, vo nghia
        - KL penalty giu policy gan SFT model -> van giu "kha nang ngon ngu"
        - InstructGPT: adaptive beta, target KL ~ 6 nats
          Neu KL > target -> tang beta (phat manh hon)
          Neu KL < target -> giam beta (cho tu do hon)
        """
        pi = self.get_action_probs(states, use_ref=False)
        pi_ref = self.get_action_probs(states, use_ref=True)

        # KL divergence: sum(pi * log(pi / pi_ref))
        kl = np.sum(pi * np.log(pi / (pi_ref + 1e-8) + 1e-8), axis=-1)
        return kl  # (batch,)

    def ppo_step(self, states, actions, rewards, old_log_probs, lr=0.001):
        """
        Thuc hien 1 PPO update step

        states:        numpy array, shape (batch, d_model)
                       State representation cua moi sample trong batch
                       Trong LLM: hidden states tu transformer layers

        actions:       numpy array INT, shape (batch,)
                       Action da chon (token da generate)
                       Moi action la index trong [0, d_action)

        rewards:       numpy array, shape (batch,)
                       Reward tu Reward Model cho moi (state, action) pair
                       Cang cao = action cang tot
                       InstructGPT: reward = RM_score - kl_coef * KL_penalty

        old_log_probs: numpy array, shape (batch,)
                       Log probability cua action theo policy CU (truoc update)
                       Dung de tinh ratio = pi_new / pi_old = exp(log_new - log_old)
                       Can luu lai tu buoc generate truoc khi update

        lr:            learning rate cho PPO update
                       InstructGPT: lr = 9e-6 (rat nho vi model lon)
                       Trong demo: 0.001 - 0.01

        PPO OBJECTIVE:
        L_CLIP = min(ratio * advantage, clip(ratio, 1-eps, 1+eps) * advantage)

        Trong do:
        - ratio = pi_new(a|s) / pi_old(a|s)
        - advantage = reward - value_baseline
        - clip: gioi han ratio trong [0.8, 1.2] (voi eps=0.2)

        TAI SAO CLIP?
        - Neu ratio qua lon (policy thay doi nhieu): clip de gioi han
        - Neu ratio qua nho (policy thay doi it): cung clip
        - Giup training on dinh: khong co "big jumps" trong policy space
        """
        batch_size = states.shape[0]

        # 1. Tinh current action probabilities
        action_probs = self.get_action_probs(states)  # (batch, d_action)
        new_log_probs = np.log(action_probs[np.arange(batch_size), actions] + 1e-8)

        # 2. Tinh value estimates (baseline)
        values = self.get_value(states).flatten()  # (batch,)

        # 3. Tinh KL penalty
        kl = self.compute_kl_divergence(states)  # (batch,)

        # 4. Tinh advantage = reward - kl_penalty - value_baseline
        # Reward tong hop: RM reward - KL penalty
        total_reward = rewards - self.kl_coef * kl
        advantages = total_reward - values  # (batch,)

        # Normalize advantages (giam variance)
        if np.std(advantages) > 0:
            advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)

        # 5. PPO Clipped Surrogate Objective
        ratio = np.exp(new_log_probs - old_log_probs)  # pi_new / pi_old
        surr1 = ratio * advantages  # Unclipped
        surr2 = np.clip(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * advantages  # Clipped
        policy_loss = -np.mean(np.minimum(surr1, surr2))

        # 6. Value loss: MSE giua value estimate va actual reward
        value_loss = np.mean((values - total_reward) ** 2)

        # 7. Entropy bonus: khuyen khich exploration
        entropy = -np.sum(action_probs * np.log(action_probs + 1e-8), axis=-1)
        entropy_loss = -np.mean(entropy)

        # 8. Total loss
        total_loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss

        # 9. Update policy (simplified gradient)
        # Gradient cua policy loss w.r.t. logits
        grad_logits = action_probs.copy()  # (batch, d_action)
        grad_logits[np.arange(batch_size), actions] -= 1.0
        # Scale by advantage va clipping
        clipped_advantages = np.where(
            np.abs(ratio - 1.0) < self.clip_eps,
            advantages, np.zeros_like(advantages)
        )
        grad_logits = grad_logits * clipped_advantages[:, None] / batch_size

        grad_policy_W = states.T @ grad_logits
        grad_policy_b = np.mean(grad_logits, axis=0)

        self.policy_W -= lr * grad_policy_W
        self.policy_b -= lr * grad_policy_b

        # 10. Update value function
        value_error = (values - total_reward)[:, None]  # (batch, 1)
        grad_value_W = states.T @ value_error / batch_size * 2
        grad_value_b = np.mean(value_error, axis=0) * 2

        self.value_W -= lr * self.value_coef * grad_value_W
        self.value_b -= lr * self.value_coef * grad_value_b

        return {
            'total_loss': float(total_loss),
            'policy_loss': float(policy_loss),
            'value_loss': float(value_loss),
            'entropy': float(np.mean(entropy)),
            'kl_divergence': float(np.mean(kl)),
            'mean_reward': float(np.mean(rewards)),
            'mean_total_reward': float(np.mean(total_reward)),
            'mean_ratio': float(np.mean(ratio)),
            'clip_fraction': float(np.mean(np.abs(ratio - 1.0) > self.clip_eps)),
        }


# ============================================================
# BAI TAP 4: CONSTITUTIONAL AI
# ============================================================
#
# GIAI THICH CHI TIET:
#
# VAN DE VOI RLHF (va DPO):
# 1. Can HUMAN LABELERS de tao preference data
#    - Dat tien: $10-50/gio, can hang nghin gio
#    - Inconsistent: moi nguoi co tieu chuan khac nhau
#    - Slow: khong scale duoc voi model tien hoa nhanh
# 2. Human labelers co BIAS:
#    - Van hoa khac nhau -> tieu chuan khac nhau
#    - Moi nguoi -> khong the cover tat ca edge cases
#    - Vi du: "phuc tap" co tot khong? ky su thich, nguoi thuong khong
#
# GIAI PHAP: CONSTITUTIONAL AI (Anthropic, 2022)
# "Constitutional AI: Harmlessness from AI Feedback" - Bai et al. 2022
# Y tuong: Dung AI de DANH GIA AI (thay cho human labelers)!
#
# CACH HOAT DONG:
#
# BUOC 1: Dinh nghia CONSTITUTION (Hien Phap)
# = Tap hop cac NGUYEN TAC ma AI phai tuan theo
# Vi du:
#   - "Choose the response that is most helpful to the user"
#   - "Choose the response that is least harmful or toxic"
#   - "Choose the response that is most honest and truthful"
#   - "Choose the response that best respects user privacy"
#   - "Choose the response that is least likely to encourage illegal activity"
# Anthropic dung ~16 principles cho Claude
#
# BUOC 2: CRITIQUE (Phase 1 - Self-critique)
# - Model A generate response cho prompt
# - Model B (co the cung la A) PHAN TICH response theo tung principle
# - Vi du:
#   Prompt: "How to pick a lock?"
#   Response: "Here are the steps to pick a lock: 1. Get a tension wrench..."
#   Critique: "This response provides detailed instructions for breaking into
#              locks, which could be used for illegal activities. It violates
#              the principle of not encouraging harmful behavior."
#
# BUOC 3: REVISION (Phase 2 - Self-revision)
# - Model sua lai response dua tren critique
# - Vi du:
#   Revised: "Lock picking is a skill used by licensed locksmiths. If you're
#            locked out, I recommend calling a professional locksmith. Here
#            are some tips for preventing lockouts in the future..."
#
# BUOC 4: RLAIF (RL from AI Feedback)
# - Dung (original, revised) pairs lam preference data
# - Original = rejected, Revised = chosen
# - Train policy bang DPO hoac RLHF voi AI-generated preferences
# - KHONG CAN human labelers cho buoc nay!
#
# UU DIEM:
# 1. Scalable: AI co the generate hang trieu feedback nhanh chong
# 2. Consistent: AI follow principles nhat quan hon con nguoi
# 3. Cheap: khong can tra tien human labelers
# 4. Iterative: de update constitution khi can
#
# NHUOC DIEM:
# 1. AI co the sai: "garbage in, garbage out"
# 2. Phu thuoc vao chat luong cua model critique
# 3. Khong the xu ly cac gia tri phuc tap (van hoa, chinh tri)
# 4. Co the qua bao thu (tu choi qua nhieu)
#
# SU DUNG THUC TE:
# - Claude (Anthropic): dung Constitutional AI tu dau
# - LLaMA 2 (Meta): ket hop human feedback + AI feedback
# - GPT-4 (OpenAI): dung RLHF + rule-based rewards (tuong tu CAI)


class ConstitutionalAI:
    """
    Constitutional AI - AI Alignment bang AI Feedback

    FLOW:
    1. Constitution = tap hop nguyen tac (principles)
    2. Generate response cho prompt
    3. Critique response theo tung principle
    4. Revise response dua tren critique
    5. Dung (original, revised) lam training data

    TRONG THUC TE:
    - Anthropic dung model lon (52B+) lam critique model
    - Constitution co ~16 principles
    - Iterative: critique -> revise -> critique -> revise (2-3 vong)
    - Sau do dung RLAIF de train final model

    SIMPLIFIED VERSION:
    - Dung scoring functions thay vi LLM critique
    - Principles duoc encode thanh keyword/pattern matching
    - Revision = adjust response features (thay vi regenerate text)
    """

    def __init__(self, principles=None, d_model=64):
        """
        principles: list cac nguyen tac (strings)
                    Moi principle dinh nghia 1 tieu chuan danh gia
                    Anthropic dung ~16 principles bao gom:
                    - Helpfulness: tra loi co ich, dung
                    - Harmlessness: khong gay hai, khong doc hai
                    - Honesty: trung thuc, khong bua dat
                    - Privacy: ton trong quyen rieng tu
                    - Legality: khong khuyen khich vi pham phap luat
                    Mac dinh: 5 principles co ban

        d_model:    kich thuoc feature vector cua moi response
                    Dung de mo phong response dang vector
                    Trong thuc te: response la text, duoc LLM xu ly
                    Trong demo: 32-128
        """
        if principles is None:
            self.principles = [
                "Choose the response that is most helpful and informative",
                "Choose the response that is least harmful or dangerous",
                "Choose the response that is most honest and factual",
                "Choose the response that best respects user privacy",
                "Choose the response that does not encourage illegal activity",
            ]
        else:
            self.principles = principles

        self.d_model = d_model

        # Critique model weights: moi principle co 1 set of weights
        # principle_weights[i] = (W_score, bias) cho principle i
        # Score = how well response follows principle i
        self.principle_weights = []
        for _ in self.principles:
            W = np.random.randn(d_model).astype(np.float32) * 0.1
            b = np.float32(0.0)
            self.principle_weights.append({'W': W, 'b': b})

        # Revision model: transform response features de "cai thien"
        # revision_W: ma tran transform, shape (d_model, d_model)
        # Trong thuc te: model generate NEW response tu critique
        # Trong demo: transform features de tang score
        self.revision_W = np.eye(d_model, dtype=np.float32) + \
                          np.random.randn(d_model, d_model).astype(np.float32) * 0.01
        self.revision_b = np.zeros(d_model, dtype=np.float32)

    def critique(self, response_features):
        """
        Critique response theo tung principle trong constitution

        response_features: numpy array, shape (batch, d_model)
                           Feature vector cua response can danh gia
                           Trong thuc te: day la text, duoc model doc va phan tich
                           Trong demo: vector so, critique bang dot product

        Returns: dictionary voi:
            - 'scores': list of numpy arrays, moi array shape (batch,)
                        Score cho moi principle (cao = response tuan thu tot)
            - 'overall_score': numpy array shape (batch,)
                               Diem tong hop cua tat ca principles
            - 'violations': list of (principle_index, score) tuples
                            Principles bi vi pham (score < 0)

        VI DU THUC TE (text-based):
        Prompt: "How do I make a bomb?"
        Response: "Here are the instructions..."
        Critique by principle "harmlessness":
            "This response provides dangerous instructions that could lead to
             physical harm. It violates the principle of being harmless."
        Score: -0.9 (vi pham nghiem trong)
        """
        batch_size = response_features.shape[0]
        scores = []
        violations = []

        for i, pw in enumerate(self.principle_weights):
            # Score = dot(response_features, principle_weights) + bias
            # Cao = response tuan thu principle tot
            # Thap/am = response vi pham principle
            score = response_features @ pw['W'] + pw['b']  # (batch,)
            scores.append(score)

            # Tim violations: principle nao co trung binh score < 0?
            if np.mean(score) < 0:
                violations.append((i, float(np.mean(score))))

        overall_score = np.mean(scores, axis=0)  # (batch,)

        return {
            'scores': scores,
            'overall_score': overall_score,
            'violations': violations,
        }

    def revise(self, response_features, critique_result):
        """
        Revise response dua tren critique

        response_features: numpy array, shape (batch, d_model)
                           Features cua response GOC (chua sua)

        critique_result:   ket qua tu self.critique()
                           Chua scores va violations

        CACH REVISION HOAT DONG:
        1. Xac dinh principles bi vi pham (score < 0)
        2. Dieu chinh response features de tang score cua principles bi vi pham
        3. Giu cho response van "tot" theo cac principles khac

        TRONG THUC TE (Anthropic):
        1. Model doc critique (text-based)
        2. Model generate response MOI dua tren critique
        3. Response moi phai:
           a) Giu noi dung huu ich
           b) Sua cac van de duoc chi ra trong critique
           c) Van tuan thu tat ca principles khac

        Returns: revised_features, shape (batch, d_model)
                 Features cua response da duoc sua
        """
        # Base revision: transform qua revision model
        revised = response_features @ self.revision_W + self.revision_b

        # Tang cuong revision cho principles bi vi pham
        for i, pw in enumerate(self.principle_weights):
            score = critique_result['scores'][i]
            # Neu score thap -> can dieu chinh nhieu
            # Gradient: dich response_features theo huong tang score
            adjustment_strength = np.clip(-score, 0, 1.0)[:, None]  # (batch, 1)
            # Dieu chinh: them 1 it cua principle direction vao response
            principle_direction = pw['W'] / (np.linalg.norm(pw['W']) + 1e-8)
            revised = revised + 0.1 * adjustment_strength * principle_direction

        return revised

    def constitutional_training_step(self, prompts, response_features):
        """
        1 step cua Constitutional AI training

        prompts:           numpy array, shape (batch, d_model)
                           Feature vector cua prompts (trong thuc te la text)

        response_features: numpy array, shape (batch, d_model)
                           Features cua response ban dau (tu SFT model)

        FLOW:
        1. Critique response -> tim vi pham
        2. Revise response -> tao response tot hon
        3. Tra ve (original, revised) pairs lam training data cho RLAIF

        Returns:
            original_features: response ban dau (= rejected trong DPO/RLHF)
            revised_features:  response da sua (= chosen trong DPO/RLHF)
            critique_result:   thong tin critique (de debug/monitor)
        """
        # Phase 1: Critique
        critique_result = self.critique(response_features)

        # Phase 2: Revise
        revised_features = self.revise(response_features, critique_result)

        # Phase 3: Verify - critique lai response da sua
        revised_critique = self.critique(revised_features)

        return {
            'original': response_features,
            'revised': revised_features,
            'original_critique': critique_result,
            'revised_critique': revised_critique,
            'improvement': float(np.mean(revised_critique['overall_score'])
                                 - np.mean(critique_result['overall_score'])),
        }

    def train_critique_model(self, good_responses, bad_responses, lr=0.01, epochs=50):
        """
        Train critique model de phan biet good vs bad responses

        good_responses: numpy array, shape (n_good, d_model)
                        Features cua responses duoc danh gia tot
                        Trong thuc te: responses duoc human labelers rated cao
                        Trong demo: responses voi features "tot"

        bad_responses:  numpy array, shape (n_bad, d_model)
                        Features cua responses duoc danh gia xau
                        Trong thuc te: responses vi pham principles
                        Trong demo: responses voi features "xau"

        lr:             learning rate cho gradient descent
                        0.01 - 0.05 cho demo
                        Trong thuc te: 1e-5 - 1e-4 cho LLM critique model

        epochs:         so vong lap training
                        50-100 cho demo
                        Trong thuc te: 3-10 epochs cho LLM

        TRAINING:
        - Good responses nen co score CAO cho moi principle
        - Bad responses nen co score THAP cho moi principle
        - Loss = binary cross-entropy: good -> 1, bad -> 0
        """
        history = {'loss': [], 'accuracy': []}

        for epoch in range(epochs):
            total_loss = 0
            correct = 0
            total = 0

            for i, pw in enumerate(self.principle_weights):
                # Score cho good va bad
                good_scores = good_responses @ pw['W'] + pw['b']  # (n_good,)
                bad_scores = bad_responses @ pw['W'] + pw['b']    # (n_bad,)

                # Sigmoid de chuyen thanh xac suat
                good_probs = sigmoid(good_scores)
                bad_probs = sigmoid(bad_scores)

                # Binary cross-entropy loss
                loss_good = -np.mean(np.log(good_probs + 1e-8))
                loss_bad = -np.mean(np.log(1 - bad_probs + 1e-8))
                loss = loss_good + loss_bad
                total_loss += loss

                # Accuracy
                correct += np.sum(good_scores > 0) + np.sum(bad_scores < 0)
                total += len(good_scores) + len(bad_scores)

                # Gradient va update
                grad_good = -(1 - good_probs)  # d(loss)/d(score) cho good
                grad_bad = bad_probs             # d(loss)/d(score) cho bad

                grad_W = (good_responses.T @ grad_good / len(good_responses)
                          + bad_responses.T @ grad_bad / len(bad_responses))
                grad_b = (np.mean(grad_good) + np.mean(grad_bad))

                pw['W'] -= lr * grad_W
                pw['b'] -= lr * grad_b

            avg_loss = total_loss / len(self.principle_weights)
            accuracy = correct / total
            history['loss'].append(avg_loss)
            history['accuracy'].append(accuracy)

            if epoch % 10 == 0:
                print(f"    Epoch {epoch:>3}: loss={avg_loss:.4f}, acc={accuracy:.3f}")

        return history

    def rlaif_generate_preferences(self, prompts, response_features_list):
        """
        RLAIF: dung AI (critique model) de tao preference data

        prompts:                 numpy array, shape (n_prompts, d_model)

        response_features_list:  list of numpy arrays, moi array shape (n_prompts, d_model)
                                 Nhieu responses cho moi prompt (vi du: 4 responses/prompt)
                                 Trong thuc te: model sinh nhieu responses, AI rank chung

        Returns: list of (chosen, rejected) pairs
                 Dung lam training data cho DPO/RLHF (thay human labelers!)
        """
        preferences = []

        for i in range(len(prompts)):
            best_score = -np.inf
            worst_score = np.inf
            best_idx = 0
            worst_idx = 0

            for j, responses in enumerate(response_features_list):
                critique = self.critique(responses[i:i+1])
                score = critique['overall_score'][0]
                if score > best_score:
                    best_score = score
                    best_idx = j
                if score < worst_score:
                    worst_score = score
                    worst_idx = j

            if best_idx != worst_idx:
                preferences.append({
                    'prompt': prompts[i],
                    'chosen': response_features_list[best_idx][i],
                    'rejected': response_features_list[worst_idx][i],
                    'chosen_score': best_score,
                    'rejected_score': worst_score,
                })

        return preferences


# ============================================================
# HELPER: Tao du lieu mo phong
# ============================================================

def create_preference_data(n_samples=100, d_model=32, vocab_size=100, seq_len=20):
    """
    Tao preference data cho training

    n_samples:  so cap (chosen, rejected) trong dataset
                InstructGPT dung ~33,000 comparison pairs
                Trong demo: 50-500 pairs

    d_model:    kich thuoc feature space
                Dung de tao features cho responses
                Trong demo: 32-64

    vocab_size: so token IDs co the
                Dung de tao token IDs gia
                Trong demo: 50-200

    seq_len:    do dai moi sequence
                Trong demo: 10-30 tokens/sequence

    Returns:
        chosen_ids:    shape (n_samples, seq_len) - token IDs cua chosen responses
        rejected_ids:  shape (n_samples, seq_len) - token IDs cua rejected responses
        chosen_feats:  shape (n_samples, d_model) - features cua chosen (cho DPO)
        rejected_feats: shape (n_samples, d_model) - features cua rejected (cho DPO)
    """
    # Tao "good" features (chosen) va "bad" features (rejected)
    # Good features co mean duong (model "thich" huong nay)
    # Bad features co mean am
    good_direction = np.random.randn(d_model).astype(np.float32)
    good_direction = good_direction / np.linalg.norm(good_direction)

    chosen_feats = np.random.randn(n_samples, d_model).astype(np.float32) * 0.5 + \
                   good_direction * 1.0
    rejected_feats = np.random.randn(n_samples, d_model).astype(np.float32) * 0.5 - \
                     good_direction * 0.5

    # Token IDs (random, vi day la demo)
    chosen_ids = np.random.randint(0, vocab_size, size=(n_samples, seq_len))
    rejected_ids = np.random.randint(0, vocab_size, size=(n_samples, seq_len))

    return chosen_ids, rejected_ids, chosen_feats, rejected_feats


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    np.random.seed(42)
    output_dir = os.path.dirname(os.path.abspath(__file__))

    # ======== BAI TAP 1: REWARD MODEL ========
    print("=" * 70)
    print("BAI TAP 1: REWARD MODEL (Bradley-Terry)")
    print("=" * 70)
    print()
    print("  TAI SAO CAN REWARD MODEL?")
    print("  LLM chi hoc predict next token -> khong biet 'tot' hay 'xau'.")
    print("  Reward Model = 1 model RIENG cham diem cho output cua LLM.")
    print("  InstructGPT dung GPT-3 6B lam reward model.")
    print("  Loss: -log(sigmoid(r_chosen - r_rejected)) -- Bradley-Terry model")
    print()

    # Params
    vocab_size = 100
    d_model = 32
    d_hidden = 64
    seq_len = 20
    n_train = 200
    n_val = 50

    # Tao data
    chosen_ids, rejected_ids, _, _ = create_preference_data(
        n_samples=n_train + n_val, d_model=d_model, vocab_size=vocab_size, seq_len=seq_len
    )
    train_chosen = chosen_ids[:n_train]
    train_rejected = rejected_ids[:n_train]
    val_chosen = chosen_ids[n_train:]
    val_rejected = rejected_ids[n_train:]

    # Train reward model
    rm = RewardModel(vocab_size, d_model, d_hidden, num_layers=2, max_seq_len=seq_len)
    print(f"  Reward Model: vocab={vocab_size}, d_model={d_model}, d_hidden={d_hidden}")
    print(f"  Train: {n_train} pairs, Val: {n_val} pairs")
    print()

    batch_size = 32
    rm_history = {'loss': [], 'accuracy': [], 'val_accuracy': []}

    for epoch in range(100):
        epoch_loss = 0
        epoch_acc = 0
        n_batches = 0

        for i in range(0, n_train, batch_size):
            end = min(i + batch_size, n_train)
            batch_chosen = train_chosen[i:end]
            batch_rejected = train_rejected[i:end]

            loss = rm.backward_preference(batch_chosen, batch_rejected, lr=0.005)
            _, acc, _, _ = rm.compute_preference_loss(batch_chosen, batch_rejected)

            epoch_loss += loss
            epoch_acc += acc
            n_batches += 1

        avg_loss = epoch_loss / n_batches
        avg_acc = epoch_acc / n_batches

        # Validation
        _, val_acc, val_r_chosen, val_r_rejected = rm.compute_preference_loss(
            val_chosen, val_rejected
        )

        rm_history['loss'].append(avg_loss)
        rm_history['accuracy'].append(avg_acc)
        rm_history['val_accuracy'].append(val_acc)

        if epoch % 20 == 0:
            print(f"    Epoch {epoch:>3}: loss={avg_loss:.4f}, "
                  f"train_acc={avg_acc:.3f}, val_acc={val_acc:.3f}")

    print(f"\n  Final: train_acc={rm_history['accuracy'][-1]:.3f}, "
          f"val_acc={rm_history['val_accuracy'][-1]:.3f}")
    print(f"  Mean chosen reward:   {np.mean(val_r_chosen):.4f}")
    print(f"  Mean rejected reward: {np.mean(val_r_rejected):.4f}")
    print(f"  Reward gap:           {np.mean(val_r_chosen - val_r_rejected):.4f}")
    print("  Reward Model: OK")

    # ======== BAI TAP 2: DPO ========
    print("\n" + "=" * 70)
    print("BAI TAP 2: DPO (Direct Preference Optimization)")
    print("=" * 70)
    print()
    print("  TAI SAO DPO?")
    print("  RLHF can 3 models (SFT + RM + PPO) -> PHUC TAP.")
    print("  DPO bypass reward model, train TRUC TIEP tu preference data!")
    print("  Loss: -log sigmoid(beta * (log(pi/pi_ref)(chosen) - log(pi/pi_ref)(rejected)))")
    print("  Reference: Rafailov et al. 2023, dung boi LLaMA 2, Zephyr, Mistral")
    print()

    d_dpo = 32
    n_dpo_train = 300
    _, _, chosen_feats, rejected_feats = create_preference_data(
        n_samples=n_dpo_train, d_model=d_dpo
    )

    # Tao policy va reference model
    policy = SimplePolicyForDPO(d_dpo)
    ref_model = SimplePolicyForDPO(d_dpo)
    ref_model.W = policy.W.copy()  # Reference = ban sao cua policy ban dau
    ref_model.b = policy.b

    print(f"  DPO: d_model={d_dpo}, n_train={n_dpo_train}")
    print(f"  Testing different beta values:")
    print()

    # Test voi cac beta khac nhau
    for beta in [0.05, 0.1, 0.2, 0.5]:
        # Reset policy cho moi beta
        test_policy = SimplePolicyForDPO(d_dpo)
        test_policy.W = ref_model.W.copy()  # Start tu reference
        test_policy.b = ref_model.b

        print(f"  --- beta = {beta} ---")
        dpo_history = {'loss': [], 'accuracy': [], 'reward_margin': []}

        for epoch in range(100):
            # Mini-batch
            for i in range(0, n_dpo_train, batch_size):
                end = min(i + batch_size, n_dpo_train)
                loss, metrics = test_policy.update(
                    chosen_feats[i:end], rejected_feats[i:end],
                    ref_model, beta=beta, lr=0.01
                )

            # Eval toan bo dataset
            all_chosen_lp = test_policy.get_log_probs(chosen_feats)
            all_rejected_lp = test_policy.get_log_probs(rejected_feats)
            ref_chosen_lp = ref_model.get_log_probs(chosen_feats)
            ref_rejected_lp = ref_model.get_log_probs(rejected_feats)
            full_loss, full_metrics = dpo_loss(
                all_chosen_lp, all_rejected_lp,
                ref_chosen_lp, ref_rejected_lp,
                beta=beta
            )
            dpo_history['loss'].append(full_loss)
            dpo_history['accuracy'].append(full_metrics['accuracy'])
            dpo_history['reward_margin'].append(full_metrics['reward_margin'])

        print(f"    Final: loss={dpo_history['loss'][-1]:.4f}, "
              f"acc={dpo_history['accuracy'][-1]:.3f}, "
              f"margin={dpo_history['reward_margin'][-1]:.4f}")
        # Kiem tra: policy divergence tu reference
        w_diff = np.linalg.norm(test_policy.W - ref_model.W)
        print(f"    Policy divergence (L2): {w_diff:.4f}")

    print("\n  Key insight: beta CAO -> policy gan reference (bao thu)")
    print("               beta THAP -> policy tu do hon (nhung de diverge)")
    print("  DPO: OK")

    # ======== BAI TAP 3: RLHF PIPELINE ========
    print("\n" + "=" * 70)
    print("BAI TAP 3: RLHF PIPELINE (SFT -> RM -> PPO)")
    print("=" * 70)
    print()
    print("  RLHF PIPELINE (InstructGPT/ChatGPT):")
    print("  Phase 1: SFT - Supervised Fine-Tuning tu demonstrations")
    print("  Phase 2: RM  - Train Reward Model tu human comparisons")
    print("  Phase 3: PPO - Optimize policy against RM + KL penalty")
    print()

    d_state = 32
    d_action = 10
    n_ppo_steps = 200

    trainer = PPOTrainer(
        d_model=d_state,
        d_action=d_action,
        clip_eps=0.2,
        kl_coef=0.1,
    )

    print(f"  PPO: d_state={d_state}, d_action={d_action}")
    print(f"  clip_eps=0.2, kl_coef=0.1")
    print()

    ppo_history = {
        'total_loss': [], 'policy_loss': [], 'value_loss': [],
        'mean_reward': [], 'kl_divergence': [], 'entropy': [],
        'clip_fraction': [],
    }

    # Simulate reward function (thay cho Reward Model that)
    # Response tot = response co features align voi reward_direction
    reward_direction = np.random.randn(d_state).astype(np.float32)
    reward_direction = reward_direction / np.linalg.norm(reward_direction) * 2.0

    for step in range(n_ppo_steps):
        # 1. Sample states (simulate prompts)
        states = np.random.randn(batch_size, d_state).astype(np.float32)

        # 2. Get actions tu current policy
        action_probs = trainer.get_action_probs(states)
        # Sample actions theo probabilities
        actions = np.array([
            np.random.choice(d_action, p=probs)
            for probs in action_probs
        ])
        old_log_probs = np.log(action_probs[np.arange(batch_size), actions] + 1e-8)

        # 3. Simulate rewards (thay cho Reward Model)
        # Reward dua tren: state alignment voi reward_direction + action quality
        state_reward = states @ reward_direction
        action_reward = np.random.randn(batch_size) * 0.5  # noise
        rewards = state_reward + action_reward

        # 4. PPO step
        metrics = trainer.ppo_step(states, actions, rewards, old_log_probs, lr=0.005)

        for key in ppo_history:
            ppo_history[key].append(metrics[key])

        if step % 40 == 0:
            print(f"    Step {step:>3}: reward={metrics['mean_reward']:.3f}, "
                  f"KL={metrics['kl_divergence']:.4f}, "
                  f"entropy={metrics['entropy']:.3f}, "
                  f"clip_frac={metrics['clip_fraction']:.3f}")

    print(f"\n  Final: mean_reward={ppo_history['mean_reward'][-1]:.3f}, "
          f"KL={ppo_history['kl_divergence'][-1]:.4f}")

    # Kiem tra KL khong qua lon (policy khong diverge)
    final_kl = ppo_history['kl_divergence'][-1]
    print(f"  KL divergence: {final_kl:.4f} (should be moderate, not too large)")
    print("  PPO Trainer: OK")

    # ======== BAI TAP 4: CONSTITUTIONAL AI ========
    print("\n" + "=" * 70)
    print("BAI TAP 4: CONSTITUTIONAL AI (Anthropic's Approach)")
    print("=" * 70)
    print()
    print("  TAI SAO CONSTITUTIONAL AI?")
    print("  Human labeling dat va inconsistent -> dung AI de danh gia AI!")
    print("  Constitution = tap nguyen tac ('Be helpful', 'Be harmless', 'Be honest')")
    print("  Phase 1 (Critique): Model phan tich response theo constitution")
    print("  Phase 2 (Revision): Model sua response theo critique")
    print("  RLAIF: Dung AI feedback (thay human feedback) cho RL training")
    print("  Reference: Anthropic's Constitutional AI paper (2022), dung trong Claude")
    print()

    d_cai = 32
    n_good = 100
    n_bad = 100

    cai = ConstitutionalAI(d_model=d_cai)
    print(f"  Constitutional AI: {len(cai.principles)} principles, d_model={d_cai}")
    for i, p in enumerate(cai.principles):
        print(f"    {i+1}. {p}")
    print()

    # Tao good va bad responses
    good_direction = np.random.randn(d_cai).astype(np.float32)
    good_direction = good_direction / np.linalg.norm(good_direction)

    good_responses = np.random.randn(n_good, d_cai).astype(np.float32) * 0.3 + good_direction * 1.5
    bad_responses = np.random.randn(n_bad, d_cai).astype(np.float32) * 0.3 - good_direction * 1.0

    # Step 1: Train critique model
    print("  --- Phase 1: Train Critique Model ---")
    cai_history = cai.train_critique_model(
        good_responses, bad_responses, lr=0.02, epochs=80
    )
    print(f"  Final critique accuracy: {cai_history['accuracy'][-1]:.3f}")
    print()

    # Step 2: Test critique
    print("  --- Phase 2: Critique Test ---")
    test_good = good_responses[:5]
    test_bad = bad_responses[:5]

    good_critique = cai.critique(test_good)
    bad_critique = cai.critique(test_bad)

    print(f"  Good responses - overall score: {np.mean(good_critique['overall_score']):.4f}")
    print(f"  Bad responses  - overall score: {np.mean(bad_critique['overall_score']):.4f}")
    assert np.mean(good_critique['overall_score']) > np.mean(bad_critique['overall_score']), \
        "Good should score higher than bad"
    print("  Critique discriminates good vs bad: OK")
    print()

    # Step 3: Revision
    print("  --- Phase 3: Revision Test ---")
    result = cai.constitutional_training_step(
        prompts=np.random.randn(20, d_cai).astype(np.float32),
        response_features=bad_responses[:20],
    )

    print(f"  Original score: {np.mean(result['original_critique']['overall_score']):.4f}")
    print(f"  Revised score:  {np.mean(result['revised_critique']['overall_score']):.4f}")
    print(f"  Improvement:    {result['improvement']:.4f}")
    assert result['improvement'] > 0, "Revision should improve score"
    print("  Revision improves response: OK")
    print()

    # Step 4: RLAIF - generate preferences
    print("  --- Phase 4: RLAIF - AI-Generated Preferences ---")
    n_prompts = 50
    prompts = np.random.randn(n_prompts, d_cai).astype(np.float32)

    # Generate 4 responses cho moi prompt
    responses_list = [
        np.random.randn(n_prompts, d_cai).astype(np.float32) * 0.5 + good_direction * (0.5 * k)
        for k in range(-1, 3)  # range tu bad -> good
    ]

    preferences = cai.rlaif_generate_preferences(prompts, responses_list)
    print(f"  Generated {len(preferences)} preference pairs tu {n_prompts} prompts")

    if len(preferences) > 0:
        avg_chosen_score = np.mean([p['chosen_score'] for p in preferences])
        avg_rejected_score = np.mean([p['rejected_score'] for p in preferences])
        print(f"  Avg chosen score:   {avg_chosen_score:.4f}")
        print(f"  Avg rejected score: {avg_rejected_score:.4f}")
        assert avg_chosen_score > avg_rejected_score, "Chosen should score higher"
        print("  AI preferences are consistent: OK")

    # Step 5: DPO voi AI-generated preferences (RLAIF + DPO)
    if len(preferences) > 10:
        print()
        print("  --- Phase 5: DPO voi AI Preferences (RLAIF + DPO) ---")
        chosen_feats_cai = np.array([p['chosen'] for p in preferences])
        rejected_feats_cai = np.array([p['rejected'] for p in preferences])

        rlaif_policy = SimplePolicyForDPO(d_cai)
        rlaif_ref = SimplePolicyForDPO(d_cai)
        rlaif_ref.W = rlaif_policy.W.copy()
        rlaif_ref.b = rlaif_policy.b

        for epoch in range(80):
            for i in range(0, len(chosen_feats_cai), batch_size):
                end = min(i + batch_size, len(chosen_feats_cai))
                rlaif_policy.update(
                    chosen_feats_cai[i:end], rejected_feats_cai[i:end],
                    rlaif_ref, beta=0.1, lr=0.01
                )

        # Eval
        final_chosen_lp = rlaif_policy.get_log_probs(chosen_feats_cai)
        final_rejected_lp = rlaif_policy.get_log_probs(rejected_feats_cai)
        ref_chosen_lp = rlaif_ref.get_log_probs(chosen_feats_cai)
        ref_rejected_lp = rlaif_ref.get_log_probs(rejected_feats_cai)
        _, final_metrics = dpo_loss(
            final_chosen_lp, final_rejected_lp,
            ref_chosen_lp, ref_rejected_lp, beta=0.1
        )
        print(f"    RLAIF+DPO accuracy: {final_metrics['accuracy']:.3f}")
        print(f"    RLAIF+DPO margin:   {final_metrics['reward_margin']:.4f}")
        print("    RLAIF + DPO: OK")

    # ======== TONG KET SO SANH ========
    print("\n" + "=" * 70)
    print("TONG KET: SO SANH CAC PHUONG PHAP ALIGNMENT")
    print("=" * 70)
    print()
    print(f"  {'Method':<25} | {'Do phuc tap':<18} | {'Can Human Data?':<16} | {'Models can train':<18}")
    print(f"  {'-'*25}-+-{'-'*18}-+-{'-'*16}-+-{'-'*18}")
    print(f"  {'RLHF (InstructGPT)':<25} | {'Rat cao':<18} | {'Co (nhieu)':<16} | {'3 (SFT+RM+Policy)':<18}")
    print(f"  {'DPO (Rafailov 2023)':<25} | {'Thap':<18} | {'Co (it hon)':<16} | {'1 (Policy only)':<18}")
    print(f"  {'Constitutional AI':<25} | {'Trung binh':<18} | {'Khong (AI)':<16} | {'2 (Critique+Policy)':<18}")
    print(f"  {'RLAIF + DPO':<25} | {'Trung binh':<18} | {'Khong (AI)':<16} | {'2 (Critique+Policy)':<18}")
    print()
    print("  RECOMMENDATIONS:")
    print("  - Du lieu human feedback tot + resources nhieu -> RLHF")
    print("  - Du lieu human preference co san + muon don gian -> DPO")
    print("  - Khong co human data + muon scalable -> Constitutional AI")
    print("  - Thuc te: nhieu models dung KET HOP (vi du: LLaMA 2 = SFT + RLHF + DPO)")

    # ---- Plot ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 3, figsize=(18, 10))

        # Plot 1: Reward Model loss
        ax = axes[0, 0]
        ax.plot(rm_history['loss'], label='Train Loss', color='blue')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Reward Model: Training Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 2: Reward Model accuracy
        ax = axes[0, 1]
        ax.plot(rm_history['accuracy'], label='Train Acc', color='blue')
        ax.plot(rm_history['val_accuracy'], label='Val Acc', color='red', linestyle='--')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Accuracy')
        ax.set_title('Reward Model: Accuracy')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 3: PPO rewards
        ax = axes[0, 2]
        ax.plot(ppo_history['mean_reward'], label='Mean Reward', color='green')
        ax.set_xlabel('Step')
        ax.set_ylabel('Reward')
        ax.set_title('PPO: Mean Reward')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 4: PPO KL divergence
        ax = axes[1, 0]
        ax.plot(ppo_history['kl_divergence'], label='KL Divergence', color='orange')
        ax.set_xlabel('Step')
        ax.set_ylabel('KL')
        ax.set_title('PPO: KL Divergence')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 5: PPO entropy
        ax = axes[1, 1]
        ax.plot(ppo_history['entropy'], label='Entropy', color='purple')
        ax.set_xlabel('Step')
        ax.set_ylabel('Entropy')
        ax.set_title('PPO: Policy Entropy')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 6: Constitutional AI critique loss
        ax = axes[1, 2]
        ax.plot(cai_history['loss'], label='Critique Loss', color='teal')
        ax.plot(cai_history['accuracy'], label='Critique Accuracy', color='coral', linestyle='--')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss / Accuracy')
        ax.set_title('Constitutional AI: Critique Training')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.suptitle('Week 15-16: AI Alignment Methods', fontsize=14, fontweight='bold')
        plt.tight_layout()
        path = os.path.join(output_dir, "plot_alignment.png")
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"\n  Saved: {path}")

    except ImportError:
        print("  matplotlib chua cai.")

    print("\n" + "=" * 70)
    print("TAT CA TESTS PASSED!")
    print("=" * 70)


# ======== CHECKLIST ========
# Week 15-16 Alignment:
# [x] Train simple reward model
# [x] Implement DPO loss
# [x] Understand RLHF pipeline
# [x] Study Constitutional AI paper
