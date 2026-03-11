# ai_path Code Walkthrough (Detailed)

Tài liệu này đi qua **từng mô-đun và từng hàm/chức năng chính** trong thư mục `ai_path/`. Với mỗi file, bạn sẽ thấy:

1. Đường dẫn + số dòng khởi đầu (dựa trên `rg -n`) để dễ mở trong editor.
2. Giải thích theo trình tự từng khối lệnh (gần như “từng dòng”) để bạn hiểu rõ hàm làm gì.

> Tip: mở file song song với phần dưới, đọc mô tả rồi bước qua code sẽ nắm cực nhanh.

---

## 01_linear_algebra.py — Matrix tối giản

### `class Matrix` (`01_linear_algebra.py:20`)
- `__init__(self, data)`
  1. Lưu dữ liệu thô `data` (list of lists).
  2. Đếm số hàng `rows = len(data)`.
  3. Lấy số cột `cols = len(data[0])` nếu có dữ liệu, ngược lại 0.
- `__repr__`: trả string `Matrix([...])` để debug.
- `dot(self, other)`
  1. Validate: số cột của self phải bằng số hàng của other; nếu không raise `ValueError`.
  2. Tạo `result` là ma trận `rows x other.cols` toàn 0.
  3. Ba vòng lặp `i/j/k`: tích lũy `self.data[i][k] * other.data[k][j]` vào `result[i][j]`.
  4. Trả `Matrix(result)`.
- `transpose`
  1. Duyệt từng cột `i`, tạo hàng mới bằng cách lấy `self.data[j][i]`.
  2. Trả ma trận hoán vị (dùng nhiều trong backprop).
- `add` / `subtract`
  1. Kiểm tra kích thước khớp.
  2. Dùng list comprehension cộng/trừ từng phần tử `self.data[i][j] ± other.data[i][j]`.
- `scalar_multiply`
  1. Nhân mọi phần tử với `scalar`.
  2. Dùng trong bước update: `lr * gradient`.
- `element_multiply`
  1. Hadamard product: nhân từng cặp phần tử (ReLU mask, dropout).
- `apply(func)`
  1. Duyệt từng phần tử, áp dụng hàm kích hoạt bất kỳ (ReLU, sigmoid…).

### Helper (`01_linear_algebra.py:159`)
- `random_matrix(rows, cols, low, high)`: sinh ma trận random lấy từ `uniform(low, high)`, trả `Matrix`.

### `__main__`
- Chạy loạt test: nhân ma trận, transpose, subtract, mô phỏng forward NN, kiểm tra ValueError.

---

## 02_gradient_descent.py — Optimizer nền tảng

### Hàm tiện ích
- `numerical_gradient(f, x, h=1e-7)` (`line 19`)
  1. Sao chép `x` thành list để tránh mutate.
  2. Với mỗi chiều `i`, cộng `h`, trừ `h`, tính `(f(x+h) - f(x-h)) / (2h)`.
  3. Trả list gradient.
- `gradient_descent(f, grad_f, x0, lr, epochs, verbose)` (`line 42`)
  1. Initialize `x = x0.copy()`, `history = []`.
  2. Lặp `epochs`: tính loss, append vào `history`.
  3. Lấy gradient từ `grad_f(x)` và cập nhật `x[i] -= lr * grad[i]`.
  4. Trả nghiệm cuối + history loss.
- `sgd_with_momentum(..., momentum=0.9)` (`line 82`)
  1. Giữ vector `velocity` ban đầu 0.
  2. Mỗi epoch: `velocity = momentum*velocity - lr*grad`.
  3. Cập nhật `x += velocity`.
- `adam(...)` (`line 117`)
  1. Khởi tạo `m`, `v` (first/second moment) = 0.
  2. Mỗi bước: update `m`, `v` bằng exponential moving average.
  3. Bias-correct: `m_hat = m / (1 - beta1^t)`; tương tự cho `v_hat`.
  4. Update `x -= lr * m_hat / (sqrt(v_hat)+eps)`.
- `simple_quadratic`, `grad_simple_quadratic` (`line 165`) và `rosenbrock`, `grad_rosenbrock` (`line 174` & `188`)
  - Hàm test để bạn chạy optimizer xem hội tụ thế nào.

---

## 03_neural_network.py — MLP thủ công

### Activation (`03_neural_network.py:26`)
- `sigmoid`, `relu`, `softmax`, cùng hàm đạo hàm tương ứng: mỗi hàm áp dụng numpy trên array, clip giá trị để tránh overflow.

### `class Layer` (`line 49`)
- `__init__(input_size, output_size, activation)`
  1. Chọn scale (He vs Xavier) dựa trên activation.
  2. Khởi tạo `weights` Gaussian * scale, `bias` = 0.
  3. Lưu activation name + các biến cache (`input`, `z`, `output`, gradients).
- `forward(x)`
  1. Lưu `input = x`, tính `z = x @ weights + bias`.
  2. Áp dụng activation: ReLU/Sigmoid/Softmax/linear, lưu `output`.
  3. Trả `output`.
- `backward(grad_output)`
  1. Tính `grad_z` = `grad_output * activation_derivative(z)`.
  2. `self.grad_weights = input.T @ grad_z / batch_size`.
  3. `self.grad_bias = sum(grad_z)/batch_size`.
  4. Tính gradient truyền về = `grad_z @ weights.T`, trả về cho layer trước.
- `update(lr)`: `weights -= lr * grad_weights`, `bias -= lr * grad_bias`.

### `class Dropout` (`line 142`)
- `forward(x, training)`
  1. Nếu training: tạo mask Bernoulli `(rand > p)`; scale `x * mask / (1-p)`.
  2. Nếu inference: trả nguyên `x`.
- `backward(grad_output)`: nhân lại với mask để truyền gradient.

### `class BatchNorm` (`line 182`)
- `forward(x, training)`:
  1. Training: tính mean/var theo batch, update running stats.
  2. Normalize `x_hat = (x - mean)/sqrt(var+eps)`, scale + shift với gamma/beta.
  3. Cache `x_hat`, `std_inv` cho backward.
- `backward(grad_output)`: triển khai công thức chuẩn BN (tính grad gamma/beta và grad_input).

### `class NeuralNetwork` (`line 263`)
- `__init__(layer_sizes, hidden_activation)`:
  1. Loop qua `layer_sizes`, build `Layer` tương ứng (last layer softmax).
  2. Lưu list layer vào `self.layers`.
- `forward(x)`:
  1. Cho `x` đi qua từng layer, update `x = layer.forward(x)`.
- `backward(y_true)`:
  1. Bắt đầu từ output layer: `grad = last.output - y_true` (cross-entropy + softmax).
  2. Loop reversed layers, gọi `layer.backward(grad)` để propagate.
- `update(lr)`: gọi `layer.update(lr)` cho từng layer.
- `compute_loss`: cross-entropy với clipping `1e-15`.
- `train`: mini-batch training (shuffle, forward, backward, update, log history).

### `one_hot(labels, num_classes)` (`line 390`)
- Dựng ma trận zero, set 1 ở cột tương ứng label → feed vào loss.

---

## 04_autograd.py — Tensor & mini autograd

### `_unbroadcast(grad, shape)` (`line 17`)
- Thu gọn gradient về đúng shape bằng cách sum theo axis đã broadcast.

### `class Tensor` (`line 33`)
- Thuộc tính: `data`, `requires_grad`, `grad`, `creators`, `creation_op`.
- Toán tử (`__add__`, `__sub__`, `__mul__`, `__matmul__`, slicing, reshape…)
  1. Tạo Tensor mới với `creators=[self, other]` và `creation_op` mô tả phép toán.
- `backward(grad=None)`
  1. Nếu grad None và tensor là scalar, dùng ones_like.
  2. Cộng dồn gradient (nếu Tensor dùng nhiều lần).
  3. Dựa vào `creation_op` để tính grad cho creators:
     - `"add"`: pass-through grad cho cả hai.
     - `"sub"`: grad và -grad.
     - `"mul"`: nhân với giá trị đối tác.
     - `"matmul"`: sử dụng công thức `grad @ other.T`…
  4. Gọi `creator.backward(...)` đệ quy.

### `softmax`, `cross_entropy_loss` (`line 306, 324`)
- Sử dụng Tensor operations; `cross_entropy_loss` flatten logits, tính log-prob, so sánh với one-hot target.

### `AutogradLayer` (`line 372`) & `AutogradMLP` (`line 395`)
- Wrap MLP nhưng thay mọi numpy array bằng Tensor để minh họa autograd custom.

---

## 05_tokenizer.py — BPE & WordPiece

### `class BPETokenizer` (`line 26`)
- `__init__(vocab_size)`: giữ `merges`, `vocab`, `inverse_vocab`.
- `get_pairs(word)`:
  1. Duyệt tuple ký tự, đếm tất cả cặp liền kề bằng `Counter`.
- `merge_pair(tokens, pair, merged)`:
  1. Loop tokens; khi gặp cặp đúng `pair` thì ghép vào `merged`.
- `train(texts, verbose=True)`:
  1. Split text → word, thêm `</w>` → tuplify.
  2. Lặp `num_merges`: đếm pair frequency (có trọng số theo word frequency).
  3. Chọn pair phổ biến nhất, tạo token mới, merge trong toàn corpus.
  4. Lưu rule vào `self.merges`.
  5. Sau khi merge đủ, build vocab (kí tự cơ bản + tokens mới) và map ID.
- `encode(text)`:
  1. Với mỗi word, khởi tạo tuple ký tự + `</w>`.
  2. Áp dụng merge rules (theo thứ tự học được) để thu token.
  3. Map token sang id bằng `self.vocab`.
- `decode(ids)`:
  1. Map id → token, nối lại thành text (bỏ `</w>`).

### `class WordPieceTokenizer` (`line 201`)
- `build_vocab(texts)`:
  1. Đếm tần suất token theo subword; score = frequency / (token_length^alpha).
  2. Lặp thêm subword mới dựa trên cặp char maximize score (khác BPE).
- `tokenize(word)`:
  1. Greedy match subword dài nhất có trong vocab, trượt từ đầu đến cuối.

### `download_shakespeare()` (`line 380`)
- Kiểm tra file, nếu chưa có thì tải; trả path.

---

## 06_attention.py — Self/Cross Attention

### `softmax(x, axis=-1)` (`line 78`)
- Trừ max để ổn định, exponentiate, chia tổng theo axis.

### `class SelfAttention` (`line 83`)
- `__init__(d_model, mask_future=True)`:
  1. Init weights `W_q`, `W_k`, `W_v` ngẫu nhiên.
  2. Lưu `mask_future` để biết có cần causal mask.
- `forward(x, mask=None)`:
  1. Tính Q = x @ W_q, K = x @ W_k, V = x @ W_v.
  2. Scores = Q @ K.T / sqrt(d_k).
  3. Nếu `mask_future`: zero-out upper triangle; nếu `mask` cung cấp, thêm vào scores.
  4. Softmax → weights, nhân với V để lấy output.

### `class MultiHeadAttention` (`line 157`)
- Chia `d_model` thành `num_heads` phần bằng nhau.
- Trong `forward`:
  1. Reshape Q/K/V thành `(batch, num_heads, seq, head_dim)`.
  2. Chạy attention từng head, concat lại.
  3. Chiếu qua `W_o`.

### `class CrossAttention` (`line 203`)
- Giống SelfAttention nhưng K/V lấy từ `context`, Q từ `x`.

### `create_causal_mask(seq_len)` (`line 258`)
- Tạo ma trận `(-inf)` phần trên tam giác để dùng với softmax.

---

## 07_gpt.py — GPT numpy

### Helpers
- `softmax`, `gelu`: phiên bản numpy.

### `class LayerNorm` (`line 39`)
- `forward(x, eps)`:
  1. Tính mean/var theo last dim.
  2. Chuẩn hóa, scale bằng `gamma`, shift bằng `beta`.

### `class FeedForward` (`line 66`)
- `forward(x)`:
  1. `x @ W1 + b1`, apply GELU.
  2. `hidden @ W2 + b2`, optional dropout.

### `class MultiHeadAttention` (`line 106`)
- Gần giống file 06 nhưng gộp trong block.
- `forward(x, mask)`:
  1. Linear projections cho q/k/v.
  2. Reshape → attention → combine.

### `class TransformerBlock` (`line 181`)
- `forward(x, mask)`:
  1. Pre-norm: `x = x + attn(LN(x))`.
  2. `x = x + ff(LN(x))`.

### `class PositionalEncoding` (`line 220`)
- Tạo bảng sin/cos, slice theo độ dài sequence.

### `class GPT` (`line 254`)
- `__init__`: embed token/pos, stack blocks, final layer norm & projection.
- `forward(input_ids, mask)`:
  1. Lookup embeddings, cộng positional.
  2. Qua từng block, LN cuối, nhân với output weight.
- `generate(start_tokens, max_new_tokens, temperature)`:
  1. Lặp: feed tokens, lấy logits cuối, sample token mới, append.

### Training helpers
- `cross_entropy_loss` (`line 357`): flatten logits -> softmax -> CE.
- `simple_train_step` (`line 376`): forward, tính loss, backward thủ công (theo numpy) rồi update weights (simplified demo).

---

## 08_training.py — Training loop PyTorch

### `cross_entropy_loss_numpy` (`line 30`)
- Giống hàm trong AI_PATH.MD: reshape `(B*T, vocab)`, softmax, log, lấy loss.

### `class TextDataset` (`line 70`)
- `__init__`: đọc text, xây char vocab, chuyển text thành list id.
- `__len__`: số sample = len(tokens) - block_size.
- `__getitem__(idx)`: trả `x` = tokens[idx:idx+block], `y` = shift một bước.

### `clip_grad_norm(grads, max_norm)` (`line 129`)
- Tính norm chung sqrt(sum||grad||^2). Nếu vượt `max_norm`: scale tất cả grad theo hệ số `max_norm / (norm + eps)`.

### `get_lr(step, warmup_steps, max_steps, max_lr, min_lr)` (`line 177`)
- Nếu step < warmup: tuyến tính từ 0 → max_lr.
- Sau đó cosine decay về `min_lr`.

### `train_with_pytorch()` (`line 234`)
- Load text, dataset, dataloader.
- Khởi tạo `GPTTorch` (import từ `ai_path/07_gpt.py` hoặc module PyTorch song song).
- Loop epochs:
  1. Forward `logits, loss = model(x, y)`.
  2. `optimizer.zero_grad()`, `loss.backward()`.
  3. Optional `clip_grad_norm`.
  4. `optimizer.step()`, log metrics.

---

## 09_kv_cache.py — Cache & Beam Search

### `softmax` (`line 37`): như trước.

### `class CachedMultiHeadAttention` (`line 55`)
- Lưu các projection matrices.
- `forward(x)`:
  1. Compute q/k/v, reshape, attention bình thường.
- `forward_with_cache(x, kv_cache, use_cache)`:
  1. Nếu cache None: khởi tạo dict `{'k': [], 'v': []}`.
  2. Append key/value mới vào cache (theo thời gian).
  3. Khi `use_cache` True: chỉ tính attention giữa query mới và toàn bộ key cached → suy luận nhanh.

### `class PagedKVCache` (`line 161`)
- Mô phỏng buffer chia page:
  1. `append(layer_id, token_index, key, value)` lưu K/V theo layer và block.
  2. `get_range(layer_id, start, end)` trả về phân đoạn K/V để attention.

### `beam_search` (`line 342`)
- Tham số: `score_fn` (model), `start_tokens`, `beam_width`, `max_length`, `eos_token`.
- Thuật toán:
  1. Khởi tạo beam với token start, score 0.
  2. Lặp cho tới `max_length`:
     - Với mỗi beam, gọi `score_fn` để lấy log-prob token tiếp.
     - Mở rộng tất cả khả năng, chọn top `beam_width`.
     - Nếu gặp `eos`, đưa vào danh sách hoàn thành.
  3. Nếu chưa đủ chuỗi hoàn thành, dùng beam hiện tại create output.

---

## 10_quantization.py — Precision tricks

### `class Quantizer` (`line 33`)
- `quantize_absmax_int8(weights)`:
  1. Lấy `abs_max`, `scale = abs_max / 127`.
  2. `round(weights / scale)` rồi clip.
- `quantize_symmetric_int8`:
  1. Giống trên nhưng cho phép scale khác `abs_max/127`.
- `quantize_asymmetric_int8`:
  1. Tính `min/max`, scale = (max-min)/255, zero_point = round(-min/scale).
  2. Lưu zero-point để dequantize chính xác hơn cho dữ liệu lệch.
- `quantize_per_channel_int8(weights, axis)`:
  1. Loop từng channel (row hoặc column), tính `abs_max`.
  2. Lưu scale riêng cho từng channel, quantize channel đó.
- `quantize_int4(weights)`:
  1. scale = abs_max / 7, round/clip vào [-8,7], lưu `np.int8` (pack 2 weight/byte sau đó).

### `measure_quantization_error` (`line 2218`)
- Dequantize lại bằng scale → tính `mse`, `mae`, `max_error`, `compression_ratio`.

### GPTQ / AWQ (`line 229+`)
- `gptq_quantize`: chia tensor thành nhóm (group_size), giải hệ least-squares để giảm sai số.
- `gptq_dequantize`: reconstruct lại weights từ quantized + scales.
- `awq_quantize`: điều chỉnh scale dựa trên activation statistics.
- `awq_dequantize`: khôi phục weights theo group/channel scale.

### `measure_inference_speed` (`line 460`)
- Benchmark `x @ weights_fp32` vs `x @ (weights_int8 * scale)` trong vòng lặp, đo thời gian.

### `quantize_gpt_model` (`line 508`)
- Đi qua dict tham số GPT, áp dụng phương pháp (absmax/gptq/awq) cho từng matrix, lưu scale/meta.

---

## 11_lora.py — LoRA & QLoRA

### `class LoRALayer` (`line 34`)
- Thuộc tính: `W` frozen, matrices `A (d_in, r)`, `B (r, d_out)`, scale `alpha/r`, dropout.
- `forward(x, training)`:
  1. Cache input.
  2. Tính `original_output = x @ W`.
  3. `lora_hidden = x @ A`; nếu training và dropout > 0 → áp dụng mask.
  4. `lora_output = lora_hidden @ B`; lưu cache.
  5. Trả `original_output + scaling * lora_output`.
- `backward(grad_output)`:
  1. Gradient LoRA = `grad_output * scaling`.
  2. `grad_B = lora_hidden^T @ grad_lora`.
  3. `grad_A = input^T @ (grad_lora @ B^T)`.
  4. Không update W (frozen).

### `class LoRAAttention` (`line 223`)
- Wrap quanh attention base: thay `W_q`, `W_v` thành `LoRALayer`, reuse forward/backward.

### `class QLoRALayer` (`line 331`)
- Giống LoRA nhưng weights base được quantize 4-bit (double quantization) và LoRA branch vẫn FP16.
- Giữ `quantized_W`, `scales`, `zero_points`, dequantize on-the-fly trong forward.

### Training utilities (`line 464+`)
- `create_classification_task`: tạo dataset synthetic.
- `train_lora_classifier`: freeze base W, chỉ train LoRA, so sánh accuracy.
- `train_full_finetune`: train toàn bộ W để đối chiếu.

---

## 12_advanced_architectures.py — RoPE, Flash, GQA, MoE

### Rotary (`lines 111-226`)
- `precompute_freqs_cis(dim, max_seq_len, base)`:
  1. Sinh tần số cho từng chiều, tạo cos/sin ma trận.
  2. Dùng base ~ 10000.
- `apply_rope(x, freqs_cos, freqs_sin)`:
  1. Chia channel thành cặp (real, imag), áp dụng phép quay (cos*x_even + sin*x_odd, ...).
- `rope_attention(Q, K, V, ...)`: áp dụng RoPE lên Q/K rồi tính attention.

### Standard vs Flash (`lines 339-430`)
- `standard_attention`: tính softmax(QK^T / sqrt(d)) @ V theo cách thông thường (O(n^2) memory).
- `flash_attention(Q, K, V, block_size)`:
  1. Duyệt Q theo block.
  2. Với mỗi block, tính scores từng phần, cập nhật `m` (max logits) và `l` (normalizer) online.
  3. Trả output đã chuẩn hóa mà không phải lưu toàn bộ matrix.

### GQA (`lines 547-716`)
- `GroupedQueryAttention`: cho phép `num_q_heads` lớn, `num_kv_heads` nhỏ.
  1. Expand K/V để match Q.
  2. Tính attention như thường.
  3. Trả output + benchmark qua `compare_attention_variants`.

### MoE (`lines 816-885`)
- `ExpertFFN`: mỗi expert = FFN riêng (Linear -> activation -> Linear).
- `MoELayer`:
  1. Router (softmax) chọn expert theo xác suất, chọn top-k.
  2. Chạy inputs qua expert tương ứng.
  3. Load-balancing loss đảm bảo expert nào cũng được sử dụng.

---

## 13_alignment.py — Reward, DPO, PPO, Constitutional AI

### Math helpers (`lines 42-64`)
- `sigmoid`, `softmax`, `log_softmax`.

### `class RewardModel` (`line 112`)
- `__init__`: embedding -> mean pool -> MLP ra scalar.
- `forward(input_ids)`:
  1. Lookup embedding, optional positional bias.
  2. Mean pool theo seq dim.
  3. Qua MLP 2 tầng (activation + linear) thành score.
- `score_pair(chosen, rejected)`: trả logit chênh lệch.

### `dpo_loss` & `dpo_gradient` (`lines 434, 512`)
- Tính loss theo công thức DPO:
  1. Lấy log-prob chosen/rejected từ policy và reference.
  2. Loss = `-log(sigmoid(beta * (Δπ - Δref)))`.
  3. Gradient: đạo hàm w.r.t logits (cho training custom).

### `class SimplePolicyForDPO` (`line 549`)
- Cấu trúc nhỏ: embedding -> transformer block -> linear head.
- `forward` trả logits; `log_probs` lấy log softmax theo chuỗi.

### `class PPOTrainer` (`line 698`)
- Chứa policy + value model.
- `collect_rollouts`: chạy policy sinh dữ liệu, tính reward/advantage.
- `update`: tính loss PPO (policy clip, value loss, entropy bonus), backward và step optimizer.

### `class ConstitutionalAI` (`line 1029`)
- `generate_response(prompt)`: model -> reply.
- `critique(response, principle)`: dùng “hiến pháp” để tự đánh giá.
- `revise(response, critique)`: sửa lại câu trả lời dựa trên critique, mô phỏng self-refine.

### `create_preference_data` (`line 1347`)
- Sinh synthetic pair (chosen vs rejected) bằng cách thêm noise, dùng làm dữ liệu cho DPO/PPO.

---

## Cách sử dụng tài liệu

1. Khi muốn hiểu hàm nào, nhảy tới file tương ứng ở trên rồi đọc từng bullet.
2. Kết hợp với editor: `:set number`, nhảy tới line ghi trong ngoặc để so sánh.
3. Nếu bạn bổ sung hàm mới trong `ai_path/`, hãy mở file này và thêm section tương tự (ghi line, mô tả từng bước) để giữ tài liệu cập nhật.
