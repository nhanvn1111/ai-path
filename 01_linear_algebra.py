# File: 01_linear_algebra.py
# Matrix Operations tu Scratch - Week 1-2
# WALKTHROUGH QUICK NOTES:
# - Matrix.__init__: lưu raw data + kích thước để các phép nhân kiểm tra hợp lệ.
# - Matrix.dot: ba vòng for nhân ma trận classic, trả Matrix mới.
# - Matrix.transpose/add/subtract/scalar_multiply/element_multiply/apply: thao tác từng phần tử, dùng cho forward/backward.
# - random_matrix: helper khởi tạo weights thủ công.
# - __main__ section: chạy test minh họa forward pass, subtract, error handling.
#
# TAI SAO HOC CAI NAY?
# Moi thu trong deep learning deu la phep toan ma tran:
# - Neural network forward pass: output = input @ weights
# - Backpropagation: gradient = grad_output @ weights.T
# - Attention: scores = Q @ K.T
# Hieu ma tran = hieu duoc nen tang cua AI.
#
# TAI SAO KHONG DUNG NUMPY?
# De hieu cach no hoat dong ben trong.
# Numpy lam nhanh hon 100x nhung "giau" logic di.
# Code bang tay 1 lan -> hieu suot doi.

import random
import time


class Matrix:
    """
    Ma tran la nen tang cua moi thu trong deep learning.
    Neural network = chuoi cac phep nhan ma tran + activation functions
    """

    def __init__(self, data):
        """
        data: list 2D, vd [[1,2],[3,4]] = ma tran 2x2
              Moi list con = 1 row, len(list con) = so cols
              Trong neural network: data co the la weights, input, gradient, v.v.
        """
        self.data = data
        self.rows = len(data)       # so hang
        self.cols = len(data[0]) if data else 0  # so cot

    def __repr__(self):
        return f"Matrix({self.data})"

    def dot(self, other):
        """
        Matrix multiplication: C = A @ B

        other: Matrix khac de nhan voi self
               Dieu kien: self.cols == other.rows (so cot A = so hang B)
               Ket qua: ma tran (self.rows x other.cols)

        Day la phep toan QUAN TRONG NHAT trong neural network.
        - Forward pass: output = input @ weights  (input: 1x3, weights: 3x2 -> output: 1x2)
        - Moi layer = 1 phep nhan ma tran

        Complexity: O(n^3) - rat cham, vi vay can GPU
        """
        if self.cols != other.rows:
            raise ValueError(
                f"Cannot multiply {self.rows}x{self.cols} with {other.rows}x{other.cols}"
            )

        result = [[0.0] * other.cols for _ in range(self.rows)]

        for i in range(self.rows):
            for j in range(other.cols):
                for k in range(self.cols):
                    result[i][j] += self.data[i][k] * other.data[k][j]

        return Matrix(result)

    def transpose(self):
        """
        Transpose: doi rows thanh cols

        Dung trong backpropagation:
        - dL/dW = input.T @ grad_output
        - dL/dinput = grad_output @ W.T
        """
        result = [[self.data[j][i] for j in range(self.rows)] for i in range(self.cols)]
        return Matrix(result)

    def add(self, other):
        """
        Element-wise addition: A + B (cong tung phan tu tuong ung)
        other: Matrix cung kich thuoc voi self
        Dung trong: cong bias vao output (output + bias), residual connection (x + attn_output)
        """
        if self.rows != other.rows or self.cols != other.cols:
            raise ValueError(
                f"Cannot add {self.rows}x{self.cols} with {other.rows}x{other.cols}"
            )
        result = [
            [self.data[i][j] + other.data[i][j] for j in range(self.cols)]
            for i in range(self.rows)
        ]
        return Matrix(result)

    def subtract(self, other):
        """
        Element-wise subtraction: A - B

        Dung nhieu trong:
        - Tinh error/loss: error = prediction - target
        - Update weights: W_new = W_old - learning_rate * gradient
        """
        if self.rows != other.rows or self.cols != other.cols:
            raise ValueError(
                f"Cannot subtract {self.rows}x{self.cols} with {other.rows}x{other.cols}"
            )
        result = [
            [self.data[i][j] - other.data[i][j] for j in range(self.cols)]
            for i in range(self.rows)
        ]
        return Matrix(result)

    def scalar_multiply(self, scalar):
        """
        Nhan moi element voi 1 so.

        scalar: so thuc (float/int), vd: learning_rate = 0.01
        Dung trong: learning_rate * gradient (scale gradient truoc khi update)
        """
        result = [
            [self.data[i][j] * scalar for j in range(self.cols)]
            for i in range(self.rows)
        ]
        return Matrix(result)

    def element_multiply(self, other):
        """
        Hadamard product - nhan tung phan tu tuong ung.
        Khac voi dot(): dot la nhan ma tran (O(n^3)), element_multiply la tung cap (O(n^2)).
        Dung trong: gradient cua ReLU (grad * mask), dropout (output * drop_mask).
        """
        result = [
            [self.data[i][j] * other.data[i][j] for j in range(self.cols)]
            for i in range(self.rows)
        ]
        return Matrix(result)

    def apply(self, func):
        """
        Apply function len moi element cua ma tran.

        func: ham nhan 1 so, tra ve 1 so. Vd: lambda x: max(0, x)

        Day la cach activation functions hoat dong:
        - ReLU: matrix.apply(lambda x: max(0, x))   -> bo gia tri am, giu gia tri duong
        - Sigmoid: matrix.apply(lambda x: 1/(1+exp(-x))) -> nen ve [0, 1]
        - Tanh: matrix.apply(lambda x: tanh(x))     -> nen ve [-1, 1]

        Moi neuron trong network se apply activation function
        len output cua phep nhan ma tran.
        """
        result = [
            [func(self.data[i][j]) for j in range(self.cols)]
            for i in range(self.rows)
        ]
        return Matrix(result)


# ============ HELPER ============
def random_matrix(rows, cols, low=-1.0, high=1.0):
    """
    Tao ma tran ngau nhien - dung de khoi tao weights

    rows: so hang (vd: so neurons input)
    cols: so cot (vd: so neurons output)
    low, high: khoang gia tri random. Default [-1, 1]
               Trong thuc te dung He/Xavier init thay vi random deu
    """
    data = [
        [random.uniform(low, high) for _ in range(cols)]
        for _ in range(rows)
    ]
    return Matrix(data)


# ============ TEST ============
if __name__ == "__main__":
    import math

    print("=" * 60)
    print("TEST CO BAN - Matrix operations")
    print("=" * 60)

    # Test matrix multiplication
    A = Matrix([[1, 2], [3, 4]])
    B = Matrix([[5, 6], [7, 8]])
    C = A.dot(B)
    print(f"A @ B = {C.data}")  # [[19, 22], [43, 50]]
    assert C.data == [[19, 22], [43, 50]], "dot() FAILED"

    # Test transpose
    print(f"A.T = {A.transpose().data}")  # [[1, 3], [2, 4]]
    assert A.transpose().data == [[1, 3], [2, 4]], "transpose() FAILED"

    # Neural network simulation
    X = Matrix([[1.0, 2.0, 3.0]])  # 1x3 input
    W = Matrix([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])  # 3x2 weights
    output = X.dot(W)
    print(f"X @ W = {output.data}")  # [[2.2, 2.8]]

    print("\n" + "=" * 60)
    print("BAI TAP 1: subtract()")
    print("=" * 60)

    A = Matrix([[10, 20], [30, 40]])
    B = Matrix([[1, 2], [3, 4]])
    result = A.subtract(B)
    print(f"A - B = {result.data}")  # [[9, 18], [27, 36]]
    assert result.data == [[9, 18], [27, 36]], "subtract() FAILED"

    # Ung dung: tinh error trong neural network
    prediction = Matrix([[0.8, 0.1, 0.1]])
    target = Matrix([[1.0, 0.0, 0.0]])
    error = prediction.subtract(target)
    print(f"Error (pred - target) = {error.data}")  # [[-0.2, 0.1, 0.1]]

    # Test dimension mismatch
    try:
        Matrix([[1, 2]]).subtract(Matrix([[1], [2]]))
        assert False, "Should have raised ValueError"
    except ValueError as e:
        print(f"Dimension check OK: {e}")

    print("\n" + "=" * 60)
    print("BAI TAP 2: apply(func)")
    print("=" * 60)

    M = Matrix([[-2, -1, 0], [1, 2, 3]])

    # ReLU activation
    relu_result = M.apply(lambda x: max(0, x))
    print(f"ReLU({M.data}) = {relu_result.data}")
    assert relu_result.data == [[0, 0, 0], [1, 2, 3]], "ReLU FAILED"

    # Sigmoid activation
    sigmoid = lambda x: 1.0 / (1.0 + math.exp(-x))
    sig_result = M.apply(sigmoid)
    print(f"Sigmoid({M.data}) = {sig_result.data}")
    # Verify sigmoid output range [0, 1]
    for row in sig_result.data:
        for val in row:
            assert 0 <= val <= 1, f"Sigmoid out of range: {val}"
    print("  -> All sigmoid values in [0, 1]: OK")

    # Square each element
    sq_result = M.apply(lambda x: x ** 2)
    print(f"Square({M.data}) = {sq_result.data}")
    assert sq_result.data == [[4, 1, 0], [1, 4, 9]], "Square FAILED"

    # Tanh activation
    tanh_result = M.apply(math.tanh)
    print(f"Tanh({M.data}) = {tanh_result.data}")
    for row in tanh_result.data:
        for val in row:
            assert -1 <= val <= 1, f"Tanh out of range: {val}"
    print("  -> All tanh values in [-1, 1]: OK")

    print("\n" + "=" * 60)
    print("BAI TAP 3: Test voi ma tran 100x100")
    print("=" * 60)

    random.seed(42)  # Reproducible results

    # Tao 2 ma tran 100x100
    big_A = random_matrix(100, 100)
    big_B = random_matrix(100, 100)

    # Test dot product 100x100
    start = time.time()
    big_C = big_A.dot(big_B)
    dot_time = time.time() - start
    print(f"100x100 dot product: {dot_time:.3f}s")
    assert big_C.rows == 100 and big_C.cols == 100
    print(f"  Result shape: {big_C.rows}x{big_C.cols} OK")

    # Test transpose 100x100
    start = time.time()
    big_T = big_A.transpose()
    trans_time = time.time() - start
    print(f"100x100 transpose: {trans_time:.4f}s")
    assert big_T.rows == 100 and big_T.cols == 100
    # Verify: (A^T)^T == A
    big_TT = big_T.transpose()
    for i in range(100):
        for j in range(100):
            assert abs(big_TT.data[i][j] - big_A.data[i][j]) < 1e-10
    print("  (A^T)^T == A: OK")

    # Test subtract 100x100
    start = time.time()
    big_sub = big_A.subtract(big_B)
    sub_time = time.time() - start
    print(f"100x100 subtract: {sub_time:.4f}s")
    # Verify: A - B + B == A
    big_check = big_sub.add(big_B)
    for i in range(100):
        for j in range(100):
            assert abs(big_check.data[i][j] - big_A.data[i][j]) < 1e-10
    print("  A - B + B == A: OK")

    # Test apply 100x100 voi ReLU
    start = time.time()
    big_relu = big_A.apply(lambda x: max(0, x))
    apply_time = time.time() - start
    print(f"100x100 apply(ReLU): {apply_time:.4f}s")
    # Verify: tat ca values >= 0
    for row in big_relu.data:
        for val in row:
            assert val >= 0
    print("  All ReLU values >= 0: OK")

    # Test A - A == zero matrix
    zero = big_A.subtract(big_A)
    for row in zero.data:
        for val in row:
            assert abs(val) < 1e-10
    print("  A - A == 0: OK")

    # Mini neural network forward pass voi 100-dim
    print("\n--- Mini Neural Network (100-dim) ---")
    X_big = random_matrix(1, 100)         # 1 sample, 100 features
    W1 = random_matrix(100, 64)           # Layer 1: 100 -> 64
    W2 = random_matrix(64, 10)            # Layer 2: 64 -> 10

    start = time.time()
    hidden = X_big.dot(W1)                           # Linear layer 1
    hidden_activated = hidden.apply(lambda x: max(0, x))  # ReLU
    output = hidden_activated.dot(W2)                # Linear layer 2
    nn_time = time.time() - start

    print(f"  Input:  1x100")
    print(f"  Hidden: 1x64 (after ReLU)")
    print(f"  Output: 1x{output.cols}")
    print(f"  Forward pass time: {nn_time:.4f}s")
    print(f"  Output values: {[round(v, 4) for v in output.data[0]]}")

    print("\n" + "=" * 60)
    print("TAT CA TESTS PASSED!")
    print("=" * 60)


# ============ CHECKLIST ============
# Week 1-2 (Bai 01):
# [x] Code duoc matrix multiply khong dung numpy
#     -> Matrix.dot(): 3 vong for long nhau (i, j, k)
# [x] Giai thich duoc tai sao matrix multiply la O(n^3)
#     -> Vi co 3 vong for long nhau: i(rows) x j(cols) x k(inner_dim)
#        Moi phan tu result[i][j] can duyet k lan nhan + cong
#        Ma tran 100x100: 100 * 100 * 100 = 1,000,000 phep tinh
#        Ma tran 1000x1000: 1 ty phep tinh -> can GPU de song song hoa
