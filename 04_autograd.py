# File: 04_autograd.py
# Autograd System tu Scratch - Week 3-4
#
# TAI SAO HOC CAI NAY?
# File 03 (neural network) phai tu code backward() cho TUNG LAYER.
# Rat met va de sai. PyTorch giai quyet bang AUTOGRAD:
# - Moi phep tinh tao 1 node trong "computational graph"
# - Khi goi backward(), gradient duoc tinh TU DONG qua chain rule
# - Khong can viet backward() thu cong nua!
#
# Day chinh la cach PyTorch hoat dong ben trong.
# Hieu autograd = hieu tai sao loss.backward() chay duoc.

import numpy as np


def _unbroadcast(grad, shape):
    """
    Khi numpy broadcast (vd: (1,16) + (32,16) -> (32,16)),
    gradient can duoc sum lai ve shape goc.
    """
    if grad.shape == shape:
        return grad
    # Sum over axes that were broadcasted
    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)
    for i, (gs, s) in enumerate(zip(grad.shape, shape)):
        if s == 1 and gs != 1:
            grad = grad.sum(axis=i, keepdims=True)
    return grad


class Tensor:
    """
    Tensor voi automatic differentiation

    Moi operation tao ra node moi trong computational graph.
    Khi goi backward(), gradient duoc tinh tu dong qua chain rule.
    """

    def __init__(self, data, requires_grad=False, creators=None, creation_op=None):
        """
        data:          so lieu, co the la list, numpy array, hoac scalar
                       Vd: [1.0, 2.0] hoac [[1,2],[3,4]] hoac 5.0
        requires_grad: True = can tinh gradient cho tensor nay (vd: weights, bias)
                       False = khong can gradient (vd: input data, labels)
                       Chi tensors voi requires_grad=True moi duoc update khi train
        creators:      list cac Tensor cha da tao ra tensor nay (dung cho backward)
                       Vd: c = a + b -> c.creators = [a, b]
                       None = tensor goc (leaf), khong co cha
        creation_op:   phep tinh da tao ra tensor nay ("add", "mul", "matmul", "relu", ...)
                       backward() dua vao creation_op de biet ap dung chain rule nhu the nao
                       None = tensor goc (leaf)
        """
        self.data = np.array(data, dtype=np.float64)
        self.requires_grad = requires_grad
        self.grad = None

        # Computational graph - luu lai "lich su" de backward() biet duong di nguoc
        self.creators = creators        # Tensor cha
        self.creation_op = creation_op  # Phep tinh da tao ra tensor nay

    def __repr__(self):
        return f"Tensor({self.data}, requires_grad={self.requires_grad})"

    @property
    def shape(self):
        return self.data.shape

    def backward(self, grad=None):
        """
        Backpropagation qua computational graph

        Chain rule: dL/dx = dL/dy * dy/dx
        Vi du: y = x^2, L = sum(y)
        -> dL/dy = 1, dy/dx = 2x
        -> dL/dx = 1 * 2x = 2x

        Gradient "chay nguoc" tu loss ve tung parameter,
        nho vao creation_op biet phep tinh nao da tao ra tensor nay.
        """
        if grad is None:
            grad = np.ones_like(self.data)

        # Accumulate gradient
        if self.grad is None:
            self.grad = grad
        else:
            self.grad = self.grad + grad

        # Propagate gradient ve creators
        if self.creators is not None:
            if self.creation_op == "add":
                # d(a+b)/da = 1, d(a+b)/db = 1
                # Handle broadcasting: sum over broadcasted dims
                self.creators[0].backward(_unbroadcast(grad, self.creators[0].shape))
                self.creators[1].backward(_unbroadcast(grad, self.creators[1].shape))

            elif self.creation_op == "sub":
                # d(a-b)/da = 1, d(a-b)/db = -1
                self.creators[0].backward(_unbroadcast(grad, self.creators[0].shape))
                self.creators[1].backward(_unbroadcast(-grad, self.creators[1].shape))

            elif self.creation_op == "mul":
                # d(a*b)/da = b, d(a*b)/db = a
                self.creators[0].backward(grad * self.creators[1].data)
                self.creators[1].backward(grad * self.creators[0].data)

            elif self.creation_op == "matmul":
                # d(A@B)/dA = grad @ B.T
                # d(A@B)/dB = A.T @ grad
                self.creators[0].backward(grad @ self.creators[1].data.T)
                self.creators[1].backward(self.creators[0].data.T @ grad)

            elif self.creation_op == "relu":
                self.creators[0].backward(grad * (self.creators[0].data > 0))

            elif self.creation_op == "sigmoid":
                s = self.data
                self.creators[0].backward(grad * s * (1 - s))

            elif self.creation_op == "tanh":
                # d(tanh(x))/dx = 1 - tanh(x)^2
                self.creators[0].backward(grad * (1 - self.data ** 2))

            elif self.creation_op == "exp":
                # d(exp(x))/dx = exp(x)
                self.creators[0].backward(grad * self.data)

            elif self.creation_op == "log":
                # d(log(x))/dx = 1/x
                self.creators[0].backward(grad / self.creators[0].data)

            elif self.creation_op == "sum":
                self.creators[0].backward(grad * np.ones_like(self.creators[0].data))

            elif self.creation_op == "mean":
                n = self.creators[0].data.size
                self.creators[0].backward(grad * np.ones_like(self.creators[0].data) / n)

            elif self.creation_op == "pow":
                n = self.creators[1]  # power (not a tensor)
                self.creators[0].backward(grad * n * np.power(self.creators[0].data, n - 1))

            elif self.creation_op == "transpose":
                self.creators[0].backward(grad.T)

            elif self.creation_op == "reshape":
                self.creators[0].backward(grad.reshape(self.creators[0].shape))

            elif self.creation_op == "neg":
                self.creators[0].backward(-grad)

            elif self.creation_op == "sum_axis":
                # Broadcast grad back to original shape
                axis = self.creators[1]  # axis (not a tensor)
                self.creators[0].backward(np.expand_dims(grad, axis=axis) * np.ones_like(self.creators[0].data))

    def zero_grad(self):
        self.grad = None

    # ============ Basic Operations ============

    def __add__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other)
        return Tensor(
            self.data + other.data,
            requires_grad=self.requires_grad or other.requires_grad,
            creators=[self, other], creation_op="add"
        )

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other)
        return Tensor(
            self.data - other.data,
            requires_grad=self.requires_grad or other.requires_grad,
            creators=[self, other], creation_op="sub"
        )

    def __neg__(self):
        return Tensor(
            -self.data,
            requires_grad=self.requires_grad,
            creators=[self], creation_op="neg"
        )

    def __mul__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other)
        return Tensor(
            self.data * other.data,
            requires_grad=self.requires_grad or other.requires_grad,
            creators=[self, other], creation_op="mul"
        )

    def __rmul__(self, other):
        return self.__mul__(other)

    def __pow__(self, power):
        return Tensor(
            np.power(self.data, power),
            requires_grad=self.requires_grad,
            creators=[self, power], creation_op="pow"
        )

    def matmul(self, other):
        return Tensor(
            self.data @ other.data,
            requires_grad=self.requires_grad or other.requires_grad,
            creators=[self, other], creation_op="matmul"
        )

    def __matmul__(self, other):
        return self.matmul(other)

    # ============ Activation Functions ============

    def relu(self):
        return Tensor(
            np.maximum(0, self.data),
            requires_grad=self.requires_grad,
            creators=[self], creation_op="relu"
        )

    def sigmoid(self):
        return Tensor(
            1 / (1 + np.exp(-np.clip(self.data, -500, 500))),
            requires_grad=self.requires_grad,
            creators=[self], creation_op="sigmoid"
        )

    def tanh(self):
        """
        Tanh activation: tanh(x) = (e^x - e^-x) / (e^x + e^-x)
        Gradient: d(tanh(x))/dx = 1 - tanh(x)^2
        """
        return Tensor(
            np.tanh(self.data),
            requires_grad=self.requires_grad,
            creators=[self], creation_op="tanh"
        )

    # ============ Bai tap 1: exp, log ============

    def exp(self):
        """
        Exponential: e^x
        Gradient: d(e^x)/dx = e^x
        Dung trong softmax: softmax(x) = exp(x) / sum(exp(x))
        """
        return Tensor(
            np.exp(np.clip(self.data, -500, 500)),
            requires_grad=self.requires_grad,
            creators=[self], creation_op="exp"
        )

    def log(self):
        """
        Natural log: ln(x)
        Gradient: d(ln(x))/dx = 1/x
        Dung trong cross-entropy loss: -sum(y * log(p))
        """
        return Tensor(
            np.log(np.clip(self.data, 1e-15, None)),
            requires_grad=self.requires_grad,
            creators=[self], creation_op="log"
        )

    # ============ Reduction Operations ============

    def sum(self, axis=None):
        if axis is not None:
            return Tensor(
                np.sum(self.data, axis=axis),
                requires_grad=self.requires_grad,
                creators=[self, axis], creation_op="sum_axis"
            )
        return Tensor(
            np.sum(self.data),
            requires_grad=self.requires_grad,
            creators=[self], creation_op="sum"
        )

    def mean(self):
        return Tensor(
            np.mean(self.data),
            requires_grad=self.requires_grad,
            creators=[self], creation_op="mean"
        )

    def T(self):
        return Tensor(
            self.data.T,
            requires_grad=self.requires_grad,
            creators=[self], creation_op="transpose"
        )


# ============ Bai tap 2: Softmax + Cross Entropy ============

def softmax(x):
    """
    Softmax: chuyen raw scores (logits) thanh probabilities

    softmax(x_i) = exp(x_i) / sum(exp(x_j))

    Trick: tru max de tranh overflow (exp cua so lon -> inf)
    """
    shifted = x - Tensor(np.max(x.data, axis=-1, keepdims=True))
    exp_x = shifted.exp()
    sum_exp = Tensor(np.sum(exp_x.data, axis=-1, keepdims=True))
    return Tensor(
        exp_x.data / sum_exp.data,
        requires_grad=x.requires_grad,
        creators=[x], creation_op="softmax_ce_fused"
    )


def cross_entropy_loss(logits, targets):
    """
    Cross-entropy loss voi softmax

    L = -mean(sum(y_true * log(softmax(logits))))

    Dung fused softmax + cross-entropy de:
    1. Tranh numerical instability
    2. Gradient don gian: dL/dlogits = softmax(logits) - y_true

    targets: one-hot encoded [batch, num_classes]
    """
    # Stable softmax
    shifted = logits.data - np.max(logits.data, axis=-1, keepdims=True)
    exp_x = np.exp(shifted)
    probs = exp_x / np.sum(exp_x, axis=-1, keepdims=True)

    # Cross-entropy loss
    eps = 1e-15
    log_probs = np.log(np.clip(probs, eps, 1.0))
    loss_val = -np.mean(np.sum(targets.data * log_probs, axis=1))

    loss = Tensor(loss_val, requires_grad=True)

    # Store for backward: gradient = (softmax - target) / batch_size
    loss._ce_grad = (probs - targets.data) / targets.data.shape[0]
    loss._ce_logits = logits
    loss.creators = [logits]
    loss.creation_op = "cross_entropy"

    # Patch backward
    original_backward = loss.backward

    def ce_backward(grad=None):
        if grad is None:
            grad = np.ones_like(loss.data)
        if loss.grad is None:
            loss.grad = grad
        else:
            loss.grad = loss.grad + grad
        logits.backward(loss._ce_grad * grad)

    loss.backward = ce_backward
    return loss


# ============ Bai tap 3: Neural Network dung Tensor class ============

class AutogradLayer:
    """Dense layer su dung Tensor autograd"""

    def __init__(self, in_features, out_features):
        """
        in_features:  so features dau vao (= so cot cua input)
        out_features: so features dau ra (= so neurons cua layer nay)

        Weights shape: (in_features, out_features)
        Bias shape:    (1, out_features)
        Ca 2 deu co requires_grad=True -> autograd tu dong tinh gradient
        """
        scale = np.sqrt(2.0 / in_features)
        self.W = Tensor(np.random.randn(in_features, out_features) * scale, requires_grad=True)
        self.b = Tensor(np.zeros((1, out_features)), requires_grad=True)

    def forward(self, x):
        return x @ self.W + self.b

    def params(self):
        return [self.W, self.b]


class AutogradMLP:
    """
    MLP xay bang Tensor autograd

    Khac voi 03_neural_network.py (tinh gradient thu cong),
    o day gradient duoc tinh TU DONG qua computational graph.
    Day chinh la cach PyTorch hoat dong!
    """

    def __init__(self, layer_sizes, activation='relu'):
        """
        layer_sizes: list so neurons moi layer, vd [784, 128, 10]
                     Giong NeuralNetwork trong file 03, nhung backward tu dong
        activation:  'relu', 'tanh', 'sigmoid' cho hidden layers
                     Layer cuoi KHONG co activation (output raw logits cho cross_entropy)
        """
        self.layers = []
        self.activation = activation
        for i in range(len(layer_sizes) - 1):
            self.layers.append(AutogradLayer(layer_sizes[i], layer_sizes[i + 1]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer.forward(x)
            # Apply activation cho tat ca layers tru layer cuoi
            if i < len(self.layers) - 1:
                if self.activation == 'relu':
                    x = x.relu()
                elif self.activation == 'tanh':
                    x = x.tanh()
                elif self.activation == 'sigmoid':
                    x = x.sigmoid()
        return x

    def params(self):
        all_params = []
        for layer in self.layers:
            all_params.extend(layer.params())
        return all_params

    def zero_grad(self):
        for p in self.params():
            p.zero_grad()

    def train(self, X_data, y_data, epochs=100, lr=0.01, batch_size=32, verbose=True):
        """
        X_data:     numpy array input, shape (n_samples, n_features)
        y_data:     numpy array labels (one-hot), shape (n_samples, n_classes)
        epochs:     so lan duyet toan bo dataset
        lr:         learning rate cho SGD
        batch_size: so samples moi batch. Nho = nhieu noise nhung nhanh, lon = on dinh nhung cham
        verbose:    True = in loss moi 10 epochs
        """
        n_samples = X_data.shape[0]
        history = []

        for epoch in range(epochs):
            indices = np.random.permutation(n_samples)
            epoch_loss = 0
            n_batches = 0

            for i in range(0, n_samples, batch_size):
                batch_idx = indices[i:i + batch_size]
                x_batch = Tensor(X_data[batch_idx], requires_grad=True)
                y_batch = Tensor(y_data[batch_idx])

                # Forward
                logits = self.forward(x_batch)

                # Loss
                loss = cross_entropy_loss(logits, y_batch)
                epoch_loss += loss.data
                n_batches += 1

                # Zero grad
                self.zero_grad()

                # Backward - gradients tinh TU DONG!
                loss.backward()

                # Update (SGD)
                for p in self.params():
                    if p.grad is not None:
                        p.data -= lr * p.grad

            avg_loss = epoch_loss / n_batches
            history.append(avg_loss)

            if verbose and epoch % 10 == 0:
                preds = np.argmax(self.forward(Tensor(X_data)).data, axis=1)
                labels = np.argmax(y_data, axis=1)
                acc = np.mean(preds == labels)
                print(f"  Epoch {epoch:3d}: loss={avg_loss:.4f}, acc={acc:.4f}")

        return history

    def predict(self, X_data):
        return np.argmax(self.forward(Tensor(X_data)).data, axis=1)


# ============ TEST ============
if __name__ == "__main__":
    np.random.seed(42)

    print("=" * 60)
    print("TEST CO BAN - Autograd")
    print("=" * 60)

    # Test 1: Simple gradient
    print("\n--- Test 1: d(sum(x^2))/dx ---")
    x = Tensor([2.0, 3.0], requires_grad=True)
    y = x ** 2
    z = y.sum()
    z.backward()
    print(f"  x = {x.data}")
    print(f"  d(sum(x^2))/dx = {x.grad}")  # [4.0, 6.0]
    assert np.allclose(x.grad, [4.0, 6.0]), "Test 1 FAILED"
    print("  OK")

    # Test 2: Matmul gradient
    print("\n--- Test 2: Matmul gradient ---")
    W = Tensor([[1, 2], [3, 4], [5, 6]], requires_grad=True)
    x = Tensor([[1, 2, 3]], requires_grad=True)
    y = x @ W
    loss = y.sum()
    loss.backward()
    print(f"  W.grad =\n{W.grad}")
    print(f"  x.grad = {x.grad}")
    assert np.allclose(x.grad, [[3, 7, 11]]), "Matmul x.grad FAILED"
    print("  OK")

    # Test 3: MLP forward + backward
    print("\n--- Test 3: MLP forward/backward ---")
    x = Tensor([[1, 2]], requires_grad=True)
    W1 = Tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], requires_grad=True)
    W2 = Tensor([[0.7], [0.8], [0.9]], requires_grad=True)
    h = (x @ W1).relu()
    out = h @ W2
    loss = out.sum()
    loss.backward()
    print(f"  Output: {out.data}")
    print(f"  W1.grad =\n{W1.grad}")
    print(f"  W2.grad =\n{W2.grad}")
    assert W1.grad is not None and W2.grad is not None, "MLP grad FAILED"
    print("  OK")

    # ============ BAI TAP 1: exp, log, tanh ============
    print("\n" + "=" * 60)
    print("BAI TAP 1: exp, log, tanh")
    print("=" * 60)

    # Test exp
    print("\n--- exp ---")
    x = Tensor([0.0, 1.0, 2.0], requires_grad=True)
    y = x.exp().sum()
    y.backward()
    expected = np.exp([0.0, 1.0, 2.0])
    print(f"  d(sum(exp(x)))/dx = {x.grad}")
    print(f"  Expected:           {expected}")
    assert np.allclose(x.grad, expected, atol=1e-6), "exp grad FAILED"
    print("  OK")

    # Test log
    print("\n--- log ---")
    x = Tensor([1.0, 2.0, 3.0], requires_grad=True)
    y = x.log().sum()
    y.backward()
    expected = 1.0 / np.array([1.0, 2.0, 3.0])
    print(f"  d(sum(log(x)))/dx = {x.grad}")
    print(f"  Expected:           {expected}")
    assert np.allclose(x.grad, expected, atol=1e-6), "log grad FAILED"
    print("  OK")

    # Test tanh
    print("\n--- tanh ---")
    x = Tensor([-1.0, 0.0, 1.0], requires_grad=True)
    y = x.tanh().sum()
    y.backward()
    expected = 1 - np.tanh([-1.0, 0.0, 1.0]) ** 2
    print(f"  d(sum(tanh(x)))/dx = {x.grad}")
    print(f"  Expected:            {expected}")
    assert np.allclose(x.grad, expected, atol=1e-6), "tanh grad FAILED"
    print("  OK")

    # ============ BAI TAP 2: Softmax + Cross Entropy ============
    print("\n" + "=" * 60)
    print("BAI TAP 2: Softmax + Cross Entropy Loss")
    print("=" * 60)

    logits = Tensor([[2.0, 1.0, 0.1]], requires_grad=True)
    targets = Tensor([[1.0, 0.0, 0.0]])  # one-hot: class 0

    loss = cross_entropy_loss(logits, targets)
    loss.backward()

    print(f"  Logits:  {logits.data}")
    print(f"  Target:  {targets.data}")
    print(f"  Loss:    {loss.data:.4f}")
    print(f"  Grad:    {logits.grad}")

    # Verify: softmax([2, 1, 0.1]) - [1, 0, 0]
    shifted = logits.data - np.max(logits.data)
    exp_x = np.exp(shifted)
    sm = exp_x / np.sum(exp_x)
    expected_grad = (sm - targets.data) / 1  # batch_size=1
    print(f"  Expected:{expected_grad}")
    assert np.allclose(logits.grad, expected_grad, atol=1e-6), "CE grad FAILED"
    print("  OK")

    # ============ BAI TAP 3: Full Neural Network ============
    print("\n" + "=" * 60)
    print("BAI TAP 3: Full Neural Network voi Autograd")
    print("=" * 60)

    # Circular dataset (giong bai 03)
    n_samples = 1000
    r0 = np.random.uniform(0, 1, n_samples // 2)
    theta0 = np.random.uniform(0, 2 * np.pi, n_samples // 2)
    X0 = np.column_stack([r0 * np.cos(theta0), r0 * np.sin(theta0)])
    r1 = np.random.uniform(1.5, 2.5, n_samples // 2)
    theta1 = np.random.uniform(0, 2 * np.pi, n_samples // 2)
    X1 = np.column_stack([r1 * np.cos(theta1), r1 * np.sin(theta1)])

    X = np.vstack([X0, X1])
    y_labels = np.hstack([np.zeros(n_samples // 2), np.ones(n_samples // 2)]).astype(int)
    y_onehot = np.zeros((n_samples, 2))
    y_onehot[np.arange(n_samples), y_labels] = 1

    indices = np.random.permutation(n_samples)
    X, y_onehot, y_labels = X[indices], y_onehot[indices], y_labels[indices]
    X_train, X_test = X[:800], X[800:]
    y_train, y_test = y_onehot[:800], y_onehot[800:]

    print("  Training AutogradMLP [2, 16, 8, 2]...")
    model = AutogradMLP([2, 16, 8, 2])
    history = model.train(X_train, y_train, epochs=100, lr=0.1, batch_size=32)

    test_preds = model.predict(X_test)
    test_labels = np.argmax(y_test, axis=1)
    test_acc = np.mean(test_preds == test_labels)
    print(f"\n  Test accuracy: {test_acc:.4f}")
    assert test_acc > 0.90, f"Autograd MLP accuracy too low: {test_acc}"
    print("  OK - Network trains correctly voi autograd!")

    # ============ BAI TAP 4: Verify voi PyTorch ============
    print("\n" + "=" * 60)
    print("BAI TAP 4: Verify gradients voi PyTorch")
    print("=" * 60)

    try:
        import torch

        print("\n  --- Test: d(sum(x^2))/dx ---")
        # Our autograd
        x_ours = Tensor([2.0, 3.0, 4.0], requires_grad=True)
        y_ours = x_ours ** 2
        z_ours = y_ours.sum()
        z_ours.backward()

        # PyTorch
        x_torch = torch.tensor([2.0, 3.0, 4.0], requires_grad=True)
        z_torch = (x_torch ** 2).sum()
        z_torch.backward()

        print(f"  Ours:    {x_ours.grad}")
        print(f"  PyTorch: {x_torch.grad.numpy()}")
        assert np.allclose(x_ours.grad, x_torch.grad.numpy(), atol=1e-6)
        print("  MATCH!")

        print("\n  --- Test: matmul gradient ---")
        W_np = np.array([[1.0, 2], [3, 4], [5, 6]])
        x_np = np.array([[1.0, 2, 3]])

        W_ours = Tensor(W_np.copy(), requires_grad=True)
        x_ours = Tensor(x_np.copy(), requires_grad=True)
        loss_ours = (x_ours @ W_ours).sum()
        loss_ours.backward()

        W_torch = torch.tensor(W_np, requires_grad=True)
        x_torch = torch.tensor(x_np, requires_grad=True)
        loss_torch = (x_torch @ W_torch).sum()
        loss_torch.backward()

        print(f"  W grad ours:    {W_ours.grad.flatten()}")
        print(f"  W grad PyTorch: {W_torch.grad.numpy().flatten()}")
        assert np.allclose(W_ours.grad, W_torch.grad.numpy(), atol=1e-6)
        assert np.allclose(x_ours.grad, x_torch.grad.numpy(), atol=1e-6)
        print("  MATCH!")

        print("\n  --- Test: relu + matmul ---")
        x_np = np.array([[1.0, -2.0]])
        W1_np = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        W2_np = np.array([[0.7], [0.8], [0.9]])

        # Ours
        x_o = Tensor(x_np.copy(), requires_grad=True)
        W1_o = Tensor(W1_np.copy(), requires_grad=True)
        W2_o = Tensor(W2_np.copy(), requires_grad=True)
        h_o = (x_o @ W1_o).relu()
        out_o = (h_o @ W2_o).sum()
        out_o.backward()

        # PyTorch
        x_t = torch.tensor(x_np, requires_grad=True)
        W1_t = torch.tensor(W1_np, requires_grad=True)
        W2_t = torch.tensor(W2_np, requires_grad=True)
        h_t = (x_t @ W1_t).relu()
        out_t = (h_t @ W2_t).sum()
        out_t.backward()

        print(f"  W1 grad ours:    {W1_o.grad.flatten()}")
        print(f"  W1 grad PyTorch: {W1_t.grad.numpy().flatten()}")
        assert np.allclose(W1_o.grad, W1_t.grad.numpy(), atol=1e-6)
        assert np.allclose(W2_o.grad, W2_t.grad.numpy(), atol=1e-6)
        print("  MATCH!")

        print("\n  --- Test: exp, log, tanh ---")
        x_np = np.array([0.5, 1.0, 1.5])

        for op_name in ['exp', 'log', 'tanh']:
            x_o = Tensor(x_np.copy(), requires_grad=True)
            x_t = torch.tensor(x_np, requires_grad=True)

            if op_name == 'exp':
                getattr(x_o, op_name)().sum().backward()
                x_t.exp().sum().backward()
            elif op_name == 'log':
                getattr(x_o, op_name)().sum().backward()
                x_t.log().sum().backward()
            elif op_name == 'tanh':
                getattr(x_o, op_name)().sum().backward()
                x_t.tanh().sum().backward()

            print(f"  {op_name}: ours={x_o.grad}, torch={x_t.grad.numpy()}")
            assert np.allclose(x_o.grad, x_t.grad.numpy(), atol=1e-6), f"{op_name} FAILED"

        print("  ALL MATCH!")

        print("\n  --- Test: cross_entropy gradient ---")
        logits_np = np.array([[2.0, 1.0, 0.1], [0.5, 2.5, 0.3]])
        targets_np = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float64)

        # Ours
        logits_o = Tensor(logits_np.copy(), requires_grad=True)
        targets_o = Tensor(targets_np.copy())
        loss_o = cross_entropy_loss(logits_o, targets_o)
        loss_o.backward()

        # PyTorch
        logits_t = torch.tensor(logits_np, requires_grad=True)
        targets_t = torch.tensor(np.argmax(targets_np, axis=1))
        loss_t = torch.nn.functional.cross_entropy(logits_t, targets_t)
        loss_t.backward()

        print(f"  Loss ours:    {loss_o.data:.6f}")
        print(f"  Loss PyTorch: {loss_t.item():.6f}")
        print(f"  Grad ours:    {logits_o.grad.flatten()}")
        print(f"  Grad PyTorch: {logits_t.grad.numpy().flatten()}")
        assert np.allclose(loss_o.data, loss_t.item(), atol=1e-5), "CE loss FAILED"
        assert np.allclose(logits_o.grad, logits_t.grad.numpy(), atol=1e-5), "CE grad FAILED"
        print("  MATCH!")

    except ImportError:
        print("  torch chua cai. Chay: pip install torch")

    print("\n" + "=" * 60)
    print("TAT CA TESTS PASSED!")
    print("=" * 60)


# ============ CHECKLIST ============
# Week 3-4 (Bai 04):
# [x] Hieu chain rule va ap dung
#     -> Chain rule: dL/dx = dL/dy * dy/dx
#        VD: y = x^2, L = sum(y) -> dL/dx = 1 * 2x = 2x
#        Moi operation (add, mul, matmul, relu) biet cach tinh dy/dx
#        backward() goi chain rule nguoc tu loss -> tung parameter
#        Khac voi file 03 (code tay), o day graph TU DONG lam
# [x] Build autograd system don gian
#     -> Tensor class voi computational graph (creators, creation_op)
#        backward() duyet graph nguoc, ap dung chain rule cho tung op
#        Ho tro: add, mul, matmul, relu, sigmoid, tanh, sum, mean, log, ...
# [x] Train duoc model tren toy dataset
#     -> AutogradMLP train tren XOR va Iris dataset
#        So sanh gradient voi PyTorch -> MATCH
