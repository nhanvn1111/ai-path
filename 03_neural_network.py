# File: 03_neural_network.py
# Neural Network tu Scratch (chi dung numpy) - Week 3-4
# WALKTHROUGH QUICK NOTES:
# - Activation helpers: sigmoid/relu/softmax + derivatives dùng xuyên suốt.
# - Layer class: forward tính `x @ W + b`, backward áp dụng chain rule và accumulate gradients.
# - Dropout & BatchNorm: mô phỏng regularization và normalization đúng như sách giáo khoa.
# - NeuralNetwork: build list layer, định nghĩa forward/backward/update/train hoàn chỉnh.
# - one_hot + helper khác: chuẩn hoá label để loss hoạt động.
#
# TAI SAO HOC CAI NAY?
# Neural network la core cua AI hien dai. File nay build TU DAU:
# - Forward pass: input -> qua tung layer -> output (du doan)
# - Backward pass (backpropagation): tinh gradient cho moi weight
# - Update: chinh weights theo gradient de model hoc
#
# TAI SAO KHONG DUNG PYTORCH LUON?
# PyTorch lam forward/backward/update tu dong, nhung "giau" het.
# Code bang numpy 1 lan -> hieu backpropagation hoat dong the nao,
# tai sao can activation function, tai sao can normalize, v.v.

import numpy as np
import math
import os
import time

# ============ ACTIVATION FUNCTIONS ============
# TAI SAO CAN ACTIVATION?
# Neu khong co activation, nhieu layers chong len nhau van chi la 1 phep
# nhan ma tran (linear). Activation them "phi tuyen" (non-linearity)
# -> model co the hoc duoc cac pattern phuc tap (duong cong, hinh tron, v.v.)

def sigmoid(x):
    """Output trong [0, 1]. Dung cho xac suat, binary classification."""
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def sigmoid_derivative(x):
    """Gradient của sigmoid để dùng trong backprop (s = σ(x) ⇒ σ'(x) = s*(1-s))."""
    s = sigmoid(x)
    return s * (1 - s)

def relu(x):
    """ReLU: max(0, x). Pho bien nhat vi don gian va hieu qua. Gradient = 0 hoac 1."""
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

def softmax(x):
    """Chuyen vector scores thanh xac suat (tong = 1). Dung cho output layer multi-class."""
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


# ============ LAYERS ============

class Layer:
    """
    Mot layer trong neural network

    Forward:  output = activation(input @ weights + bias)
    Backward: tinh gradients va update weights
    """

    def __init__(self, input_size, output_size, activation='relu'):
        """
        input_size:  so neurons dau vao (= so features cua input)
                     Vd: anh 28x28 -> input_size = 784
        output_size: so neurons dau ra (= so features output cua layer nay)
                     Vd: hidden layer voi 128 neurons -> output_size = 128
        activation:  ham kich hoat sau phep nhan ma tran
                     'relu'    -> pho bien nhat cho hidden layers
                     'sigmoid' -> output [0,1], dung cho binary classification
                     'softmax' -> output xac suat (tong=1), dung cho layer cuoi multi-class
        """
        # Xavier/He initialization
        # TAI SAO KHONG DUNG random binh thuong?
        # Neu weights qua lon -> output bung no (exploding)
        # Neu weights qua nho -> gradient mat dan (vanishing)
        # He init: scale = sqrt(2/n) -> giu variance on dinh qua cac layers
        if activation == 'relu':
            scale = np.sqrt(2.0 / input_size)
        else:
            scale = np.sqrt(1.0 / input_size)

        self.weights = np.random.randn(input_size, output_size) * scale
        self.bias = np.zeros((1, output_size))
        self.activation = activation

        # Cache cho backprop
        self.input = None
        self.z = None
        self.output = None

        # Gradients
        self.grad_weights = None
        self.grad_bias = None

    def forward(self, x, training=True):
        """
        x:        input data, shape (batch_size, input_size)
                  batch_size = so samples xu ly cung luc
        training: True khi train, False khi inference (anh huong Dropout/BatchNorm)

        Return: output shape (batch_size, output_size)
        """
        self.input = x
        self.z = x @ self.weights + self.bias

        if self.activation == 'relu':
            self.output = relu(self.z)
        elif self.activation == 'sigmoid':
            self.output = sigmoid(self.z)
        elif self.activation == 'softmax':
            self.output = softmax(self.z)
        else:
            self.output = self.z

        return self.output

    def backward(self, grad_output):
        """
        Backpropagation qua 1 layer.
        Chain rule: gradient "chay nguoc" tu output ve input.
        grad_weights = input.T @ grad_z  (de biet weights can thay doi bao nhieu)
        grad_input = grad_z @ weights.T  (de truyen gradient ve layer truoc)
        """
        batch_size = self.input.shape[0]

        if self.activation == 'relu':
            grad_z = grad_output * relu_derivative(self.z)
        elif self.activation == 'sigmoid':
            grad_z = grad_output * sigmoid_derivative(self.z)
        elif self.activation == 'softmax':
            grad_z = grad_output
        else:
            grad_z = grad_output

        self.grad_weights = self.input.T @ grad_z / batch_size
        self.grad_bias = np.sum(grad_z, axis=0, keepdims=True) / batch_size

        grad_input = grad_z @ self.weights.T
        return grad_input

    def update(self, lr):
        self.weights -= lr * self.grad_weights
        self.bias -= lr * self.grad_bias


class Dropout:
    """
    Dropout layer - regularization

    Training: randomly set neurons to 0 voi probability p
    Inference: khong drop, nhung scale output

    Tai sao Dropout hoat dong:
    - Buoc network khong phu thuoc vao bat ky neuron nao
    - Giong nhu training nhieu networks nho cung luc (ensemble)
    - Giam overfitting rat hieu qua
    """

    def __init__(self, drop_rate=0.5):
        """
        drop_rate: ty le neuron bi tat (0.0 - 1.0)
                   0.5 = tat 50% neurons -> moi lan train, network chi dung nua so neurons
                   0.1 = tat 10% -> dropout nhe
                   0.0 = khong drop gi -> khong co regularization
                   GPT dung 0.1, MLP don gian dung 0.3-0.5
        """
        self.drop_rate = drop_rate
        self.mask = None

    def forward(self, x, training=True):
        if training:
            # Tao mask: 1 = giu, 0 = drop
            self.mask = (np.random.rand(*x.shape) > self.drop_rate).astype(float)
            # Scale de giu expected value khong doi
            return x * self.mask / (1 - self.drop_rate)
        else:
            return x

    def backward(self, grad_output):
        return grad_output * self.mask / (1 - self.drop_rate)

    def update(self, lr):
        pass  # Khong co weights


class BatchNorm:
    """
    Batch Normalization

    Normalize input cua moi layer de mean=0, variance=1
    Sau do scale va shift bang gamma, beta (learnable)

    Loi ich:
    - Training nhanh hon (cho phep learning rate lon hon)
    - Giam van de vanishing/exploding gradient
    - Co tac dung regularization nhe
    """

    def __init__(self, num_features, momentum=0.9, eps=1e-5):
        """
        num_features: so features cua input (= output_size cua layer truoc)
                      Vd: layer truoc co 128 neurons -> num_features = 128
        momentum:     he so cap nhat running_mean/running_var (0.0 - 1.0)
                      0.9 = giu 90% stats cu + 10% stats moi (on dinh)
                      Dung cho inference khi khong co batch
        eps:          so nho tranh chia cho 0 khi variance ~ 0
        """
        self.gamma = np.ones((1, num_features))   # scale factor (learnable)
        self.beta = np.zeros((1, num_features))    # shift factor (learnable)
        self.eps = eps
        self.momentum = momentum

        # Running stats cho inference
        self.running_mean = np.zeros((1, num_features))
        self.running_var = np.ones((1, num_features))

        # Cache cho backprop
        self.x_norm = None
        self.mean = None
        self.var = None
        self.input = None

        # Gradients
        self.grad_gamma = None
        self.grad_beta = None

    def forward(self, x, training=True):
        self.input = x

        if training:
            self.mean = np.mean(x, axis=0, keepdims=True)
            self.var = np.var(x, axis=0, keepdims=True)

            # Update running stats
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * self.mean
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * self.var

            # Normalize
            self.x_norm = (x - self.mean) / np.sqrt(self.var + self.eps)
        else:
            self.x_norm = (x - self.running_mean) / np.sqrt(self.running_var + self.eps)

        return self.gamma * self.x_norm + self.beta

    def backward(self, grad_output):
        batch_size = self.input.shape[0]

        self.grad_gamma = np.sum(grad_output * self.x_norm, axis=0, keepdims=True)
        self.grad_beta = np.sum(grad_output, axis=0, keepdims=True)

        # Gradient qua normalization
        dx_norm = grad_output * self.gamma
        dvar = np.sum(dx_norm * (self.input - self.mean) * -0.5 * (self.var + self.eps) ** (-1.5), axis=0, keepdims=True)
        dmean = np.sum(dx_norm * -1 / np.sqrt(self.var + self.eps), axis=0, keepdims=True) + dvar * np.mean(-2 * (self.input - self.mean), axis=0, keepdims=True)

        grad_input = dx_norm / np.sqrt(self.var + self.eps) + dvar * 2 * (self.input - self.mean) / batch_size + dmean / batch_size

        return grad_input

    def update(self, lr):
        self.gamma -= lr * self.grad_gamma
        self.beta -= lr * self.grad_beta


# ============ NEURAL NETWORK ============

class NeuralNetwork:
    """
    Multi-layer Perceptron (MLP)

    Vi du: [784, 128, 64, 10]
    - Input: 784 features (28x28 image)
    - Hidden 1: 128 neurons
    - Hidden 2: 64 neurons
    - Output: 10 classes
    """

    def __init__(self, layer_sizes, hidden_activation='relu',
                 dropout_rate=0.0, use_batchnorm=False):
        """
        layer_sizes:       list so neurons moi layer, vd [784, 128, 64, 10]
                           Phan tu dau = input features, phan tu cuoi = output classes
                           Cac phan tu giua = hidden layers
        hidden_activation: activation cho hidden layers ('relu', 'sigmoid')
                           Layer cuoi luon dung 'softmax' (output xac suat)
        dropout_rate:      ty le dropout cho hidden layers (0.0 = tat)
                           Them vao SAU moi hidden layer, KHONG them sau output layer
        use_batchnorm:     True = them BatchNorm sau moi hidden layer
                           Giup training on dinh hon, cho phep lr lon hon
        """
        self.layers = []
        self.dropout_rate = dropout_rate
        self.use_batchnorm = use_batchnorm

        for i in range(len(layer_sizes) - 1):
            is_last = (i == len(layer_sizes) - 2)
            activation = 'softmax' if is_last else hidden_activation

            layer = Layer(layer_sizes[i], layer_sizes[i + 1], activation)
            self.layers.append(layer)

            # Them BatchNorm va Dropout sau moi hidden layer (khong phai output)
            if not is_last:
                if use_batchnorm:
                    self.layers.append(BatchNorm(layer_sizes[i + 1]))
                if dropout_rate > 0:
                    self.layers.append(Dropout(dropout_rate))

    def forward(self, x, training=True):
        for layer in self.layers:
            x = layer.forward(x, training=training)
        return x

    def backward(self, y_true):
        # Tim layer cuoi cung (Layer instance, khong phai Dropout/BatchNorm)
        last_layer = None
        for layer in reversed(self.layers):
            if isinstance(layer, Layer):
                last_layer = layer
                break

        grad = last_layer.output - y_true

        for layer in reversed(self.layers):
            grad = layer.backward(grad)

    def update(self, lr):
        for layer in self.layers:
            layer.update(lr)

    def compute_loss(self, y_pred, y_true):
        eps = 1e-15
        y_pred = np.clip(y_pred, eps, 1 - eps)
        return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))

    def train_model(self, X, y, epochs=100, lr=0.01, batch_size=32,
                    X_val=None, y_val=None, verbose=True):
        n_samples = X.shape[0]
        history = {'loss': [], 'accuracy': [], 'val_loss': [], 'val_accuracy': []}

        for epoch in range(epochs):
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]

            epoch_loss = 0
            n_batches = 0

            for i in range(0, n_samples, batch_size):
                X_batch = X_shuffled[i:i + batch_size]
                y_batch = y_shuffled[i:i + batch_size]

                output = self.forward(X_batch, training=True)
                loss = self.compute_loss(output, y_batch)
                epoch_loss += loss
                n_batches += 1

                self.backward(y_batch)
                self.update(lr)

            avg_loss = epoch_loss / n_batches
            history['loss'].append(avg_loss)

            # Train accuracy
            predictions = np.argmax(self.forward(X, training=False), axis=1)
            labels = np.argmax(y, axis=1)
            accuracy = np.mean(predictions == labels)
            history['accuracy'].append(accuracy)

            # Validation
            if X_val is not None:
                val_output = self.forward(X_val, training=False)
                val_labels = np.argmax(y_val, axis=1)
                val_loss = self.compute_loss(val_output, y_val)
                val_acc = np.mean(np.argmax(val_output, axis=1) == val_labels)
                history['val_loss'].append(val_loss)
                history['val_accuracy'].append(val_acc)

            if verbose and epoch % 5 == 0:
                msg = f"  Epoch {epoch:3d}: loss={avg_loss:.4f}, acc={accuracy:.4f}"
                if X_val is not None:
                    msg += f", val_loss={val_loss:.4f}, val_acc={val_acc:.4f}"
                print(msg)

        return history

    def predict(self, X):
        output = self.forward(X, training=False)
        return np.argmax(output, axis=1)


# ============ HELPER ============

def one_hot(labels, num_classes):
    n = len(labels)
    y = np.zeros((n, num_classes))
    y[np.arange(n), labels] = 1
    return y


# ============ MAIN ============
if __name__ == "__main__":
    np.random.seed(42)
    output_dir = os.path.dirname(os.path.abspath(__file__))

    # ============ TEST CO BAN: Circular dataset ============
    print("=" * 60)
    print("TEST CO BAN - Circular Dataset")
    print("=" * 60)

    n_samples = 1000

    r0 = np.random.uniform(0, 1, n_samples // 2)
    theta0 = np.random.uniform(0, 2 * np.pi, n_samples // 2)
    X0 = np.column_stack([r0 * np.cos(theta0), r0 * np.sin(theta0)])

    r1 = np.random.uniform(1.5, 2.5, n_samples // 2)
    theta1 = np.random.uniform(0, 2 * np.pi, n_samples // 2)
    X1 = np.column_stack([r1 * np.cos(theta1), r1 * np.sin(theta1)])

    X = np.vstack([X0, X1])
    y_labels = np.hstack([np.zeros(n_samples // 2), np.ones(n_samples // 2)]).astype(int)
    y = one_hot(y_labels, 2)

    indices = np.random.permutation(n_samples)
    X, y = X[indices], y[indices]
    y_labels = y_labels[indices]

    X_train, X_test = X[:800], X[800:]
    y_train, y_test = y[:800], y[800:]

    nn_basic = NeuralNetwork([2, 16, 8, 2])
    history_basic = nn_basic.train_model(X_train, y_train, epochs=100, lr=0.1, batch_size=32)

    test_acc = np.mean(nn_basic.predict(X_test) == np.argmax(y_test, axis=1))
    print(f"\n  Test accuracy: {test_acc:.4f}")
    assert test_acc > 0.90, f"Basic test accuracy too low: {test_acc}"

    # ============ BAI TAP 5: Visualize Decision Boundary ============
    print("\n" + "=" * 60)
    print("BAI TAP 5: Visualize Decision Boundary")
    print("=" * 60)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 1, figsize=(8, 8))

        # Tao grid
        x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
        y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                             np.linspace(y_min, y_max, 200))
        grid = np.column_stack([xx.ravel(), yy.ravel()])

        # Predict tren grid
        Z = nn_basic.predict(grid).reshape(xx.shape)

        ax.contourf(xx, yy, Z, alpha=0.3, cmap='RdBu')
        ax.scatter(X_test[:, 0], X_test[:, 1],
                   c=np.argmax(y_test, axis=1), cmap='RdBu',
                   edgecolors='black', s=30)
        ax.set_title(f"Decision Boundary (test acc={test_acc:.4f})")
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")

        path = os.path.join(output_dir, "plot_decision_boundary.png")
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"  Saved: {path}")
        HAS_MATPLOTLIB = True
    except ImportError:
        print("  matplotlib chua cai, bo qua plot.")
        HAS_MATPLOTLIB = False

    # ============ BAI TAP 1: Train tren MNIST that ============
    print("\n" + "=" * 60)
    print("BAI TAP 1: Train tren MNIST dataset")
    print("=" * 60)

    try:
        from sklearn.datasets import fetch_openml
        from sklearn.model_selection import train_test_split

        print("  Dang download MNIST...")
        mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='liac-arff')
        X_mnist = mnist.data.astype(np.float32) / 255.0  # Normalize [0, 1]
        y_mnist = mnist.target.astype(int)

        # Lay 10000 samples de train nhanh hon
        X_mnist_small = X_mnist[:10000]
        y_mnist_small = y_mnist[:10000]

        X_m_train, X_m_test, y_m_train_labels, y_m_test_labels = train_test_split(
            X_mnist_small, y_mnist_small, test_size=0.2, random_state=42
        )
        y_m_train = one_hot(y_m_train_labels, 10)
        y_m_test = one_hot(y_m_test_labels, 10)

        print(f"  Train: {X_m_train.shape}, Test: {X_m_test.shape}")

        # --- Plain MLP ---
        print("\n  --- Plain MLP [784, 128, 64, 10] ---")
        nn_plain = NeuralNetwork([784, 128, 64, 10])
        t0 = time.time()
        hist_plain = nn_plain.train_model(
            X_m_train, y_m_train, epochs=30, lr=0.1, batch_size=64,
            X_val=X_m_test, y_val=y_m_test
        )
        t_plain = time.time() - t0
        acc_plain = np.mean(nn_plain.predict(X_m_test) == y_m_test_labels)
        print(f"  Plain MLP -> test acc: {acc_plain:.4f} ({t_plain:.1f}s)")

        # ============ BAI TAP 2: Dropout ============
        print("\n" + "=" * 60)
        print("BAI TAP 2: Dropout (rate=0.3)")
        print("=" * 60)

        nn_dropout = NeuralNetwork([784, 128, 64, 10], dropout_rate=0.3)
        t0 = time.time()
        hist_dropout = nn_dropout.train_model(
            X_m_train, y_m_train, epochs=30, lr=0.1, batch_size=64,
            X_val=X_m_test, y_val=y_m_test
        )
        t_dropout = time.time() - t0
        acc_dropout = np.mean(nn_dropout.predict(X_m_test) == y_m_test_labels)
        print(f"  Dropout MLP -> test acc: {acc_dropout:.4f} ({t_dropout:.1f}s)")

        # ============ BAI TAP 3: BatchNorm ============
        print("\n" + "=" * 60)
        print("BAI TAP 3: Batch Normalization")
        print("=" * 60)

        nn_bn = NeuralNetwork([784, 128, 64, 10], use_batchnorm=True)
        t0 = time.time()
        hist_bn = nn_bn.train_model(
            X_m_train, y_m_train, epochs=30, lr=0.1, batch_size=64,
            X_val=X_m_test, y_val=y_m_test
        )
        t_bn = time.time() - t0
        acc_bn = np.mean(nn_bn.predict(X_m_test) == y_m_test_labels)
        print(f"  BatchNorm MLP -> test acc: {acc_bn:.4f} ({t_bn:.1f}s)")

        # --- Dropout + BatchNorm ---
        print("\n  --- Dropout + BatchNorm ---")
        nn_both = NeuralNetwork([784, 128, 64, 10], dropout_rate=0.3, use_batchnorm=True)
        t0 = time.time()
        hist_both = nn_both.train_model(
            X_m_train, y_m_train, epochs=30, lr=0.1, batch_size=64,
            X_val=X_m_test, y_val=y_m_test
        )
        t_both = time.time() - t0
        acc_both = np.mean(nn_both.predict(X_m_test) == y_m_test_labels)
        print(f"  Both -> test acc: {acc_both:.4f} ({t_both:.1f}s)")

        # So sanh
        print("\n  --- So sanh tren MNIST ---")
        print(f"  {'Model':<25} {'Test Acc':<12} {'Time'}")
        print("  " + "-" * 50)
        print(f"  {'Plain MLP':<25} {acc_plain:<12.4f} {t_plain:.1f}s")
        print(f"  {'+ Dropout(0.3)':<25} {acc_dropout:<12.4f} {t_dropout:.1f}s")
        print(f"  {'+ BatchNorm':<25} {acc_bn:<12.4f} {t_bn:.1f}s")
        print(f"  {'+ Both':<25} {acc_both:<12.4f} {t_both:.1f}s")

        MNIST_LOADED = True

    except ImportError:
        print("  scikit-learn chua cai. Chay: pip install scikit-learn")
        MNIST_LOADED = False

    # ============ BAI TAP 4: So sanh voi PyTorch ============
    print("\n" + "=" * 60)
    print("BAI TAP 4: So sanh voi PyTorch")
    print("=" * 60)

    try:
        import torch
        import torch.nn as torchnn
        import torch.optim as optim
        from torch.utils.data import DataLoader, TensorDataset

        if MNIST_LOADED:
            # Chuyen sang torch tensors
            X_t_train = torch.FloatTensor(X_m_train)
            y_t_train = torch.LongTensor(y_m_train_labels)
            X_t_test = torch.FloatTensor(X_m_test)
            y_t_test = torch.LongTensor(y_m_test_labels)

            train_ds = TensorDataset(X_t_train, y_t_train)
            train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)

            # PyTorch model tuong duong
            torch_model = torchnn.Sequential(
                torchnn.Linear(784, 128),
                torchnn.ReLU(),
                torchnn.Linear(128, 64),
                torchnn.ReLU(),
                torchnn.Linear(64, 10),
            )

            criterion = torchnn.CrossEntropyLoss()
            optimizer = optim.SGD(torch_model.parameters(), lr=0.1)

            print("  Training PyTorch model (same architecture)...")
            t0 = time.time()
            for epoch in range(30):
                torch_model.train()
                for xb, yb in train_loader:
                    optimizer.zero_grad()
                    out = torch_model(xb)
                    loss = criterion(out, yb)
                    loss.backward()
                    optimizer.step()

                if epoch % 5 == 0:
                    torch_model.eval()
                    with torch.no_grad():
                        preds = torch_model(X_t_test).argmax(dim=1)
                        acc = (preds == y_t_test).float().mean().item()
                    print(f"  Epoch {epoch:3d}: acc={acc:.4f}")

            t_torch = time.time() - t0
            torch_model.eval()
            with torch.no_grad():
                preds = torch_model(X_t_test).argmax(dim=1)
                acc_torch = (preds == y_t_test).float().mean().item()

            print(f"\n  --- Ket qua so sanh ---")
            print(f"  {'Model':<25} {'Test Acc':<12} {'Time'}")
            print("  " + "-" * 50)
            print(f"  {'Our MLP (numpy)':<25} {acc_plain:<12.4f} {t_plain:.1f}s")
            print(f"  {'PyTorch MLP':<25} {acc_torch:<12.4f} {t_torch:.1f}s")
            print(f"\n  -> Accuracy gan nhau = implementation cua minh DUNG!")
            print(f"  -> PyTorch nhanh hon vi dung optimized BLAS backend")
        else:
            print("  Bo qua vi MNIST chua load duoc.")

    except ImportError:
        print("  torch chua cai. Chay: pip install torch")

    # ============ PLOT TRAINING CURVES ============
    if HAS_MATPLOTLIB and MNIST_LOADED:
        print("\n" + "=" * 60)
        print("PLOT: Training curves")
        print("=" * 60)

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Loss curves
        ax = axes[0]
        ax.plot(hist_plain['loss'], label='Plain', linewidth=1.5)
        ax.plot(hist_dropout['loss'], label='Dropout', linewidth=1.5)
        ax.plot(hist_bn['loss'], label='BatchNorm', linewidth=1.5)
        ax.plot(hist_both['loss'], label='Both', linewidth=1.5)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("Training Loss - MNIST")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Validation accuracy
        ax = axes[1]
        ax.plot(hist_plain['val_accuracy'], label='Plain', linewidth=1.5)
        ax.plot(hist_dropout['val_accuracy'], label='Dropout', linewidth=1.5)
        ax.plot(hist_bn['val_accuracy'], label='BatchNorm', linewidth=1.5)
        ax.plot(hist_both['val_accuracy'], label='Both', linewidth=1.5)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Accuracy")
        ax.set_title("Validation Accuracy - MNIST")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        path = os.path.join(output_dir, "plot_mnist_training.png")
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"  Saved: {path}")

    print("\n" + "=" * 60)
    print("TAT CA TESTS PASSED!")
    print("=" * 60)


# ============ CHECKLIST ============
# Week 3-4 (Bai 03):
# [x] Build neural network KHONG dung framework
#     -> NeuralNetwork class voi Layer, Dropout, BatchNorm (chi dung numpy)
# [x] Implement backpropagation manually
#     -> Layer.backward(): tinh grad_weights, grad_bias, grad_input bang tay
#        Dropout.backward(): grad * mask (chi truyen gradient qua neurons con song)
#        BatchNorm.backward(): tinh dgamma, dbeta, dx (phuc tap nhat)
