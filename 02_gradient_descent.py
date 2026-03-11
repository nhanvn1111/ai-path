# File: 02_gradient_descent.py
# Gradient Descent tu Scratch - Week 1-2
# WALKTHROUGH QUICK NOTES:
# - numerical_gradient: finite difference từng chiều, tiện để kiểm chứng autograd.
# - gradient_descent/sgd_with_momentum/adam: ba biến thể tối ưu, mỗi hàm lặp epochs rồi cập nhật vector tham số.
# - simple_quadratic/rosenbrock + gradients tương ứng: benchmark để chạy thử optimizers.
#
# TAI SAO HOC CAI NAY?
# Gradient Descent la CACH DUY NHAT de model hoc:
# 1. Model du doan sai -> tinh loss (do sai bao nhieu)
# 2. Tinh gradient (huong nao giam loss nhanh nhat)
# 3. Update weights theo huong do: W = W - lr * gradient
# 4. Lap lai -> loss giam dan -> model du doan dung hon
#
# Moi optimizer (SGD, Momentum, Adam) deu la bien the cua y tuong nay.
# Khong co gradient descent = khong co AI.

import math
import os

# ============ OPTIMIZERS ============

def numerical_gradient(f, x, h=1e-7):
    """
    Tinh gradient bang finite difference

    f: ham can tinh gradient, nhan list so -> tra ve 1 so (loss)
    x: list cac gia tri hien tai, vd [0.5, 1.2] (vi tri dang dung)
    h: buoc nho de xap xi dao ham (cang nho cang chinh xac, nhung qua nho -> sai so so)

    gradient ~ (f(x+h) - f(x-h)) / (2h)

    Day la cach don gian nhat de tinh gradient.
    PyTorch dung autograd (symbolic differentiation) - nhanh hon nhieu.
    """
    grad = []
    for i in range(len(x)):
        x_plus = x.copy()
        x_minus = x.copy()
        x_plus[i] += h
        x_minus[i] -= h
        grad.append((f(x_plus) - f(x_minus)) / (2 * h))
    return grad


def gradient_descent(f, grad_f, x0, lr=0.01, epochs=1000, verbose=False):
    """
    Vanilla Gradient Descent - phien ban don gian nhat

    f:       ham loss, nhan list so -> tra ve loss (float)
    grad_f:  ham tinh gradient cua f, nhan list so -> tra ve list gradient
    x0:      diem bat dau (list so), vd [0.0, 0.0]
    lr:      learning rate - buoc nhay moi lan update
             Qua lon (>1.0) -> phan ky (diverge), qua nho (<0.001) -> hoi tu cham
    epochs:  so lan lap (moi lan = 1 buoc update)
    verbose: True -> in loss moi 100 epochs de theo doi

    x_new = x_old - learning_rate * gradient

    TAI SAO TRU?
    - Gradient chi huong TANG nhanh nhat cua ham
    - Muon GIAM loss -> di NGUOC lai -> tru gradient

    VAN DE:
    - De bi ket o local minima (diem thap cuc bo)
    - Toc do hoi tu phu thuoc nhieu vao learning rate
    - Khong co "da" -> dao dong quanh minimum
    """
    x = x0.copy()
    history = []

    for epoch in range(epochs):
        loss = f(x)
        history.append(loss)
        grad = grad_f(x)

        for i in range(len(x)):
            x[i] = x[i] - lr * grad[i]

        if verbose and epoch % 100 == 0:
            print(f"  Epoch {epoch}: loss = {loss:.6f}, x = {[round(v, 4) for v in x]}")

    return x, history


def sgd_with_momentum(f, grad_f, x0, lr=0.01, momentum=0.9, epochs=1000):
    """
    SGD with Momentum - them "da" vao gradient descent

    f, grad_f, x0, lr, epochs: giong gradient_descent()
    momentum: he so "da" (0.0 - 1.0)
              0.9 = giu 90% huong di cu, chi them 10% gradient moi
              0.0 = khong co da = giong vanilla GD
              0.99 = da rat lon, vuot qua nhieu local minima nhung kho dung lai

    velocity = momentum * velocity - lr * gradient
    x = x + velocity

    TAI SAO CAN MOMENTUM?
    - Vanilla GD de bi ket o local minima
    - Momentum giong qua bong lan xuong doi: co da nen vuot qua duoc ho nho
    - momentum=0.9: giu 90% velocity cu + 10% gradient moi
    - Giup hoi tu nhanh hon va on dinh hon vanilla GD
    """
    x = x0.copy()
    velocity = [0.0] * len(x)  # ban dau dung yen, khong co huong di
    history = []

    for epoch in range(epochs):
        loss = f(x)
        history.append(loss)
        grad = grad_f(x)

        for i in range(len(x)):
            velocity[i] = momentum * velocity[i] - lr * grad[i]
            x[i] = x[i] + velocity[i]

    return x, history


def adam(f, grad_f, x0, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8, epochs=1000):
    """
    Adam Optimizer - pho bien nhat hien nay (GPT, BERT, LLaMA deu dung)

    f, grad_f, x0, epochs: giong gradient_descent()
    lr:    learning rate, default 0.001 (nho hon vanilla GD vi Adam tu adapt)
    beta1: he so trung binh cho momentum (first moment), default 0.9
           Giong momentum: giu 90% huong di cu
    beta2: he so trung binh cho variance (second moment), default 0.999
           Theo doi gradient^2 de biet chieu nao dao dong nhieu
    eps:   so nho tranh chia cho 0, default 1e-8

    Ket hop 2 y tuong:
    - m (first moment = Momentum): trung binh gradient -> huong di on dinh
    - v (second moment = RMSprop): trung binh gradient^2 -> tu dong dieu chinh lr
      * Chieu nao gradient lon -> lr giam (tranh nhay qua manh)
      * Chieu nao gradient nho -> lr tang (tang toc)
    - Bias correction: vi m, v bat dau = 0, can chinh lai cho cac buoc dau

    TAI SAO ADAM TOT?
    - Khong can chon lr cau ky nhu vanilla GD
    - Tu dong adapt lr cho tung parameter
    - Hoat dong tot voi hau het bai toan
    """
    x = x0.copy()
    m = [0.0] * len(x)  # First moment
    v = [0.0] * len(x)  # Second moment
    history = []

    for t in range(1, epochs + 1):
        loss = f(x)
        history.append(loss)
        grad = grad_f(x)

        for i in range(len(x)):
            m[i] = beta1 * m[i] + (1 - beta1) * grad[i]
            v[i] = beta2 * v[i] + (1 - beta2) * grad[i] ** 2

            m_hat = m[i] / (1 - beta1 ** t)
            v_hat = v[i] / (1 - beta2 ** t)

            x[i] = x[i] - lr * m_hat / (math.sqrt(v_hat) + eps)

    return x, history


# ============ TEST FUNCTIONS ============

def simple_quadratic(params):
    """f(x,y) = (x-3)^2 + (y-2)^2, minimum tai (3, 2)"""
    x, y = params
    return (x - 3) ** 2 + (y - 2) ** 2

def grad_simple_quadratic(params):
    x, y = params
    return [2 * (x - 3), 2 * (y - 2)]

def rosenbrock(params):
    """
    Rosenbrock function: f(x,y) = (a-x)^2 + b*(y-x^2)^2
    Voi a=1, b=100 -> minimum tai (1, 1)

    Day la ham test kinh dien cho optimizers vi:
    - Co narrow curved valley
    - Tim minimum rat kho
    - Gradient nho gan minimum -> hoi tu cham
    """
    x, y = params
    a, b = 1.0, 100.0
    return (a - x) ** 2 + b * (y - x ** 2) ** 2

def grad_rosenbrock(params):
    x, y = params
    a, b = 1.0, 100.0
    dx = -2 * (a - x) + b * 2 * (y - x ** 2) * (-2 * x)
    dy = b * 2 * (y - x ** 2)
    return [dx, dy]


# ============ MAIN ============
if __name__ == "__main__":
    print("=" * 60)
    print("TEST CO BAN - Gradient Descent")
    print("=" * 60)

    x0 = [0.0, 0.0]

    print("\n--- Vanilla GD ---")
    result_gd, hist_gd = gradient_descent(
        simple_quadratic, grad_simple_quadratic, x0, lr=0.1, epochs=500, verbose=True
    )
    print(f"  Result: [{result_gd[0]:.6f}, {result_gd[1]:.6f}]")
    assert abs(result_gd[0] - 3.0) < 1e-4 and abs(result_gd[1] - 2.0) < 1e-4, "GD FAILED"

    print("\n--- SGD with Momentum ---")
    result_mom, hist_mom = sgd_with_momentum(
        simple_quadratic, grad_simple_quadratic, x0, lr=0.01, momentum=0.9, epochs=500
    )
    print(f"  Result: [{result_mom[0]:.6f}, {result_mom[1]:.6f}]")
    assert abs(result_mom[0] - 3.0) < 1e-2 and abs(result_mom[1] - 2.0) < 1e-2, "Momentum FAILED"

    print("\n--- Adam ---")
    result_adam, hist_adam = adam(
        simple_quadratic, grad_simple_quadratic, x0, lr=0.5, epochs=500
    )
    print(f"  Result: [{result_adam[0]:.6f}, {result_adam[1]:.6f}]")
    assert abs(result_adam[0] - 3.0) < 1e-4 and abs(result_adam[1] - 2.0) < 1e-4, "Adam FAILED"

    # ============ BAI TAP 2: So sanh toc do hoi tu ============
    print("\n" + "=" * 60)
    print("BAI TAP 2: So sanh toc do hoi tu")
    print("=" * 60)

    def epochs_to_converge(history, threshold=1e-4):
        """Dem so epochs de loss < threshold"""
        for i, loss in enumerate(history):
            if loss < threshold:
                return i
        return len(history)

    epochs_gd = epochs_to_converge(hist_gd)
    epochs_mom = epochs_to_converge(hist_mom)
    epochs_adam = epochs_to_converge(hist_adam)

    print(f"  Vanilla GD : hoi tu sau {epochs_gd} epochs (lr=0.1)")
    print(f"  Momentum   : hoi tu sau {epochs_mom} epochs (lr=0.01, mom=0.9)")
    print(f"  Adam       : hoi tu sau {epochs_adam} epochs (lr=0.5)")
    print(f"  -> {'Adam' if epochs_adam <= min(epochs_gd, epochs_mom) else 'GD' if epochs_gd <= epochs_mom else 'Momentum'} nhanh nhat!")

    # ============ BAI TAP 3: Learning rate khac nhau ============
    print("\n" + "=" * 60)
    print("BAI TAP 3: Learning rate khac nhau")
    print("=" * 60)

    learning_rates = [0.001, 0.01, 0.1, 0.5, 0.9, 1.0, 1.1]

    print(f"\n  {'LR':<8} {'Final Loss':<15} {'Converged?':<12} {'Epochs':<10} Ghi chu")
    print("  " + "-" * 65)

    lr_results = {}
    for lr in learning_rates:
        result, history = gradient_descent(
            simple_quadratic, grad_simple_quadratic, x0, lr=lr, epochs=1000
        )
        final_loss = history[-1]
        converged = final_loss < 1e-4
        epochs_needed = epochs_to_converge(history)

        if lr < 0.05:
            note = "Qua nho -> hoi tu cham"
        elif lr > 1.0:
            note = "Qua lon -> PHAN KY (diverge)!"
        elif lr >= 0.9:
            note = "Gan gioi han -> dao dong"
        else:
            note = "Tot"

        # Kiem tra diverge
        if math.isinf(final_loss) or math.isnan(final_loss) or final_loss > 1e10:
            print(f"  {lr:<8} {'DIVERGED':<15} {'No':<12} {'N/A':<10} {note}")
        else:
            print(f"  {lr:<8} {final_loss:<15.6f} {'Yes' if converged else 'No':<12} {epochs_needed if converged else 'N/A':<10} {note}")

        lr_results[lr] = history

    # ============ BAI TAP 4: Rosenbrock function ============
    print("\n" + "=" * 60)
    print("BAI TAP 4: Rosenbrock function")
    print("=" * 60)
    print("  f(x,y) = (1-x)^2 + 100*(y-x^2)^2")
    print("  Minimum tai (1, 1), narrow curved valley -> kho optimize")

    x0_rosen = [-1.0, 1.0]
    rosen_epochs = 10000

    print(f"\n  Start: {x0_rosen}, Epochs: {rosen_epochs}")

    print("\n  --- Vanilla GD (lr=0.001) ---")
    result_gd_r, hist_gd_r = gradient_descent(
        rosenbrock, grad_rosenbrock, x0_rosen, lr=0.001, epochs=rosen_epochs
    )
    print(f"  Result: [{result_gd_r[0]:.6f}, {result_gd_r[1]:.6f}]")
    print(f"  Final loss: {hist_gd_r[-1]:.6f}")

    print("\n  --- Momentum (lr=0.001, mom=0.9) ---")
    result_mom_r, hist_mom_r = sgd_with_momentum(
        rosenbrock, grad_rosenbrock, x0_rosen, lr=0.001, momentum=0.9, epochs=rosen_epochs
    )
    print(f"  Result: [{result_mom_r[0]:.6f}, {result_mom_r[1]:.6f}]")
    print(f"  Final loss: {hist_mom_r[-1]:.6f}")

    print("\n  --- Adam (lr=0.01) ---")
    result_adam_r, hist_adam_r = adam(
        rosenbrock, grad_rosenbrock, x0_rosen, lr=0.01, epochs=rosen_epochs
    )
    print(f"  Result: [{result_adam_r[0]:.6f}, {result_adam_r[1]:.6f}]")
    print(f"  Final loss: {hist_adam_r[-1]:.6f}")

    # So sanh
    print("\n  --- So sanh tren Rosenbrock ---")
    print(f"  {'Optimizer':<15} {'Final Loss':<15} {'Distance to (1,1)'}")
    print("  " + "-" * 50)
    for name, result, hist in [
        ("Vanilla GD", result_gd_r, hist_gd_r),
        ("Momentum", result_mom_r, hist_mom_r),
        ("Adam", result_adam_r, hist_adam_r),
    ]:
        dist = math.sqrt((result[0] - 1) ** 2 + (result[1] - 1) ** 2)
        print(f"  {name:<15} {hist[-1]:<15.6f} {dist:.6f}")

    # ============ BAI TAP 1: Plot loss history ============
    print("\n" + "=" * 60)
    print("BAI TAP 1: Plot loss history voi matplotlib")
    print("=" * 60)

    try:
        import matplotlib
        matplotlib.use("Agg")  # Non-interactive backend
        import matplotlib.pyplot as plt

        output_dir = os.path.dirname(os.path.abspath(__file__))

        # --- Plot 1: So sanh 3 optimizers tren simple quadratic ---
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        ax = axes[0]
        ax.plot(hist_gd, label="Vanilla GD (lr=0.1)", linewidth=1.5)
        ax.plot(hist_mom, label="Momentum (lr=0.01)", linewidth=1.5)
        ax.plot(hist_adam, label="Adam (lr=0.5)", linewidth=1.5)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("Simple Quadratic: (x-3)^2 + (y-2)^2")
        ax.set_yscale("log")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # --- Plot 2: Rosenbrock ---
        ax = axes[1]
        ax.plot(hist_gd_r, label="Vanilla GD (lr=0.001)", linewidth=1.5)
        ax.plot(hist_mom_r, label="Momentum (lr=0.001)", linewidth=1.5)
        ax.plot(hist_adam_r, label="Adam (lr=0.01)", linewidth=1.5)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("Rosenbrock: (1-x)^2 + 100*(y-x^2)^2")
        ax.set_yscale("log")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        path1 = os.path.join(output_dir, "plot_optimizer_comparison.png")
        plt.savefig(path1, dpi=150)
        plt.close()
        print(f"  Saved: {path1}")

        # --- Plot 3: Learning rate comparison ---
        fig, ax = plt.subplots(figsize=(10, 6))
        for lr in [0.001, 0.01, 0.1, 0.5, 0.9]:
            hist = lr_results[lr]
            # Chi plot 200 epochs dau de thay ro su khac biet
            ax.plot(hist[:200], label=f"lr={lr}", linewidth=1.5)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("Learning Rate Comparison (Vanilla GD)")
        ax.set_yscale("log")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        path2 = os.path.join(output_dir, "plot_learning_rates.png")
        plt.savefig(path2, dpi=150)
        plt.close()
        print(f"  Saved: {path2}")

    except ImportError:
        print("  matplotlib chua cai. Chay: pip install matplotlib")
        print("  Cac test khac van PASSED, chi thieu plot.")

    print("\n" + "=" * 60)
    print("TAT CA TESTS PASSED!")
    print("=" * 60)


# ============ CHECKLIST ============
# Week 1-2 (Bai 02):
# [x] Implement 3 optimizers: SGD, Momentum, Adam
#     -> gradient_descent(), sgd_with_momentum(), adam()
# [x] Biet learning rate qua lon/nho thi sao
#     -> Qua lon (>1.0): loss tang vo han (diverge), model khong hoc duoc
#        Qua nho (<0.001): hoi tu cuc cham, co the ket o local minima
#        Vua phai (0.01-0.1): hoi tu tot cho vanilla GD
#        Adam tu adapt lr nen it nhay cam hon (default 0.001 la du)
# [x] Visualize duoc loss curve
#     -> plot_loss_comparison(): ve loss cua 3 optimizer tren cung 1 do thi
#     -> plot_learning_rate(): ve loss voi nhieu lr khac nhau
