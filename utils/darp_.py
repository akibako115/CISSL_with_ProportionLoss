import math
import numpy as np
import torch
from scipy import optimize

def estimate_pseudo(q_y, saved_q, num_class=10, alpha=2):
    pseudo_labels = torch.zeros(len(saved_q), num_class)
    k_probs = torch.zeros(num_class)

    for i in range(1, num_class + 1):
        i = num_class - i
        num_i = int(alpha * q_y[i])
        sorted_probs, idx = saved_q[:, i].sort(dim=0, descending=True)
        pseudo_labels[idx[: num_i], i] = 1
        k_probs[i] = sorted_probs[:num_i].sum()

    return pseudo_labels, (q_y + 1e-6) / (k_probs + 1e-6)

def f(x, a, b, c, d):
    return np.sum(a * b * np.exp(-1 * x/c)) - d

def opt_solver(probs, target_distb, num_iter=10, th=0.1, num_newton=30):
    # Tensor -> numpy
    if isinstance(probs, torch.Tensor):
        probs_np = probs.cpu().numpy()
    else:
        probs_np = probs
    if isinstance(target_distb, torch.Tensor):
        target_distb_np = target_distb.cpu().numpy()
    else:
        target_distb_np = target_distb

    N, K = probs_np.shape
    A = probs_np
    A_e = A / math.e

    # エントロピー重み
    entropy = -np.sum(A * np.log(A + 1e-6), axis=1)
    w = 1 / (entropy + 1e-6)

    r = np.ones(N)
    c = target_distb_np
    prev_Y = np.zeros(K)

    X_t = np.ones(N)
    Y_t = np.ones((1, K))

    for n in range(num_iter):
        # 正規化
        denom = np.sum(A_e * Y_t, axis=1)
        X_t = r / (denom + 1e-12)

        # Newton 更新
        Y_new = np.zeros(K)
        for i in range(K):
            try:
                # prev_Y[i] を初期値に Newton 法
                Y_new[i] = optimize.newton(
                    f, prev_Y[i], maxiter=num_newton,
                    args=(A_e[:, i], X_t, w, c[i]), tol=th
                )
            except RuntimeError:
                # 収束しなければ前回の値を利用
                Y_new[i] = prev_Y[i]
        prev_Y = Y_new
        # Y_t 更新
        Y_t = np.exp(-Y_new.reshape(1, -1) / w.reshape(-1, 1))

    # 最終 M を計算
    denom = np.sum(A_e * Y_t, axis=1)
    X_t = r / (denom + 1e-12)
    M = torch.Tensor(A_e * X_t.reshape(-1, 1) * Y_t)

    # 数値安定化のため正規化
    M /= M.sum(dim=1, keepdim=True) + 1e-12

    return M