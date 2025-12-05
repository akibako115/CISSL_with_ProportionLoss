import torch
import torch.utils.data as data
import numpy as np
import math
from collections import defaultdict
import os

def compute_sampling_rates_from_labeled_counts(counts, alpha: float):
    """μ_l = (N_{L+1-l}/N_1)^α をクラス頻度降順の鏡映ランクで計算"""
    L = len(counts)
    idx_sorted = sorted(range(L), key=lambda i: counts[i], reverse=True)
    counts_sorted = [counts[i] for i in idx_sorted]
    rank = {cls: r for r, cls in enumerate(idx_sorted)}
    N1 = float(max(1, counts_sorted[0]))
    rates = [0.0] * L
    for c in range(L):
        r = rank[c]
        mirror_r = L - 1 - r
        Nm = float(max(1, counts_sorted[mirror_r]))
        mu = (Nm / N1) ** alpha
        rates[c] = float(max(0.0, min(1.0, mu)))
    return rates

def labeled_hist_from_dataset(ds, num_classes: int, device='cpu'):
    """(inputs, targets, idx) を返す dataset を一周してラベルヒストグラムを数える"""
    loader = data.DataLoader(ds, batch_size=512, shuffle=False, drop_last=False, num_workers=0)
    hist = np.zeros(num_classes, dtype=np.int64)
    with torch.no_grad():
        for batch in loader:
            # 既存ラベルドは (x, y, idx) 形式を前提
            _, y, _ = batch
            y = torch.as_tensor(y).cpu().numpy()
            hist += np.bincount(y, minlength=num_classes)
    return hist.tolist()

class PseudoLabeledFromU(data.Dataset):
    """未ラベルdataset上の indices を、弱ビューu1を使った『ラベル付き』として供給する"""
    def __init__(self, u_dataset, indices, labels):
        self.u_dataset = u_dataset
        self.indices = list(indices)
        self.labels = list(map(int, labels))
        assert len(self.indices) == len(self.labels)
    def __len__(self): return len(self.indices)
    def __getitem__(self, i):
        idx = self.indices[i]
        sample = self.u_dataset[idx]
        views = sample[0]
        u1 = views[0] if isinstance(views, (list, tuple)) else views
        return u1, self.labels[i], idx

@torch.no_grad()
def crest_select(args, model_ema, u_dataset, labeled_dataset, num_classes: int, t_cur: float = None):
    """
    EMAでUを推論 →（任意）PDAで分布整列 → τで閾値 → クラス別に μ 割合で上位採用
    デバッグ用: 選ばれた candidate の global/local index と confidence を表示
    """
    model_ema.eval()

    # 1) μ を計算（鏡映ランク）
    labeled_counts = labeled_hist_from_dataset(labeled_dataset, num_classes)
    rates = compute_sampling_rates_from_labeled_counts(
        labeled_counts, alpha=getattr(args, "crest_alpha", 1/3)
    )

    # 2) p(y)^t をGPUで準備（t は進行温度）
    if t_cur is None:
        t_cur = getattr(args, "crest_t_min", 0.5)
    p_y = torch.tensor(labeled_counts, dtype=torch.float32, device=args.device)
    p_y = p_y / p_y.sum().clamp_min(1e-12)
    p_y_t = p_y.pow(t_cur)
    p_y_t = p_y_t / p_y_t.sum().clamp_min(1e-12)

    if not hasattr(crest_select, "_pt_ma"):
        crest_select._pt_ma = torch.ones(num_classes, device=args.device) / num_classes
    momentum = 0.9

    loader = data.DataLoader(
        u_dataset,
        batch_size=args.batch_size * args.mu,
        shuffle=False,
        drop_last=False,
        num_workers=min(16, os.cpu_count() or 8),
        pin_memory=True,
        persistent_workers=True
    )

    class_to_cands = defaultdict(list)

    for batch in loader:
        views, _gt, idx = batch
        u1 = views[0] if isinstance(views, (list, tuple)) else views
        u1 = u1.to(args.device, non_blocking=True)
        out = model_ema(u1)
        logits = out[0] if isinstance(out, (tuple, list)) else out
        probs = torch.softmax(logits, dim=-1)

        p_tilde = probs.mean(0).clamp_min(1e-12)
        crest_select._pt_ma = momentum * crest_select._pt_ma + (1.0 - momentum) * p_tilde
        scale = (p_y_t / crest_select._pt_ma.clamp_min(1e-12)).view(1, -1)
        probs = (probs * scale)
        probs = probs / probs.sum(1, keepdim=True).clamp_min(1e-12)

        conf, pred = probs.max(dim=-1)
        conf = conf.detach().cpu().tolist()
        pred = pred.detach().cpu().tolist()

        idx_local = map_to_base_local_indices(idx, u_dataset)

        tau = float(getattr(args, "crest_tau", getattr(args, "tau", 0.95)))
        for j in range(len(idx_local)):
            if conf[j] >= tau:
                c = int(pred[j])
                class_to_cands[c].append({
                    'global_idx': int(idx[j]),
                    'local_idx': int(idx_local[j]),
                    'conf': conf[j],
                    'pred': c
                })

    # 4) クラス別に μ 割合で上位採用 & ログ出力
    chosen_idx, chosen_lbl = [], []
    total_candidates = sum(len(class_to_cands.get(c, [])) for c in range(num_classes))
    print(f"Total candidates above threshold: {total_candidates}")

    for c in range(num_classes):
        cand = class_to_cands.get(c, [])
        if not cand:
            continue
        cand.sort(key=lambda x: x['conf'], reverse=True)
        k = int(math.ceil(rates[c] * len(cand)))
        pick = cand[:k]

        chosen_idx.extend([p['local_idx'] for p in pick])
        chosen_lbl.extend([p['pred'] for p in pick])

        # デバッグログ: 選ばれた candidate の情報
        confs = [p['conf'] for p in pick]
        avg_conf = sum(confs) / len(confs) if confs else 0.0
        print(f"Class {c}: {len(cand)} candidates, rate={rates[c]:.3f}, selected={k}, avg_conf={avg_conf:.4f}")
        for p in pick:
            print(f"  -> global_idx {p['global_idx']}, local_idx {p['local_idx']}, conf {p['conf']:.4f}")

    print(f"Total selected: {len(chosen_idx)}")
    return chosen_idx, chosen_lbl, rates, {c: len(class_to_cands.get(c, [])) for c in range(num_classes)}


def map_to_base_local_indices(idx_batch, u_dataset):
    """DataLoader から返ってきた idx（意味は実装依存）を
    常に base_ds(train_unlabeled_set) のローカル添字に正規化して返す。
    """
    # リスト化
    idx_list = idx_batch.tolist() if hasattr(idx_batch, "tolist") else list(idx_batch)

    # base_ds と必要ならグローバル→ローカル写像を用意
    base_ds = u_dataset.dataset if isinstance(u_dataset, data.Subset) else u_dataset
    global2local = None
    if hasattr(base_ds, "orig_indices"):
        arr = np.asarray(base_ds.orig_indices)
        global2local = {int(g): int(i) for i, g in enumerate(arr)}

    # 1) グローバルIDとみなせるなら最優先で map
    if global2local is not None and all(int(x) in global2local for x in idx_list):
        return [global2local[int(x)] for x in idx_list]

    # 2) Subset ローカル（= u_dataset の位置）なら base ローカルへ
    if isinstance(u_dataset, data.Subset) and all(0 <= int(x) < len(u_dataset) for x in idx_list):
        base_idx = u_dataset.indices  # これ自体が base ローカル添字
        return [int(base_idx[int(x)]) for x in idx_list]

    # 3) すでに base ローカルならそのまま
    if all(0 <= int(x) < len(base_ds) for x in idx_list):
        return [int(x) for x in idx_list]

    # 4) どれでもない（データセット実装が特殊）→ 例外にして気づけるように
    raise ValueError(
        "Cannot interpret indices from dataset: not global, not subset-local, not base-local."
    )