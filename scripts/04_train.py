"""
ขั้น 4: เทรน MLP กับข่าวจริง 

ทุก epoch:
  1. สุ่มลำดับข่าว train แล้วหยิบทีละ BATCH_SIZE ข่าว → forward_batch → backward_batch → Adam
  2. วัดผลบน validation (loss + macro/micro-F1 ที่ threshold 0.5)
  3. เก็บ weight ของ epoch ที่ val macro-F1 ดีที่สุดไว้ (early stopping)
     ถ้าไม่ดีขึ้นติดกัน PATIENCE epoch → หยุด

หลังเทรน:
  4. เลือก threshold ที่ได้ val macro-F1 สูงสุด
  5. พิมพ์ตาราง F1 ทีละแท็ก เทียบกับ baseline
  6. บันทึก model (models/mlp.npz + models/config.json) และกราฟ (reports/04_training_curves.png)

"""

import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy import sparse

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from thai_news.data import LABELS, LABELS_TH  # noqa: E402
from thai_news.metrics import macro_f1, micro_f1, per_label_counts, report  # noqa: E402
from thai_news.model import backward_batch, bce_batch, forward_batch, init_params  # noqa: E402
from thai_news.optimizers import adam_step, init_state  # noqa: E402

HIDDEN = 128
LR = 0.001
BATCH_SIZE = 64
MAX_EPOCHS = 10
PATIENCE = 2            # val macro-F1 ไม่ดีขึ้นติดกันกี่ epoch แล้วหยุด
WEIGHT_DECAY = 0.0      # ลองเปลี่ยนเป็น 1e-4 ดูทีหลัง (notes Stage 1 หัวข้อ 14)
SEED = 0

BASELINE = {"macro": 0.262, "micro": 0.434}   # จาก scripts/03_baseline.py


def load():
    X = {s: sparse.load_npz(ROOT / "data" / f"X_{s}.npz") for s in ("train", "validation")}
    Y = np.load(ROOT / "data" / "Y.npz")
    return X, {s: Y[s] for s in ("train", "validation")}


def predict(X, P, batch_size=2048):
    # ทำทีละก้อน เพราะแปลง sparse เป็น array ปกติทั้งกองทีเดียวกิน RAM เยอะ
    return np.vstack([forward_batch(X[i:i + batch_size].toarray(), P)["probs"]
                      for i in range(0, X.shape[0], batch_size)])


def evaluate(X, Y, P, threshold=0.5):
    probs = predict(X, P)
    counts = per_label_counts(Y, (probs > threshold).astype(int))
    return {"loss": bce_batch(probs, Y), "macro": macro_f1(counts), "micro": micro_f1(counts), "probs": probs}


def train_one_epoch(X, Y, P, state, rng):
    order = rng.permutation(X.shape[0])                        # สุ่มลำดับข่าวทุก epoch
    for start in range(0, len(order), BATCH_SIZE):
        idx = order[start:start + BATCH_SIZE]                  # หมายเลขข่าวใน batch นี้
        X_batch = X[idx].toarray()                             # sparse → array ปกติ (64, 5000)
        cache = forward_batch(X_batch, P)
        grads = backward_batch(Y[idx], cache, P)
        if WEIGHT_DECAY:
            for name in ("W1", "W2"):
                grads[name] = grads[name] + WEIGHT_DECAY * P[name]
        adam_step(P, grads, state, LR)


def best_threshold(probs, Y):
    results = []
    for th in np.arange(0.1, 0.91, 0.05):
        counts = per_label_counts(Y, (probs > th).astype(int))
        results.append((round(float(th), 2), macro_f1(counts), micro_f1(counts)))
    return max(results, key=lambda r: r[1]), results


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    X, Y = load()
    print(f"train {X['train'].shape} · validation {X['validation'].shape}\n")

    P = init_params(X["train"].shape[1], HIDDEN, len(LABELS), seed=SEED)
    state = init_state(P)
    rng = np.random.default_rng(SEED)

    history, best = [], {"macro": -1}
    for epoch in range(1, MAX_EPOCHS + 1):
        start = time.time()
        train_one_epoch(X["train"], Y["train"], P, state, rng)
        tr = evaluate(X["train"][:5000], Y["train"][:5000], P)         # train 5,000 ข่าวแรกพอดูแนวโน้ม
        va = evaluate(X["validation"], Y["validation"], P)
        history.append({"epoch": epoch, "train_loss": tr["loss"], "val_loss": va["loss"], "val_macro": va["macro"]})
        print(f"epoch {epoch:2d} ({time.time() - start:3.0f}s)  train loss {tr['loss']:.3f}"
              f" | val loss {va['loss']:.3f} · macro-F1 {va['macro']:.3f} · micro-F1 {va['micro']:.3f}")

        if va["macro"] > best["macro"]:
            best = {"macro": va["macro"], "epoch": epoch, "P": {k: v.copy() for k, v in P.items()}, "probs": va["probs"]}
        elif epoch - best["epoch"] >= PATIENCE:
            print(f"→ val macro-F1 ไม่ดีขึ้น {PATIENCE} epoch ติด หยุดเทรน (early stopping)")
            break

    print(f"\nใช้ weight จาก epoch {best['epoch']} (val macro-F1 ดีที่สุด)")
    (th, macro, micro), all_th = best_threshold(best["probs"], Y["validation"])
    print("threshold → macro-F1: " + " · ".join(f"{t:.2f}→{m:.3f}" for t, m, _ in all_th[::2]))
    print(f"เลือก threshold = {th:.2f}\n")

    names = [LABELS_TH[label] for label in LABELS]
    print("=== validation ===")
    print(report(Y["validation"], (best["probs"] > th).astype(int), names))
    print(f"\nเทียบ baseline: macro-F1 {BASELINE['macro']:.3f} → {macro:.3f} · micro-F1 {BASELINE['micro']:.3f} → {micro:.3f}")

    save(best["P"], th, best["epoch"], macro, micro)
    plot(history, best["epoch"])


def save(P, threshold, epoch, macro, micro):
    models = ROOT / "models"
    models.mkdir(exist_ok=True)
    np.savez(models / "mlp.npz", **{k: v.astype(np.float32) for k, v in P.items()})
    config = {
        "labels": LABELS, "threshold": threshold, "hidden": HIDDEN, "lr": LR, "batch_size": BATCH_SIZE,
        "weight_decay": WEIGHT_DECAY, "best_epoch": epoch, "val_macro_f1": round(macro, 4),
        "val_micro_f1": round(micro, 4), "features": {"vocab": "vocab.json", "mode": "binary", "max_chars": 3000},
    }
    with open(models / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    print("\nบันทึก model ไว้ที่ models/mlp.npz + models/config.json")


def plot(history, best_epoch):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ep = [h["epoch"] for h in history]
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 3.6))
    a.plot(ep, [h["train_loss"] for h in history], "o-", label="train")
    a.plot(ep, [h["val_loss"] for h in history], "o-", label="validation")
    a.axvline(best_epoch, color="tab:red", ls="--", lw=1, label=f"best epoch {best_epoch}")
    a.set(title="Binary cross-entropy (lower is better)", xlabel="epoch"); a.legend()
    b.plot(ep, [h["val_macro"] for h in history], "o-", color="tab:green", label="MLP (threshold 0.5)")
    b.axhline(BASELINE["macro"], color="gray", ls=":", label="best baseline")
    b.set(title="Validation macro-F1 (higher is better)", xlabel="epoch", ylim=(0, 1)); b.legend()
    path = ROOT / "reports" / "04_training_curves.png"
    plt.savefig(path, dpi=110, bbox_inches="tight")
    print(f"บันทึกกราฟไว้ที่ {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
