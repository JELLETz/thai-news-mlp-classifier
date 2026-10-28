"""
ขั้น 4: เทรน MLP กับข่าวจริง
    python scripts/04_train.py                                   (ค่าเริ่มต้น: คลังคำ 5,000, hidden 128, ไม่มี weight decay)
    python scripts/04_train.py --vocab-size 20000 --weight-decay 0.001 --name vocab20k_wd

ทุก epoch:
  1. สุ่มลำดับข่าว train แล้วหยิบทีละ BATCH_SIZE ข่าว → forward_batch → backward_batch → Adam
  2. วัดผลบน validation (loss + macro/micro-F1 ที่ threshold 0.5)
  3. เก็บ weight ของ epoch ที่ val macro-F1 ดีที่สุดไว้ (early stopping)
     ถ้าไม่ดีขึ้นติดกัน PATIENCE epoch → หยุด

หลังเทรน (เลือกทุกอย่างจาก validation เท่านั้น):
  4. threshold ค่าเดียวทุกแท็ก และ threshold แยกต่อแท็ก (metrics.tune_thresholds)
  5. พิมพ์ตาราง F1 ทีละแท็ก เทียบกับ baseline
  6. บันทึกรันนี้แยกไว้ที่ models/runs/<name>/ และ reports/runs/<name>_curves.png (ไม่ทับรันอื่น)

⚠️ ไม่แตะ test set (เก็บไว้วัดครั้งเดียวตอนจบ)
"""

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
from scipy import sparse

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from thai_news.data import LABELS, LABELS_TH  # noqa: E402
from thai_news.metrics import macro_f1, micro_f1, per_label_counts, report, tune_thresholds  # noqa: E402
from thai_news.model import backward_batch, bce_batch, forward_batch, init_params  # noqa: E402
from thai_news.optimizers import adam_step, init_state  # noqa: E402

LR = 0.001
BATCH_SIZE = 64
MAX_EPOCHS = 10
PATIENCE = 2            # val macro-F1 ไม่ดีขึ้นติดกันกี่ epoch แล้วหยุด
SEED = 0

BASELINE = {"macro": 0.262, "micro": 0.434}   # จาก scripts/03_baseline.py


def load(vocab_size):
    folder = ROOT / "data" / f"vocab_{vocab_size}"
    X = {s: sparse.load_npz(folder / f"X_{s}.npz") for s in ("train", "validation")}
    Y = np.load(ROOT / "data" / "Y.npz")
    return X, {s: Y[s] for s in ("train", "validation")}, folder / "vocab.json"


def predict(X, P, batch_size=2048):
    # ทำทีละก้อน เพราะแปลง sparse เป็น array ปกติทั้งกองทีเดียวกิน RAM เยอะ
    return np.vstack([forward_batch(X[i:i + batch_size].toarray(), P)["probs"]
                      for i in range(0, X.shape[0], batch_size)])


def scores(probs, Y, thresholds):
    counts = per_label_counts(Y, (probs > thresholds).astype(int))
    return macro_f1(counts), micro_f1(counts)


def train_one_epoch(X, Y, P, state, rng, weight_decay):
    order = rng.permutation(X.shape[0])                        # สุ่มลำดับข่าวทุก epoch
    for start in range(0, len(order), BATCH_SIZE):
        idx = order[start:start + BATCH_SIZE]                  # หมายเลขข่าวใน batch นี้
        X_batch = X[idx].toarray()                             # sparse → array ปกติ (64, ขนาดคลังคำ)
        cache = forward_batch(X_batch, P)
        grads = backward_batch(Y[idx], cache, P)
        if weight_decay:
            for name in ("W1", "W2"):                          # เฉพาะ weight ไม่ทำกับ bias
                grads[name] = grads[name] + weight_decay * P[name]
        adam_step(P, grads, state, LR)


def best_global_threshold(probs, Y):
    results = [(round(float(th), 2), *scores(probs, Y, th)) for th in np.arange(0.05, 0.96, 0.05)]
    return max(results, key=lambda r: r[1])


def run(vocab_size=5000, hidden=128, weight_decay=0.0, name=None, verbose=True):
    """เทรน 1 รัน แล้วคืนผลบน validation (ใช้ทั้งจาก command line และจาก 05_experiments.py)"""
    name = name or f"vocab{vocab_size}_h{hidden}_wd{weight_decay:g}"
    log = print if verbose else (lambda *a, **k: None)
    X, Y, vocab_path = load(vocab_size)
    log(f"=== {name} · train {X['train'].shape} · validation {X['validation'].shape} ===")

    P = init_params(X["train"].shape[1], hidden, len(LABELS), seed=SEED)
    state = init_state(P)
    rng = np.random.default_rng(SEED)

    history, best = [], {"macro": -1}
    for epoch in range(1, MAX_EPOCHS + 1):
        start = time.time()
        train_one_epoch(X["train"], Y["train"], P, state, rng, weight_decay)
        train_probs = predict(X["train"][:5000], P)                  # train 5,000 ข่าวแรกพอดูแนวโน้ม
        val_probs = predict(X["validation"], P)
        val_macro, val_micro = scores(val_probs, Y["validation"], 0.5)
        history.append({"epoch": epoch, "train_loss": bce_batch(train_probs, Y["train"][:5000]),
                        "val_loss": bce_batch(val_probs, Y["validation"]), "val_macro": val_macro})
        log(f"epoch {epoch:2d} ({time.time() - start:3.0f}s)  train loss {history[-1]['train_loss']:.3f}"
            f" | val loss {history[-1]['val_loss']:.3f} · macro-F1 {val_macro:.3f} · micro-F1 {val_micro:.3f}")

        if val_macro > best["macro"]:
            best = {"macro": val_macro, "epoch": epoch, "P": {k: v.copy() for k, v in P.items()}, "probs": val_probs}
        elif epoch - best["epoch"] >= PATIENCE:
            log(f"→ val macro-F1 ไม่ดีขึ้น {PATIENCE} epoch ติด หยุดเทรน (early stopping)")
            break

    probs = best["probs"]
    global_th, global_macro, global_micro = best_global_threshold(probs, Y["validation"])
    label_th = tune_thresholds(probs, Y["validation"])
    label_macro, label_micro = scores(probs, Y["validation"], label_th)

    names = [LABELS_TH[label] for label in LABELS]
    log(f"\nใช้ weight จาก epoch {best['epoch']}")
    log(f"threshold ค่าเดียว {global_th:.2f}: macro-F1 {global_macro:.3f} · micro-F1 {global_micro:.3f}")
    log(f"threshold แยกต่อแท็ก: macro-F1 {label_macro:.3f} · micro-F1 {label_micro:.3f}")
    log("  " + " · ".join(f"{n} {t:.2f}" for n, t in zip(names, label_th)))
    log("\n=== validation (threshold แยกต่อแท็ก) ===")
    log(report(Y["validation"], (probs > label_th).astype(int), names))

    result = {
        "name": name, "vocab_size": vocab_size, "hidden": hidden, "weight_decay": weight_decay,
        "best_epoch": best["epoch"], "epochs_run": len(history),
        "global_threshold": global_th, "val_macro_global": round(global_macro, 4), "val_micro_global": round(global_micro, 4),
        "label_thresholds": [round(float(t), 2) for t in label_th],
        "val_macro_per_label": round(label_macro, 4), "val_micro_per_label": round(label_micro, 4),
        "history": history,
    }
    save(name, best["P"], result, vocab_path)
    plot(name, history, best["epoch"])
    return result


def save(name, P, result, vocab_path):
    out = ROOT / "models" / "runs" / name
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "mlp.npz", **{k: v.astype(np.float32) for k, v in P.items()})
    shutil.copy(vocab_path, out / "vocab.json")              # model กับคลังคำต้องอยู่คู่กันเสมอ
    config = {"labels": LABELS, "lr": LR, "batch_size": BATCH_SIZE,
              "features": {"mode": "binary", "max_chars": 3000}, **result}
    with open(out / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)


def plot(name, history, best_epoch):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ep = [h["epoch"] for h in history]
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 3.6))
    a.plot(ep, [h["train_loss"] for h in history], "o-", label="train")
    a.plot(ep, [h["val_loss"] for h in history], "o-", label="validation")
    a.axvline(best_epoch, color="tab:red", ls="--", lw=1, label=f"best epoch {best_epoch}")
    a.set(title=f"{name}: binary cross-entropy (lower is better)", xlabel="epoch"); a.legend()
    b.plot(ep, [h["val_macro"] for h in history], "o-", color="tab:green", label="MLP (threshold 0.5)")
    b.axhline(BASELINE["macro"], color="gray", ls=":", label="best baseline")
    b.set(title="Validation macro-F1 (higher is better)", xlabel="epoch", ylim=(0, 1)); b.legend()
    path = ROOT / "reports" / "runs" / f"{name}_curves.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocab-size", type=int, default=5000)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--name")
    args = parser.parse_args()
    result = run(args.vocab_size, args.hidden, args.weight_decay, args.name)
    print(f"\nบันทึกไว้ที่ models/runs/{result['name']}/ และ reports/runs/{result['name']}_curves.png")


if __name__ == "__main__":
    main()
