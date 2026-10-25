"""
ขั้น 3: Baseline — model ที่ง่ายที่สุดได้คะแนนเท่าไหร่ 

ไม่ได้เทรนอะไรเลย แค่ "ทาย" ด้วยกฎง่ายๆ แล้ววัดผลบน validation
MLP ที่จะเทรนต่อไปต้องชนะตัวเลขเหล่านี้ให้ขาด ไม่งั้นความซับซ้อนที่เพิ่มเข้าไปไม่คุ้ม

  A. ทายแท็กที่พบบ่อยที่สุดแท็กเดียวทุกข่าว (การเมือง)
  B. ทาย 2 แท็กที่พบบ่อยที่สุดทุกข่าว (ข่าวเฉลี่ยมี ~2 แท็ก)
  C. ทายทุกแท็กทุกข่าว (recall = 100% แน่นอน แล้ว precision ล่ะ?)
"""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from thai_news.data import LABELS, LABELS_TH  # noqa: E402
from thai_news.metrics import macro_f1, micro_f1, per_label_counts, report  # noqa: E402


def predict_top_k(train_Y, n_rows, k):
    top = np.argsort(-train_Y.sum(axis=0))[:k]          # แท็กที่พบบ่อยสุด k แท็กใน train
    Y_pred = np.zeros((n_rows, train_Y.shape[1]), dtype=int)
    Y_pred[:, top] = 1                                  # ทุกข่าวทายแท็กเดิม
    return Y_pred


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    Y = np.load(ROOT / "data" / "Y.npz")
    train_Y, val_Y = Y["train"], Y["validation"]
    names = [LABELS_TH[label] for label in LABELS]

    baselines = {
        "A. การเมืองทุกข่าว": predict_top_k(train_Y, len(val_Y), k=1),
        "B. 2 แท็กยอดนิยมทุกข่าว": predict_top_k(train_Y, len(val_Y), k=2),
        "C. ทุกแท็กทุกข่าว": np.ones_like(val_Y, dtype=int),
    }

    summary = []
    for name, Y_pred in baselines.items():
        print(f"\n=== {name} (validation) ===")
        print(report(val_Y, Y_pred, names))
        counts = per_label_counts(val_Y, Y_pred)
        summary.append((name, macro_f1(counts), micro_f1(counts)))

    print("\n=== สรุป: ตัวเลขที่ MLP ต้องชนะ ===")
    for name, macro, micro in summary:
        print(f"{name:26s} macro-F1 {macro:.3f} · micro-F1 {micro:.3f}")


if __name__ == "__main__":
    main()
