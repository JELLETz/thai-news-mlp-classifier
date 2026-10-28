"""
ขั้น 6: วัดผลบน test set (ครั้งเดียว กับ model ที่เลือกแล้วเท่านั้น)

ใช้ threshold ที่เลือกไว้จาก validation ตามเดิม (ไม่ได้เลือกใหม่จาก test)
แล้วเทียบ validation กับ test → ดูว่า val สูงเกินจริงแค่ไหน

ผลสรุป: reports/06_test_results.md
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy import sparse

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

from thai_news.data import LABELS, LABELS_TH  # noqa: E402
from thai_news.metrics import per_label_counts, precision_recall_f1, report  # noqa: E402

train = __import__("04_train") 


def load_run(name):
    folder = ROOT / "models" / "runs" / name if name else ROOT / "models"   # ไม่ระบุ = model ที่เลือก (ขึ้น GitHub)
    with open(folder / "config.json", encoding="utf-8") as f:
        config = json.load(f)
    P = dict(np.load(folder / "mlp.npz"))
    return config, P


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", help="ชื่อรันใน models/runs/ (ไม่ใส่ = model ใน models/)")
    config, P = load_run(parser.parse_args().run)
    name = config["name"]
    features = ROOT / "data" / f"vocab_{config['vocab_size']}"
    Y = np.load(ROOT / "data" / "Y.npz")
    global_th = config["global_threshold"]
    label_th = np.array(config["label_thresholds"])

    rows, f1_per_label = [], {}
    for split in ("validation", "test"):
        probs = train.predict(sparse.load_npz(features / f"X_{split}.npz"), P)
        macro_g, micro_g = train.scores(probs, Y[split], global_th)
        macro_l, micro_l = train.scores(probs, Y[split], label_th)
        rows.append((split, macro_g, macro_l, micro_l))
        counts = per_label_counts(Y[split], (probs > label_th).astype(int))
        f1_per_label[split] = precision_recall_f1(counts["tp"], counts["fp"], counts["fn"])[2]
        if split == "test":
            names = [LABELS_TH[label] for label in LABELS]
            print(f"=== test · {name} (threshold แยกต่อแท็ก จาก validation) ===")
            print(report(Y["test"], (probs > label_th).astype(int), names))

    write_report(name, config, rows, f1_per_label, Y["test"].sum(axis=0))


def write_report(name, config, rows, f1, support):
    lines = [
        f"model: `{name}` · คลังคำ {config['vocab_size']:,} · weight decay {config['weight_decay']:g}"
        f" · threshold เลือกจาก validation",
        "",
        "| ชุดข้อมูล | macro-F1 (threshold เดียว) | macro-F1 (แยกต่อแท็ก) | micro-F1 (แยกต่อแท็ก) |",
        "|---|---|---|---|",
    ]
    for split, macro_g, macro_l, micro_l in rows:
        label = "**test**" if split == "test" else split
        lines.append(f"| {label} | {macro_g:.3f} | {macro_l:.3f} | {micro_l:.3f} |")

    lines += ["", "| แท็ก | ข่าวใน test | F1 validation | F1 test | ต่าง |", "|---|---|---|---|---|"]
    for i in np.argsort(-support):
        lines.append(f"| {LABELS_TH[LABELS[i]]} | {int(support[i]):,} | {f1['validation'][i]:.2f}"
                     f" | {f1['test'][i]:.2f} | {f1['test'][i] - f1['validation'][i]:+.2f} |")

    table = "\n".join(lines)
    path = ROOT / "reports" / "06_test_results.md"
    path.write_text(table + "\n", encoding="utf-8")
    print("\n" + table)
    print(f"\nบันทึกไว้ที่ {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
