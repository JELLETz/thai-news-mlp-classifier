"""
ขั้น 5: การทดลอง (วัดบน validation ทั้งหมด · ยังไม่แตะ test)

รันทุกการตั้งค่าใน EXPERIMENTS ต่อกัน (~25 นาที) แล้วสรุปเป็นตารางเดียว
  - สร้าง features ของคลังคำขนาดที่ยังไม่มีให้อัตโนมัติ (ใช้ผลตัดคำที่บันทึกไว้ ~1 นาทีต่อขนาด)
  - รันที่เคยเสร็จแล้วจะข้าม (อ่านผลจาก models/runs/<name>/config.json) → หยุดกลางทางแล้วรันต่อได้
  - ผลแต่ละรันอยู่ที่ models/runs/<name>/ และ reports/runs/<name>_curves.png (ไม่ทับกัน)

ผลสรุป: reports/05_experiments.md + reports/05_val_loss.png
"""

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

train = __import__("04_train")  

WEIGHT_DECAY = 0.001

# ชื่อ/คลังคำ/hidden/weight decay  
EXPERIMENTS = [
    ("v1_baseline_mlp",      5000,   128,   0.0), # ตัวตั้งต้น (ผลเดิม: overfit ตั้งแต่ epoch 3)
    ("weight_decay",         5000,   128,   WEIGHT_DECAY), 
    ("vocab_10k",            10000,  128,   0.0),     
    ("vocab_20k",            20000,  128,   0.0),
    ("vocab_20k_wd",         20000,  128,   WEIGHT_DECAY),  
]


def ensure_features(vocab_size):
    if not (ROOT / "data" / f"vocab_{vocab_size}" / "X_train.npz").exists():
        print(f"สร้าง features คลังคำ {vocab_size:,} คำ ...")
        subprocess.run([sys.executable, str(ROOT / "scripts" / "02_build_features.py"),
                        "--vocab-size", str(vocab_size)], check=True)


def load_existing(name):
    path = ROOT / "models" / "runs" / name / "config.json"
    if path.exists():
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return None


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    results = []
    for name, vocab_size, hidden, wd in EXPERIMENTS:
        existing = load_existing(name)
        if existing:
            print(f"ข้าม {name} (เทรนไว้แล้ว)")
            results.append(existing)
            continue
        ensure_features(vocab_size)
        results.append(train.run(vocab_size, hidden, wd, name))
        print()

    write_table(results)
    plot_val_loss(results)


def write_table(results):
    base = train.BASELINE
    best = max(results, key=lambda r: r["val_macro_per_label"])
    lines = [
        "| การตั้งค่า | คลังคำ | weight decay | epoch ที่ดีสุด | macro-F1 (threshold เดียว) | macro-F1 (แยกต่อแท็ก) | micro-F1 (แยกต่อแท็ก) |",
        "|---|---|---|---|---|---|---|",
        f"| baseline: ทุกแท็กทุกข่าว | – | – | – | {base['macro']:.3f} | – | {base['micro']:.3f}* |",
    ]
    for r in results:
        mark = " ⭐" if r is best else ""
        lines.append(f"| {r['name']}{mark} | {r['vocab_size']:,} | {r['weight_decay']:g} | {r['best_epoch']} / {r['epochs_run']}"
                     f" | {r['val_macro_global']:.3f} | **{r['val_macro_per_label']:.3f}** | {r['val_micro_per_label']:.3f} |")
    lines += ["", "ทุกค่าวัดบน **validation** · * micro-F1 ของ baseline ที่ดีที่สุด (2 แท็กยอดนิยม) · ⭐ = macro-F1 สูงสุด"]
    table = "\n".join(lines)
    path = ROOT / "reports" / "05_experiments.md"
    path.write_text(table + "\n", encoding="utf-8")
    print(table)
    print(f"\nบันทึกตารางไว้ที่ {path.relative_to(ROOT)}")


def plot_val_loss(results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 3.8))
    for r in results:
        ep = [h["epoch"] for h in r["history"]]
        a.plot(ep, [h["val_loss"] for h in r["history"]], "o-", label=r["name"])
        b.plot(ep, [h["val_macro"] for h in r["history"]], "o-", label=r["name"])
    a.set(title="Validation loss (lower is better)", xlabel="epoch")
    b.set(title="Validation macro-F1 @ 0.5 (higher is better)", xlabel="epoch")
    a.legend(fontsize=8); b.legend(fontsize=8)
    path = ROOT / "reports" / "05_val_loss.png"
    plt.savefig(path, dpi=110, bbox_inches="tight")
    print(f"บันทึกกราฟไว้ที่ {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
