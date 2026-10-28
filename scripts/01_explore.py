"""
ขั้น 1: สำรวจข้อมูลก่อนสร้าง model
    python scripts/01_explore.py

ผลการสำรวจนี้นำไปสู่การตัดสินใจ 2 ข้อของโปรเจค:
  1. ข่าวราวครึ่งหนึ่งมีหลายแท็ก → ทำแบบ multi-label (sigmoid)
  2. แท็กของข่าวปี 2004–2008 ผิดเพี้ยน (ทั้งปีติดแท็กเดียวกันแบบเหมารวม) → ใช้เฉพาะข่าวตั้งแต่ปี 2009

ครั้งแรกจะดาวน์โหลดข้อมูลจาก Hugging Face (ประมาณ 1–2 นาที) ครั้งต่อไปใช้ของที่เก็บไว้ในเครื่อง
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from thai_news.data import LABELS, LABELS_TH, load_splits, multilabel  # noqa: E402


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    splits = load_splits()
    train = splits["train"]

    print("=== 1. ข่าวหนึ่งชิ้นมีกี่แท็ก → ทำไมต้อง multi-label ===")
    for name, df in splits.items():
        n_tags = df[LABELS].sum(axis=1)
        print(f"{name:10s} ทั้งหมด {len(df):6,d} · แท็กเดียว {(n_tags == 1).sum():6,d}"
              f" · หลายแท็ก {(n_tags > 1).sum():6,d} · ไม่มีแท็ก {(n_tags == 0).sum():5,d}")

    print("\n=== 2. ข่าวแท็กเดียว แยกตามปี → ทำไมตัดข่าวก่อนปี 2009 ===")
    one_tag = train[train[LABELS].sum(axis=1) == 1]
    by_year = one_tag.assign(year=one_tag["date"].str[:4]).groupby("year")[LABELS].sum()
    show = ["politics", "environment", "international", "quality_of_life"]
    print("ปี    " + "".join(f"{LABELS_TH[c]:>12s}" for c in show))
    for year, row in by_year.iterrows():
        print(f"{year}  " + "".join(f"{row[c]:12,d}" for c in show))
    print("→ 2004–2005 แทบทุกข่าวแท็กเดียวเป็น 'สิ่งแวดล้อม' · 2006–2008 เป็น 'การเมือง' ทั้งหมด")
    print("  ตัวอย่างหัวข่าว 'สิ่งแวดล้อม' ปี 2005 (เนื้อหาไม่เกี่ยวกับสิ่งแวดล้อม):")
    old_env = one_tag[(one_tag["environment"] == 1) & (one_tag["date"].str[:4] == "2005")]
    for title in old_env["title"].head(3):
        print("    -", title)

    data = multilabel(train)
    print(f"\n=== 3. ข้อมูลที่ใช้จริง: train ตั้งแต่ปี 2009 = {len(data):,} ข่าว ===")
    n_tags = data[LABELS].sum(axis=1)
    print("จำนวนแท็กต่อข่าว: " + " · ".join(f"{k} แท็ก {v:,d}" for k, v in n_tags.value_counts().sort_index().items()))
    counts = data[LABELS].sum().sort_values(ascending=False)
    print("ข่าวที่มีแต่ละแท็ก:")
    for label, n in counts.items():
        bar = "█" * int(40 * n / counts.max())
        print(f"  {LABELS_TH[label]:12s} {n:6,d} ({n / len(data):5.1%})  {bar}")
    print(f"→ ไม่สมดุล: '{LABELS_TH[counts.index[0]]}' อยู่ใน {counts.iloc[0] / len(data):.0%} ของข่าว"
          f" แต่ '{LABELS_TH[counts.index[-1]]}' แค่ {counts.iloc[-1] / len(data):.0%}"
          " → accuracy หลอกได้ ต้องวัดด้วย F1 แยกทีละแท็ก")

    print("\n=== 4. ตัวอย่างหัวข่าวแต่ละแท็ก (ข่าวแท็กเดียว 2009+) ===")
    single = data[n_tags == 1]
    for label in counts.index[:4]:
        print(f"[{LABELS_TH[label]}]")
        for title in single[single[label] == 1]["title"].head(2):
            print("  -", title)

    print("\n=== 5. ความยาว (จำนวนตัวอักษร, train 2009+) ===")
    for col in ("title", "body_text"):
        lengths = data[col].str.len()
        print(f"{col:10s} median {lengths.median():6,.0f} · 90% ไม่เกิน {lengths.quantile(0.9):7,.0f}"
              f" · ยาวสุด {lengths.max():8,d}")
    print("→ ข่าวยาวมาก ตัดคำทั้งข่าวช้า → ใช้หัวข่าว + 3,000 ตัวอักษรแรก (features.clip_text)")

    plot_counts(counts, len(data), ROOT / "reports" / "01_label_distribution.png")


def plot_counts(counts, n_articles, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path.parent.mkdir(exist_ok=True)
    plt.figure(figsize=(8, 4))
    plt.bar(counts.index, counts.values / n_articles * 100, color="#534AB7")
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("% of articles with this tag")
    plt.title(f"Prachathai-67k (train, 2009+, {n_articles:,} articles): multi-label tag frequency")
    plt.savefig(path, dpi=110, bbox_inches="tight")
    print(f"\nบันทึกกราฟไว้ที่ {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
