"""
ขั้น 2: ข่าว → ตัวเลข 
    python scripts/02_build_features.py

  1. โหลดข่าวตั้งแต่ปี 2009 (multi-label)
  2. ตัดคำทุกข่าว (หัวข่าว + 3,000 ตัวอักษรแรก · ใช้ CPU ทุก core พร้อมกัน)
     → เก็บผลไว้ที่ data/tokens.pkl (ครั้งแรกใช้เวลาหลายนาที ครั้งต่อไปโหลดจากไฟล์)
  3. สร้าง vocabulary จาก **train อย่างเดียว** → models/vocab.json
  4. แปลงทุกกองเป็น bag-of-words → data/features.npz

data/ อยู่ใน .gitignore (ไฟล์ใหญ่ สร้างใหม่ได้) · models/vocab.json ขึ้น GitHub ได้ (API ต้องใช้)
"""

import pickle
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy import sparse

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from thai_news.data import LABELS, load_splits, multilabel  # noqa: E402
from thai_news.features import build_vocab, clip_text, save_vocab, tokenize, vectorize_corpus  # noqa: E402

VOCAB_SIZE = 5000
MODE = "binary"     # "binary" / "count" / "log" 

DATA = ROOT / "data"
TOKENS_CACHE = DATA / "tokens.pkl"


def tokenize_all(splits):
    if TOKENS_CACHE.exists():
        print(f"โหลดผลตัดคำจาก {TOKENS_CACHE.relative_to(ROOT)}")
        with open(TOKENS_CACHE, "rb") as f:
            return pickle.load(f)

    tokens = {}
    with Pool() as pool:                     # แบ่งข่าวให้ CPU ทุก core ตัดคำพร้อมกัน
        for name, df in splits.items():
            start = time.time()
            texts = [clip_text(t) for t in df["text"]]
            tokens[name] = pool.map(tokenize, texts, chunksize=50)
            print(f"{name}: ตัดคำ {len(df):,} ข่าว ใน {time.time() - start:.0f} วินาที")

    DATA.mkdir(exist_ok=True)
    with open(TOKENS_CACHE, "wb") as f:
        pickle.dump(tokens, f)
    return tokens


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    splits = {name: multilabel(df) for name, df in load_splits().items()}
    for name, df in splits.items():
        print(f"{name:10s} {len(df):6,d} ข่าว · เฉลี่ย {df[LABELS].sum(axis=1).mean():.2f} แท็กต่อข่าว")

    tokens = tokenize_all(splits)

    vocab = build_vocab(tokens["train"], max_size=VOCAB_SIZE)       # train อย่างเดียว (กัน data leakage)
    (ROOT / "models").mkdir(exist_ok=True)
    save_vocab(vocab, ROOT / "models" / "vocab.json")
    print(f"\nvocabulary: {len(vocab):,} คำ · 10 คำแรก: {list(vocab)[:10]}")

    arrays = {}
    for name in splits:
        X = vectorize_corpus(tokens[name], vocab, MODE)
        arrays[f"X_{name}"] = X
        arrays[f"Y_{name}"] = splits[name][LABELS].to_numpy(dtype=np.float32)
        oov = 1 - X.getnnz(axis=1).mean() / np.mean([len(set(t)) for t in tokens[name]])
        print(f"{name:10s} X {X.shape} · ช่องที่ไม่ใช่ 0 เฉลี่ย {X.getnnz(axis=1).mean():.0f} ต่อข่าว"
              f" · คำที่ไม่อยู่ในคลังคำ ~{oov:.0%}")

    sparse.save_npz(DATA / "X_train.npz", arrays["X_train"])
    sparse.save_npz(DATA / "X_validation.npz", arrays["X_validation"])
    sparse.save_npz(DATA / "X_test.npz", arrays["X_test"])
    np.savez(DATA / "Y.npz", train=arrays["Y_train"], validation=arrays["Y_validation"], test=arrays["Y_test"])
    print(f"\nบันทึก features ไว้ที่ {DATA.relative_to(ROOT)}/ · vocabulary ที่ models/vocab.json")


if __name__ == "__main__":
    main()
