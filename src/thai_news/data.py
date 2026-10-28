"""
โหลดข้อมูล Prachathai-67k (ข่าวประชาไท) จาก Hugging Face

ข่าวหนึ่งชิ้นมีได้หลายแท็ก (multi-label) แต่ละแท็กเป็นคอลัมน์ 0/1
"""

import os

import pandas as pd

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

DATASET = "pythainlp/prachathai67k"
REVISION = "refs/convert/parquet"

LABELS = [
    "politics", "human_rights", "quality_of_life", "international", "social", "environment",
    "economics", "culture", "labor", "national_security", "ict", "education",
]

LABELS_TH = {
    "politics": "การเมือง", "human_rights": "สิทธิมนุษยชน", "quality_of_life": "คุณภาพชีวิต",
    "international": "ต่างประเทศ", "social": "สังคม", "environment": "สิ่งแวดล้อม",
    "economics": "เศรษฐกิจ", "culture": "วัฒนธรรม", "labor": "แรงงาน",
    "national_security": "ความมั่นคง", "ict": "ไอซีที", "education": "การศึกษา",
}


def load_splits():
    from datasets import load_dataset

    ds = load_dataset(DATASET, revision=REVISION)
    splits = {}
    for name, split in ds.items():
        df = split.to_pandas()
        df[LABELS] = df[LABELS].astype(int)
        splits[name] = df
    return splits


def multilabel(df, min_year=2009):
    """
    เตรียมข้อมูลแบบ multi-label
      - ตัดข่าวที่ไม่มีแท็กเลยทิ้ง (ไม่มีเฉลยให้เรียน)
      - เพิ่มคอลัมน์ "text" = หัวข่าว + เนื้อข่าว
    คำตอบที่ถูก (multi-hot) อยู่ใน df[LABELS] รูปแบบเป็น [1, 1, 0, 0, ...]
    """
    keep = (df["date"].str[:4] >= str(min_year)) & (df[LABELS].sum(axis=1) > 0)
    out = df[keep].copy()
    out["text"] = out["title"] + "\n" + out["body_text"]
    return out.reset_index(drop=True)
