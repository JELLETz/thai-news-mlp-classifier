"""
ข้อความ → ตัวเลข: tokenization → vocabulary → bag-of-words

ไฟล์นี้ใช้ทั้งตอนเทรนและตอน API รับข่าวใหม่ → ข่าวทุกชิ้นผ่านขั้นตอนเดียวกันเป๊ะ
"""

import json
from collections import Counter

import numpy as np
from pythainlp.corpus import thai_stopwords
from pythainlp.tokenize import word_tokenize

STOPWORDS = frozenset(thai_stopwords())

# ใช้แค่หัวข่าว + ต้นข่าว 3,000 ตัวอักษรแรก
#   - ข่าวมักบอกประเด็นหลักไว้ตอนต้น (lead) → หมวดข่าวยังดูออก
#   - ตอน API รับข่าวใหม่ ต้องตัดที่ 3,000 ตัวอักษรเหมือนกัน → ใช้ clip_text() ที่เดียวกัน
MAX_CHARS = 3000


def clip_text(text, max_chars=MAX_CHARS):
    return text[:max_chars]


# ---------------------------------------------------------------
# Tokenization
# ---------------------------------------------------------------
def tokenize(text):
    """
    ข้อความ → list ของ token เก็บเฉพาะ token ที่มีตัวอักษรหรือตัวเลขอย่างน้อย 1 ตัว
    """
    tokens = word_tokenize(text, engine="newmm")
    result = []                                 
    for token in tokens:                          
        token = token.lower()                      
        if any(ch.isalnum() for ch in token):      
            result.append(token)                  
    return result


# ---------------------------------------------------------------
# Vocabulary (สร้างจาก train อย่างเดียว)
# ---------------------------------------------------------------
def build_vocab(token_lists, max_size=5000, min_df=5, max_df_ratio=0.5, stopwords=STOPWORDS):
    """
    token_lists = list ของข่าว แต่ละข่าวเป็น list ของ token   เช่น [["ชาวบ้าน", "ค้าน"], ["รัฐบาล", ...], ...]
    คืนค่า dict {คำ: หมายเลขช่อง}  เช่น {"รัฐบาล": 0, "เขื่อน": 1, ...}

    ขั้นตอน:
      1. นับ document frequency (df) = แต่ละคำอยู่ใน "กี่ข่าว" (ไม่ใช่กี่ครั้ง)
           df = Counter()
           for tokens in token_lists:
               df.update(set(tokens))          ← set() ทำให้ข่าวเดียวนับคำละ 1 ครั้งพอ
      2. กรองทิ้ง:
           - คำใน stopwords
           - df < min_df                         (เจอน้อยเกิน เช่นชื่อคนที่โผล่ข่าวเดียว)
           - df > max_df_ratio × จำนวนข่าว       (อยู่เกินครึ่งของข่าว ไม่ช่วยแยกหมวด)
      3. เรียงจาก df มากไปน้อย แล้วเก็บ max_size คำแรก
           ใบ้: df.most_common() คืนค่า [(คำ, df), ...] เรียงให้แล้ว
      4. ให้หมายเลขช่อง 0, 1, 2, ... ตามลำดับ        ใบ้: enumerate()
    """
    df = Counter()
    for tokens in token_lists:
        df.update(set(tokens))  
    n_docs = len(token_lists)                 
    words = []
    vocab = {}
    for word, count in df.most_common():     
        if word in stopwords:
            continue                        
        if count < min_df:
            continue
        if count > max_df_ratio * n_docs:
            continue
        words.append(word)                     
    words = words[:max_size]         

    for i, word in enumerate(words):
        vocab[word] = i
    return vocab         


# ---------------------------------------------------------------
# Bag-of-words
# ---------------------------------------------------------------
def bag_of_words(tokens, vocab, mode="binary"):

    x = np.zeros(len(vocab), dtype=np.float32)
    for token in tokens:
        if token in vocab:
            x[vocab[token]] += 1

    if mode == "count":
        return x                            
    elif mode == "binary":
        return (x > 0).astype(np.float32)    
    elif mode == "log":
        return np.log1p(x)  


# ---------------------------------------------------------------
# บันทึก/โหลด vocabulary (ให้ API ใช้คลังคำชุดเดียวกับตอนเทรน)
# ---------------------------------------------------------------
def save_vocab(vocab, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False)


def load_vocab(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def vectorize_corpus(token_lists, vocab, mode="binary"):
    """
    ข่าวทั้งกอง → matrix ขนาด (จำนวนข่าว, len(vocab)) แบบ sparse

    ทำไม sparse: ข่าวหนึ่งชิ้นมีคำจากคลังคำแค่หลักร้อย จาก 5,000 ช่อง → เกือบทุกช่องเป็น 0
    ถ้าเก็บเต็ม 38,000 × 5,000 แบบ float32 ใช้ RAM ~760 MB · sparse เก็บเฉพาะช่องที่ไม่ใช่ 0 → เหลือไม่กี่สิบ MB
    ตอนเทรนค่อยแปลงทีละ batch เป็น array ปกติด้วย .toarray()
    """
    from scipy import sparse

    rows = [sparse.csr_matrix(bag_of_words(tokens, vocab, mode)) for tokens in token_lists]
    return sparse.vstack(rows, format="csr")
