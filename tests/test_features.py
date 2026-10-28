import numpy as np

from thai_news.features import bag_of_words, build_vocab, tokenize

VOCAB = {"รัฐบาล": 0, "เขื่อน": 1, "ชาวบ้าน": 2, "เลือกตั้ง": 3, "ค้าน": 4}


def test_tokenize_drops_spaces_and_punctuation():
    tokens = tokenize('ชาวบ้าน "ค้าน" เขื่อน!')
    assert tokens == ["ชาวบ้าน", "ค้าน", "เขื่อน"]


def test_tokenize_lowercases_english():
    assert tokenize("iPhone") == tokenize("iphone")


def test_bag_of_words_count_matches_lesson():
    news = ["ชาวบ้าน", "ค้าน", "เขื่อน", "ชาวบ้าน", "ร้องเรียน", "รัฐบาล", "เขื่อน", "เขื่อน"]
    assert bag_of_words(news, VOCAB, mode="count").tolist() == [1, 3, 2, 0, 1]


def test_bag_of_words_skips_oov_word():
    news = ["เลือกตั้ง", "รัฐบาล", "ประกาศ", "เลือกตั้ง", "ค้าน"]
    assert bag_of_words(news, VOCAB, mode="count").tolist() == [1, 0, 0, 2, 1]


def test_bag_of_words_binary_and_log():
    news = ["เขื่อน", "เขื่อน", "เขื่อน", "ค้าน"]
    assert bag_of_words(news, VOCAB, mode="binary").tolist() == [0, 1, 0, 0, 1]
    assert np.allclose(bag_of_words(news, VOCAB, mode="log"), np.log1p([0, 3, 0, 0, 1]))


def test_build_vocab_filters_and_orders_by_document_frequency():
    docs = [
        ["ที่", "เขื่อน", "ชาวบ้าน"],
        ["ที่", "เขื่อน", "เขื่อน"],            # นับ df ของ "เขื่อน" แค่ 1 ในข่าวนี้
        ["ที่", "ชาวบ้าน", "รัฐบาล"],
        ["ที่", "เขื่อน", "เลือกตั้ง"],
    ]
    vocab = build_vocab(docs, max_size=10, min_df=2, max_df_ratio=0.9, stopwords=frozenset())
    # "ที่" อยู่ 4/4 ข่าว (> 90%) → ตัด · "รัฐบาล", "เลือกตั้ง" อยู่ข่าวเดียว (< min_df) → ตัด
    # "เขื่อน" df = 3 มากกว่า "ชาวบ้าน" df = 2 → ได้ช่อง 0
    assert vocab == {"เขื่อน": 0, "ชาวบ้าน": 1}


def test_build_vocab_removes_stopwords_and_respects_max_size():
    docs = [["และ", "เขื่อน", "ชาวบ้าน"]] * 3
    vocab = build_vocab(docs, max_size=1, min_df=1, max_df_ratio=1.0, stopwords=frozenset({"และ"}))
    assert len(vocab) == 1 and "และ" not in vocab
