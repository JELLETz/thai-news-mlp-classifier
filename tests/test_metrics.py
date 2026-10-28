import numpy as np

from thai_news.metrics import macro_f1, micro_f1, per_label_counts, precision_recall_f1, tune_thresholds


def test_per_label_counts_example_from_docstring():
    Y_true = np.array([[1, 0], [1, 1], [0, 1]])
    Y_pred = np.array([[1, 1], [0, 1], [0, 1]])
    c = per_label_counts(Y_true, Y_pred)
    assert c["tp"].tolist() == [1, 2]
    assert c["fp"].tolist() == [0, 1]
    assert c["fn"].tolist() == [1, 0]


def test_precision_recall_f1_labor_example():
    p, r, f1 = precision_recall_f1(15, 10, 5)
    assert np.isclose(p, 0.60) and np.isclose(r, 0.75) and np.isclose(f1, 0.667, atol=0.001)


def test_precision_recall_f1_works_per_label_and_handles_zero():
    p, r, f1 = precision_recall_f1(np.array([6, 0]), np.array([2, 0]), np.array([4, 10]))
    assert np.allclose(p, [0.75, 0.0])
    assert np.allclose(r, [0.60, 0.0])
    assert np.all(np.isfinite(f1)) and np.isclose(f1[1], 0.0)


def test_macro_hides_nothing_micro_hides_rare_label():
    counts = {"tp": np.array([900, 5]), "fp": np.array([100, 5]), "fn": np.array([100, 45])}
    assert np.isclose(macro_f1(counts), (0.90 + 1 / 6) / 2, atol=0.001)    
    assert np.isclose(micro_f1(counts), 0.879, atol=0.001)


def test_tune_thresholds_picks_best_threshold_per_label():
    Y = np.array([[1, 0], [1, 1], [0, 1], [0, 0]])
    probs = np.array([[0.35, 0.60], [0.40, 0.80], [0.20, 0.75], [0.10, 0.65]])
    th = tune_thresholds(probs, Y, candidates=np.array([0.1, 0.3, 0.5, 0.7, 0.9]))
    assert np.allclose(th, [0.3, 0.7])
    assert np.isclose(macro_f1(per_label_counts(Y, (probs > th).astype(int))), 1.0)
