import numpy as np

from thai_news.model import (
    backward, backward_batch, bce_batch, binary_cross_entropy, cross_entropy, forward, forward_batch,
    init_params, sigmoid, softmax,
)
from thai_news.optimizers import adam_step, init_state, momentum_step


def test_softmax_matches_hand_calculation():
    p = softmax(np.array([2.0, -1.0, 1.0]))
    assert np.allclose(p, [0.705, 0.035, 0.259], atol=0.002)
    assert np.isclose(p.sum(), 1.0)


def test_softmax_does_not_overflow():
    assert np.all(np.isfinite(softmax(np.array([1000.0, 999.0, 998.0]))))


def test_cross_entropy():
    p = np.array([0.9, 0.05, 0.05])
    assert np.isclose(cross_entropy(p, 0), 0.105, atol=0.001)


def test_sigmoid_is_independent_per_label():
    p = sigmoid(np.array([2.0, -1.0, 1.0]))
    assert np.allclose(p, [0.881, 0.269, 0.731], atol=0.001)   # ไม่ต้องรวมกันได้ 1
    assert np.all(np.isfinite(sigmoid(np.array([-1000.0, 1000.0]))))


def test_binary_cross_entropy_matches_lesson():
    t = np.array([1.0, 0.0, 1.0])
    p = np.array([0.9, 0.2, 0.3])
    assert np.isclose(binary_cross_entropy(p, t), 1.532, atol=0.001)


def test_gradient_check():
    """backward (dz2 = p − t) ต้องตรงกับ gradient ที่วัดจริงจาก binary cross-entropy"""
    rng = np.random.default_rng(0)
    P = init_params(n_input=6, n_hidden=5, n_classes=4)
    x, t, eps = rng.normal(size=6), np.array([1.0, 0.0, 1.0, 0.0]), 1e-5
    grads = backward(t, forward(x, P), P)
    for name in P:
        for idx in np.ndindex(P[name].shape):
            old = P[name][idx]
            P[name][idx] = old + eps
            c_plus = binary_cross_entropy(forward(x, P)["p"], t)
            P[name][idx] = old - eps
            c_minus = binary_cross_entropy(forward(x, P)["p"], t)
            P[name][idx] = old
            numeric = (c_plus - c_minus) / (2 * eps)
            assert np.isclose(grads[name][idx], numeric, rtol=1e-4, atol=1e-7), (name, idx)


def test_momentum_accumulates_velocity():
    P = {"w": np.array([0.0])}
    state = init_state(P)
    for _ in range(3):
        momentum_step(P, {"w": np.array([1.0])}, state, lr=1.0)
    assert np.isclose(state["v"]["w"][0], 2.71)
    assert np.isclose(P["w"][0], -5.61)


def test_adam_first_step_is_lr_regardless_of_gradient_size():
    for g in (100.0, 0.01):
        P = {"w": np.array([0.0])}
        state = init_state(P)
        adam_step(P, {"w": np.array([g])}, state, lr=0.001)
        assert state["t"] == 1
        assert np.isclose(P["w"][0], -0.001, rtol=1e-4)


def _batch():
    rng = np.random.default_rng(1)
    P = init_params(n_input=6, n_hidden=5, n_classes=4)
    X = rng.normal(size=(3, 6))
    T = np.array([[1, 0, 1, 0], [0, 0, 1, 1], [1, 1, 0, 0]], dtype=float)
    return P, X, T


def test_forward_batch_matches_forward_one_by_one():
    """แต่ละแถวของ batch ต้องได้ผลเท่ากับ forward() ทีละข่าวที่เขียนไว้แล้ว"""
    P, X, _ = _batch()
    probs = forward_batch(X, P)["probs"]
    assert probs.shape == (3, 4)
    for i in range(3):
        assert np.allclose(probs[i], forward(X[i], P)["p"])


def test_backward_batch_is_average_of_backward_one_by_one():
    """gradient ของ batch = ค่าเฉลี่ยของ backward() ทีละข่าว"""
    P, X, T = _batch()
    grads = backward_batch(T, forward_batch(X, P), P)
    for name in P:
        assert grads[name].shape == P[name].shape, f"d{name} ต้องขนาดเท่ากับ {name}"
        one_by_one = np.mean([backward(T[i], forward(X[i], P), P)[name] for i in range(3)], axis=0)
        assert np.allclose(grads[name], one_by_one), name


def test_backward_batch_gradient_check():
    P, X, T = _batch()
    eps = 1e-5
    grads = backward_batch(T, forward_batch(X, P), P)
    for name in P:
        for idx in np.ndindex(P[name].shape):
            old = P[name][idx]
            P[name][idx] = old + eps
            c_plus = bce_batch(forward_batch(X, P)["probs"], T, eps=1e-15)
            P[name][idx] = old - eps
            c_minus = bce_batch(forward_batch(X, P)["probs"], T, eps=1e-15)
            P[name][idx] = old
            numeric = (c_plus - c_minus) / (2 * eps)
            assert np.isclose(grads[name][idx], numeric, rtol=1e-4, atol=1e-7), (name, idx)
