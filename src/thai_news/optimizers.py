import numpy as np


def init_state(P):
    return {
        "t": 0,                                              # จำนวนก้าว (Adam ใช้ทำ bias correction)
        "m": {name: np.zeros_like(P[name]) for name in P},   # Adam: ค่าเฉลี่ยของ g
        "v": {name: np.zeros_like(P[name]) for name in P},   # Momentum: ความเร็ว · Adam: ค่าเฉลี่ยของ g²
    }


def sgd_step(P, grads, state, lr):
    for name in P:
        P[name] = P[name] - lr * grads[name]


def momentum_step(P, grads, state, lr, beta=0.9):
    for name in P:
        state["v"][name] = beta * state["v"][name] + grads[name]
        P[name] = P[name] - lr * state["v"][name]


def adam_step(P, grads, state, lr, beta1=0.9, beta2=0.999, eps=1e-8):
    state["t"] += 1
    t = state["t"]
    for name in P:
        state["m"][name] = beta1 * state["m"][name] + (1 - beta1) * grads[name]
        state["v"][name] = beta2 * state["v"][name] + (1 - beta2) * grads[name] ** 2
        m_hat = state["m"][name] / (1 - beta1 ** t)
        v_hat = state["v"][name] / (1 - beta2 ** t)
        P[name] = P[name] - lr * m_hat / (np.sqrt(v_hat) + eps)


OPTIMIZERS = {"sgd": sgd_step, "momentum": momentum_step, "adam": adam_step}
