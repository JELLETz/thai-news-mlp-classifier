"""
MLP 2 layer เขียนเองด้วย NumPy แบบ multi-label (ข่าวหนึ่งชิ้นมีได้หลายแท็ก)

    x (n_input) → [W1, b1] → ReLU → h (n_hidden) → [W2, b2] → sigmoid → p (n_labels)
    cost: binary cross-entropy = −Σ [t·log(p) + (1 − t)·log(1 − p)]
    t = เฉลยแบบ multi-hot เช่น [1, 0, 1, ...] · gradient ที่ output: dz2 = p − t

softmax / cross_entropy / one_hot เก็บไว้สำหรับแบบ single-label (ไม่ได้ใช้ใน forward/backward แล้ว)
"""

import numpy as np


def init_params(n_input, n_hidden, n_classes, seed=1, scale=0.5):
    rng = np.random.default_rng(seed)
    return {
        "W1": rng.normal(0, scale, (n_hidden, n_input)),   
        "b1": np.zeros(n_hidden),
        "W2": rng.normal(0, scale, (n_classes, n_hidden)),  
        "b2": np.zeros(n_classes),
    }


def relu(z):
    return np.maximum(0, z)


def relu_grad(z):
    return (z > 0).astype(float)


def softmax(z):
    z_shift = z - np.max(z)
    exp_z = np.exp(z_shift)
    return exp_z / np.sum(exp_z)

def sigmoid(z):
    z = np.clip(z, -30, 30)         
    return 1 / (1 + np.exp(-z))

def cross_entropy(p, y):
    return -np.log(p[y])

def binary_cross_entropy(p, t, eps=1e-12):
    p = np.clip(p, eps, 1 - eps)    
    return -np.sum(t * np.log(p) + (1 - t) * np.log(1 - p))


def one_hot(y, n_classes):
    t = np.zeros(n_classes)
    t[y] = 1.0
    return t


def forward(x, P):
    z1 = P["W1"] @ x + P["b1"]
    h = relu(z1)
    z2 = P["W2"] @ h + P["b2"]
    p = sigmoid(z2)
    return {"x": x, "z1": z1, "h": h, "z2": z2, "p": p}


def backward(t, cache, P):
    dz2 = cache["p"] - t 
    dW2 = np.outer(dz2, cache["h"])
    db2 = dz2
    dh = P["W2"].T @ dz2
    dz1 = dh * relu_grad(cache["z1"])
    dW1 = np.outer(dz1, cache["x"])
    db1 = dz1
    return {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}


def predict_proba(x, P):
    return forward(x, P)["p"]
