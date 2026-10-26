"""
MLP 2 layer เขียนเองด้วย NumPy แบบ multi-label (ข่าวหนึ่งชิ้นมีได้หลายแท็ก)

    x (n_input) → [W1, b1] → ReLU → h (n_hidden) → [W2, b2] → sigmoid → p (n_labels)
    cost: binary cross-entropy = −Σ [t·log(p) + (1 − t)·log(1 − p)]
    t = เฉลยแบบ multi-hot เช่น [1, 0, 1, ...] · gradient ที่ output: dz2 = p − t

softmax / cross_entropy / one_hot เก็บไว้สำหรับแบบ single-label (ไม่ได้ใช้ใน forward/backward แล้ว)
"""

import numpy as np


def init_params(n_input, n_hidden, n_classes, seed=1, scale=None):
    # scale=None → ปรับความกว้างของการสุ่มตามจำนวน input (He initialization)
    #   ใน 02 input มีแค่ 2 ตัว ใช้ 0.5 ได้ · ข่าวมี input 5,000 ตัว ถ้ายังใช้ 0.5
    #   z1 = ผลรวมของ weight × input หลายร้อยตัว → ค่าใหญ่มาก → sigmoid/ReLU ทำงานผิดปกติตั้งแต่เริ่ม
    #   std = √(2 / n_input) ทำให้ z1 มีขนาดพอดีๆ ไม่ว่า input จะมีกี่ตัว
    rng = np.random.default_rng(seed)
    s1 = np.sqrt(2 / n_input) if scale is None else scale
    s2 = np.sqrt(1 / n_hidden) if scale is None else scale
    return {
        "W1": rng.normal(0, s1, (n_hidden, n_input)),
        "b1": np.zeros(n_hidden),
        "W2": rng.normal(0, s2, (n_classes, n_hidden)),
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


# =================================================================
# Mini-batch: ทำทีละหลายข่าวพร้อมกัน (ใช้ตอนเทรนจริง)
#   X = (B, n_input) แถวละข่าว · T = (B, n_labels) เฉลย multi-hot แถวละข่าว
#   หลักการเหมือน forward / backward ด้านบนทุกอย่าง แค่ "ซ้อน" ข่าวเป็นแถว
# =================================================================

def forward_batch(X, P):
    """
    เหมือน forward() แต่ X มีหลายแถว
    """
    Z1 = X @ P["W1"].T + P["b1"] # (B, n_input) @ (n_input, n_hidden) = (B, n_hidden) / B คือ จำนวน batch, n_input คือ จำนวนคลังคำ, n_hidden คือจำนวน neuron ใน hidden layers
    H  = relu(Z1)
    Z2 = H @ P["W2"].T + P["b2"] # (B, n_hidden) @ (n_hidden, n_labels) = (B, n_labels) / n_labels คือ จำนวนแท็ก
    probs  = sigmoid(Z2)
    return {"X": X, "Z1": Z1, "H": H, "probs": probs}

def backward_batch(T, cache, P):
    """
    เหมือน backward() แต่ทำทีละ B ข่าว แล้วเฉลี่ย gradient ของทั้ง batch
    """

    B = len(T)
    dZ2 = (cache["probs"] - T) / B  # (B, n_labels)
    dW2 = dZ2.T @ cache["H"] # (n_labels, n_hidden)
    db2 = dZ2.sum(axis=0) # (n_labels,)
    dH = dZ2 @ P["W2"]  # (B, n_hidden)  
    dZ1 = dH * relu_grad(cache["Z1"]) # (B, n_hidden)
    dW1 = dZ1.T @ cache["X"] # (n_hidden, n_input)
    db1 = dZ1.sum(axis=0) # (n_hidden,)
    return {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}


def bce_batch(probs, T, eps=1e-7):
    """binary cross-entropy เฉลี่ยต่อข่าว (ใช้ดูกราฟ loss) · ไม่ต้องแก้"""
    probs = np.clip(probs, eps, 1 - eps)
    return float(-np.mean(np.sum(T * np.log(probs) + (1 - T) * np.log(1 - probs), axis=1)))
