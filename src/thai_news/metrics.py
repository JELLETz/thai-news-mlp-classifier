"""
วัดผล multi-label: นับ TP / FP / FN แยกทีละแท็กก่อน → precision / recall / F1 ทีละแท็ก → ค่อยรวมเป็น micro / macro

Y_true, Y_pred เป็น array 0/1 ขนาด (จำนวนข่าว, จำนวนแท็ก)
    แถว = ข่าว 1 ชิ้น · คอลัมน์ = แท็ก 1 แท็ก
"""

import numpy as np


# ---------------------------------------------------------------
# นับ TP / FP / FN แยกทีละแท็ก
# ---------------------------------------------------------------
def per_label_counts(Y_true, Y_pred):
    is_tp = (Y_pred == 1) & (Y_true == 1)  
    is_fp = (Y_pred == 1) & (Y_true == 0)
    is_fn = (Y_pred == 0) & (Y_true == 1)

    return {
        "tp": np.sum(is_tp, axis=0),         
        "fp": np.sum(is_fp, axis=0),
        "fn": np.sum(is_fn, axis=0),
    }


# ---------------------------------------------------------------
# precision / recall / F1 
# ---------------------------------------------------------------
def precision_recall_f1(tp, fp, fn):

    precision = tp / np.maximum((tp + fp), 1)
    recall    = tp / np.maximum((tp + fn), 1)
    f1        = 2 * precision * recall / np.maximum((precision + recall), 1e-12)
    return (precision, recall, f1)



def macro_f1(counts):
    p, r, f1 = precision_recall_f1(counts["tp"], counts["fp"], counts["fn"])
    return np.mean(f1)


def micro_f1(counts):
    tp = np.sum(counts["tp"])   
    fp = np.sum(counts["fp"])    
    fn = np.sum(counts["fn"])  
    precision, recall, f1 = precision_recall_f1(tp, fp, fn)
    return f1


# ---------------------------------------------------------------
# threshold แยกต่อแท็ก (เลือกจาก validation เท่านั้น)
# ---------------------------------------------------------------
def tune_thresholds(probs, Y_true, candidates=np.arange(0.05, 0.96, 0.05)):
    f1_list = []
    for th in candidates:
        Y_pred = (probs > th).astype(int)
        Y_pred = (probs > th).astype(int)
        counts = per_label_counts(Y_true, Y_pred)
        _, _, f1 = precision_recall_f1(counts["tp"], counts["fp"], counts["fn"])
        f1_list.append(f1) 
    f1_table = np.array(f1_list)
    best_row = np.argmax(f1_table, axis=0)
    return candidates[best_row] 


def report(Y_true, Y_pred, label_names):
    counts = per_label_counts(Y_true, Y_pred)
    precision, recall, f1 = precision_recall_f1(counts["tp"], counts["fp"], counts["fn"])
    support = np.sum(Y_true, axis=0).astype(int)

    lines = [f"{'แท็ก':14s} {'ข่าวจริง':>8s} {'precision':>10s} {'recall':>8s} {'F1':>6s}"]
    for i in np.argsort(-support):
        lines.append(f"{label_names[i]:14s} {support[i]:8,d} {precision[i]:10.2f} {recall[i]:8.2f} {f1[i]:6.2f}")
    lines.append(f"{'macro-F1':14s} {'':8s} {'':10s} {'':8s} {macro_f1(counts):6.3f}")
    lines.append(f"{'micro-F1':14s} {'':8s} {'':10s} {'':8s} {micro_f1(counts):6.3f}")
    return "\n".join(lines)
