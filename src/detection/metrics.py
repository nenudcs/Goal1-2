"""Internal metrics. Official interpolation and negative universe may differ."""
import math
from itertools import groupby


def binary_metrics(labels, scores):
    if len(labels) != len(scores) or not labels:
        raise ValueError("Metrics require nonempty, equally sized labels and scores")
    if any(y not in (0, 1) for y in labels) or set(labels) != {0, 1}:
        raise ValueError("Metrics require known positive AND negative examples")
    if not all(math.isfinite(float(s)) for s in scores):
        raise ValueError("Nonfinite prediction score")
    positive, negative = sum(labels), len(labels) - sum(labels)
    tp = fp = 0
    ap = auc = recall_at_fpr = 0.0
    previous_recall = previous_fpr = 0.0
    precision_at_recall = None
    for _, group in groupby(sorted(zip(scores, labels), reverse=True), key=lambda p: p[0]):
        values = [p[1] for p in group]
        tp += sum(values)
        fp += len(values) - sum(values)
        recall, fpr = tp / positive, fp / negative
        precision = tp / (tp + fp)
        ap += (recall - previous_recall) * precision
        auc += (fpr - previous_fpr) * (recall + previous_recall) / 2
        if fpr <= 0.1:
            recall_at_fpr = recall
        if precision_at_recall is None and recall >= 0.15:
            precision_at_recall = precision
        previous_recall, previous_fpr = recall, fpr
    return {"n": len(labels), "positive": positive, "negative": negative,
            "AP": ap, "ROC_AUC": auc, "Recall@10%FPR": recall_at_fpr,
            "Precision@15%Recall": precision_at_recall,
            "definition": "AP=sum(delta recall * precision); ties grouped; discrete operating points"}


def classification_metrics(labels, predictions):
    if not labels or len(labels) != len(predictions):
        raise ValueError("Empty or mismatched classification evaluation")
    confusion = [[0] * 3 for _ in range(3)]
    for y, p in zip(labels, predictions):
        if y not in (0, 1, 2) or p not in (0, 1, 2):
            raise ValueError("Unknown class")
        confusion[y][p] += 1
    result = {}
    for k, name in enumerate(("true", "fake", "compositing")):
        tp = confusion[k][k]
        support = sum(confusion[k])
        if support == 0:
            raise ValueError(f"Validation missing class {name}")
        predicted = sum(row[k] for row in confusion)
        precision = tp / predicted if predicted else 0.0
        recall = tp / support
        result[name] = {"precision": precision, "recall": recall, "support": support,
                        "F1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0}
    return {"classes": result, "confusion_matrix": confusion,
            "accuracy": sum(confusion[i][i] for i in range(3)) / len(labels)}


def logit(probability):
    value = min(1 - 1e-6, max(1e-6, float(probability)))
    return math.log(value / (1 - value))


def fit_calibration(labels, scores):
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    binary_metrics(labels, scores)  # Reject single-class or invalid calibration data.
    model = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000)
    model.fit(np.asarray(scores, dtype=np.float64).reshape(-1, 1), labels)
    coefficient = float(model.coef_[0, 0])
    if coefficient <= 0 or not math.isfinite(coefficient):
        raise ValueError("Calibration has no positive association with truth; inspect model/data before submission")
    return {"kind": "platt", "coefficient": coefficient,
            "intercept": float(model.intercept_[0]), "count": len(labels)}


def calibrated(calibration, score):
    z = calibration["coefficient"] * float(score) + calibration["intercept"]
    if not math.isfinite(z):
        raise ValueError("Invalid calibrated logit")
    return 1 / (1 + math.exp(-z)) if z >= 0 else math.exp(z) / (1 + math.exp(z))
