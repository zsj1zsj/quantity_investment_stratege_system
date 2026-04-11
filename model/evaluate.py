import numpy as np
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    accuracy_score,
)


def evaluate_window(model, X_test, y_test, window_name: str = "") -> dict:
    """Evaluate model on a single test window."""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # Handle edge case where test set has only one class
    if len(np.unique(y_test)) < 2:
        auc = 0.5
    else:
        auc = roc_auc_score(y_test, y_prob)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "auc_roc": auc,
        "samples": len(y_test),
    }

    print(f"    {window_name}: "
          f"Acc={metrics['accuracy']:.3f} "
          f"Prec={metrics['precision']:.3f} "
          f"Rec={metrics['recall']:.3f} "
          f"F1={metrics['f1']:.3f} "
          f"AUC={metrics['auc_roc']:.3f}")

    return metrics


def print_summary(results: list[dict]) -> None:
    """Print aggregated evaluation summary across all symbols and windows."""
    for result in results:
        symbol = result["symbol"]
        metrics_list = result["metrics"]

        print(f"\n{'='*50}")
        print(f"  {symbol} - Evaluation Summary ({result['windows']} windows)")
        print(f"{'='*50}")

        for key in ["accuracy", "precision", "recall", "f1", "auc_roc"]:
            values = [m[key] for m in metrics_list]
            print(f"  {key:>10s}: mean={np.mean(values):.3f}  std={np.std(values):.3f}  "
                  f"min={np.min(values):.3f}  max={np.max(values):.3f}")

        print(f"\n  Feature Importance (top 5):")
        importance = result["feature_importance"]
        sorted_feat = sorted(importance.items(), key=lambda x: x[1], reverse=True)
        for name, score in sorted_feat[:5]:
            print(f"    {name:>25s}: {score:.4f}")
