"""
Module 5 Week B — Applied Lab: Trees & Ensembles

Build and evaluate decision tree and random forest models on the Petra
Telecom churn dataset. Handle class imbalance honestly (class_weight as an
operating-point tool at a fixed threshold), evaluate with PR-AUC and
calibration, and demonstrate what tree models capture that linear models
cannot.

Complete the 12 functions below. See the lab guide for task-by-task detail.
Run with:  python lab_trees.py
Tests:     pytest tests/ -v
"""

import os

# Use a non-interactive matplotlib backend so plots save cleanly in CI
# and on headless environments.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
from sklearn.calibration import CalibrationDisplay
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (PrecisionRecallDisplay, average_precision_score,
                             classification_report, f1_score, recall_score,precision_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.inspection import permutation_importance



NUMERIC_FEATURES = ["tenure", "monthly_charges", "total_charges",
                    "num_support_calls", "senior_citizen",
                    "has_partner", "has_dependents", "contract_months"]


def load_and_split(filepath="data/telecom_churn.csv", random_state=42):
    """Load dataset, select numeric features, and split into train/test sets.

    Args:
        filepath: Path to the CSV file.

    Returns:
        Tuple of (X_train, X_test, y_train, y_test).
    """
    df = pd.read_csv(filepath)

    X = df[NUMERIC_FEATURES]
    y = df["churned"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42
    )

    return X_train, X_test, y_train, y_test


def build_decision_tree(X_train, y_train, max_depth=5, random_state=42):
    """Train a DecisionTreeClassifier.

    Args:
        max_depth: Maximum tree depth (None means unconstrained).
        random_state: Random seed.

    Returns:
        Fitted DecisionTreeClassifier.
    """
    # TODO: Fit a DecisionTreeClassifier with the given max_depth and seed.
    model = DecisionTreeClassifier(max_depth=max_depth,random_state=random_state)
    model.fit(X_train, y_train)
    return model


def compute_ece(y_true, y_prob, n_bins=10):
    """Compute Expected Calibration Error (ECE)."""
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)

    order = np.argsort(y_prob)
    y_true = y_true[order]
    y_prob = y_prob[order]

    n = len(y_true)
    bins = np.array_split(np.arange(n), n_bins)

    ece = 0.0

    for bin_idx in bins:
        if len(bin_idx) == 0:
            continue

        mean_predicted_prob = np.mean(y_prob[bin_idx])
        fraction_actual_positive = np.mean(y_true[bin_idx])

        ece += abs(mean_predicted_prob - fraction_actual_positive) * (len(bin_idx) / n)

    return ece

def compare_dt_calibration(X_train, X_test, y_train, y_test):
    tree_unbounded = build_decision_tree(
        X_train, y_train, max_depth=None, random_state=42
    )
    y_prob_unbounded = tree_unbounded.predict_proba(X_test)[:, 1]
    ece_unbounded = compute_ece(y_test, y_prob_unbounded)

    tree_depth_5 = build_decision_tree(
        X_train, y_train, max_depth=5, random_state=42
    )
    y_prob_depth_5 = tree_depth_5.predict_proba(X_test)[:, 1]
    ece_depth_5 = compute_ece(y_test, y_prob_depth_5)

    return {
        "ece_unbounded": ece_unbounded,
        "ece_depth_5": ece_depth_5
    }


def build_random_forest(X_train, y_train, n_estimators=100, max_depth=10,
                        class_weight=None, random_state=42):
    """Train a RandomForestClassifier.

    Args:
        class_weight: None for default, 'balanced' to reweight the loss
            so minority-class samples count more during training.
        random_state: Random seed.

    Returns:
        Fitted RandomForestClassifier.
    """
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        class_weight=class_weight,
        random_state=random_state
    )
    model.fit(X_train, y_train)
    return model


def get_feature_importances(model, feature_names):
    """Return a dict of feature_name -> importance, sorted descending."""
    feature_importances = zip(feature_names, model.feature_importances_)
    sorted_importances = sorted(feature_importances, key=lambda i: i[1], reverse=True)
    return dict(sorted_importances)


def evaluate_recall_at_threshold(model, X_test, y_test, threshold=0.5):
    """Recall for class 1 at a specified decision threshold.

    Standard .predict() uses threshold 0.5. Passing a different threshold
    lets you observe how recall responds to operating-point choice — which
    is what `class_weight='balanced'` effectively shifts.

    Returns:
        Recall as a float in [0, 1].
    """
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)
    return recall_score(y_test, y_pred, zero_division=0)


def compute_pr_auc(model, X_test, y_test):
    """PR-AUC (average precision) for the positive class.

    Threshold-independent: measures the model's ability to rank positives
    above negatives across all thresholds. Unlike recall at a specific
    threshold, PR-AUC does not change when you apply class_weight='balanced'
    in a way that merely shifts predicted probabilities uniformly — the
    ranking is what matters.

    Returns:
        Float in [0, 1].
    """
    y_prob = model.predict_proba(X_test)[:, 1]
    return average_precision_score(y_test, y_prob)


def plot_pr_curves(rf_default, rf_balanced, X_test, y_test, output_path):
    """Plot PR curves for both RF models on the same axes and save as PNG.

    Args:
        output_path: Destination path (e.g., 'results/pr_curves.png').
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    fig, ax = plt.subplots()

    PrecisionRecallDisplay.from_estimator(
        rf_default, X_test, y_test, ax=ax, name="RF Default"
    )
    PrecisionRecallDisplay.from_estimator(
        rf_balanced, X_test, y_test, ax=ax, name="RF Balanced"
    )

    ax.set_title("Precision-Recall Curves")
    plt.savefig(output_path)
    plt.close(fig)


def plot_calibration_curves(rf_default, rf_balanced, X_test, y_test, output_path):
    """Plot calibration curves for both RF models and save as PNG."""
    fig, ax = plt.subplots()

    CalibrationDisplay.from_estimator(
        rf_default, X_test, y_test, n_bins=10, ax=ax, name="RF Default"
    )
    CalibrationDisplay.from_estimator(
        rf_balanced, X_test, y_test, n_bins=10, ax=ax, name="RF Balanced"
    )

    ax.set_title("Calibration Curves")
    plt.savefig(output_path)
    plt.close(fig)


def build_logistic_regression(X_train_scaled, y_train, random_state=42):
    """Train a LogisticRegression baseline on scaled features.

    Linear models need their inputs on a common scale, otherwise features
    with larger numeric ranges (total_charges ~ 0-9000) swamp features with
    smaller ranges (binary indicators at 0/1). Apply StandardScaler to the
    training features BEFORE calling this function.

    Returns:
        Fitted LogisticRegression(max_iter=1000).
    """
    model = LogisticRegression(max_iter=1000, random_state=random_state)
    model.fit(X_train_scaled, y_train)
    return model


def find_tree_vs_linear_disagreement(rf_model, lr_model, X_test_raw,
                                     X_test_scaled, y_test, feature_names,
                                     min_diff=0.15):
    """Find ONE test sample where RF and LR predicted probabilities differ most.

    The tree-vs-linear capability demonstration. The random forest can
    capture feature interactions, non-monotonic relationships, and threshold
    effects that a linear model cannot express with per-feature coefficients.
    Finding a sample where the two models disagree — and explaining WHY in
    structural terms — is the lab's evidence that trees have capabilities
    linear models don't, regardless of aggregate PR-AUC.

    Args:
        rf_model: Trained RF (takes raw features).
        lr_model: Trained LR (takes scaled features).
        X_test_raw: Unscaled test features (what RF consumes).
        X_test_scaled: Scaled test features (what LR consumes).
        y_test: True labels for the test set.
        feature_names: List of feature name strings.
        min_diff: Minimum probability difference to count as disagreement.

    Returns:
        Dict with keys:
          - sample_idx (int): test-set row index of the selected sample
          - feature_values (dict): {name: value} for the sample's features
          - rf_proba (float): RF's predicted P(churn=1)
          - lr_proba (float): LR's predicted P(churn=1)
          - prob_diff (float): |rf_proba - lr_proba|
          - true_label (int): 0 or 1
    """
    rf_proba = rf_model.predict_proba(X_test_raw)[:, 1]
    lr_proba = lr_model.predict_proba(X_test_scaled)[:, 1]

    prob_diff = np.abs(rf_proba - lr_proba)

    max_idx = np.argmax(prob_diff)

    if prob_diff[max_idx] < min_diff:
        return None

    if hasattr(X_test_raw, "iloc"):
        row_values = X_test_raw.iloc[max_idx]
        feature_values = dict(zip(feature_names, row_values))
    else:
        feature_values = dict(zip(feature_names, X_test_raw[max_idx]))

    if hasattr(y_test, "iloc"):
        true_label = y_test.iloc[max_idx]
    else:
        true_label = y_test[max_idx]

    return {
        "sample_idx": int(max_idx),
        "feature_values": feature_values,
        "rf_proba": float(rf_proba[max_idx]),
        "lr_proba": float(lr_proba[max_idx]),
        "prob_diff": float(prob_diff[max_idx]),
        "true_label": int(true_label)
    }


def threshold_sweep(rf_balanced, X_test, y_test, output_path):
    """Sweep thresholds from 0.1 to 0.9 and plot precision/recall/F1."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    y_prob = rf_balanced.predict_proba(X_test)[:, 1]
    thresholds = np.arange(0.1, 0.91, 0.05)

    precisions = []
    recalls = []
    f1s = []

    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        precisions.append(precision_score(y_test, y_pred, zero_division=0))
        recalls.append(recall_score(y_test, y_pred, zero_division=0))
        f1s.append(f1_score(y_test, y_pred, zero_division=0))

    best_f1_idx = int(np.argmax(f1s))
    best_f1_threshold = float(thresholds[best_f1_idx])
    best_f1_value = float(f1s[best_f1_idx])

    recall_80_threshold = None
    for t, r in zip(thresholds, recalls):
        if r >= 0.80:
            recall_80_threshold = float(t)
            break

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(thresholds, precisions, marker="o", label="Precision")
    ax.plot(thresholds, recalls, marker="o", label="Recall")
    ax.plot(thresholds, f1s, marker="o", label="F1")
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Score")
    ax.set_title("Threshold Sweep — Balanced Random Forest")
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close(fig)

    return {
        "thresholds": thresholds.tolist(),
        "precisions": precisions,
        "recalls": recalls,
        "f1s": f1s,
        "best_f1_threshold": best_f1_threshold,
        "best_f1_value": best_f1_value,
        "recall_80_threshold": recall_80_threshold,
    }
def compare_mdi_vs_permutation(rf_balanced, X_test, y_test, feature_names, output_path):
    """Compare MDI importance vs permutation importance and save a chart."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    mdi = pd.Series(
        rf_balanced.feature_importances_,
        index=feature_names
    ).sort_values(ascending=False)

    perm = permutation_importance(
        rf_balanced,
        X_test,
        y_test,
        n_repeats=10,
        random_state=42,
        scoring="average_precision"
    )

    perm_series = pd.Series(
        perm.importances_mean,
        index=feature_names
    ).sort_values(ascending=False)

    top_features = list(mdi.head(10).index)
    for f in perm_series.head(10).index:
        if f not in top_features:
            top_features.append(f)
    top_features = top_features[:10]

    mdi_top = mdi.reindex(top_features).fillna(0)
    perm_top = perm_series.reindex(top_features).fillna(0)

    x = np.arange(len(top_features))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor("#0e0303")
    ax.set_facecolor("#b90505")
    ax.bar(x - width/2, mdi_top.values, width, label="MDI")
    ax.bar(x + width/2, perm_top.values, width, label="Permutation")
    ax.set_xticks(x)
    ax.set_xticklabels(top_features, rotation=45, ha="right")
    ax.set_ylabel("Importance")
    ax.set_title("MDI vs Permutation Importance", color="darkred")
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close(fig)

    return {
        "mdi": mdi.to_dict(),
        "permutation": perm_series.to_dict()
    }
class ScaledModelWrapper:
    """Wrap a scaler + fitted model so it can consume raw X."""
    def __init__(self, scaler, model):
        self.scaler = scaler
        self.model = model
        self.classes_ = model.classes_

    def predict_proba(self, X):
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)

    def predict(self, X):
        probs = self.predict_proba(X)
        return self.classes_[np.argmax(probs, axis=1)]


class CustomVotingEnsemble:
    """Simple soft-voting ensemble for fitted classifiers."""
    def __init__(self, models):
        self.models = models
        self.classes_ = np.array([0, 1])

    def _aligned_proba(self, model, X):
        probs = model.predict_proba(X)
        aligned = np.zeros((len(X), len(self.classes_)))

        for i, cls in enumerate(model.classes_):
            target_idx = np.where(self.classes_ == cls)[0][0]
            aligned[:, target_idx] = probs[:, i]

        return aligned

    def predict_proba(self, X):
        all_probs = [self._aligned_proba(model, X) for model in self.models]
        return np.mean(all_probs, axis=0)

    def predict(self, X):
        probs = self.predict_proba(X)
        return self.classes_[np.argmax(probs, axis=1)]
def main():
    """Orchestrate all 7 lab tasks. Run with: python lab_trees.py"""
    os.makedirs("results", exist_ok=True)

    # Task 1: Load + split
    result = load_and_split()
    if result is None:
        print("load_and_split not implemented. Exiting.")
        return
    X_train, X_test, y_train, y_test = result
    print(f"Train: {len(X_train)}  Test: {len(X_test)}  Churn rate: {y_train.mean():.2%}")

    # Task 2: Decision tree + calibration comparison
    dt = build_decision_tree(X_train, y_train)
    if result is None:
        print(f"\n--- Decision Tree (max_depth=5) ---")
        print(classification_report(y_test, dt.predict(X_test), zero_division=0))
        # Plot tree (first 3 levels)
        plt.figure(figsize=(14, 8))
        plot_tree(dt, feature_names=NUMERIC_FEATURES, max_depth=3,
                  filled=True, fontsize=8)
        plt.savefig("results/decision_tree.png", dpi=100, bbox_inches="tight")
        plt.close()

    cal = compare_dt_calibration(X_train, X_test, y_train, y_test)
    if cal:
        print(f"DT ECE (max_depth=None): {cal['ece_unbounded']:.3f}")
        print(f"DT ECE (max_depth=5):    {cal['ece_depth_5']:.3f}")

    # Task 3: Random forest + feature importances
    rf = build_random_forest(X_train, y_train)
    if rf is not None:
        print(f"\n--- Random Forest (max_depth=10) ---")
        imp = get_feature_importances(rf, NUMERIC_FEATURES)
        if imp:
            print("Feature importances:")
            for name, value in imp.items():
                print(f"  {name:<22s} {value:.3f}")

    # Task 4: Balanced RF + recall@0.5 comparison + PR-AUC
    rf_bal = build_random_forest(X_train, y_train, class_weight="balanced")
    if rf is not None and rf_bal is not None:
        r_def = evaluate_recall_at_threshold(rf, X_test, y_test, threshold=0.5)
        r_bal = evaluate_recall_at_threshold(rf_bal, X_test, y_test, threshold=0.5)
        print(f"\n--- class_weight effect at default 0.5 threshold ---")
        print(f"  RF default recall@0.5:  {r_def:.3f}")
        print(f"  RF balanced recall@0.5: {r_bal:.3f}  (ratio: {r_bal / max(r_def, 1e-9):.2f}x)")

        auc_def = compute_pr_auc(rf, X_test, y_test)
        auc_bal = compute_pr_auc(rf_bal, X_test, y_test)
        print(f"\n--- PR-AUC (threshold-independent ranking quality) ---")
        print(f"  RF default:  {auc_def:.3f}")
        print(f"  RF balanced: {auc_bal:.3f}")
        print("Note: class_weight='balanced' shifts the operating point at a fixed "
              "threshold; it does not improve the underlying ranking (PR-AUC).")

        # Task 5: PR curves + calibration curves
        plot_pr_curves(rf, rf_bal, X_test, y_test, "results/pr_curves.png")
        plot_calibration_curves(rf, rf_bal, X_test, y_test, "results/calibration_curves.png")

    # Task 6: Tree-vs-linear disagreement
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    lr = build_logistic_regression(X_train_scaled, y_train)
    if rf is not None and lr is not None:
        d = find_tree_vs_linear_disagreement(
            rf, lr, X_test, X_test_scaled, y_test, NUMERIC_FEATURES
        )
        if d:
            print(f"\n--- Tree-vs-linear disagreement (sample idx={d['sample_idx']}) ---")
            print(f"  RF P(churn=1)={d['rf_proba']:.3f}  LR P(churn=1)={d['lr_proba']:.3f}")
            print(f"  |diff| = {d['prob_diff']:.3f}   true label = {d['true_label']}")
            print(f"  Feature values: {d['feature_values']}")
    # Challenge Tier 1: Threshold tuning
    if rf_bal is not None:
        sweep = threshold_sweep(
            rf_bal, X_test, y_test, "results/threshold_sweep.png"
        )
        print("\n--- Threshold Sweep (Balanced RF) ---")
        print(f"Best F1 threshold: {sweep['best_f1_threshold']:.2f}")
        print(f"Best F1 value:     {sweep['best_f1_value']:.3f}")
        print(f"Threshold for recall >= 0.80: {sweep['recall_80_threshold']}")
    # Challenge Tier 2: Permutation importance
    if rf_bal is not None:
        perm_results = compare_mdi_vs_permutation(
            rf_bal, X_test, y_test, NUMERIC_FEATURES,
            "results/permutation_vs_mdi.png"
        )
        print("\n--- Permutation vs MDI Importance ---")
        print("Top MDI features:")
        for k, v in list(perm_results["mdi"].items())[:5]:
            print(f"  {k:<22s} {v:.3f}")
        print("Top Permutation features:")
        for k, v in list(perm_results["permutation"].items())[:5]:
            print(f"  {k:<22s} {v:.3f}")
    # Challenge Tier 3: Custom voting ensemble
    dt_bal = DecisionTreeClassifier(
        max_depth=5,
        class_weight="balanced",
        random_state=42
    )
    dt_bal.fit(X_train, y_train)

    lr_wrapped = ScaledModelWrapper(scaler, lr)

    ensemble = CustomVotingEnsemble([lr_wrapped, dt_bal, rf_bal])

    ens_pred = ensemble.predict(X_test)
    ens_prob = ensemble.predict_proba(X_test)[:, 1]

    print("\n--- Custom Voting Ensemble ---")
    print(classification_report(y_test, ens_pred, zero_division=0))
    print(f"Ensemble PR-AUC: {average_precision_score(y_test, ens_prob):.3f}")
if __name__ == "__main__":
    main()