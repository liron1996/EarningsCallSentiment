"""Phase E: train models that predict forward excess returns from features.

For each horizon (1d / 5d / 21d / 63d), we train 9 models:
  - ridge       RidgeCV  (regularized linear; alpha auto-tuned via CV)
  - lasso       LassoCV  (L1 linear; alpha auto-tuned, performs implicit feature selection)
  - elasticnet  ElasticNetCV  (L1+L2 mix; both alpha and l1_ratio auto-tuned)
  - logreg      LogisticRegressionCV  (sign classifier; C auto-tuned)
  - knn         KNeighborsRegressor + GridSearchCV  (non-parametric baseline)
  - rf          RandomForestRegressor + GridSearchCV  (bagging; variance reduction)
  - histgb      HistGradientBoostingRegressor + GridSearchCV
  - xgb         XGBRegressor + GridSearchCV
  - lgbm        LGBMRegressor + GridSearchCV

Metrics (per model, per split):
  Regression: r2, mse, mae
  Classification (on sign of return): dir_acc, f1, precision, recall, mcc
  Ranking:    ic (Spearman of pred vs realized), auc (sign-truth vs continuous score)

Train/test split is PER-TICKER chronological:
  - For each ticker, the first N calls (default N=6, configurable) -> train.
  - All later calls for the same ticker -> test.
  - Avoids look-ahead leakage AND ensures every ticker contributes to both splits.

Feature selection (auto, train-only):
  Before fitting, we drop columns with <=1 unique value or >= 95% zero+NaN
  on the train portion. Reduces the 54-column feature matrix to ~30 informative
  columns, lifting the train-rows / features ratio.

Hyperparameter CV:
  Linear models use *CV variants (alpha / C selection internal).
  Tree models use GridSearchCV with TimeSeriesSplit so inner CV folds respect time.

Outputs (under data/model/):
  metrics.csv               long format: horizon, model, split, n, r2, mse, dir_acc, ic
  metrics_summary.csv       wide format: train+test side-by-side per (horizon, model)
  predictions.csv           per-call: filename, horizon, model, split, y_true, y_pred
  feature_importance.csv    per (horizon, model, feature)
  plots/overfit_h{H}d.png   train vs test direction-accuracy bar chart per horizon
  models/h{H}d_{model}.joblib  pickled fitted pipelines
  _summary.json             schema version, config, selected features per horizon
"""
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from typing import Any

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import (
    ElasticNet,
    ElasticNetCV,
    Lasso,
    LassoCV,
    LogisticRegression,
    LogisticRegressionCV,
    Ridge,
    RidgeCV,
)
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import lightgbm as lgb
import xgboost as xgb

# Quiet down a benign warning that fires once per LGBM call.
warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names",
    category=UserWarning,
)
warnings.filterwarnings("ignore", category=ConvergenceWarning)

REPO_ROOT = Path(__file__).resolve().parents[2]
FEATURES_CSV = REPO_ROOT / "data" / "features" / "features.csv"
FEATURES_SUMMARY = REPO_ROOT / "data" / "features" / "_summary.json"
MODEL_DIR = REPO_ROOT / "data" / "model"
PLOTS_DIR = MODEL_DIR / "plots"
MODELS_DIR = MODEL_DIR / "models"

DEFAULT_HORIZONS = [1, 5, 21, 63]
DEFAULT_TRAIN_PER_TICKER = 6
DEFAULT_MODELS = ["ridge", "lasso", "elasticnet", "logreg", "knn", "rf",
                  "histgb", "xgb", "lgbm"]

# Inner-CV folds used by the *CV variants and GridSearchCV. With ~80 train rows
# split per ticker, 3 folds gives ~26 rows per fold which is the smallest sane size.
INNER_CV_SPLITS = 3


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data() -> tuple[pd.DataFrame, list[str]]:
    df = pd.read_csv(FEATURES_CSV)
    summary = json.loads(FEATURES_SUMMARY.read_text(encoding="utf-8"))
    return df, summary["columns"]["features"]


# ---------------------------------------------------------------------------
# Per-ticker chronological split
# ---------------------------------------------------------------------------

def split_per_ticker(
    df: pd.DataFrame,
    target_col: str,
    n_train_per_ticker: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """For each ticker, first n_train_per_ticker calls (oldest -> newest) are train,
    the rest are test. Drops rows where the target is NaN before splitting so
    horizons with insufficient forward data still produce a clean split."""
    sub = df.dropna(subset=[target_col]).copy()
    sub["report_date"] = pd.to_datetime(sub["report_date"])
    sub = sub.sort_values(["ticker", "report_date"]).reset_index(drop=True)
    sub["_rank"] = sub.groupby("ticker").cumcount()
    train = sub[sub["_rank"] < n_train_per_ticker].drop(columns="_rank")
    test = sub[sub["_rank"] >= n_train_per_ticker].drop(columns="_rank")
    # Resort each by date so inner CV (TimeSeriesSplit) respects time order globally.
    train = train.sort_values("report_date").reset_index(drop=True)
    test = test.sort_values("report_date").reset_index(drop=True)
    return train, test


# ---------------------------------------------------------------------------
# Feature selection (auto, train-only)
# ---------------------------------------------------------------------------

def select_features(
    train: pd.DataFrame,
    feature_cols: list[str],
    *,
    sparsity_threshold: float = 0.95,
    min_unique: int = 2,
) -> list[str]:
    """Drop columns with <= min_unique-1 unique values OR >= sparsity_threshold zero+NaN."""
    kept: list[str] = []
    for c in feature_cols:
        col = train[c]
        if col.nunique(dropna=True) < min_unique:
            continue
        zero_or_nan = ((col == 0) | col.isna()).mean()
        if zero_or_nan >= sparsity_threshold:
            continue
        kept.append(c)
    return kept


# ---------------------------------------------------------------------------
# Pipelines
# ---------------------------------------------------------------------------

def make_pipeline(model_name: str, n_inner_cv: int = INNER_CV_SPLITS) -> Pipeline:
    """Returns an unfitted pipeline. Inner CV uses TimeSeriesSplit when applicable."""
    impute = SimpleImputer(strategy="constant", fill_value=0.0)
    inner_cv = TimeSeriesSplit(n_splits=n_inner_cv)

    if model_name == "ridge":
        return Pipeline([
            ("imputer", impute),
            ("scaler", StandardScaler()),
            ("model", RidgeCV(alphas=[0.1, 1.0, 10.0, 100.0, 1000.0], cv=inner_cv)),
        ])
    if model_name == "lasso":
        return Pipeline([
            ("imputer", impute),
            ("scaler", StandardScaler()),
            ("model", LassoCV(
                alphas=[0.0001, 0.001, 0.01, 0.1, 1.0],
                cv=inner_cv,
                max_iter=10000,
                random_state=42,
            )),
        ])
    if model_name == "elasticnet":
        return Pipeline([
            ("imputer", impute),
            ("scaler", StandardScaler()),
            ("model", ElasticNetCV(
                l1_ratio=[0.1, 0.5, 0.9, 1.0],
                alphas=[0.0001, 0.001, 0.01, 0.1, 1.0],
                cv=inner_cv,
                max_iter=10000,
                random_state=42,
            )),
        ])
    if model_name == "knn":
        base = KNeighborsRegressor()
        grid = {"n_neighbors": [3, 5, 7, 11], "weights": ["uniform", "distance"]}
        return Pipeline([
            ("imputer", impute),
            ("scaler", StandardScaler()),
            ("model", GridSearchCV(base, grid, cv=inner_cv, scoring="r2", n_jobs=1)),
        ])
    if model_name == "rf":
        base = RandomForestRegressor(random_state=42, n_jobs=1)
        grid = {
            "n_estimators": [100, 200],
            "max_depth": [3, 5, None],
            "min_samples_leaf": [5, 10],
        }
        return Pipeline([
            ("imputer", impute),
            ("model", GridSearchCV(base, grid, cv=inner_cv, scoring="r2", n_jobs=1)),
        ])
    if model_name == "logreg":
        return Pipeline([
            ("imputer", impute),
            ("scaler", StandardScaler()),
            ("model", LogisticRegressionCV(
                Cs=[0.01, 0.1, 1.0, 10.0, 100.0],
                cv=inner_cv,
                max_iter=2000,
                random_state=42,
            )),
        ])
    if model_name == "histgb":
        base = HistGradientBoostingRegressor(random_state=42)
        grid = {
            "max_depth": [2, 3],
            "max_iter": [80, 150],
            "learning_rate": [0.03, 0.08],
            "min_samples_leaf": [10, 20],
        }
        return Pipeline([
            ("imputer", impute),
            ("model", GridSearchCV(base, grid, cv=inner_cv, scoring="r2", n_jobs=1)),
        ])
    if model_name == "xgb":
        base = xgb.XGBRegressor(
            random_state=42, verbosity=0, tree_method="hist", n_jobs=1)
        grid = {
            "max_depth": [2, 3],
            "n_estimators": [80, 150],
            "learning_rate": [0.03, 0.08],
            "min_child_weight": [5, 10],
        }
        return Pipeline([
            ("model", GridSearchCV(base, grid, cv=inner_cv, scoring="r2", n_jobs=1)),
        ])
    if model_name == "lgbm":
        base = lgb.LGBMRegressor(random_state=42, verbose=-1, n_jobs=1)
        grid = {
            "max_depth": [2, 3],
            "n_estimators": [80, 150],
            "learning_rate": [0.03, 0.08],
            "min_child_samples": [10, 20],
        }
        return Pipeline([
            ("model", GridSearchCV(base, grid, cv=inner_cv, scoring="r2", n_jobs=1)),
        ])
    raise ValueError(f"Unknown model: {model_name}")


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def metric_block(
    y_true_cont: np.ndarray,
    y_pred: np.ndarray,
    is_classifier: bool,
    y_score: np.ndarray | None = None,
) -> dict:
    """Compute regression + classification + ranking metrics.

    y_pred:  for regressors, continuous predicted excess returns;
             for the LogReg classifier, predicted class labels in {-1, +1}.
    y_score: a continuous score used only for AUC (regressor: same as y_pred;
             classifier: predict_proba(class=+1)). If None, AUC is skipped.
    """
    out: dict[str, Any] = {}
    sign_true = np.sign(y_true_cont)

    if is_classifier:
        # y_pred is already in {-1, +1}; compare directly to sign_true.
        sign_pred = y_pred.astype(int)
        out["r2"] = None
        out["mse"] = None
        out["mae"] = None
    else:
        sign_pred = np.sign(y_pred).astype(int)
        out["r2"] = float(r2_score(y_true_cont, y_pred)) if len(y_true_cont) > 1 else None
        out["mse"] = float(mean_squared_error(y_true_cont, y_pred))
        out["mae"] = float(mean_absolute_error(y_true_cont, y_pred))

    # Direction accuracy + class-imbalance-robust metrics.
    out["dir_acc"] = float(accuracy_score(sign_true, sign_pred))
    out["f1"] = float(f1_score(sign_true, sign_pred, pos_label=1, zero_division=0))
    out["precision"] = float(precision_score(sign_true, sign_pred, pos_label=1, zero_division=0))
    out["recall"] = float(recall_score(sign_true, sign_pred, pos_label=1, zero_division=0))
    if len(np.unique(sign_true)) > 1 and len(np.unique(sign_pred)) > 1:
        out["mcc"] = float(matthews_corrcoef(sign_true, sign_pred))
    else:
        out["mcc"] = None

    # Ranking metrics.
    if not is_classifier and len(np.unique(y_true_cont)) > 1 and len(np.unique(y_pred)) > 1:
        ic, _ = spearmanr(y_true_cont, y_pred)
        out["ic"] = float(ic) if not np.isnan(ic) else None
    else:
        out["ic"] = None

    if y_score is not None:
        binary_true = (sign_true >= 0).astype(int)
        if len(np.unique(binary_true)) > 1:
            try:
                out["auc"] = float(roc_auc_score(binary_true, y_score))
            except Exception:  # noqa: BLE001
                out["auc"] = None
        else:
            out["auc"] = None
    else:
        out["auc"] = None

    return out


def extract_importance(pipe: Pipeline, feature_cols: list[str]) -> pd.DataFrame:
    model = pipe.named_steps["model"]
    # Unwrap GridSearchCV
    if hasattr(model, "best_estimator_"):
        model = model.best_estimator_
    if hasattr(model, "coef_"):
        coef = np.asarray(model.coef_).ravel()
        return pd.DataFrame({"feature": feature_cols, "importance": coef})
    if hasattr(model, "feature_importances_"):
        return pd.DataFrame({
            "feature": feature_cols,
            "importance": np.asarray(model.feature_importances_, dtype=float),
        })
    return pd.DataFrame({"feature": feature_cols, "importance": np.zeros(len(feature_cols))})


def best_param_summary(pipe: Pipeline) -> dict[str, Any]:
    """Capture the auto-tuned hyperparameter for logging."""
    model = pipe.named_steps["model"]
    if hasattr(model, "best_params_"):
        return dict(model.best_params_)
    if hasattr(model, "alpha_"):
        return {"alpha": float(model.alpha_)}
    if hasattr(model, "C_"):
        c = model.C_
        return {"C": float(c[0]) if hasattr(c, "__len__") else float(c)}
    return {}


# ---------------------------------------------------------------------------
# Train one (horizon, model) on a pre-computed split
# ---------------------------------------------------------------------------

def train_one(
    train: pd.DataFrame,
    test: pd.DataFrame,
    selected_features: list[str],
    horizon: int,
    model_name: str,
) -> dict[str, Any]:
    target_col = f"excess_{horizon}d"

    if len(train) < 5:
        return {"status": "insufficient_train", "n_train": len(train), "n_test": len(test)}
    if len(test) < 1:
        return {"status": "no_test_rows", "n_train": len(train), "n_test": len(test)}

    X_train = train[selected_features].to_numpy(dtype=float)
    X_test = test[selected_features].to_numpy(dtype=float)
    y_train_cont = train[target_col].to_numpy(dtype=float)
    y_test_cont = test[target_col].to_numpy(dtype=float)

    is_classifier = model_name == "logreg"

    if is_classifier:
        y_train_target = np.sign(y_train_cont).astype(int)
        y_train_target = np.where(y_train_target == 0, 1, y_train_target)
        if len(np.unique(y_train_target)) < 2:
            return {"status": "single_class_train", "n_train": len(train), "n_test": len(test)}
    else:
        y_train_target = y_train_cont

    pipe = make_pipeline(model_name)
    pipe.fit(X_train, y_train_target)

    y_pred_train = pipe.predict(X_train)
    y_pred_test = pipe.predict(X_test)

    # For AUC we need a continuous score. Regressors: use y_pred directly.
    # Classifier: predict_proba(class=+1) centered around 0.5.
    if is_classifier and hasattr(pipe, "predict_proba"):
        try:
            y_score_train = pipe.predict_proba(X_train)[:, 1]
            y_score_test = pipe.predict_proba(X_test)[:, 1]
        except Exception:  # noqa: BLE001
            y_score_train = y_pred_train.astype(float)
            y_score_test = y_pred_test.astype(float)
    else:
        y_score_train = y_pred_train.astype(float)
        y_score_test = y_pred_test.astype(float)

    m_train = metric_block(y_train_cont, y_pred_train, is_classifier, y_score=y_score_train)
    m_test = metric_block(y_test_cont, y_pred_test, is_classifier, y_score=y_score_test)

    importance = extract_importance(pipe, selected_features)
    best_params = best_param_summary(pipe)

    predictions = pd.concat([
        pd.DataFrame({
            "filename": train["filename"].values,
            "ticker": train["ticker"].values,
            "report_date": train["report_date"].dt.strftime("%Y-%m-%d").values,
            "horizon": f"{horizon}d",
            "model": model_name,
            "split": "train",
            "y_true": y_train_cont,
            "y_pred": y_pred_train,
        }),
        pd.DataFrame({
            "filename": test["filename"].values,
            "ticker": test["ticker"].values,
            "report_date": test["report_date"].dt.strftime("%Y-%m-%d").values,
            "horizon": f"{horizon}d",
            "model": model_name,
            "split": "test",
            "y_true": y_test_cont,
            "y_pred": y_pred_test,
        }),
    ], ignore_index=True)

    joblib.dump(pipe, MODELS_DIR / f"h{horizon}d_{model_name}.joblib")

    return {
        "status": "ok",
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "metrics_train": m_train,
        "metrics_test": m_test,
        "best_params": best_params,
        "predictions": predictions,
        "importance": importance,
    }


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def make_simple_estimator(model_name: str) -> Pipeline:
    """Frozen-hyperparam variant of make_pipeline(); used only for learning curves
    so we don't pay the GridSearchCV / *CV cost at every train fraction. Hyperparams
    are picked from what the CV-tuned models converged to in earlier runs.
    """
    impute = SimpleImputer(strategy="constant", fill_value=0.0)
    if model_name == "ridge":
        return Pipeline([("imputer", impute), ("scaler", StandardScaler()),
                         ("model", Ridge(alpha=100.0, random_state=42))])
    if model_name == "lasso":
        return Pipeline([("imputer", impute), ("scaler", StandardScaler()),
                         ("model", Lasso(alpha=0.01, max_iter=10000, random_state=42))])
    if model_name == "elasticnet":
        return Pipeline([("imputer", impute), ("scaler", StandardScaler()),
                         ("model", ElasticNet(alpha=0.01, l1_ratio=0.5,
                                              max_iter=10000, random_state=42))])
    if model_name == "logreg":
        return Pipeline([("imputer", impute), ("scaler", StandardScaler()),
                         ("model", LogisticRegression(C=0.1, max_iter=2000, random_state=42))])
    if model_name == "knn":
        return Pipeline([("imputer", impute), ("scaler", StandardScaler()),
                         ("model", KNeighborsRegressor(n_neighbors=11, weights="uniform"))])
    if model_name == "rf":
        return Pipeline([("imputer", impute),
                         ("model", RandomForestRegressor(
                             n_estimators=200, max_depth=3, min_samples_leaf=10,
                             random_state=42, n_jobs=1))])
    if model_name == "histgb":
        return Pipeline([("imputer", impute),
                         ("model", HistGradientBoostingRegressor(
                             max_iter=80, max_depth=2, learning_rate=0.03,
                             min_samples_leaf=10, random_state=42))])
    if model_name == "xgb":
        return Pipeline([("model", xgb.XGBRegressor(
            n_estimators=80, max_depth=2, learning_rate=0.03, min_child_weight=10,
            random_state=42, verbosity=0, tree_method="hist", n_jobs=1))])
    if model_name == "lgbm":
        return Pipeline([("model", lgb.LGBMRegressor(
            n_estimators=80, max_depth=3, learning_rate=0.08, min_child_samples=10,
            random_state=42, verbose=-1, n_jobs=1))])
    raise ValueError(f"Unknown model: {model_name}")


def plot_learning_curves(
    df: pd.DataFrame,
    feature_cols: list[str],
    horizon: int,
    models: list[str],
    train_per_ticker: int,
    out_path: Path,
) -> None:
    """3x3 grid of learning curves at the given horizon.

    For each model: refit at 25%, 50%, 75%, 100% of the training set (random
    subsample with fixed seed). Plot direction-accuracy on the (sub)train and on
    the FULL held-out test set.
    """
    target_col = f"excess_{horizon}d"
    if target_col not in df.columns:
        return
    train, test = split_per_ticker(df, target_col, train_per_ticker)
    if len(train) < 20 or len(test) < 5:
        return
    selected = select_features(train, feature_cols)

    fractions = [0.25, 0.50, 0.75, 1.00]
    n_train = len(train)
    rng = np.random.default_rng(42)
    full_idx = np.arange(n_train)
    indices_per_frac: dict[float, np.ndarray] = {}
    for f in fractions:
        n = max(int(round(n_train * f)), 5)
        indices_per_frac[f] = full_idx if f == 1.0 else rng.choice(full_idx, size=n, replace=False)

    X_test = test[selected].to_numpy(dtype=float)
    y_test_cont = test[target_col].to_numpy(dtype=float)

    n_models = len(models)
    cols = 3
    rows = (n_models + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(13, 3.4 * rows), sharey=True)
    axes_flat = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for i, model_name in enumerate(models):
        ax = axes_flat[i]
        train_accs: list[float] = []
        test_accs: list[float] = []
        sizes: list[int] = []
        for f in fractions:
            idx = indices_per_frac[f]
            sub = train.iloc[idx]
            X_tr = sub[selected].to_numpy(dtype=float)
            y_tr_cont = sub[target_col].to_numpy(dtype=float)

            is_clf = model_name == "logreg"
            if is_clf:
                y_tr_target = np.sign(y_tr_cont).astype(int)
                y_tr_target = np.where(y_tr_target == 0, 1, y_tr_target)
                if len(np.unique(y_tr_target)) < 2:
                    train_accs.append(np.nan); test_accs.append(np.nan); sizes.append(len(idx))
                    continue
            else:
                y_tr_target = y_tr_cont

            try:
                pipe = make_simple_estimator(model_name)
                pipe.fit(X_tr, y_tr_target)
                yp_tr = pipe.predict(X_tr)
                yp_te = pipe.predict(X_test)
                if is_clf:
                    tr_acc = accuracy_score(np.sign(y_tr_cont), yp_tr)
                    te_acc = accuracy_score(np.sign(y_test_cont), yp_te)
                else:
                    tr_acc = accuracy_score(np.sign(y_tr_cont), np.sign(yp_tr))
                    te_acc = accuracy_score(np.sign(y_test_cont), np.sign(yp_te))
            except Exception:  # noqa: BLE001
                tr_acc = np.nan
                te_acc = np.nan

            train_accs.append(tr_acc)
            test_accs.append(te_acc)
            sizes.append(len(idx))

        ax.plot(sizes, train_accs, "o-", color="#1f77b4", label="train")
        ax.plot(sizes, test_accs, "s--", color="#ff7f0e", label="test")
        ax.set_title(model_name, fontsize=10)
        ax.set_ylim(0.2, 1.05)
        ax.axhline(0.5, color="gray", linestyle=":", linewidth=0.7)
        ax.set_xlabel("train set size")
        ax.set_ylabel("dir_acc")
        ax.grid(alpha=0.25)
        if i == 0:
            ax.legend(fontsize=9, loc="lower right")

    for j in range(n_models, len(axes_flat)):
        axes_flat[j].axis("off")

    fig.suptitle(f"Learning curves @ horizon = {horizon}d   "
                 f"(test on full {len(test)}-row held-out set)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_overfitting(metrics_df: pd.DataFrame, horizon: int, out_path: Path) -> None:
    sub = metrics_df[metrics_df["horizon"] == f"{horizon}d"]
    if sub.empty:
        return
    pivot = sub.pivot_table(index="model", columns="split", values="dir_acc")
    if pivot.empty:
        return
    cols = [c for c in ("train", "test") if c in pivot.columns]
    pivot = pivot[cols]
    fig, ax = plt.subplots(figsize=(9, 5))
    pivot.plot(kind="bar", ax=ax, color=["#1f77b4", "#ff7f0e"])
    ax.set_title(f"Direction accuracy: train vs test  (horizon = {horizon}d)")
    ax.set_ylabel("Accuracy")
    ax.set_xlabel("Model")
    ax.set_ylim(0, 1.0)
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, label="chance (0.50)")
    ax.legend(loc="lower right")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Wide-format summary CSV
# ---------------------------------------------------------------------------

def write_metrics_summary(metrics_df: pd.DataFrame, out_path: Path) -> None:
    if metrics_df.empty:
        return
    wide = metrics_df.pivot_table(
        index=["horizon", "model"],
        columns="split",
        values=["n", "r2", "mse", "mae", "dir_acc", "f1",
                "precision", "recall", "mcc", "auc", "ic"],
        aggfunc="first",
    ).reset_index()
    wide.columns = [
        c if isinstance(c, str) else (f"{c[0]}_{c[1]}" if c[1] else c[0])
        for c in wide.columns
    ]
    column_order = [
        "horizon", "model",
        "n_train", "n_test",
        "dir_acc_train", "dir_acc_test",
        "f1_train", "f1_test",
        "precision_train", "precision_test",
        "recall_train", "recall_test",
        "auc_train", "auc_test",
        "mcc_train", "mcc_test",
        "ic_train", "ic_test",
        "r2_train", "r2_test",
        "mse_train", "mse_test",
        "mae_train", "mae_test",
    ]
    wide = wide[[c for c in column_order if c in wide.columns]]
    if "dir_acc_train" in wide.columns and "dir_acc_test" in wide.columns:
        wide["dir_acc_gap"] = wide["dir_acc_train"] - wide["dir_acc_test"]
    wide["_h"] = wide["horizon"].str.replace("d", "").astype(int)
    wide = wide.sort_values(["_h", "model"]).drop(columns="_h").reset_index(drop=True)
    wide.to_csv(out_path, index=False)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Phase E: train forward-return models")
    parser.add_argument("--horizons", nargs="+", type=int, default=DEFAULT_HORIZONS)
    parser.add_argument("--train-per-ticker", type=int, default=DEFAULT_TRAIN_PER_TICKER,
                        help="First N calls per ticker go to train, rest to test (default: 6)")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    args = parser.parse_args(argv)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    df, feature_cols = load_data()
    print(f"Loaded {len(df)} feature rows; {len(feature_cols)} feature columns "
          f"({df['ticker'].nunique()} tickers)")
    print(f"Split: first {args.train_per_ticker} calls per ticker -> train; rest -> test")
    print(f"Horizons: {args.horizons}    Models: {args.models}")
    print()

    metrics_rows: list[dict[str, Any]] = []
    all_predictions: list[pd.DataFrame] = []
    all_importance: list[pd.DataFrame] = []
    skipped: list[dict[str, Any]] = []
    selected_per_horizon: dict[str, list[str]] = {}
    best_params_log: dict[str, dict[str, Any]] = {}

    for horizon in args.horizons:
        target_col = f"excess_{horizon}d"
        if target_col not in df.columns:
            print(f"  h={horizon}d  missing target column {target_col}; skipping")
            continue

        train, test = split_per_ticker(df, target_col, args.train_per_ticker)
        selected = select_features(train, feature_cols)
        selected_per_horizon[f"{horizon}d"] = selected
        print(f"  h={horizon:2d}d  train={len(train):3d}  test={len(test):3d}  "
              f"features kept={len(selected):2d}/{len(feature_cols)}")

        for model_name in args.models:
            try:
                r = train_one(train, test, selected, horizon, model_name)
            except Exception as exc:  # noqa: BLE001
                print(f"    {model_name:8s}  FAILED: {exc}")
                skipped.append({"horizon": f"{horizon}d", "model": model_name, "error": str(exc)})
                continue
            if r["status"] != "ok":
                print(f"    {model_name:8s}  skipped: {r['status']}")
                skipped.append({"horizon": f"{horizon}d", "model": model_name, "reason": r["status"]})
                continue

            ta = r["metrics_train"]["dir_acc"]
            te = r["metrics_test"]["dir_acc"]
            f1 = r["metrics_test"]["f1"]
            auc = r["metrics_test"].get("auc")
            ic = r["metrics_test"].get("ic")
            best = r.get("best_params", {})
            best_str = " ".join(f"{k}={v}" for k, v in best.items())
            auc_s = f"{auc:.3f}" if auc is not None else "  -  "
            ic_s = f"{ic:+.3f}" if ic is not None else "  -  "
            print(f"    {model_name:10s}  train={ta:.3f}  test={te:.3f}  "
                  f"gap={ta-te:+.3f}  f1={f1:.3f}  auc={auc_s}  ic={ic_s}  "
                  f"[{best_str}]")
            best_params_log[f"{horizon}d_{model_name}"] = best

            for split_name, m in (("train", r["metrics_train"]), ("test", r["metrics_test"])):
                metrics_rows.append({
                    "horizon": f"{horizon}d",
                    "model": model_name,
                    "split": split_name,
                    "n": r[f"n_{split_name}"],
                    **m,
                })
            all_predictions.append(r["predictions"])
            imp = r["importance"].copy()
            imp["horizon"] = f"{horizon}d"
            imp["model"] = model_name
            all_importance.append(imp)

    metrics_df = pd.DataFrame(metrics_rows)

    def _safe_to_csv(df_to_write: pd.DataFrame, path: Path) -> None:
        try:
            df_to_write.to_csv(path, index=False)
        except PermissionError:
            alt = path.with_name(path.stem + "_v2" + path.suffix)
            df_to_write.to_csv(alt, index=False)
            print(f"  (locked: {path.name} -> wrote {alt.name})")

    _safe_to_csv(metrics_df, MODEL_DIR / "metrics.csv")
    try:
        write_metrics_summary(metrics_df, MODEL_DIR / "metrics_summary.csv")
    except PermissionError:
        write_metrics_summary(metrics_df, MODEL_DIR / "metrics_summary_v2.csv")
        print(f"  (locked: metrics_summary.csv -> wrote metrics_summary_v2.csv)")

    if all_predictions:
        _safe_to_csv(pd.concat(all_predictions, ignore_index=True),
                     MODEL_DIR / "predictions.csv")
    if all_importance:
        _safe_to_csv(pd.concat(all_importance, ignore_index=True),
                     MODEL_DIR / "feature_importance.csv")

    for horizon in args.horizons:
        plot_overfitting(metrics_df, horizon, PLOTS_DIR / f"overfit_h{horizon}d.png")
        plot_learning_curves(
            df, feature_cols, horizon, args.models,
            args.train_per_ticker,
            PLOTS_DIR / f"learning_curve_h{horizon}d.png",
        )

    summary = {
        "schema_version": 2,
        "split_mode": "per_ticker_chronological",
        "train_per_ticker": args.train_per_ticker,
        "horizons": args.horizons,
        "models": args.models,
        "n_total_rows": int(len(df)),
        "n_features_in_input": len(feature_cols),
        "selected_features_per_horizon": selected_per_horizon,
        "best_hyperparams": best_params_log,
        "skipped": skipped,
    }
    (MODEL_DIR / "_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")

    print()
    print(f"Wrote: {MODEL_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
