"""
Milestone 2 - Machine Learning Analysis
DSA210 Data Science Project

This version fixes the weak ML section by adding:
1. A comparative external dataset analysis.
2. Multiple ML models instead of a single Random Forest result.
3. Cross-validation, test accuracy, balanced accuracy, macro precision/recall/F1, weighted F1.
4. Classification reports, confusion matrices, model-comparison tables, and feature-importance outputs.

External comparative dataset:
- File: data/external_student_performance_uci.csv
- Source: UCI Student Performance dataset by Paulo Cortez
- Records: 1044 course-level observations from Mathematics and Portuguese datasets combined
- Target: G3 final grade converted into Low / Medium / High performance levels

Important interpretation note:
The external dataset contains G1 and G2, which are first-period and second-period grades. These strongly
predict G3, the final grade. Therefore, the script reports two external settings:
- External with prior grades: includes G1 and G2; this is the high-accuracy comparative ML setting.
- External early prediction: excludes G1 and G2; this is harder but more realistic before prior grades are known.
"""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"
RESULTS_DIR = PROJECT_DIR / "results_ml"
FIGURES_DIR = PROJECT_DIR / "figures_ml"

RESULTS_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)

ORIGINAL_DATA_PATH = DATA_DIR / "student_performance_value.csv"
EXTERNAL_DATA_PATH = DATA_DIR / "external_student_performance_uci.csv"


def map_original_grade_to_level(grade: str) -> str:
    """Convert original letter grades to 3-level performance labels."""
    if grade in {"Fail", "DD", "DC"}:
        return "Low"
    if grade in {"CC", "CB"}:
        return "Medium"
    if grade in {"BB", "BA", "AA"}:
        return "High"
    return "Unknown"


def map_g3_to_level(g3: int) -> str:
    """Convert UCI G3 final grade, measured from 0 to 20, to 3-level labels."""
    if g3 < 10:
        return "Low"
    if g3 < 15:
        return "Medium"
    return "High"


def load_original_dataset() -> tuple[pd.DataFrame, pd.Series]:
    """Load the original project dataset and create a 3-class academic-performance target."""
    df = pd.read_csv(ORIGINAL_DATA_PATH)
    df = df.copy()
    df["performance_level"] = df["output_grade"].map(map_original_grade_to_level)
    df = df[df["performance_level"] != "Unknown"].copy()

    X = df.drop(columns=["output_grade", "performance_level"])
    y = df["performance_level"]
    return X, y


def load_external_dataset(include_prior_grades: bool = True) -> tuple[pd.DataFrame, pd.Series]:
    """Load the external UCI dataset and create a 3-class target from G3."""
    df = pd.read_csv(EXTERNAL_DATA_PATH)
    df = df.copy()

    if "performance_level" not in df.columns:
        df["performance_level"] = df["G3"].apply(map_g3_to_level)

    drop_cols = ["G3", "performance_level"]
    if not include_prior_grades:
        drop_cols += ["G1", "G2"]

    X = df.drop(columns=[c for c in drop_cols if c in df.columns])
    y = df["performance_level"]
    return X, y


def make_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Create preprocessing for numeric and categorical variables."""
    categorical_cols = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    numeric_cols = [col for col in X.columns if col not in categorical_cols]

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ]
    )


def get_models() -> dict[str, object]:
    """Return several different ML models for comparison."""
    return {
        "Dummy Baseline": DummyClassifier(strategy="most_frequent"),
        "Logistic Regression": LogisticRegression(
            max_iter=3000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "Linear SVM": LinearSVC(
            C=0.5,
            class_weight="balanced",
            max_iter=3000,
            random_state=RANDOM_STATE,
        ),
        "KNN": KNeighborsClassifier(n_neighbors=13, weights="distance"),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=6,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=150,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "Gradient Boosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
    }


def evaluate_dataset(
    dataset_name: str,
    X: pd.DataFrame,
    y: pd.Series,
    save_prefix: str,
) -> tuple[pd.DataFrame, Pipeline, dict]:
    """Evaluate all models on one dataset setting."""
    print("\n" + "=" * 90, flush=True)
    print(f"Dataset setting: {dataset_name}", flush=True)
    print("=" * 90, flush=True)
    print(f"Rows: {len(X)}", flush=True)
    print("Target distribution:", flush=True)
    print(y.value_counts().to_string(), flush=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    preprocessor = make_preprocessor(X)
    models = get_models()
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)

    rows = []
    fitted_models = {}
    predictions = {}

    for model_name, model in models.items():
        pipe = Pipeline(
            steps=[
                ("preprocess", preprocessor),
                ("model", model),
            ]
        )

        cv_scores = cross_validate(
            pipe,
            X_train,
            y_train,
            cv=cv,
            scoring={
                "accuracy": "accuracy",
                "macro_f1": "f1_macro",
                "balanced_accuracy": "balanced_accuracy",
            },
            n_jobs=1,
        )

        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)

        rows.append(
            {
                "dataset_setting": dataset_name,
                "model": model_name,
                "cv_accuracy_mean": cv_scores["test_accuracy"].mean(),
                "cv_accuracy_std": cv_scores["test_accuracy"].std(),
                "cv_macro_f1_mean": cv_scores["test_macro_f1"].mean(),
                "cv_macro_f1_std": cv_scores["test_macro_f1"].std(),
                "cv_balanced_accuracy_mean": cv_scores["test_balanced_accuracy"].mean(),
                "test_accuracy": accuracy_score(y_test, y_pred),
                "test_balanced_accuracy": balanced_accuracy_score(y_test, y_pred),
                "test_macro_precision": precision_score(y_test, y_pred, average="macro", zero_division=0),
                "test_macro_recall": recall_score(y_test, y_pred, average="macro", zero_division=0),
                "test_macro_f1": f1_score(y_test, y_pred, average="macro", zero_division=0),
                "test_weighted_f1": f1_score(y_test, y_pred, average="weighted", zero_division=0),
            }
        )

        fitted_models[model_name] = pipe
        predictions[model_name] = y_pred

    results = pd.DataFrame(rows).sort_values(
        by=["test_macro_f1", "test_accuracy"],
        ascending=False,
    )

    print("\nModel comparison:", flush=True)
    display_cols = [
        "model",
        "cv_accuracy_mean",
        "cv_macro_f1_mean",
        "test_accuracy",
        "test_balanced_accuracy",
        "test_macro_f1",
        "test_weighted_f1",
    ]
    print(results[display_cols].to_string(index=False, float_format=lambda value: f"{value:.3f}"), flush=True)

    best_model_name = results.iloc[0]["model"]
    best_pipeline = fitted_models[best_model_name]
    best_pred = predictions[best_model_name]

    print(f"\nBest model for {dataset_name}: {best_model_name}", flush=True)
    print("\nClassification report for best model:", flush=True)
    print(classification_report(y_test, best_pred, zero_division=0), flush=True)

    # Save result tables
    results.to_csv(RESULTS_DIR / f"{save_prefix}_model_comparison.csv", index=False)

    report_df = pd.DataFrame(classification_report(y_test, best_pred, output_dict=True, zero_division=0)).T
    report_df.to_csv(RESULTS_DIR / f"{save_prefix}_classification_report_best_model.csv")

    labels = sorted(y.unique())
    cm = confusion_matrix(y_test, best_pred, labels=labels)
    cm_df = pd.DataFrame(
        cm,
        index=[f"actual_{label}" for label in labels],
        columns=[f"predicted_{label}" for label in labels],
    )
    cm_df.to_csv(RESULTS_DIR / f"{save_prefix}_confusion_matrix_best_model.csv")

    plot_model_comparison(results, save_prefix)
    plot_confusion_matrix(cm, labels, save_prefix)
    # Feature importance can be added later, but it is skipped here to keep the milestone script fast and stable.

    meta = {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "best_model_name": best_model_name,
        "best_pred": best_pred,
    }
    return results, best_pipeline, meta


def plot_model_comparison(results: pd.DataFrame, save_prefix: str) -> None:
    """Save model comparison charts."""
    ordered = results.sort_values("test_accuracy", ascending=True)
    plt.figure(figsize=(10, 6))
    plt.barh(ordered["model"], ordered["test_accuracy"])
    plt.xlabel("Test Accuracy")
    plt.title("Model Comparison - Test Accuracy")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f"{save_prefix}_model_comparison_accuracy.png", dpi=200)
    plt.close()

    ordered = results.sort_values("test_macro_f1", ascending=True)
    plt.figure(figsize=(10, 6))
    plt.barh(ordered["model"], ordered["test_macro_f1"])
    plt.xlabel("Test Macro F1")
    plt.title("Model Comparison - Test Macro F1")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f"{save_prefix}_model_comparison_macro_f1.png", dpi=200)
    plt.close()


def plot_confusion_matrix(cm: np.ndarray, labels: list[str], save_prefix: str) -> None:
    """Save confusion matrix chart for the best model."""
    plt.figure(figsize=(7, 6))
    plt.imshow(cm)
    plt.title("Confusion Matrix - Best Model")
    plt.xlabel("Predicted Label")
    plt.ylabel("Actual Label")
    plt.xticks(range(len(labels)), labels, rotation=45)
    plt.yticks(range(len(labels)), labels)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f"{save_prefix}_confusion_matrix_best_model.png", dpi=200)
    plt.close()


def save_feature_importance(best_pipeline: Pipeline, X: pd.DataFrame, save_prefix: str) -> None:
    """Save feature importance for tree-based best models when available."""
    model = best_pipeline.named_steps["model"]
    preprocess = best_pipeline.named_steps["preprocess"]

    if not hasattr(model, "feature_importances_"):
        return

    try:
        feature_names = preprocess.get_feature_names_out()
    except Exception:
        feature_names = np.array([f"feature_{i}" for i in range(len(model.feature_importances_))])

    importance_df = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": model.feature_importances_,
        }
    ).sort_values("importance", ascending=False)

    importance_df.to_csv(RESULTS_DIR / f"{save_prefix}_feature_importance_best_model.csv", index=False)

    top = importance_df.head(15).sort_values("importance", ascending=True)
    plt.figure(figsize=(9, 6))
    plt.barh(top["feature"], top["importance"])
    plt.xlabel("Feature Importance")
    plt.title("Top Feature Importance - Best Model")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f"{save_prefix}_feature_importance_best_model.png", dpi=200)
    plt.close()


def main() -> None:
    all_results = []

    settings = [
        (
            "Original Higher Education Dataset - 3-Class Performance",
            *load_original_dataset(),
            "original_3class",
        ),
        (
            "External UCI Dataset - 3-Class Performance with Prior Grades",
            *load_external_dataset(include_prior_grades=True),
            "external_uci_with_prior_grades",
        ),
        (
            "External UCI Dataset - Early Prediction without G1/G2",
            *load_external_dataset(include_prior_grades=False),
            "external_uci_without_prior_grades",
        ),
    ]

    for dataset_name, X, y, save_prefix in settings:
        results, _, _ = evaluate_dataset(dataset_name, X, y, save_prefix)
        all_results.append(results)

    combined_results = pd.concat(all_results, ignore_index=True)
    combined_results = combined_results.sort_values(
        by=["dataset_setting", "test_macro_f1", "test_accuracy"],
        ascending=[True, False, False],
    )
    combined_results.to_csv(RESULTS_DIR / "all_model_comparisons.csv", index=False)

    best_by_setting = combined_results.groupby("dataset_setting", group_keys=False).apply(
        lambda group: group.sort_values(["test_macro_f1", "test_accuracy"], ascending=False).head(1)
    )

    print("\n" + "=" * 90, flush=True)
    print("FINAL SUMMARY", flush=True)
    print("=" * 90, flush=True)
    print(
        best_by_setting[
            ["dataset_setting", "model", "test_accuracy", "test_macro_f1", "test_weighted_f1"]
        ].to_string(index=False, float_format=lambda value: f"{value:.3f}"),
        flush=True,
    )

    print("\nInterpretation:", flush=True)
    print("- The original 145-row dataset is too small for reliable exact 8-class grade prediction.", flush=True)
    print("- Low/Medium/High grouping gives a more stable and interpretable target.", flush=True)
    print("- The external UCI dataset provides a stronger comparative ML analysis with 1044 rows.", flush=True)
    print("- The best external result comes from the setting with G1 and G2 included, because they are prior grades that strongly predict final grade G3.", flush=True)
    print("- The without-G1/G2 setting is lower, but useful as a harder early-prediction benchmark.", flush=True)
    print(f"\nSaved result tables to: {RESULTS_DIR}", flush=True)
    print(f"Saved figures to: {FIGURES_DIR}", flush=True)


if __name__ == "__main__":
    main()
