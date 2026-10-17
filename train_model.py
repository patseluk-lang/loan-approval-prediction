"""Навчання моделі, що прогнозує схвалення займу.

Запуск: python train_model.py
Результат: ml/loan_model.joblib (модель для Django-застосунку),
ml/metrics.json і два графіки в loans/static/loans/charts (для сторінки "Про модель").
"""

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    recall_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA_PATH = Path("loan_data.csv")
MODEL_PATH = Path("ml/loan_model.joblib")
METRICS_PATH = Path("ml/metrics.json")
# Графіки кладу в static застосунку loans, звідки їх віддає Django.
CHARTS_DIR = Path("loans/static/loans/charts")

NUMERIC_FEATURES = ["ApplicantIncome", "CoapplicantIncome", "LoanAmount"]
# Credit_History і Loan_Amount_Term записані числами, але це набір фіксованих варіантів,
# тому кодую їх як категорії, а не масштабую як неперервні величини.
CATEGORICAL_FEATURES = [
    "Gender",
    "Married",
    "Dependents",
    "Education",
    "Self_Employed",
    "Property_Area",
    "Credit_History",
    "Loan_Amount_Term",
]


def load_data():
    df = pd.read_csv(DATA_PATH)
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = (df["Loan_Status"] == "Y").astype(int)
    return X, y


def build_pipeline():
    # Пропуски заповнюються всередині Pipeline: медіану й найчастіше значення
    # він бере лише з навчальної частини, тож тестові дані не "підглядаються".
    numeric = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            # ignore: якщо у формі прийде значення, якого не було в даних, модель не впаде.
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    preprocessor = ColumnTransformer(
        [
            ("num", numeric, NUMERIC_FEATURES),
            ("cat", categorical, CATEGORICAL_FEATURES),
        ]
    )
    return Pipeline(
        [
            ("preprocessor", preprocessor),
            ("classifier", RandomForestClassifier(random_state=42)),
        ]
    )


def feature_importances(model, top=10):
    # Важливість ознак є лише в деревних моделей; якщо переможе логістична регресія,
    # сторінка "Про модель" просто не покаже цей блок.
    classifier = model.named_steps["classifier"]
    if not hasattr(classifier, "feature_importances_"):
        return []
    names = model.named_steps["preprocessor"].get_feature_names_out()
    names = [name.split("__", 1)[1] for name in names]
    values = classifier.feature_importances_
    ranked = sorted(zip(names, values), key=lambda pair: pair[1], reverse=True)[:top]
    return [
        {"feature": name, "importance": round(float(value), 3)}
        for name, value in ranked
    ]


def save_feature_importance_chart(importances):
    names = [item["feature"] for item in importances][::-1]
    values = [item["importance"] for item in importances][::-1]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(names, values, color="#2f5d8a")
    ax.set_xlabel("важливість")
    ax.set_title("10 найважливіших ознак (RandomForest)")
    fig.tight_layout()
    fig.savefig(CHARTS_DIR / "feature_importance.png", dpi=120)
    plt.close(fig)


def save_confusion_matrix_chart(matrix):
    labels = ["відмова", "схвалено"]
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel("прогноз моделі")
    ax.set_ylabel("фактичне рішення")
    ax.set_title("Матриця помилок на тестовій вибірці")
    fig.tight_layout()
    fig.savefig(CHARTS_DIR / "confusion_matrix.png", dpi=120)
    plt.close(fig)


def main():
    X, y = load_data()
    # stratify зберігає співвідношення 71/29 і в навчальній, і в тестовій частині.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Точка відліку для файнтюну: той самий Pipeline з параметрами лісу за замовчуванням.
    default_score = cross_val_score(
        build_pipeline(), X_train, y_train, scoring="f1_macro", cv=cv
    ).mean()
    print(f"RandomForest з параметрами за замовчуванням, F1 macro: {default_score:.3f}")

    # Порівнюю два різні підходи: ліс дерев і логістичну регресію.
    param_grid = [
        {
            "classifier": [RandomForestClassifier(random_state=42)],
            "classifier__n_estimators": [100, 300],
            "classifier__max_depth": [None, 3, 5, 8],
            "classifier__min_samples_leaf": [1, 3, 5],
            "classifier__class_weight": [None, "balanced"],
        },
        {
            "classifier": [LogisticRegression(max_iter=1000)],
            "classifier__C": [0.01, 0.1, 1, 10],
            "classifier__class_weight": [None, "balanced"],
        },
    ]
    # f1_macro усереднює якість по обох класах. Accuracy тут не годиться:
    # модель, що завжди каже "схвалити", вже має 71%.
    search = GridSearchCV(
        build_pipeline(), param_grid, scoring="f1_macro", cv=cv, n_jobs=-1
    )
    search.fit(X_train, y_train)

    print("\nНайкращі параметри:")
    for name, value in search.best_params_.items():
        print(f"  {name}: {value}")
    print(f"F1 macro на перехресній валідації: {search.best_score_:.3f}")

    results = pd.DataFrame(search.cv_results_)
    results["model"] = results["param_classifier"].map(lambda clf: type(clf).__name__)
    best_by_model = results.groupby("model")["mean_test_score"].max()
    print("\nНайкращий результат кожного типу моделі (F1 macro, перехресна валідація):")
    print(best_by_model.round(3).to_string())

    baseline = DummyClassifier(strategy="most_frequent").fit(X_train, y_train)
    baseline_f1 = f1_score(y_test, baseline.predict(X_test), average="macro")
    print(f"\nБазова модель 'завжди схвалити', F1 macro на тесті: {baseline_f1:.3f}")

    y_pred = search.predict(X_test)
    print("\nНайкраща модель на тестовій вибірці:")
    print(classification_report(y_test, y_pred, target_names=["відмова", "схвалено"]))
    matrix = confusion_matrix(y_test, y_pred)
    print("Матриця помилок (рядки - факт, стовпці - прогноз):")
    print(matrix)

    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(search.best_estimator_, MODEL_PATH)
    print(f"\nМодель збережено: {MODEL_PATH}")

    metrics = {
        "test_size": int(len(y_test)),
        "accuracy": round(accuracy_score(y_test, y_pred), 3),
        "f1_macro": round(f1_score(y_test, y_pred, average="macro"), 3),
        "recall_rejected": round(recall_score(y_test, y_pred, pos_label=0), 3),
        "recall_approved": round(recall_score(y_test, y_pred, pos_label=1), 3),
        "baseline_f1_macro": round(baseline_f1, 3),
        "cv_default_forest": round(default_score, 3),
        "cv_best": round(search.best_score_, 3),
        "cv_by_model": {name: round(score, 3) for name, score in best_by_model.items()},
        "best_params": {
            name.replace("classifier__", ""): str(value)
            for name, value in search.best_params_.items()
            if name != "classifier"
        },
        "feature_importances": feature_importances(search.best_estimator_),
    }
    METRICS_PATH.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Показники збережено: {METRICS_PATH}")

    CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    if metrics["feature_importances"]:
        save_feature_importance_chart(metrics["feature_importances"])
    save_confusion_matrix_chart(matrix)
    print(f"Графіки збережено: {CHARTS_DIR}")


if __name__ == "__main__":
    main()
