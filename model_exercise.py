import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder, label_binarize
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    mean_absolute_error,
)
import joblib


# CONFIGURAÇÕES

CSV_PATH = "personalised_dataset.csv"
TARGET = "Exercise_Recommendation"
SELECTED_FEATURES = [
    # Top sinais para exercício
    "Glucose_Level",
    "HbA1c",
    "BMI",
    "Age",
    "Physical_Activity_Level",
    "Cholesterol",
    "LDL",
    "HDL",
    "Resting_Heart_Rate",
    "HRV",
    "Family_History_CVD",
    "PRS_Cardiometabolic",
    "PRS_Type2Diabetes",
]
RANDOM_STATE = 42
TEST_SIZE = 0.2

MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True, parents=True)
PLOTS_DIR = Path("plots")
PLOTS_DIR.mkdir(exist_ok=True, parents=True)


# CARREGAR E PREPARAR DADOS
df = pd.read_csv(CSV_PATH)

# verificar se as colunas existem
missing = [c for c in SELECTED_FEATURES + [TARGET] if c not in df.columns]
if missing:
    raise ValueError(f"Colunas ausentes no CSV: {missing}")

X = df[SELECTED_FEATURES].copy()
y = df[TARGET].copy()

cat_cols = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
num_cols = X.columns.difference(cat_cols).tolist()

preprocessor = ColumnTransformer(
    [
        (
            "cat",
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    (
                        "encoder",
                        OrdinalEncoder(
                            handle_unknown="use_encoded_value", unknown_value=-1
                        ),
                    ),
                ]
            ),
            cat_cols,
        ),
        (
            "num",
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                ]
            ),
            num_cols,
        ),
    ]
)

label_enc = LabelEncoder()
y_enc = label_enc.fit_transform(y)


# TREINO / TESTE

X_train, X_test, y_train, y_test = train_test_split(
    X, y_enc, test_size=TEST_SIZE, stratify=y_enc, random_state=RANDOM_STATE
)


# MODELO
model = Pipeline(
    [
        ("pre", preprocessor),
        (
            "rf",
            RandomForestClassifier(
                n_estimators=200,
                max_depth=15,
                min_samples_leaf=2,
                class_weight="balanced",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
        ),
    ]
)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)
proba = model.predict_proba(X_test)


# MÉTRICAS ESSENCIAIS

acc = accuracy_score(y_test, y_pred)
f1w = f1_score(y_test, y_pred, average="weighted")
top2 = (np.argsort(proba, axis=1)[:, -2:] == y_test.reshape(-1, 1)).any(axis=1).mean()

# MAE das probabilidades
y_test_bin = label_binarize(y_test, classes=np.arange(len(label_enc.classes_)))
mae = mean_absolute_error(y_test_bin, proba)

print("\n=== RESULTADOS FINAIS — Exercise_Recommendation ===")
print(f"Accuracy........: {acc:.3f}")
print(f"F1-weighted.....: {f1w:.3f}")
print(f"Top-2 Accuracy..: {top2:.3f}")
print(f"Mean Absolute Error (MAE): {mae:.4f}\n")

print("Relatório de Classificação:")
print(classification_report(y_test, y_pred, target_names=label_enc.classes_))


# MATRIZ DE CONFUSÃO

# Matriz de confusão
cm = confusion_matrix(y_test, y_pred)

# Normalizar por linha (percentual por classe real)
cm_percent = cm / cm.sum(axis=1, keepdims=True)

# Quebrar nomes longos em múltiplas linhas
labels = [s.replace("; ", ";\n") for s in label_enc.classes_]

# Plot
fig, ax = plt.subplots(figsize=(10, 6))

sns.heatmap(
    cm_percent,
    annot=True,
    fmt=".2f",
    cmap="viridis",
    xticklabels=labels,
    yticklabels=labels,
    cbar=False,
    annot_kws={"size": 12},
)

plt.xticks(rotation=25, ha="right", fontsize=11)
plt.yticks(rotation=0, fontsize=11)

plt.title("Matriz de Confusão (Percentual) — Exercise_Recommendation", fontsize=15)
plt.xlabel("Predicted label", fontsize=12)
plt.ylabel("True label", fontsize=12)

plt.tight_layout()
plt.savefig(PLOTS_DIR / "confusion_matrix_exercise_percent.png", dpi=200)
plt.show()

# distribuição das classes
print("\nDistribuição das classes no dataset:")
print(df["Exercise_Recommendation"].value_counts())


# SALVAR MODELO

joblib.dump(model, MODELS_DIR / "exercise_recommender_final.pkl")
joblib.dump(label_enc, MODELS_DIR / "exercise_label_encoder_final.pkl")
print("\nModelos salvos em pasta models/")
