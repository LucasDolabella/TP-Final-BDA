# model_final.py — versão enxuta e limpa do modelo de recomendação de dieta
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
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
)
import joblib

# ===============================
# CONFIGURAÇÕES
# ===============================
CSV_PATH = "personalised_dataset.csv"
TARGET = "Diet_Recommendation"
SELECTED_FEATURES = [
    "Alcohol_Consumption",
    "Cholesterol",
    "Glucose_Level",
    "LDL",
    "BMI",
    "Triglycerides",
    "Systolic_BP",
]
RANDOM_STATE = 42
TEST_SIZE = 0.2

MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True, parents=True)
PLOTS_DIR = Path("plots")
PLOTS_DIR.mkdir(exist_ok=True, parents=True)

# ===============================
# 1. CARREGAR E PREPARAR DADOS
# ===============================
df = pd.read_csv(CSV_PATH)
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

# ===============================
# 2. TREINO / TESTE
# ===============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y_enc, test_size=TEST_SIZE, stratify=y_enc, random_state=RANDOM_STATE
)

# ===============================
# 3. MODELO
# ===============================
model = Pipeline(
    [
        ("pre", preprocessor),
        (
            "rf",
            RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                min_samples_leaf=3,
                class_weight="balanced_subsample",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
        ),
    ]
)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)
proba = model.predict_proba(X_test)

# ===============================
# 4. MÉTRICAS ESSENCIAIS
# ===============================
acc = accuracy_score(y_test, y_pred)
f1w = f1_score(y_test, y_pred, average="weighted")
top2 = (np.argsort(proba, axis=1)[:, -2:] == y_test.reshape(-1, 1)).any(axis=1).mean()

print("\n=== RESULTADOS FINAIS — Diet_Recommendation ===")
print(f"Accuracy........: {acc:.3f}")
print(f"F1-weighted.....: {f1w:.3f}")
print(f"Top-2 Accuracy..: {top2:.3f}\n")

print("Relatório de Classificação:")
print(classification_report(y_test, y_pred, target_names=label_enc.classes_))

# ===============================
# 5. MATRIZ DE CONFUSÃO
# ===============================
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_enc.classes_)
fig, ax = plt.subplots(figsize=(7, 6))
disp.plot(ax=ax, xticks_rotation=45, colorbar=False)
plt.title("Matriz de Confusão — Diet_Recommendation (Final)")
plt.tight_layout()
plt.savefig(PLOTS_DIR / "confusion_matrix_diet_final.png", dpi=150)
plt.show()
print(df["Diet_Recommendation"].value_counts())

# ===============================
# 6. SALVAR MODELO
# ===============================
joblib.dump(model, MODELS_DIR / "diet_recommender_final.pkl")
joblib.dump(label_enc, MODELS_DIR / "label_encoder_final.pkl")
print("\nModelos salvos em pasta models/")
