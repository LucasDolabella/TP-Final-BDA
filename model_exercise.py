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

# FEATURES SELECIONADAS
# Baseado em importâncias + evidência clínica para prescrição de exercício
SELECTED_FEATURES = [
    "LDL",
    "HDL",
    "Cholesterol",
    "Triglycerides",
    "HbA1c",
    "Glucose_Level",
    "Systolic_BP",
    "Diastolic_BP",
    "HRV",
    "eGFR",
    "CRP",
    "PRS_Cardiometabolic",
    "Family_History_CVD",
    "BRCA_Pathogenic_Variant",
    "BMI",
    "Waist_Circumference",
    "Age",
]

RANDOM_STATE = 42
TEST_SIZE = 0.25

MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True, parents=True)
PLOTS_DIR = Path("plots")
PLOTS_DIR.mkdir(exist_ok=True, parents=True)


# CARREGAR E PREPARAR DADOS

df = pd.read_csv(CSV_PATH)

# Verificar se todas as features existem
missing = [f for f in SELECTED_FEATURES if f not in df.columns]
if missing:
    raise ValueError(f"Features ausentes no dataset: {missing}")

X = df[SELECTED_FEATURES].copy()
y = df[TARGET].copy()

print(f"\n{'='*70}")
print(f"MODELO DE RECOMENDAÇÃO: {TARGET}")
print(f"{'='*70}")
print(f"Features selecionadas (Top 12 de importance_exercise.py):")
for i, feat in enumerate(SELECTED_FEATURES, 1):
    print(f"  {i:2d}. {feat}")
print(f"\nTotal de features: {len(SELECTED_FEATURES)}")
print(f"Total de amostras: {len(X)}")
print(f"{'='*70}\n")

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
                max_depth=8,             # Balanceado
                min_samples_leaf=8,      # Moderado
                min_samples_split=20,    # Moderado
                max_features='sqrt',
                min_impurity_decrease=0.001,  
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

# Validação de overfitting
y_train_pred = model.predict(X_train)
acc_train = accuracy_score(y_train, y_train_pred)
acc_test = accuracy_score(y_test, y_pred)

print(f"\nValidação de Overfitting:")
print(f"  Accuracy (treino): {acc_train:.3f}")
print(f"  Accuracy (teste):  {acc_test:.3f}")
print(f"  Diferença:         {(acc_train - acc_test)*100:.1f}%")

if acc_train - acc_test > 0.15:
    print(f"  ALERTA: Possível overfitting!")
elif acc_test > 0.95:
    print(f"  ALERTA: Acurácia muito alta - verificar data leakage!")
else:
    print(f"  Modelo generaliza adequadamente")


# MÉTRICAS ESSENCIAIS

acc = acc_test
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


# IMPORTÂNCIA DE FEATURES (Feature Importance do modelo treinado)
print("\n" + "="*70)
print("IMPORTÂNCIA DAS 12 FEATURES SELECIONADAS")
print("="*70)

rf_model = model.named_steps['rf']
feature_names = SELECTED_FEATURES

importances = rf_model.feature_importances_
indices = np.argsort(importances)[::-1]

for i in range(len(feature_names)):
    idx = indices[i]
    print(f"{i+1:2d}. {feature_names[idx]:30s} → {importances[idx]:.4f}")

print("="*70 + "\n")


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