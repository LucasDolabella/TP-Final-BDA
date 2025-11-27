# model_final.py

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
from sklearn.model_selection import cross_val_score
import joblib

# CONFIGURAÇÕES

CSV_PATH = "personalised_dataset.csv"
TARGET = "Diet_Recommendation"

# TOP 12 FEATURES mais importantes (baseado em importance_diet.py)
SELECTED_FEATURES = [
    "Alcohol_Consumption",
    "LDL",
    "Gender",
    "BRCA_Pathogenic_Variant",
    "HbA1c",
    "Systolic_BP",
    "HDL",
    "Cholesterol",
    "Glucose_Level",
    "PRS_Cardiometabolic",
    "Physical_Activity_Level",
    "HRV",
    "BMI",
    "Waist_Circumference",
    "Triglycerides",      
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
print(f"Features selecionadas (Top 15 de importance_diet.py):")
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

# MODELO - CONFIGURAÇÃO ANTI-OVERFITTING

model = Pipeline(
    [
        ("pre", preprocessor),
        (
            "rf",
            RandomForestClassifier(
                n_estimators=200,
                max_depth=8,
                min_samples_leaf=8,
                min_samples_split=20,
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

# Validação Cruzada (5-fold) para métrica mais robusta
print("\n" + "="*70)
print("VALIDAÇÃO CRUZADA (5-FOLD)")
print("="*70)
cv_scores = cross_val_score(model, X, y_enc, cv=5, scoring='accuracy', n_jobs=-1)
print(f"Scores por fold: {[f'{s:.3f}' for s in cv_scores]}")
print(f"Média CV:  {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
print(f"Min/Max:   {cv_scores.min():.3f} / {cv_scores.max():.3f}")

# Se desvio padrão > 0.05, indica instabilidade
if cv_scores.std() > 0.05:
    print(f"  ALERTA: Alta variabilidade entre folds (std={cv_scores.std():.3f})")
else:
    print(f"✓ Modelo estável entre folds")
print("="*70 + "\n")

# Validação de overfitting
y_train_pred = model.predict(X_train)
acc_train = accuracy_score(y_train, y_train_pred)
acc_test = accuracy_score(y_test, y_pred)

print(f"\nValidação de Overfitting:")
print(f"  Accuracy (treino): {acc_train:.3f} ({acc_train*100:.1f}%)")
print(f"  Accuracy (teste):  {acc_test:.3f} ({acc_test*100:.1f}%)")
print(f"  Diferença:         {(acc_train - acc_test)*100:.1f}%")

# Diagnóstico detalhado
diff = acc_train - acc_test

if diff > 0.20:
    print(f"  OVERFITTING SEVERO! Diferença de {diff*100:.1f}%")
    print(f"     → Modelo memorizou treino, não generalizou")
    print(f"     → Reduzir max_depth ou aumentar min_samples_leaf")
elif diff > 0.15:
    print(f"  OVERFITTING MODERADO. Diferença de {diff*100:.1f}%")
    print(f"     → Modelo precisa de mais regularização")
elif diff > 0.10:
    print(f"  Leve overfitting. Diferença de {diff*100:.1f}%")
    print(f"     → Aceitável, mas pode melhorar")
elif acc_test > 0.95:
    print(f"  ALERTA: Acurácia muito alta ({acc_test*100:.1f}%) - verificar data leakage!")
elif diff < 0:
    print(f"  UNDERFITTING: Treino pior que teste (diferença: {diff*100:.1f}%)")
    print(f"     → Aumentar complexidade do modelo")
else:
    print(f"  Modelo generaliza bem (diferença: {diff*100:.1f}%)")


# MÉTRICAS ESSENCIAIS

acc = acc_test
f1w = f1_score(y_test, y_pred, average="weighted")
top2 = (np.argsort(proba, axis=1)[:, -2:] == y_test.reshape(-1, 1)).any(axis=1).mean()

# Cálculo do MAE das probabilidades
y_test_bin = label_binarize(y_test, classes=np.arange(len(label_enc.classes_)))
mae = mean_absolute_error(y_test_bin, proba)

print("\n=== RESULTADOS FINAIS — Diet_Recommendation ===")
print(f"Accuracy........: {acc:.3f}")
print(f"F1-weighted.....: {f1w:.3f}")
print(f"Top-2 Accuracy..: {top2:.3f}")
print(f"Mean Absolute Error (MAE): {mae:.4f}\n")

print("Relatório de Classificação:")
print(classification_report(y_test, y_pred, target_names=label_enc.classes_))


# ANÁLISE DETALHADA DA MATRIZ DE CONFUSÃO
print("\n" + "="*70)
print("ANÁLISE DETALHADA DA MATRIZ DE CONFUSÃO")
print("="*70)

cm = confusion_matrix(y_test, y_pred)
cm_percent = cm / cm.sum(axis=1, keepdims=True)

# Identificar classes problemáticas
print("\nPerformance por Classe (Diagonal da Matriz):")
for i, class_name in enumerate(label_enc.classes_):
    recall = cm_percent[i, i]
    total_samples = cm[i].sum()
    status = "✅" if recall >= 0.70 else "⚠️" if recall >= 0.50 else "❌"
    print(f"{status} {class_name:40s} → {recall:.1%} ({cm[i, i]}/{total_samples} amostras)")

# Identificar confusões principais
print("\n Principais Confusões (>10%):")
for i, true_class in enumerate(label_enc.classes_):
    for j, pred_class in enumerate(label_enc.classes_):
        if i != j and cm_percent[i, j] > 0.10:
            print(f"  {true_class[:35]:35s} → {pred_class[:35]:35s}: {cm_percent[i, j]:.1%}")

print("="*70 + "\n")


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

cm = confusion_matrix(y_test, y_pred)
cm_percent = cm / cm.sum(axis=1, keepdims=True)

fig, ax = plt.subplots(figsize=(9, 6))

sns.heatmap(
    cm_percent,
    annot=True,
    fmt=".2f",
    cmap="viridis",
    xticklabels=label_enc.classes_,
    yticklabels=label_enc.classes_,
    cbar=False,
    annot_kws={"size": 12},
)

plt.xticks(rotation=25, ha="right", fontsize=10)
plt.yticks(rotation=0, fontsize=10)

plt.title("Matriz de Confusão (Percentual) — Diet_Recommendation", fontsize=14)
plt.xlabel("Predicted", fontsize=12)
plt.ylabel("True", fontsize=12)

plt.tight_layout()
plt.savefig(PLOTS_DIR / "confusion_matrix_diet_percent.png", dpi=200)
plt.show()

# ver distribuição das classes
print("\nDistribuição das classes no dataset:")
print(df["Diet_Recommendation"].value_counts())


# SALVAR MODELO

joblib.dump(model, MODELS_DIR / "diet_recommender_final.pkl")
joblib.dump(label_enc, MODELS_DIR / "diet_label_encoder_final.pkl")
print("\nModelos salvos em pasta models/")