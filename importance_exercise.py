# Etapa 1 — Exploratória: importância de features para prever Exercise_Recommendation

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.inspection import permutation_importance
import numpy as np

CSV_PATH = "personalised_dataset.csv"  # ajuste se necessário
TARGET = "Exercise_Recommendation"  # agora estamos analisando o modelo de EXERCÍCIO
N_TOP = 12  # quantas features destacar no gráfico

# 1) carregar dados
df = pd.read_csv(CSV_PATH)

# 2) descartar colunas que causam vazamento/ID
to_drop = [
    "Diet_Recommendation",  # dropa a outra recomendação
    "Exercise_Recommendation",  # dropa o próprio target da tabela
    "id",
    "ID",
    "patient_id",
    "Patient_ID",
]
to_drop = [c for c in to_drop if c in df.columns]

y = df[TARGET].copy()
X = df.drop(columns=to_drop).copy()

# 3) remover linhas com alvo faltante e alinhar X/y
mask = y.notna()
X, y = X[mask], y[mask]

# 4) detectar tipos
cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
num_cols = X.select_dtypes(include=["number", "bool"]).columns.tolist()

# 5) pipeline: ordinal encode p/ categóricas + random forest
pre = ColumnTransformer(
    transformers=[
        (
            "cat",
            OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
            cat_cols,
        ),
        ("num", "passthrough", num_cols),
    ],
    remainder="drop",
)

rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    random_state=42,
    n_jobs=-1,
    class_weight="balanced_subsample",
)

pipe = Pipeline([("pre", pre), ("rf", rf)])

# 6) treino/validação rápida
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
pipe.fit(X_tr, y_tr)
y_pred = pipe.predict(X_te)
print(
    f"[{TARGET}] Accuracy: {accuracy_score(y_te, y_pred):.3f} | "
    f"F1-weighted: {f1_score(y_te, y_pred, average='weighted'):.3f}"
)

# 7) importância por permutação
out_feature_names = cat_cols + num_cols
r = permutation_importance(pipe, X_te, y_te, n_repeats=10, random_state=42, n_jobs=-1)
importances = pd.Series(r.importances_mean, index=out_feature_names)

# ordenar e mostrar top-N
imp_top = importances.sort_values(ascending=True).tail(N_TOP)
print("\nTop features para Exercise_Recommendation:")
print(importances.sort_values(ascending=False).head(N_TOP))

# 8) plot e salvar
plt.figure(figsize=(8, max(4, 0.4 * N_TOP)))
imp_top.plot(kind="barh")
plt.title(f"Top {N_TOP} importâncias — {TARGET}")
plt.xlabel("Importância (permutation)")
plt.tight_layout()
png_name = f"importances_{TARGET.lower().replace(' ', '_')}.png"
plt.savefig(png_name, dpi=150)
plt.show()
print(f"Gráfico salvo em: {png_name}")

df.groupby("Exercise_Recommendation")["Glucose_Level"].describe()
