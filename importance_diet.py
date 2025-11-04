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

CSV_PATH = "personalised_dataset.csv"
TARGET = "Diet_Recommendation"  #
N_TOP = 12  # quantas features destacar no gráfico

df = pd.read_csv(CSV_PATH)

# descartar colunas "irrelevantes"
to_drop = [
    "Diet_Recommendation",
    "Exercise_Recommendation",
    "Patient_ID",
]
to_drop = [c for c in to_drop if c in df.columns]

y = df[TARGET].copy()
X = df.drop(columns=to_drop).copy()

# remover linhas com alvo faltante e alinhar X/y
mask = y.notna()
X, y = X[mask], y[mask]

# detectar tipos
cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
num_cols = X.select_dtypes(include=["number", "bool"]).columns.tolist()

# pipeline -> ordinal encode p/ categóricas + random forest
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

# treino/validação rápida (só p ter uma noção de desempenho)
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
pipe.fit(X_tr, y_tr)
y_pred = pipe.predict(X_te)
print(
    f"[{TARGET}] Accuracy: {accuracy_score(y_te, y_pred):.3f} | F1-weighted: {f1_score(y_te, y_pred, average='weighted'):.3f}"
)

#  importância por permutação (mais estável p/ dados categóricos)
#    mapeando importância de volta às colunas originais (cat + num)
preprocess = pipe.named_steps["pre"]
rf_model = pipe.named_steps["rf"]

# nomes das colunas após o pré-processamento (ordinal mantém 1-1 com colunas originais)
out_feature_names = cat_cols + num_cols

# calcular importâncias
r = permutation_importance(pipe, X_te, y_te, n_repeats=10, random_state=42, n_jobs=-1)
importances = pd.Series(r.importances_mean, index=out_feature_names)

# ordenar e mostrar top-N
imp_top = importances.sort_values(ascending=True).tail(N_TOP)

print("\nTop features:")
print(importances.sort_values(ascending=False).head(N_TOP))

# plot e salvar
plt.figure(figsize=(8, max(4, 0.4 * N_TOP)))
imp_top.plot(kind="barh")
plt.title(f"Top {N_TOP} importâncias — {TARGET}")
plt.xlabel("Importância (permutation)")
plt.tight_layout()
png_name = f"importances_{TARGET.lower().replace(' ', '_')}.png"
plt.savefig(png_name, dpi=150)
plt.show()
print(f"Gráfico salvo em: {png_name}")
