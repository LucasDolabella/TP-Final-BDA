# Análise de Importância de Features para Diet_Recommendation

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.inspection import permutation_importance
import numpy as np
from pathlib import Path

# Configurações
CSV_PATH = "personalised_dataset.csv"
TARGET = "Diet_Recommendation"
N_TOP = 15
PLOTS_DIR = Path("plots")
PLOTS_DIR.mkdir(exist_ok=True, parents=True)

df = pd.read_csv(CSV_PATH)

# Descartar colunas irrelevantes e colunas derivadas que causam data leakage
to_drop = [
    "Diet_Recommendation", 
    "Exercise_Recommendation", 
    "Patient_ID",
    "Health_Risk",
    "Heart_Disease_Risk",
    "Diabetes_Risk",
    "Predicted_Insurance_Cost",
]
to_drop = [c for c in to_drop if c in df.columns]

y = df[TARGET].copy()
X = df.drop(columns=to_drop).copy()

print(f"\n Colunas removidas (data leakage): Health_Risk, Heart_Disease_Risk, Diabetes_Risk, Predicted_Insurance_Cost")

# Remover linhas com alvo faltante
mask = y.notna()
X, y = X[mask], y[mask]

# Detectar tipos de colunas
cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
num_cols = X.select_dtypes(include=["number", "bool"]).columns.tolist()

print(f"\n{'='*60}")
print(f"ANÁLISE DE IMPORTÂNCIA DE FEATURES - {TARGET}")
print(f"{'='*60}")
print(f"\nTotal de amostras: {len(X)}")
print(f"Features categóricas: {len(cat_cols)}")
print(f"Features numéricas: {len(num_cols)}")
print(f"Total de features: {len(X.columns)}")
print(f"\nDistribuição das classes:")
print(y.value_counts())

# Pipeline com imputação e encoding
preprocessor = ColumnTransformer(
    transformers=[
        (
            "cat",
            Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
            ]),
            cat_cols,
        ),
        (
            "num",
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
            ]),
            num_cols,
        ),
    ],
    remainder="drop",
)

# Modelo Random Forest para Diet
rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=8,
    min_samples_split=20,
    min_samples_leaf=8,
    max_features='sqrt',
    min_impurity_decrease=0.001,
    random_state=42,
    n_jobs=-1,
    class_weight="balanced",
)

pipe = Pipeline([("preprocessor", preprocessor), ("rf", rf)])

# Treino/validação
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

print(f"\n{'='*60}")
print("TREINAMENTO DO MODELO")
print(f"{'='*60}")
print(f"Tamanho do treino: {len(X_train)} amostras")
print(f"Tamanho do teste: {len(X_test)} amostras")
print("Treinando Random Forest (com regularização para evitar overfitting)...")
pipe.fit(X_train, y_train)

y_pred = pipe.predict(X_test)
acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='weighted')

# Calcular accuracy no treino para detectar overfitting
y_train_pred = pipe.predict(X_train)
acc_train = accuracy_score(y_train, y_train_pred)

print(f"\n✓ Modelo treinado com sucesso!")
print(f"  Accuracy (TREINO): {acc_train:.3f} ({acc_train*100:.1f}%)")
print(f"  Accuracy (TESTE):  {acc:.3f} ({acc*100:.1f}%)")
print(f"  F1-weighted:       {f1:.3f}")

# Detectar overfitting
if acc_train - acc > 0.15:
    print(f"\n  ALERTA: Possível overfitting detectado!")
    print(f"  Diferença treino-teste: {(acc_train - acc)*100:.1f}%")
elif acc > 0.95:
    print(f"\n ALERTA: Acurácia muito alta ({acc*100:.1f}%) - verificar data leakage!")
else:
    print(f"\n✓ Modelo generaliza bem (diferença treino-teste: {(acc_train - acc)*100:.1f}%)")

# Calcular importâncias por PERMUTAÇÃO
print(f"\n{'='*60}")
print("CALCULANDO IMPORTÂNCIA POR PERMUTAÇÃO")
print(f"{'='*60}")
print("Executando permutation importance (10 repetições)...")

out_feature_names = cat_cols + num_cols
result = permutation_importance(
    pipe, X_test, y_test, 
    n_repeats=10, 
    random_state=42, 
    n_jobs=-1,
    scoring='accuracy'
)

importances = pd.DataFrame({
    'feature': out_feature_names,
    'importance_mean': result.importances_mean,
    'importance_std': result.importances_std
}).sort_values('importance_mean', ascending=False)

print("\n✓ Importâncias calculadas!")

# Exibir top features
print(f"\n{'='*60}")
print(f"TOP {N_TOP} FEATURES MAIS IMPORTANTES")
print(f"{'='*60}")
for idx, row in importances.head(N_TOP).iterrows():
    print(f"{row['feature']:30s} | {row['importance_mean']:8.5f} ± {row['importance_std']:.5f}")

# VISUALIZAÇÃO 1: Barras horizontais com erro padrão
fig, ax = plt.subplots(figsize=(10, 8))

top_features = importances.head(N_TOP).sort_values('importance_mean')
colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(top_features)))

bars = ax.barh(
    range(len(top_features)), 
    top_features['importance_mean'],
    xerr=top_features['importance_std'],
    color=colors,
    edgecolor='black',
    linewidth=1.2,
    alpha=0.85,
    capsize=5
)

ax.set_yticks(range(len(top_features)))
ax.set_yticklabels(top_features['feature'], fontsize=11)
ax.set_xlabel('Importância (Permutation Importance)', fontsize=12, fontweight='bold')
ax.set_title(f'Top {N_TOP} Features Mais Importantes\n{TARGET}', 
             fontsize=14, fontweight='bold', pad=20)
ax.grid(axis='x', alpha=0.3, linestyle='--')

# Adicionar valores nas barras
for i, (idx, row) in enumerate(top_features.iterrows()):
    ax.text(row['importance_mean'] + row['importance_std'] + 0.001, i, 
            f"{row['importance_mean']:.4f}", 
            va='center', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig(PLOTS_DIR / f"importance_{TARGET.lower().replace(' ', '_')}.png", dpi=200, bbox_inches='tight')
plt.show()
print(f"\n✓ Gráfico 1 salvo: plots/importance_{TARGET.lower().replace(' ', '_')}.png")

# VISUALIZAÇÃO 2: Heatmap de importâncias (categorias vs numéricas)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

# Features categóricas
cat_importance = importances[importances['feature'].isin(cat_cols)].head(10)
if len(cat_importance) > 0:
    ax1.barh(range(len(cat_importance)), cat_importance['importance_mean'], 
             color='salmon', edgecolor='darkred', linewidth=1.5, alpha=0.8)
    ax1.set_yticks(range(len(cat_importance)))
    ax1.set_yticklabels(cat_importance['feature'], fontsize=10)
    ax1.set_xlabel('Importância', fontsize=11, fontweight='bold')
    ax1.set_title('Top Features Categóricas', fontsize=13, fontweight='bold')
    ax1.grid(axis='x', alpha=0.3)

# Features numéricas
num_importance = importances[importances['feature'].isin(num_cols)].head(10)
if len(num_importance) > 0:
    ax2.barh(range(len(num_importance)), num_importance['importance_mean'], 
             color='skyblue', edgecolor='darkblue', linewidth=1.5, alpha=0.8)
    ax2.set_yticks(range(len(num_importance)))
    ax2.set_yticklabels(num_importance['feature'], fontsize=10)
    ax2.set_xlabel('Importância', fontsize=11, fontweight='bold')
    ax2.set_title('Top Features Numéricas', fontsize=13, fontweight='bold')
    ax2.grid(axis='x', alpha=0.3)

plt.suptitle(f'Importância de Features por Tipo - {TARGET}', 
             fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(PLOTS_DIR / f"importance_by_type_{TARGET.lower().replace(' ', '_')}.png", 
            dpi=200, bbox_inches='tight')
plt.show()
print(f"✓ Gráfico 2 salvo: plots/importance_by_type_{TARGET.lower().replace(' ', '_')}.png")

# ANÁLISE ADICIONAL: Correlação entre top features e target
print(f"\n{'='*60}")
print("ANÁLISE DE CORRELAÇÃO COM TARGET")
print(f"{'='*60}")

# Converter target para numérico para correlação
label_enc = LabelEncoder()
y_numeric = label_enc.fit_transform(y)

# Pegar apenas features numéricas do top
top_num_features = [f for f in importances.head(N_TOP)['feature'] if f in num_cols]

if len(top_num_features) > 0:
    correlations = []
    for feat in top_num_features:
        corr = np.corrcoef(X[feat].fillna(X[feat].median()), y_numeric)[0, 1]
        correlations.append({'feature': feat, 'correlation': abs(corr)})
    
    df_corr = pd.DataFrame(correlations).sort_values('correlation', ascending=False)
    
    print("\nCorrelação absoluta com Diet_Recommendation:")
    for _, row in df_corr.head(10).iterrows():
        print(f"{row['feature']:30s} | {row['correlation']:.4f}")

# Salvar CSV com todas as importâncias
importances.to_csv(PLOTS_DIR / f"feature_importances_{TARGET.lower().replace(' ', '_')}.csv", 
                   index=False)
print(f"\n✓ Importâncias salvas em CSV: plots/feature_importances_{TARGET.lower().replace(' ', '_')}.csv")

print(f"\n{'='*60}")
print("ANÁLISE CONCLUÍDA COM SUCESSO!")
print(f"{'='*60}\n")