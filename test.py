import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# carregar dados
df = pd.read_csv("personalised_dataset.csv")

# garantir que temos as colunas
assert "Exercise_Recommendation" in df and "Glucose_Level" in df, "colunas ausentes"

# ver estatísticas resumidas por tipo de recomendação
summary = df.groupby("Exercise_Recommendation")["Glucose_Level"].describe()
print("\n=== Estatísticas de Glucose_Level por tipo de exercício ===")
print(summary)

# boxplot para visualizar distribuições
plt.figure(figsize=(8, 5))
sns.boxplot(data=df, x="Exercise_Recommendation", y="Glucose_Level", palette="Set2")
plt.title("Distribuição de Glucose_Level por tipo de Exercise_Recommendation")
plt.xticks(rotation=20, ha="right")
plt.tight_layout()
plt.savefig("plots/glucose_vs_exercise.png", dpi=150)
plt.show()
print("\nGráfico salvo em: plots/glucose_vs_exercise.png")
