# TP Final - Banco de Dados Avançados

Sistema de recomendação personalizado para dieta e exercícios usando Machine Learning com Random Forest.

## 📋 Descrição do Projeto

Este projeto implementa modelos de classificação para recomendar planos alimentares e programas de exercícios personalizados com base em dados de saúde dos pacientes. Utiliza algoritmos de Random Forest para prever as melhores recomendações considerando diversos fatores de saúde e estilo de vida.

## 🎯 Objetivos

- **Modelo de Dieta**: Recomenda tipos de dieta (Mediterrânea, dieta balanceada, baixo índice glicêmico, etc.)
- **Modelo de Exercícios**: Sugere programas de exercícios adequados ao perfil de saúde do paciente

## 📊 Dataset

O dataset (`personalised_dataset.csv`) contém 2002 registros de pacientes com 40 atributos, incluindo:

### Características Demográficas
- Age, Gender, BMI

### Indicadores de Saúde
- Glucose_Level, HbA1c, Cholesterol
- LDL, HDL, Triglycerides
- Systolic_BP, Diastolic_BP
- CRP, eGFR, Waist_Circumference

### Estilo de Vida
- Physical_Activity_Level
- Smoking_Status
- Alcohol_Consumption
- Diet_Type
- Sleep_Hours, Sleep_Quality

### Fatores Psicológicos
- Stress_Level
- Depression_Score
- Anxiety_Score
- Social_Isolation_Index

### Fatores Genéticos
- PRS_Cardiometabolic
- PRS_Type2Diabetes
- APOE_e4_Carrier
- Family_History_CVD, Family_History_T2D

### Variáveis Alvo
- **Diet_Recommendation**: Tipo de dieta recomendada
- **Exercise_Recommendation**: Programa de exercícios recomendado

## 🏗️ Estrutura do Projeto

```
TP-Final-BDA/
├── personalised_dataset.csv          # Dataset principal
├── model_diet.py                     # Modelo de recomendação de dieta
├── model_exercise.py                 # Modelo de recomendação de exercícios
├── importance_diet.py                # Análise de importância de features (dieta)
├── importance_exercise.py            # Análise de importância de features (exercícios)
├── test.py                           # Script de teste/visualização
├── models/                           # Modelos treinados salvos
│   ├── diet_recommender_final.pkl
│   ├── label_encoder_final.pkl
│   ├── exercise_recommender_final.pkl
│   └── exercise_label_encoder_final.pkl
├── plots/                            # Gráficos e visualizações
│   ├── confusion_matrix_diet_percent.png
│   ├── confusion_matrix_exercise_percent.png
│   └── glucose_vs_exercise.png
└── README.md
```

## 🔧 Tecnologias Utilizadas

- **Python 3.x**
- **Pandas**: Manipulação de dados
- **NumPy**: Operações numéricas
- **Scikit-learn**: Machine Learning
  - RandomForestClassifier
  - Pipeline e ColumnTransformer
  - Métricas de avaliação
- **Matplotlib & Seaborn**: Visualização de dados
- **Joblib**: Serialização de modelos

## 📦 Instalação

```bash
# Instalar dependências
pip install pandas numpy scikit-learn matplotlib seaborn joblib
```

## 🚀 Como Usar

### 1. Treinar Modelo de Dieta

```bash
python model_diet.py
```

**Saída esperada:**
- Accuracy, F1-score, Top-2 Accuracy
- Mean Absolute Error (MAE)
- Relatório de classificação
- Matriz de confusão salva em `plots/`
- Modelo salvo em `models/diet_recommender_final.pkl`

### 2. Treinar Modelo de Exercícios

```bash
python model_exercise.py
```

**Saída esperada:**
- Métricas de desempenho
- Matriz de confusão percentual
- Modelo salvo em `models/exercise_recommender_final.pkl`

### 3. Análise de Importância de Features

```bash
# Para dieta
python importance_diet.py

# Para exercícios
python importance_exercise.py
```

**Gera múltiplas visualizações:**
- Gráfico principal com top 15 features (com barras de erro)
- Gráfico separado: features categóricas vs numéricas
- CSV com todas as importâncias
- Análise de correlação com target

**Melhorias implementadas:**
- ✅ Permutation importance com 10 repetições (mais robusto que feature importance padrão)
- ✅ Barras de erro padrão (std) para confiabilidade estatística
- ✅ Visualizações separadas por tipo de feature (categóricas vs numéricas)
- ✅ Análise de correlação detalhada com target
- ✅ Outputs formatados e informativos com detecção de overfitting
- ✅ Exportação em CSV para análise posterior
- ✅ **Remoção de features derivadas** (Health_Risk, Heart_Disease_Risk, Diabetes_Risk, Predicted_Insurance_Cost) para evitar **data leakage**
- ✅ **Remoção de Gender** do modelo de exercícios para eliminar viés espúrio (sem significância estatística: p-value=0.46)
- ✅ **Regularização anti-overfitting** (max_depth=8, min_samples_leaf=8, min_samples_split=20, min_impurity_decrease=0.001)
- ✅ **Validação de overfitting** em tempo de execução (comparação treino vs teste)

## 📈 Métricas de Avaliação

### Modelo de Dieta
**Features Selecionadas (15):**
- **Perfil Lipídico**: LDL, HDL, Cholesterol, Triglycerides
- **Metabolismo**: Alcohol_Consumption, HbA1c, Glucose_Level
- **Cardiovascular**: Systolic_BP, HRV
- **Composição Corporal**: BMI, Waist_Circumference
- **Atividade & Risco**: Physical_Activity_Level, PRS_Cardiometabolic
- **Genética**: BRCA_Pathogenic_Variant, Gender

**Configuração do Modelo:**
- n_estimators: 200
- max_depth: 8 (regularização anti-overfitting)
- min_samples_leaf: 8
- min_samples_split: 20
- min_impurity_decrease: 0.001
- class_weight: balanced

**Desempenho:**
- Accuracy: ~80.6%
- F1-weighted: ~80.4%
- Top-2 Accuracy: ~96.0%
- MAE: ~0.20
- Overfitting: 5.9% (diferença treino-teste)

**Performance por Classe:**
- Balanced whole-food diet: 90%
- Calorie deficit, fiber boost: 91%
- Low-glycemic, high-fiber: 57%
- Mediterranean diet: 63%

### Modelo de Exercícios
**Features Selecionadas (18):**
- **Perfil Lipídico**: LDL, HDL, Cholesterol, Triglycerides
- **Metabolismo Glicêmico**: HbA1c, Glucose_Level
- **Cardiovascular**: Systolic_BP, Diastolic_BP, HRV
- **Função Renal/Inflamação**: eGFR, CRP
- **Risco Genético**: PRS_Cardiometabolic, Family_History_CVD, BRCA_Pathogenic_Variant
- **Composição Corporal**: BMI, Waist_Circumference, Age

**Configuração do Modelo:**
- n_estimators: 200
- max_depth: 8
- min_samples_leaf: 8
- min_samples_split: 20
- min_impurity_decrease: 0.001
- class_weight: balanced

**Desempenho Esperado:**
- Accuracy: ~78-80%
- F1-weighted: ~78%
- Top-2 Accuracy: ~97%
- Overfitting controlado: <10%

**Performance por Classe:**
- 150+ min moderate cardio: 85%
- Maintain 90+ min mixed activity: 84%
- ≥120 min cardio + strength: 63%

### Métricas Calculadas

- **Accuracy**: Precisão geral do modelo
- **F1-weighted**: F1-score ponderado por classe (ideal para classes desbalanceadas)
- **Top-2 Accuracy**: Percentual de casos onde a classe correta está entre as 2 principais predições
- **MAE (Mean Absolute Error)**: Erro médio absoluto entre classes ordinais
- **Overfitting Check**: Diferença entre accuracy de treino e teste (< 10% = saudável)

## 📊 Visualizações

O projeto gera automaticamente:

1. **Matrizes de Confusão Percentuais**
   - Visualização normalizada por linha (percentual de acerto por classe)
   - Formato heatmap com anotações de valores
   - Identificação automática de classes problemáticas (<60%)
   - Salvas em alta resolução (200 DPI)
   - Análise detalhada de confusões principais (>10%)

2. **Gráficos de Importância de Features**
   - Top 15 features mais importantes via permutation importance
   - Barras de erro (desvio padrão de 10 repetições)
   - Separação por tipo: categóricas vs numéricas
   - Cores por gradiente de importância
   - Valores anotados para precisão

3. **Análises Exploratórias (test.py)**
   - Boxplots: Glucose_Level vs Exercise_Recommendation
   - Violin plots: BMI vs Diet_Recommendation  
   - Heatmaps de correlação de biomarcadores
   - Distribuição de classes com balanceamento
   - Estatísticas descritivas por grupo

## 🔍 Pipeline de Processamento

### Pré-processamento
1. **Variáveis Categóricas**: 
   - Imputação com moda
   - OrdinalEncoder (handle_unknown=-1)

2. **Variáveis Numéricas**:
   - Imputação com mediana

3. **Encoding de Alvo**:
   - LabelEncoder para variáveis multi-classe

### Divisão de Dados
- 75% treino / 25% teste (otimizado para datasets <5000 amostras)
- Estratificação por classe para manter distribuição
- Random state fixo (42) para reprodutibilidade
- Validação de overfitting automática (treino vs teste)

## 💾 Modelos Salvos

Os modelos treinados são salvos usando Joblib:

- `diet_recommender_final.pkl`: Pipeline completo (pré-processamento + modelo)
- `label_encoder_final.pkl`: Encoder das classes de dieta
- `exercise_recommender_final.pkl`: Pipeline de exercícios
- `exercise_label_encoder_final.pkl`: Encoder das classes de exercício

### Carregar Modelo

```python
import joblib

# Carregar modelo
model = joblib.load('models/diet_recommender_final.pkl')
label_enc = joblib.load('models/label_encoder_final.pkl')

# Fazer predição
prediction = model.predict(X_new)
recommendation = label_enc.inverse_transform(prediction)
```

## 📝 Notas Importantes

### Prevenção de Overfitting e Data Leakage
- **Features Derivadas Removidas**: Health_Risk, Heart_Disease_Risk, Diabetes_Risk, Predicted_Insurance_Cost (causavam 100% accuracy artificial)
- **Gender Removido do Modelo de Exercícios**: Apresentava importância espúria de 21% sem significância estatística real (p=0.46)
- **Regularização Agressiva**: max_depth=8, min_samples_leaf=8, min_samples_split=20 para evitar memorização
- **Validação em Tempo Real**: Scripts exibem diferença treino-teste automaticamente

### Boas Práticas de ML Implementadas
- **Balanceamento de Classes**: `class_weight='balanced'` para lidar com desbalanceamento
- **Tratamento de Missing**: Imputação estratificada (mediana para numéricos, moda para categóricos)
- **Validação Estratificada**: Stratified split para manter proporção de classes no treino/teste
- **Reprodutibilidade**: Random state fixo (42) em todas as operações aleatórias
- **Permutation Importance**: Método mais confiável que feature_importances_ padrão (10 repetições)
- **Top-2 Accuracy**: Métrica crucial para sistemas de recomendação clínica (backup seguro)

### Interpretação Clínica
- **Modelo de Dieta**: Prioriza biomarcadores metabólicos (Alcohol, LDL, HbA1c) e composição corporal (BMI, Waist)
- **Modelo de Exercícios**: Baseado 100% em biomarcadores (sem viés demográfico), focado em risco cardiovascular (LDL, Systolic_BP, eGFR)
- **Decisões Médicas**: Sempre usar Top-2 predictions para segurança clínica (96-97% de cobertura)

## 🎓 Contexto Acadêmico

**Trabalho Final - Banco de Dados Avançados (2025/2)**

Este projeto reproduz e aprimora o sistema de recomendações de saúde personalizado apresentado no artigo IEEE:
- **Artigo Base**: [Personalized Health Recommendations using Machine Learning](https://ieeexplore.ieee.org/abstract/document/10774650)

### Melhorias Implementadas sobre o Artigo Original:
1. **Detecção e Correção de Data Leakage**: Remoção de features derivadas
2. **Eliminação de Viés Demográfico**: Remoção de Gender após análise estatística
3. **Regularização Anti-Overfitting**: Parâmetros otimizados para generalização
4. **Validação Robusta**: Permutation importance com 10 repetições
5. **Métricas Clínicas**: Top-2 Accuracy para decisões médicas seguras
6. **Transparência**: Análise automática de confusão e overfitting

## 📄 Licença

Este é um projeto acadêmico.

## 👥 Autores

Lucas Dolabella de Castro Lopes
Vanessa Nascimento Silva
---

**Última atualização**: Novembro 2025