# Credit Scoring - Análise de Risco de Crédito

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.0+-orange)](https://scikit-learn.org/)
[![Pandas](https://img.shields.io/badge/Pandas-1.3+-brightgreen)](https://pandas.pydata.org/)
[![Jupyter Notebook](https://img.shields.io/badge/Jupyter-Notebook-orange)](https://jupyter.org/)
[![Imbalanced-Learn](https://img.shields.io/badge/Imbalanced--Learn-0.9+-blueviolet)](https://imbalanced-learn.org/)

## Descrição do Projeto
Este projeto tem como objetivo desenvolver um modelo de machine learning para prever o risco de crédito de clientes com base em suas características financeiras e pessoais. O modelo utiliza técnicas de regressão logística e métricas específicas para avaliação de crédito, como KS (Kolmogorov-Smirnov) e GINI.

## Estrutura do Projeto
        Credit_Scoring_Consolidated.ipynb
        ├── Importação de Bibliotecas
        ├── Funções Auxiliares
        ├── Preparação e Tratamento de Dados
        ├── Modelo I - Regressão Logística Básica
        ├── Tratamento de Desbalanceamento
        ├── Modelo II - Com Balanceamento
        ├── Feature Engineering e Seleção
        ├── Modelo III - Com Features Selecionadas
        ├── Pipeline Completo com Feature Engineering
        ├── Validação com K-Fold
        └── Resultados Finais

## Funcionalidades
### **Pré-processamento de dados com tratamento de valores missing**
  **Objetivo**: Preparar os dados brutos para análise, garantindo qualidade e consistência.
  **Funcionalidades**:
- Identificação e tratamento de valores ausentes (NaN)
- Preenchimento inteligente usando estratégias específicas por coluna
      saving_accounts: preenchimento com valor 'little'
      checking_account: distribuição uniforme entre 'little' e 'moderate'
- Codificação de variáveis categóricas (ex: 'good'/'bad' → 1/0)
- Conversão de tipos de dados (strings para inteiros)
- Remoção de colunas irrelevantes (CPF, income, reference)

Importância: Dados limpos são fundamentais para modelos precisos e evita viés nas previsões

### **Análise exploratória e feature engineering**
  **Objetivo**: Compreender os dados e criar variáveis mais informativas
  **Funcionalidades**:
- Exploração inicial da estrutura dos dados (shape, tipos, valores missing)
- Criação de features temporais a partir da coluna 'reference'
- Extração de mês e ano como variáveis separadas
- One-hot encoding para variáveis categóricas
- Transformação de todas as variáveis para formato numérico
  
Importância: Melhora o poder preditivo do modelo e revela insights sobre os dados

### **Múltiplos modelos de machine learning**
**Objetivo**: Implementar e comparar diferentes abordagens de modelagem
**Abordagens**:
- Modelo I: Regressão Logística básica (baseline)
- Modelo II: Regressão Logística com balanceamento de classes
- Modelo III: Regressão Logística com features selecionadas
- Pipeline Completo: Combinação de pré-processamento e modelagem

Importância: Permite encontrar a melhor abordagem para o problema específico

### **Balanceamento de classes com SMOTE e undersampling**
**Objetivo**: Lidar com o desbalanceamento típico em credit scoring
**Técnicas**:
- SMOTE: Cria exemplos sintéticos da classe minoritária
- Random UnderSampling: Reduz a classe majoritária
- Pipeline combinado: Aplica ambas técnicas sequencialmente

Importância: Evita viés do modelo em favor da classe majoritária (bons pagadores)

### **Seleção de features baseada em importância**
**Objetivo**: Reduzir dimensionalidade e melhorar a generalização
**Método**:
- Análise dos coeficientes da regressão logística
- Seleção de features com importância acima de threshold (0.1)
- Criação de subset otimizado de variáveis

Importância:
- Reduz overfitting
- Melhora interpretabilidade do modelo
- Diminui tempo de treinamento

### **Otimização de hiperparâmetros com GridSearchCV**
**Objetivo**: Encontrar a melhor configuração para o modelo
**Parâmetros otimizados**:
- C: Força de regularização (0.001 a 100)
- penalty: Tipo de regularização (L1, L2)
- solver: Algoritmo de otimização
- class_weight: Peso das classes

Importância: Maximiza o desempenho do modelo através da configuração ideal

### **Validação cruzada robusta**
**Objetivo**: Garantir que o modelo generalize bem para dados não vistos
**Técnicas**:
- K-Fold Cross Validation: Divisão em 5 folds com embaralhamento
- Validação em múltiplos subsets: Garante robustez estatística
- Métricas consolidadas: Média e desvio padrão dos scores

Importância: Fornece estimativa confiável do desempenho em produção

### **Métricas específicas para credit scoring (KS, GINI)**
**Objetivo**: Avaliar o modelo com métricas do domínio financeiro
**Métricas implementadas**:
- KS (Kolmogorov-Smirnov): Mede separação entre distribuições de bons/maus
- GINI: Mede poder discriminativo do modelo
- AUC/ROC: Área sob a curva ROC
- Métricas tradicionais: Acurácia, Precisão, Recall, F1-Score

Importância: Alinhamento com práticas do setor financeiro e regulatórias

## Explicação Detalhada do Código
**1. Importação de Bibliotecas**
```
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, KFold
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from scipy.stats import ks_2samp
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline
```

**Propósito**: 
Importa todas as bibliotecas necessárias para:
- Manipulação de dados (pandas, numpy)
- Visualização (seaborn, matplotlib)
- Machine Learning (scikit-learn)
- Balanceamento de classes (imbalanced-learn)
- Análise estatística (scipy)

**2. Funções Auxiliares**  
`calcular_ks(y_true, y_pred)`  
Objetivo: Calcula a estatística Kolmogorov-Smirnov, que mede a distância máxima entre as distribuições de probabilidade dos clientes "bons" e "maus".  
Importância: KS > 0.3 indica bom poder discriminativo do modelo.

`calcular_gini(y_true, y_pred)`  
Objetivo: Calcula o coeficiente de Gini a partir da AUC.  
Fórmula: GINI = 2 × AUC - 1

`fill_missing_values(df, column, filler_values)`  
Objetivo: Preenche valores missing de forma inteligente, distribuindo valores de preenchimento uniformemente.  

`plot_roc_curve(y_true, y_pred_proba)`  
Objetivo: Visualiza a curva ROC para avaliar o desempenho do modelo.  

`evaluate_credit_model(y_true, y_pred, y_pred_proba)`  
Objetivo: Avaliação completa do modelo com todas as métricas relevantes para credit scoring.  

`select_important_features(model, feature_names, threshold=0.1)`  
Objetivo: Seleciona features mais importantes baseado nos coeficientes da regressão logística.  

**3. Preparação e Tratamento de Dados**  
```
# Carregamento dos dados
df = pd.read_csv('/content/drive/MyDrive/DATA_VIKING/credit_risk.csv')

# Tratamento de valores missing
df['saving_accounts'].fillna('little', inplace=True)
fill_missing_values(df, 'checking_account', ['little', 'moderate'])

# Codificação de variáveis categóricas
df['risk'] = df['risk'].map({'good': 1, 'bad': 0})
df['sex'] = df['sex'].map({'male': 1, 'female': 0})

# Feature engineering temporal
df['month'] = df['reference'].str.split('-').str[1].astype(int)
df['year'] = df['reference'].str.split('-').str[0].astype(int)

# One-hot encoding
categorical_columns = df.select_dtypes(include=['object']).columns
for column in categorical_columns:
    df = pd.get_dummies(df, columns=[column], drop_first=True, dtype=int)
```

**Processos realizados**:  
- Limpeza de dados: Tratamento de valores missing
- Codificação: Transformação de variáveis categóricas em numéricas
- Engenharia de features: Criação de variáveis temporais
- Preparação final: One-hot encoding para todas as variáveis categóricas

**4. Modelo I - Regressão Logística Básica**  
**Objetivo**: Modelo baseline sem balanceamento ou otimização.
**Etapas**:  
- Divisão treino-teste (80-20)
- Treinamento do modelo
- Avaliação com múltiplas métricas
- Validação cruzada
- Otimização com GridSearch

**5. Tratamento de Desbalanceamento**  
```
balancing_pipeline = ImbPipeline([
    ('smote', SMOTE(random_state=42)),
    ('undersampling', RandomUnderSampler(random_state=42))
])
```
Técnicas utilizadas:  
- SMOTE: Synthetic Minority Over-sampling Technique
- Random UnderSampling: Redução da classe majoritária
Propósito:
Lidar com o desbalanceamento típico em credit scoring (muitos "bons" vs poucos "maus").

**6. Modelo II - Com Balanceamento**  
Melhorias:
- Dados balanceados
- `class_weight='balanced'` na regressão logística
- Avaliação mais robusta

**7. Feature Engineering e Seleção**  
Processo:
- Análise dos coeficientes do modelo
- Seleção das features mais importantes (threshold > 0.1)
- Redução de dimensionalidade

**8. Modelo III - Com Features Selecionadas**  
Vantagens:
- Menos overfitting
- Modelo mais interpretável
- Tempo de treinamento reduzido

**9. Pipeline Completo com Feature Engineering**  
```
model_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression())
])
```
Componentes:
- Pré-processamento: StandardScaler para normalização
- Classificador: Regressão Logística
- Otimização: GridSearch com validação cruzada

**10. Validação com K-Fold**  
Técnica: K-Fold Cross Validation com 5 folds  
Objetivo: Avaliar robustez e generalização do modelo  

**11. Resultados Finais**  
Comparação entre modelos:
- Modelo Básico
- Modelo Balanceado
- Modelo com Features Selecionadas
- Pipeline Completo

## Métricas de Avaliação
- Acurácia: Percentual de previsões corretas
- Precisão: Percentual de "maus" identificados corretamente
- Recall: Percentual de "maus" corretamente detectados
- F1-Score: Média harmônica entre precisão e recall
- AUC/ROC: Área sob a curva ROC
- KS: Kolmogorov-Smirnov (0.3+ = bom, 0.4+ = excelente)
- GINI: Coeficiente de Gini (0.5+ = bom, 0.6+ = excelente)

## Estrutura do Código
- **Preparação dos dados**: Carregamento, tratamento de valores ausentes e codificação de variáveis.
- **Modelagem**: Treinamento do modelo de regressão logística com validação cruzada.
- **Avaliação**: Cálculo de métricas e visualização de resultados.

## Resultados  
| Modelo                | AUC    | KS     | GINI   |  
| --------------------- | ------ | ------ | ------ |  
| Básico                | 0.5642 | 0.1549 | 0.1285 |  
| Balanceado            | 0.7871 | 0.4769 | 0.5743 |  
| Features Selecionadas | 0.7763 | 0.5006 | 0.5526 |  
| Pipeline Completo     | 0.6858 | NaN    | NaN    |  

Para detalhes completos, consulte o notebook.

## Contribuição
Contribuições são bem-vindas! Sinta-se à vontade para abrir issues ou enviar pull requests.

## Licença
Este projeto está licenciado sob a licença MIT. Consulte o arquivo LICENSE para mais detalhes.

---

Este README fornece uma visão geral do projeto, instruções de instalação e uso, e destaca as principais funcionalidades e métricas. Adapte conforme necessário para incluir informações específicas do seu projeto.

