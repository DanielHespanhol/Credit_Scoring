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

## Métricas de Avaliação
- Acurácia: Percentual de previsões corretas
- Precisão: Percentual de "maus" identificados corretamente
- Recall: Percentual de "maus" corretamente detectados
- F1-Score: Média harmônica entre precisão e recall
- AUC/ROC: Área sob a curva ROC
- KS: Kolmogorov-Smirnov (0.3+ = bom, 0.4+ = excelente)
- GINI: Coeficiente de Gini (0.5+ = bom, 0.6+ = excelente)

## Instalação
Para executar este projeto, você precisará das seguintes bibliotecas Python:
```bash
pip install pandas numpy scikit-learn imbalanced-learn seaborn matplotlib
```

## Uso
1. Clone o repositório:
```bash
git clone https://github.com/DanielHespanhol/Credit_Scoring
cd Credit_Scoring
```

2. Execute o notebook Jupyter:
```bash
jupyter notebook Credit_Scoring_Consolidated.ipynb
```

3. Siga as etapas no notebook para carregar os dados, treinar o modelo e avaliar os resultados.

## Estrutura do Código
- **Preparação dos dados**: Carregamento, tratamento de valores ausentes e codificação de variáveis.
- **Modelagem**: Treinamento do modelo de regressão logística com validação cruzada.
- **Avaliação**: Cálculo de métricas e visualização de resultados.

## Resultados
O modelo foi avaliado com as seguintes métricas:
- Acurácia: 71,87%
- KS Score: 0,5006
- Gini Index: 0,5527

Para detalhes completos, consulte o notebook.

## Contribuição
Contribuições são bem-vindas! Sinta-se à vontade para abrir issues ou enviar pull requests.

## Licença
Este projeto está licenciado sob a licença MIT. Consulte o arquivo LICENSE para mais detalhes.

---

Este README fornece uma visão geral do projeto, instruções de instalação e uso, e destaca as principais funcionalidades e métricas. Adapte conforme necessário para incluir informações específicas do seu projeto.

