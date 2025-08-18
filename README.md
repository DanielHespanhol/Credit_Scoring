# Credit Scoring - Análise de Risco de Crédito

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.0+-orange)](https://scikit-learn.org/)
[![Pandas](https://img.shields.io/badge/Pandas-1.3+-brightgreen)](https://pandas.pydata.org/)
[![Jupyter Notebook](https://img.shields.io/badge/Jupyter-Notebook-orange)](https://jupyter.org/)
[![Imbalanced-Learn](https://img.shields.io/badge/Imbalanced--Learn-0.9+-blueviolet)](https://imbalanced-learn.org/)

## Descrição do Projeto
Este projeto tem como objetivo desenvolver um modelo de machine learning para prever o risco de crédito de clientes com base em suas características financeiras e pessoais. O modelo utiliza técnicas de regressão logística e métricas específicas para avaliação de crédito, como KS (Kolmogorov-Smirnov) e GINI.

## Funcionalidades
- **Pré-processamento de dados**: Tratamento de valores ausentes, codificação de variáveis categóricas e engenharia de features.
- **Modelagem**: Implementação de um modelo de regressão logística com otimização de hiperparâmetros.
- **Avaliação**: Métricas de avaliação como acurácia, precisão, recall, F1-score, KS e GINI.
- **Balanceamento de dados**: Uso de técnicas como SMOTE e RandomUnderSampler para lidar com classes desbalanceadas.
- **Validação cruzada**: Utilização de K-fold para garantir a robustez do modelo.

## Métricas Utilizadas
- **KS (Kolmogorov-Smirnov)**: Mede a capacidade do modelo de distinguir entre clientes bons e maus.
- **GINI**: Avalia a qualidade da classificação do modelo.
- **AUC-ROC**: Curva ROC para visualizar o desempenho do modelo em diferentes thresholds.

## Instalação
Para executar este projeto, você precisará das seguintes bibliotecas Python:
```bash
pip install pandas numpy scikit-learn imbalanced-learn seaborn matplotlib
```

## Uso
1. Clone o repositório:
```bash
git clone https://github.com/seu-usuario/credit-scoring.git
cd credit-scoring
```

2. Execute o notebook Jupyter:
```bash
jupyter notebook Credit_Scoring_EDA_03.ipynb
```

3. Siga as etapas no notebook para carregar os dados, treinar o modelo e avaliar os resultados.

## Estrutura do Código
- **Preparação dos dados**: Carregamento, tratamento de valores ausentes e codificação de variáveis.
- **Modelagem**: Treinamento do modelo de regressão logística com validação cruzada.
- **Avaliação**: Cálculo de métricas e visualização de resultados.

## Resultados
O modelo foi avaliado com as seguintes métricas:
- Acurácia: X%
- KS Score: X
- Gini Index: X

Para detalhes completos, consulte o notebook.

## Contribuição
Contribuições são bem-vindas! Sinta-se à vontade para abrir issues ou enviar pull requests.

## Licença
Este projeto está licenciado sob a licença MIT. Consulte o arquivo LICENSE para mais detalhes.

---

Este README fornece uma visão geral do projeto, instruções de instalação e uso, e destaca as principais funcionalidades e métricas. Adapte conforme necessário para incluir informações específicas do seu projeto.

