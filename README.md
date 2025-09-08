# A Anatomia de um Filme de Sucesso: Análise de Dados IMDB

## Projeto de Ciência de Dados para PProductions

Este projeto apresenta uma análise completa dos dados de filmes do IMDB para orientar as decisões estratégicas do estúdio fictício PProductions, desenvolvido como parte do desafio de Ciência de Dados da Indicium/Lighthouse.

## Objetivo

Identificar os fatores que determinam o sucesso de um filme, tanto em termos de aclamação crítica quanto de faturamento comercial, através de análise exploratória de dados (EDA), processamento de linguagem natural (NLP) e modelos de machine learning.

## Estrutura do Projeto

```
LH_CD_ADRIANOGOMESDEARAUJO/
├── README.md                           # Este arquivo
├── requirements.txt                    # Dependências do projeto
├── desafio_indicium_imdb.csv          # Dataset principal
├── analise_filmes_imdb.ipynb          # Notebook principal com análises
├── modelo_imdb.pkl                    # Modelo treinado salvo
├── Desafio Cientista de Dados.txt     # Especificações do desafio
├── Análise e Relatório de Projeto de Dados.txt  # Relatório detalhado
└── relatorio_final.pdf                # Relatório final em PDF
```

## Dataset

O dataset `desafio_indicium_imdb.csv` contém informações sobre os 1000 filmes mais bem avaliados do IMDB, incluindo:

- **Series_Title**: Nome do filme
- **Released_Year**: Ano de lançamento  
- **Certificate**: Classificação etária
- **Runtime**: Duração em minutos
- **Genre**: Gênero(s) do filme
- **IMDB_Rating**: Nota do IMDB (variável alvo principal)
- **Overview**: Sinopse do filme
- **Meta_score**: Média ponderada das críticas
- **Director**: Diretor
- **Star1-Star4**: Atores principais
- **No_of_Votes**: Número de votos
- **Gross**: Faturamento em dólares

## Principais Análises Realizadas

### 1. Análise Exploratória de Dados (EDA)
- Limpeza e pré-processamento dos dados
- Análise univariada das variáveis principais
- Matriz de correlação e análise bivariada
- Visualizações e identificação de padrões

### 2. Insights de Negócio
- Recomendação universal de filme
- Fatores relacionados ao alto faturamento
- Análise de texto (NLP) das sinopses
- Estratégias para sucesso comercial vs. crítico

### 3. Modelagem Preditiva
- Predição da nota IMDB usando técnicas de regressão
- Comparação de algoritmos (Linear Regression, Random Forest, LightGBM)
- Feature engineering e seleção de variáveis
- Classificação de gêneros a partir das sinopses (NLP)

## Como Executar o Projeto

### Pré-requisitos

- Python 3.8 ou superior
- Jupyter Notebook ou JupyterLab

### Clonando do Git

```bash
git clone <repository-url>
cd LH_CD_ADRIANOGOMESDEARAUJO
```

**Nota**: O modelo treinado (`modelo_imdb.pkl`) não está incluído no repositório devido ao tamanho. Execute o notebook ou script para gerá-lo.

### Instalação

1. Clone ou baixe este repositório
2. Navegue até o diretório do projeto
3. Instale as dependências:

```bash
pip install -r requirements.txt
```

### Execução

1. Inicie o Jupyter Notebook:
```bash
jupyter notebook
```

2. Abra o arquivo `analise_filmes_imdb.ipynb`

3. Execute todas as células sequencialmente (Cell > Run All)

### Uso do Modelo Treinado

```python
import pickle
import pandas as pd

# Carregar o modelo
with open('modelo_imdb.pkl', 'rb') as file:
    modelo = pickle.load(file)

# Fazer predições
# (dados devem estar no formato correto conforme mostrado no notebook)
predicao = modelo.predict(dados_preprocessados)
```

## Principais Descobertas

### Fatores de Sucesso Comercial
1. **Gênero**: Ação, Aventura e Animação têm maior faturamento médio
2. **Popularidade**: Correlação forte entre número de votos e faturamento
3. **Elenco/Direção**: Nomes conhecidos impactam significativamente o sucesso

### Fatores de Aclamação Crítica
1. **Meta_score**: Maior preditor da nota IMDB
2. **Gênero**: Drama predomina entre filmes bem avaliados
3. **Duração**: Filmes mais longos tendem a ter notas mais altas

### Insights de NLP
- É possível inferir gêneros com boa precisão a partir das sinopses
- Palavras-chave específicas são fortes indicadores de gênero
- TF-IDF + Regressão Logística oferece resultados satisfatórios

## Performance dos Modelos

- **Melhor modelo**: LightGBM Regressor
- **RMSE**: 0.39 pontos na nota IMDB
- **R²**: 0.74 (explica 74% da variância)
- **Precisão na previsão específica**: 9.18 vs 9.3 real (The Shawshank Redemption)

## Recomendações Estratégicas

### Para Máximo Sucesso Comercial:
- Focar em filmes de Ação/Aventura/Sci-Fi
- Utilizar propriedades intelectuais conhecidas (sequelas, adaptações)
- Investir em elenco A-list e diretores experientes
- Orçamento adequado para produção e marketing

### Para Prestígio e Crítica:
- Priorizar roteiros sólidos e execução técnica de qualidade
- Considerar dramas com desenvolvimento de personagens
- Duração adequada para narrativa complexa (120+ minutos)

## Limitações e Trabalhos Futuros

### Limitações:
- Dataset com viés de sobrevivência (apenas filmes bem-sucedidos)
- Ausência de dados de orçamento e marketing
- Amostra limitada a 1000 filmes

### Próximos Passos:
- Incorporar dados externos de orçamento e marketing
- Análise de sentimentos mais sofisticada
- Modelos especializados para diferentes objetivos (comercial vs. crítico)
- Web scraping para dados adicionais

## Tecnologias Utilizadas

- **Python**: Linguagem principal
- **Pandas**: Manipulação de dados
- **NumPy**: Computação numérica
- **Matplotlib/Seaborn**: Visualização
- **Scikit-learn**: Machine learning
- **LightGBM**: Gradient boosting
- **NLTK**: Processamento de linguagem natural
- **Jupyter**: Ambiente de desenvolvimento

## Autor

**Adriano Gomes de Araújo**

Projeto desenvolvido como parte do processo seletivo Indicium/Lighthouse para Cientista de Dados.

## Licença

Este projeto é desenvolvido para fins educacionais e de avaliação técnica.