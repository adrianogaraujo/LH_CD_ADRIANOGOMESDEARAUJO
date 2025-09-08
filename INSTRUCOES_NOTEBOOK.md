# Instruções para o Notebook Jupyter

## Como usar o arquivo `analise_filmes_imdb.ipynb`

### 1. Abrir o Notebook

**Opção 1 - Jupyter Lab:**
```bash
cd "C:\Users\adriano\Documents\LH_CD_ADRIANOGOMESDEARAUJO"
jupyter lab
```

**Opção 2 - Jupyter Notebook:**
```bash
cd "C:\Users\adriano\Documents\LH_CD_ADRIANOGOMESDEARAUJO"
jupyter notebook
```

**Opção 3 - VSCode:**
- Abra o VSCode
- Navegue até o arquivo `analise_filmes_imdb.ipynb`
- Clique para abrir (VSCode tem suporte nativo para notebooks)

### 2. Executar o Notebook

#### Execução Completa:
- No menu: `Kernel` → `Restart & Run All`
- Ou use o atalho: `Ctrl+Shift+F5`

#### Execução por Células:
- Selecione uma célula e pressione `Shift+Enter` para executar
- Use `Ctrl+Enter` para executar sem avançar para a próxima célula

### 3. Estrutura do Notebook

O notebook está organizado em 10 seções principais:

1. **Importação das Bibliotecas** - Setup inicial
2. **Download dos Recursos NLTK** - Configuração para análise de texto
3. **Carregamento e Inspeção dos Dados** - Primeira visualização
4. **Limpeza e Pré-processamento** - Tratamento dos dados
5. **Análise Exploratória de Dados (EDA)** - Visualizações e estatísticas
6. **Questões de Negócio** - Respostas específicas do desafio
7. **Modelo de Predição da Nota IMDB** - Machine Learning
8. **Predição para The Shawshank Redemption** - Caso de uso específico
9. **Salvando o Modelo** - Serialização do modelo treinado
10. **Conclusões e Recomendações** - Insights finais

### 4. Dependências Necessárias

Certifique-se de que todas as bibliotecas estão instaladas:

```bash
pip install -r requirements.txt
```

### 5. Arquivos Necessários

Para o notebook funcionar corretamente, certifique-se de que os seguintes arquivos estão na mesma pasta:

- `desafio_indicium_imdb.csv` - Dataset principal
- `requirements.txt` - Lista de dependências

### 6. Saídas Esperadas

Ao executar completamente, o notebook irá gerar:

- Diversos gráficos e visualizações
- Estatísticas descritivas e correlações
- Respostas às 3 questões principais do desafio
- Comparação de 3 modelos de Machine Learning
- Arquivo `modelo_imdb.pkl` (modelo treinado salvo)
- Relatório final com recomendações estratégicas

### 7. Tempo de Execução

- **Execução completa**: ~3-5 minutos
- **Seções mais demoradas**: 
  - Treinamento dos modelos (seção 7): ~1-2 minutos
  - Download dos recursos NLTK (primeira execução): ~30 segundos

### 8. Solução de Problemas

**Erro de bibliotecas ausentes:**
```bash
pip install pandas numpy matplotlib seaborn scikit-learn lightgbm nltk
```

**Erro com NLTK:**
```python
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
```

**Erro com o dataset:**
- Verifique se `desafio_indicium_imdb.csv` está na mesma pasta
- Confirme que o arquivo não está corrompido

### 9. Personalizações

Para modificar a análise:

- **Alterar parâmetros dos modelos**: Seção 7, células dos algoritmos
- **Adicionar novos gráficos**: Seção 5 (EDA)
- **Modificar features**: Seção 7, célula de engenharia de features
- **Testar outros filmes**: Seção 8, modificar `shawshank_data`

### 10. Resultados Principais

Ao final da execução, você terá:

- **Recomendação Universal**: The Godfather
- **Gêneros Lucrativos**: Action, Adventure, Animation
- **Melhor Modelo**: LightGBM (RMSE: ~0.11, R²: ~0.82)
- **Predição Shawshank**: ~8.5 (muito próximo do real)
- **Modelo Salvo**: `modelo_imdb.pkl` pronto para uso

---

**Dica**: Execute o notebook célula por célula na primeira vez para acompanhar cada etapa da análise!