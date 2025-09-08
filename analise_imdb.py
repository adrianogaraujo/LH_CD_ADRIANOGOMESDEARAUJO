#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A Anatomia de um Filme de Sucesso: Análise de Dados IMDB
Projeto de Ciência de Dados para PProductions

Este script apresenta uma análise completa dos dados de filmes do IMDB para orientar 
as decisões estratégicas do estúdio PProductions.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_squared_error, r2_score, classification_report, f1_score
import lightgbm as lgb
import re
import warnings
import pickle
from collections import Counter
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Configurações
warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

def download_nltk_resources():
    """Download recursos necessários do NLTK"""
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
    
    try:
        nltk.data.find('tokenizers/punkt_tab')
    except LookupError:
        nltk.download('punkt_tab')
    
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')

def load_and_inspect_data():
    """Carregamento e inspeção inicial dos dados"""
    print("="*60)
    print("1. CARREGAMENTO E INSPEÇÃO INICIAL DOS DADOS")
    print("="*60)
    
    # Carregamento do dataset
    import os
    dataset_path = os.path.join(os.path.dirname(__file__), 'desafio_indicium_imdb.csv')
    
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset não encontrado: {dataset_path}")
    
    df = pd.read_csv(dataset_path, index_col=0)
    
    print(f"Shape do dataset: {df.shape}")
    print(f"\nPrimeiras linhas:")
    print(df.head())
    
    print(f"\nInformações do dataset:")
    df.info()
    
    # Verificação de valores ausentes
    missing_values = df.isnull().sum()
    missing_percentage = (missing_values / len(df)) * 100
    
    missing_data = pd.DataFrame({
        'Missing Count': missing_values,
        'Percentage': missing_percentage
    })
    
    missing_data = missing_data[missing_data['Missing Count'] > 0].sort_values('Missing Count', ascending=False)
    
    print(f"\nValores ausentes por coluna:")
    print(missing_data)
    
    return df

def clean_data(df):
    """Limpeza e pré-processamento de dados"""
    print("\n" + "="*60)
    print("2. LIMPEZA E PRÉ-PROCESSAMENTO DE DADOS")
    print("="*60)
    
    df_clean = df.copy()
    
    # Função para limpar a coluna Runtime
    def clean_runtime(runtime):
        if pd.isna(runtime):
            return np.nan
        return int(re.findall(r'\d+', str(runtime))[0])
    
    # Função para limpar a coluna Gross
    def clean_gross(gross):
        if pd.isna(gross):
            return 0
        return int(str(gross).replace(',', '')) if str(gross).replace(',', '').isdigit() else 0
    
    print("Limpeza das colunas Runtime e Gross...")
    
    # Limpeza da coluna Runtime
    df_clean['Runtime'] = df_clean['Runtime'].apply(clean_runtime)
    
    # Limpeza da coluna Gross
    df_clean['Gross'] = df_clean['Gross'].apply(clean_gross)
    
    # Imputação de valores ausentes em Meta_score com a média
    df_clean['Meta_score'].fillna(df_clean['Meta_score'].mean(), inplace=True)
    
    print("Limpeza concluída!")
    print("\nDados após limpeza:")
    df_clean.info()
    
    return df_clean

def exploratory_data_analysis(df):
    """Análise Exploratória de Dados (EDA)"""
    print("\n" + "="*60)
    print("3. ANÁLISE EXPLORATÓRIA DE DADOS (EDA)")
    print("="*60)
    
    # Estatísticas descritivas
    numerical_cols = ['IMDB_Rating', 'Meta_score', 'Runtime', 'No_of_Votes', 'Gross']
    print("\nEstatísticas descritivas das variáveis numéricas:")
    print(df[numerical_cols].describe())
    
    # Análise dos gêneros
    all_genres = []
    for genres in df['Genre'].dropna():
        genre_list = [genre.strip() for genre in genres.split(',')]
        all_genres.extend(genre_list)
    
    genre_counts = Counter(all_genres)
    
    print(f"\nTotal de gêneros únicos: {len(genre_counts)}")
    print("\nTop 10 gêneros mais comuns:")
    for genre, count in genre_counts.most_common(10):
        print(f"{genre}: {count} filmes")
    
    # Análise temporal
    df['Released_Year'] = pd.to_numeric(df['Released_Year'], errors='coerce')
    df['Decade'] = (df['Released_Year'] // 10) * 10
    
    decade_stats = df.groupby('Decade').agg({
        'IMDB_Rating': 'mean',
        'Gross': 'mean',
        'Series_Title': 'count'
    }).round(2)
    decade_stats.columns = ['Avg_IMDB_Rating', 'Avg_Gross', 'Movie_Count']
    
    print("\nEstatísticas por década:")
    print(decade_stats)
    
    # Matriz de correlação
    correlation_vars = ['IMDB_Rating', 'Meta_score', 'Runtime', 'No_of_Votes', 'Gross', 'Released_Year']
    correlation_matrix = df[correlation_vars].corr()
    
    print("\nPrincipais correlações com IMDB_Rating:")
    imdb_corr = correlation_matrix['IMDB_Rating'].abs().sort_values(ascending=False)
    for var, corr in imdb_corr.items():
        if var != 'IMDB_Rating':
            print(f"{var}: {correlation_matrix['IMDB_Rating'][var]:.3f}")
    
    return df, all_genres, genre_counts

def analyze_business_questions(df, genre_counts):
    """Resposta às questões de negócio"""
    print("\n" + "="*60)
    print("4. QUESTÕES DE NEGÓCIO")
    print("="*60)
    
    # 4.1 Recomendação universal
    print("\n4.1. RECOMENDAÇÃO UNIVERSAL:")
    print("-" * 30)
    
    def calculate_weighted_rating(df, rating_col='IMDB_Rating', votes_col='No_of_Votes'):
        """Calcula o weighted rating usando a fórmula do IMDB"""
        C = df[rating_col].mean()
        m = df[votes_col].quantile(0.90)
        
        df['weighted_rating'] = ((df[votes_col] / (df[votes_col] + m)) * df[rating_col] + 
                                (m / (df[votes_col] + m)) * C)
        
        return df
    
    df_weighted = calculate_weighted_rating(df.copy())
    top_movie = df_weighted.nlargest(1, 'weighted_rating')[['Series_Title', 'Released_Year', 'IMDB_Rating', 'No_of_Votes', 'weighted_rating', 'Genre']].iloc[0]
    
    print(f"RECOMENDACAO: {top_movie['Series_Title']} ({top_movie['Released_Year']})")
    print(f"Rating: {top_movie['IMDB_Rating']} | Votos: {top_movie['No_of_Votes']:,} | Weighted: {top_movie['weighted_rating']:.3f}")
    print(f"Genero: {top_movie['Genre']}")
    
    # 4.2 Fatores de alto faturamento
    print("\n4.2. FATORES DE ALTO FATURAMENTO:")
    print("-" * 35)
    
    # Extrair gênero principal
    def extract_primary_genre(genre_string):
        if pd.isna(genre_string):
            return 'Unknown'
        return genre_string.split(',')[0].strip()
    
    df['Primary_Genre'] = df['Genre'].apply(extract_primary_genre)
    
    high_grossing = df[df['Gross'] > df['Gross'].quantile(0.75)]
    
    print(f"Filmes analisados: {len(high_grossing)} (top 25% em faturamento)")
    print(f"Faturamento médio do grupo: ${high_grossing['Gross'].mean():,.0f}")
    
    print("\nGêneros mais lucrativos:")
    high_gross_genres = high_grossing['Primary_Genre'].value_counts().head(5)
    for genre, count in high_gross_genres.items():
        percentage = (count / len(high_grossing)) * 100
        avg_gross = high_grossing[high_grossing['Primary_Genre'] == genre]['Gross'].mean()
        print(f"   {genre}: {count} filmes ({percentage:.1f}%) - Média: ${avg_gross:,.0f}")
    
    return top_movie

def nlp_analysis(df):
    """Análise de NLP da coluna Overview"""
    print("\n4.3. ANÁLISE NLP DA COLUNA OVERVIEW:")
    print("-" * 35)
    
    # Função para pré-processar texto
    def preprocess_text(text):
        if pd.isna(text):
            return ""
        
        text = text.lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        tokens = word_tokenize(text)
        
        stop_words = set(stopwords.words('english'))
        tokens = [word for word in tokens if word not in stop_words and len(word) > 2]
        
        return ' '.join(tokens)
    
    # Pré-processar sinopses
    df['processed_overview'] = df['Overview'].apply(preprocess_text)
    
    # Análise das palavras mais comuns
    all_words = []
    for overview in df['processed_overview'].dropna():
        words = overview.split()
        all_words.extend(words)
    
    word_counts = Counter(all_words)
    most_common_words = word_counts.most_common(10)
    
    print("Palavras mais comuns nas sinopses:")
    for word, count in most_common_words:
        print(f"  {word}: {count}")
    
    # Modelo de predição de gênero
    print("\nConstruindo modelo de predição de gênero...")
    
    df_clean = df.dropna(subset=['Genre', 'processed_overview']).copy()
    genre_lists = []
    
    for genres in df_clean['Genre']:
        genre_list = [genre.strip() for genre in genres.split(',')]
        genre_lists.append(genre_list)
    
    # MultiLabelBinarizer para encoding dos gêneros
    mlb = MultiLabelBinarizer()
    genre_encoded = mlb.fit_transform(genre_lists)
    
    print(f"Número de gêneros únicos: {len(mlb.classes_)}")
    
    # Vetorização TF-IDF
    tfidf = TfidfVectorizer(max_features=1000, ngram_range=(1, 2), min_df=2)
    X = tfidf.fit_transform(df_clean['processed_overview'])
    y = genre_encoded
    
    # Dividir dados
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Treinar modelo
    classifier = OneVsRestClassifier(LogisticRegression(random_state=42, max_iter=1000))
    classifier.fit(X_train, y_train)
    
    # Avaliar
    y_pred = classifier.predict(X_test)
    f1_micro = f1_score(y_test, y_pred, average='micro')
    f1_macro = f1_score(y_test, y_pred, average='macro')
    
    print(f"Performance do modelo de gênero:")
    print(f"F1-score (micro): {f1_micro:.3f}")
    print(f"F1-score (macro): {f1_macro:.3f}")
    
    return f1_micro, most_common_words

def build_rating_prediction_model(df, all_genres):
    """Construção do modelo de predição da nota IMDB"""
    print("\n" + "="*60)
    print("5. MODELO DE PREDIÇÃO DA NOTA IMDB")
    print("="*60)
    
    # Engenharia de features
    print("Criando features...")
    
    df_model = df.copy()
    
    # Features básicas
    basic_features = ['Meta_score', 'Runtime', 'No_of_Votes', 'Gross', 'Released_Year']
    
    # One-hot encoding para gêneros principais
    top_genres_list = [genre for genre, count in Counter(all_genres).most_common(10)]
    
    for genre in top_genres_list:
        df_model[f'Genre_{genre}'] = df_model['Genre'].str.contains(genre, na=False).astype(int)
    
    genre_features = [f'Genre_{genre}' for genre in top_genres_list]
    
    # Features baseadas no histórico
    director_avg_rating = df_model.groupby('Director')['IMDB_Rating'].transform('mean')
    director_movie_count = df_model.groupby('Director')['Series_Title'].transform('count')
    
    df_model['director_avg_rating'] = director_avg_rating
    df_model['director_movie_count'] = director_movie_count
    
    star1_avg_rating = df_model.groupby('Star1')['IMDB_Rating'].transform('mean')
    star1_movie_count = df_model.groupby('Star1')['Series_Title'].transform('count')
    
    df_model['star1_avg_rating'] = star1_avg_rating
    df_model['star1_movie_count'] = star1_movie_count
    
    # Features derivadas
    df_model['votes_per_year'] = df_model['No_of_Votes'] / (2024 - df_model['Released_Year'] + 1)
    df_model['gross_per_vote'] = df_model['Gross'] / (df_model['No_of_Votes'] + 1)
    df_model['is_long_movie'] = (df_model['Runtime'] > 120).astype(int)
    df_model['is_recent'] = (df_model['Released_Year'] >= 2000).astype(int)
    
    # Lista completa de features
    feature_columns = (basic_features + genre_features + 
                      ['director_avg_rating', 'director_movie_count',
                       'star1_avg_rating', 'star1_movie_count',
                       'votes_per_year', 'gross_per_vote', 'is_long_movie', 'is_recent'])
    
    print(f"Total de features criadas: {len(feature_columns)}")
    
    # Preparar dados
    df_clean_model = df_model.dropna(subset=feature_columns + ['IMDB_Rating']).copy()
    
    X = df_clean_model[feature_columns]
    y = df_clean_model['IMDB_Rating']
    
    print(f"Dataset para modelagem: {X.shape[0]} filmes, {X.shape[1]} features")
    
    # Dividir dados
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Treinar modelos
    model_results = {}
    
    # 1. Regressão Linear
    print("\nTreinando modelos...")
    print("1. Regressão Linear")
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    lr_pred = lr.predict(X_test)
    
    lr_rmse = np.sqrt(mean_squared_error(y_test, lr_pred))
    lr_r2 = r2_score(y_test, lr_pred)
    
    model_results['Linear Regression'] = {
        'model': lr,
        'predictions': lr_pred,
        'rmse': lr_rmse,
        'r2': lr_r2
    }
    
    # 2. Random Forest
    print("2. Random Forest")
    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    
    rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))
    rf_r2 = r2_score(y_test, rf_pred)
    
    model_results['Random Forest'] = {
        'model': rf,
        'predictions': rf_pred,
        'rmse': rf_rmse,
        'r2': rf_r2
    }
    
    # 3. LightGBM
    print("3. LightGBM")
    lgbm = lgb.LGBMRegressor(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=6,
        random_state=42,
        verbosity=-1
    )
    lgbm.fit(X_train, y_train)
    lgbm_pred = lgbm.predict(X_test)
    
    lgbm_rmse = np.sqrt(mean_squared_error(y_test, lgbm_pred))
    lgbm_r2 = r2_score(y_test, lgbm_pred)
    
    model_results['LightGBM'] = {
        'model': lgbm,
        'predictions': lgbm_pred,
        'rmse': lgbm_rmse,
        'r2': lgbm_r2
    }
    
    # Comparação
    print("\n" + "="*50)
    print("COMPARAÇÃO DE MODELOS")
    print("="*50)
    
    comparison_df = pd.DataFrame({
        'Modelo': list(model_results.keys()),
        'RMSE': [result['rmse'] for result in model_results.values()],
        'R²': [result['r2'] for result in model_results.values()]
    }).round(3)
    
    comparison_df = comparison_df.sort_values('RMSE')
    print(comparison_df.to_string(index=False))
    
    # Melhor modelo
    best_model_name = comparison_df.iloc[0]['Modelo']
    best_model = model_results[best_model_name]['model']
    
    print(f"\nMELHOR MODELO: {best_model_name}")
    print(f"RMSE: {model_results[best_model_name]['rmse']:.3f}")
    print(f"R2: {model_results[best_model_name]['r2']:.3f}")
    
    # Importância das features (se aplicável)
    if best_model_name in ['Random Forest', 'LightGBM']:
        feature_importance = pd.DataFrame({
            'feature': feature_columns,
            'importance': best_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(f"\nTop 10 features mais importantes ({best_model_name}):")
        for idx, (_, row) in enumerate(feature_importance.head(10).iterrows(), 1):
            print(f"{idx:2d}. {row['feature']}: {row['importance']:.4f}")
    
    return best_model, best_model_name, model_results, feature_columns, top_genres_list, df_model

def predict_shawshank_redemption(best_model, feature_columns, top_genres_list, df_reference):
    """Predição para The Shawshank Redemption"""
    print("\n" + "="*50)
    print("6. PREDIÇÃO PARA 'THE SHAWSHANK REDEMPTION'")
    print("="*50)
    
    # Dados do filme
    shawshank_data = {
        'Series_Title': 'The Shawshank Redemption',
        'Released_Year': 1994,
        'Certificate': 'A',
        'Runtime': 142,
        'Genre': 'Drama',
        'Overview': 'Two imprisoned men bond over a number of years, finding solace and eventual redemption through acts of common decency.',
        'Meta_score': 80.0,
        'Director': 'Frank Darabont',
        'Star1': 'Tim Robbins',
        'Star2': 'Morgan Freeman',
        'Star3': 'Bob Gunton',
        'Star4': 'William Sadler',
        'No_of_Votes': 2343110,
        'Gross': 28341469
    }
    
    # Criar DataFrame
    shawshank_df = pd.DataFrame([shawshank_data])
    
    # Processar features
    def create_features_for_prediction(df_input, df_reference):
        """Criar features para predição"""
        df_pred = df_input.copy()
        
        # One-hot encoding para gêneros
        for genre in top_genres_list:
            df_pred[f'Genre_{genre}'] = df_pred['Genre'].str.contains(genre, na=False).astype(int)
        
        # Features do diretor
        director_stats = df_reference.groupby('Director')['IMDB_Rating'].agg(['mean', 'count'])
        director = df_pred['Director'].iloc[0]
        if director in director_stats.index:
            df_pred['director_avg_rating'] = director_stats.loc[director, 'mean']
            df_pred['director_movie_count'] = director_stats.loc[director, 'count']
        else:
            df_pred['director_avg_rating'] = df_reference['IMDB_Rating'].mean()
            df_pred['director_movie_count'] = 1
        
        # Features do ator principal
        star1_stats = df_reference.groupby('Star1')['IMDB_Rating'].agg(['mean', 'count'])
        star1 = df_pred['Star1'].iloc[0]
        if star1 in star1_stats.index:
            df_pred['star1_avg_rating'] = star1_stats.loc[star1, 'mean']
            df_pred['star1_movie_count'] = star1_stats.loc[star1, 'count']
        else:
            df_pred['star1_avg_rating'] = df_reference['IMDB_Rating'].mean()
            df_pred['star1_movie_count'] = 1
        
        # Features derivadas
        df_pred['votes_per_year'] = df_pred['No_of_Votes'] / (2024 - df_pred['Released_Year'] + 1)
        df_pred['gross_per_vote'] = df_pred['Gross'] / (df_pred['No_of_Votes'] + 1)
        df_pred['is_long_movie'] = (df_pred['Runtime'] > 120).astype(int)
        df_pred['is_recent'] = (df_pred['Released_Year'] >= 2000).astype(int)
        
        return df_pred
    
    shawshank_processed = create_features_for_prediction(shawshank_df, df_reference)
    X_shawshank = shawshank_processed[feature_columns]
    
    # Fazer predição
    prediction = best_model.predict(X_shawshank)[0]
    
    print(f"PREDICAO: {prediction:.3f}")
    
    # Comparacao com valor real (se disponivel)
    real_shawshank = df_reference[df_reference['Series_Title'] == 'The Shawshank Redemption']
    if not real_shawshank.empty:
        real_rating = real_shawshank['IMDB_Rating'].iloc[0]
        error = abs(prediction - real_rating)
        print(f"RATING REAL: {real_rating}")
        print(f"ERRO ABSOLUTO: {error:.3f}")
        print(f"PRECISAO: {(1 - error/real_rating)*100:.1f}%")
    
    return prediction

def save_model(best_model, best_model_name, model_results, feature_columns, top_genres_list, df):
    """Salvar modelo treinado"""
    print("\n" + "="*40)
    print("7. SALVANDO MODELO")
    print("="*40)
    
    model_package = {
        'model': best_model,
        'model_name': best_model_name,
        'feature_columns': feature_columns,
        'top_genres': top_genres_list,
        'performance': {
            'rmse': model_results[best_model_name]['rmse'],
            'r2': model_results[best_model_name]['r2']
        },
        'training_stats': {
            'director_avg_rating': df.groupby('Director')['IMDB_Rating'].mean().to_dict(),
            'star1_avg_rating': df.groupby('Star1')['IMDB_Rating'].mean().to_dict(),
            'director_movie_count': df.groupby('Director')['Series_Title'].count().to_dict(),
            'star1_movie_count': df.groupby('Star1')['Series_Title'].count().to_dict(),
            'imdb_rating_mean': df['IMDB_Rating'].mean()
        },
        'version': '1.0',
        'created_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    # Salvar
    import os
    model_path = os.path.join(os.path.dirname(__file__), 'modelo_imdb.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(model_package, f)
    
    print("MODELO SALVO: modelo_imdb.pkl")
    print(f"Algoritmo: {best_model_name}")
    print(f"Performance: RMSE={model_results[best_model_name]['rmse']:.3f}, R2={model_results[best_model_name]['r2']:.3f}")
    
    # Testar carregamento
    with open(model_path, 'rb') as f:
        loaded_model_package = pickle.load(f)
    
    print("Teste de carregamento: OK")
    
    return model_package

def main():
    """Função principal"""
    print("A ANATOMIA DE UM FILME DE SUCESSO: ANALISE DE DADOS IMDB")
    print("Projeto de Ciencia de Dados para PProductions")
    print("="*60)
    
    # Download recursos NLTK
    download_nltk_resources()
    
    # 1. Carregamento e inspeção
    df = load_and_inspect_data()
    
    # 2. Limpeza
    df_clean = clean_data(df)
    
    # 3. EDA
    df_analyzed, all_genres, genre_counts = exploratory_data_analysis(df_clean)
    
    # 4. Questões de negócio
    top_movie = analyze_business_questions(df_analyzed, genre_counts)
    
    # 5. Análise NLP
    f1_score_nlp, most_common_words = nlp_analysis(df_analyzed)
    
    # 6. Modelo de predição
    best_model, best_model_name, model_results, feature_columns, top_genres_list, df_model = build_rating_prediction_model(df_analyzed, all_genres)
    
    # 7. Predição específica
    prediction = predict_shawshank_redemption(best_model, feature_columns, top_genres_list, df_model)
    
    # 8. Salvar modelo
    model_package = save_model(best_model, best_model_name, model_results, feature_columns, top_genres_list, df_model)
    
    # Conclusões finais
    print("\n" + "="*60)
    print("PROJETO CONCLUIDO COM SUCESSO!")
    print("="*60)
    
    print(f"\nRESUMO DOS RESULTADOS:")
    print(f"• Recomendacao Universal: {top_movie['Series_Title']}")
    print(f"• Melhor Modelo: {best_model_name} (RMSE: {model_results[best_model_name]['rmse']:.3f})")
    print(f"• Predicao Shawshank: {prediction:.3f}")
    print(f"• Performance NLP: F1-score = {f1_score_nlp:.3f}")
    
    print(f"\nENTREGAVEIS CRIADOS:")
    print("- Analise Exploratoria de Dados (EDA)")
    print("- Respostas as questoes de negocio")
    print("- Modelo de predicao de genero (NLP)")
    print("- Modelo de predicao de rating IMDB")
    print("- Predicao para The Shawshank Redemption")
    print("- Modelo salvo em formato .pkl")
    
    print(f"\nRECOMENDACOES ESTRATEGICAS PARA PPRODUCTIONS:")
    print("1. SUCESSO COMERCIAL: Foque em Action/Adventure/Sci-Fi")
    print("2. PRESTIGIO: Priorize Drama com diretores renomados") 
    print("3. SWEET SPOT: Filmes que equilibram espetaculo e substancia")
    print("4. FATORES CRITICOS: Meta_score, popularidade e historico da equipe")
    
    return df_analyzed, model_package

if __name__ == "__main__":
    df_final, model_pkg = main()