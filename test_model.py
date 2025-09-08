#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Teste de Carregamento e Uso do Modelo IMDB
"""

import pickle
import pandas as pd

def test_model():
    """Testa o carregamento e uso do modelo salvo"""
    
    print("Teste de Carregamento do Modelo IMDB")
    print("="*40)
    
    # Carregar modelo
    try:
        import os
        model_path = os.path.join(os.path.dirname(__file__), 'modelo_imdb.pkl')
        with open(model_path, 'rb') as f:
            model_package = pickle.load(f)
        print("Modelo carregado com sucesso!")
        
        # Informações do modelo
        print(f"\nInformações do Modelo:")
        print(f"- Algoritmo: {model_package['model_name']}")
        print(f"- RMSE: {model_package['performance']['rmse']:.3f}")
        print(f"- R²: {model_package['performance']['r2']:.3f}")
        print(f"- Features: {len(model_package['feature_columns'])}")
        print(f"- Versão: {model_package['version']}")
        print(f"- Data de criação: {model_package['created_date']}")
        
        print(f"\nTop 5 Features:")
        for i, feature in enumerate(model_package['feature_columns'][:5], 1):
            print(f"{i}. {feature}")
        
        print(f"\nEstatísticas de treino disponíveis:")
        for key in model_package['training_stats'].keys():
            if isinstance(model_package['training_stats'][key], dict):
                count = len(model_package['training_stats'][key])
                print(f"- {key}: {count} registros")
            else:
                print(f"- {key}: {model_package['training_stats'][key]:.3f}")
        
        print("\nModelo pronto para uso!")
        return True
        
    except Exception as e:
        print(f"Erro ao carregar modelo: {e}")
        return False

if __name__ == "__main__":
    test_model()