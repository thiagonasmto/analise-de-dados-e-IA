import os
import pandas as pd
from minisom import MiniSom
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import numpy as np

# Carregar o DataFrame
df = pd.read_csv('./caminho.csv')

# Selecionar colunas de interesse
categorical_cols = ['col4','col5','col6']
numerical_cols = ['col1','col2','col3']

df = df[categorical_cols + numerical_cols]

# Criar o transformer para one-hot encoding
categorical_transformer = OneHotEncoder(sparse_output=False, drop='first')

# Criar o transformer para normalizar variáveis numéricas
numerical_transformer = MinMaxScaler()

# Criar o preprocessor que aplica os transformadores apropriados às colunas corretas
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ]
)

# Aplicar as transformações
X_processed = preprocessor.fit_transform(df)

# Transformar os dados processados em um DataFrame
feature_names = numerical_cols + list(preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_cols))
df_processed = pd.DataFrame(X_processed, columns=feature_names)

# Definir parâmetros do SOM
som_size = 10  # Tamanho da grade do SOM (10x10)
som = MiniSom(som_size, som_size, df_processed.shape[1], sigma=1.0, learning_rate=0.5)

# Treinar o SOM
som.train_batch(df_processed.values, num_iteration=1000)

# Criar o diretório 'cluster_somf' se não existir
os.makedirs('cluster_somf', exist_ok=True)

# Mapear os dados para os neurônios do SOM
win_map = som.win_map(df_processed.values)

# Criar um DataFrame para armazenar os resultados com os clusters
df_clusters = df.copy()
df_clusters = df_clusters.reset_index(drop=True)
df_clusters['Cluster'] = [som.winner(x)[0] * som_size + som.winner(x)[1] for x in df_processed.values]

# Salvar os dados em CSVs por cluster
for cluster in df_clusters['Cluster'].unique():
    cluster_df = df_clusters[df_clusters['Cluster'] == cluster]
    cluster_df.drop(columns='Cluster', inplace=True)  # Remove a coluna 'Cluster' antes de salvar
    cluster_df.to_csv(f'cluster_somf/cluster_{cluster}.csv', index=False)

print("Arquivos CSV salvos no diretório 'cluster_somf'.")
