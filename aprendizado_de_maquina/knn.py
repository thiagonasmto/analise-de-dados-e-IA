# Importando as bibliotecas necessárias
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Carregar o arquivo CSV
df = pd.read_csv('./caminho.csv')
df = df.drop(['col_nao_numericas'], axis=1)

# Normalizando as colunas para que todas as strings estejam em letras maiúsculas
df['col1'] = df['col1'].str.upper()

# Tratamento de variáveis categoricas
df_encoded = pd.get_dummies(df, columns=['col1', 'col2', 'col3'])

df_encoded.to_csv('df_encoded_knn.csv', index=False)

# Visualizar as primeiras linhas do dataset
print(df_encoded.head())

# Supondo que o arquivo CSV tenha uma coluna 'target' que é a variável que queremos prever
# As features (variáveis independentes) são todas as outras colunas
X = df_encoded.drop('target', axis=1)
y = df_encoded['target']

# Dividir o conjunto de dados em conjunto de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalizando as features (opcional, mas recomendado para KNN)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Criando o modelo KNN
knn = KNeighborsClassifier(n_neighbors=5)  # n_neighbors é o número de vizinhos

# Treinando o modelo
knn.fit(X_train, y_train)

# Fazendo previsões no conjunto de teste
y_pred = knn.predict(X_test)

# Avaliando o modelo
print("Acurácia:", accuracy_score(y_test, y_pred))
print("Relatório de Classificação:\n", classification_report(y_test, y_pred))

# Calculando a matriz de confusão
cm = confusion_matrix(y_test, y_pred)

# Plotando a matriz de confusão
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Matriz de Confusão')
plt.xlabel('Previsto')
plt.ylabel('Real')
plt.show()