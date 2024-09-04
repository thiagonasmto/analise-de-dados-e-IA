# Importando as bibliotecas necessárias
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix
from sklearn import tree
import matplotlib.pyplot as plt
import seaborn as sns

# Carregar o arquivo CSV
df = pd.read_csv('./caminho.csv')
df = df.drop(['col_nao_numericas'], axis=1)

# Normalizando as colunas para que todas as strings estejam em letras maiúsculas
df['col1'] = df['col1'].str.upper()

# Tratamento de variáveis categoricas
df_encoded = pd.get_dummies(df, columns=['col1', 'col2', 'col3'])

# Exportanto arquivo para visualização
df_encoded.to_csv('df_encoded_decision_tree.csv', index=False)

# Visualizar as primeiras linhas do dataset
print(df_encoded.head())

# Supondo que o arquivo CSV tenha uma coluna 'target' que é a variável que queremos prever
# As features (variáveis independentes) são todas as outras colunas
X = df_encoded.drop('target', axis=1)
y = df_encoded['target']

# Dividir o conjunto de dados em conjunto de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# Criando o modelo de Árvore de Decisão
# Você pode ajustar o parâmetro 'max_depth' para limitar a profundidade da árvore
dtree = DecisionTreeClassifier(random_state=42)

# Treinando o modelo
dtree.fit(X_train, y_train)

# Fazendo previsões no conjunto de teste
y_pred = dtree.predict(X_test)

# Avaliando o modelo
print("Acurácia:", accuracy_score(y_test, y_pred))
print("Relatório de Classificação:\n", classification_report(y_test, y_pred))

# Visualizando a Árvore de Decisão
plt.figure(figsize=(20,10))
tree.plot_tree(dtree, filled=True, feature_names=X.columns, class_names=True, rounded=True)
plt.show()

# Calculando a matriz de confusão
cm = confusion_matrix(y_test, y_pred)

# Plotando a matriz de confusão
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Matriz de Confusão')
plt.xlabel('Previsto')
plt.ylabel('Real')
plt.show()