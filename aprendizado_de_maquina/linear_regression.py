import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Carregar o arquivo CSV
df = pd.read_csv('./caminho.csv')
df = df.drop(['col_nao_numericas'], axis=1)

# Normalizando as colunas para que todas as strings estejam em letras maiúsculas
df['col1'] = df['col1'].str.upper()

# Tratamento de variáveis categoricas
df_encoded = pd.get_dummies(df, columns=['col1', 'col2', 'col3'])

# Separando as features (X) e a variável target (y)
X = df_encoded.drop('target', axis=1)
y = df_encoded['target']

# Dividindo os dados em conjuntos de treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Criando o modelo de Regressão Linear
model = LinearRegression()

# Treinando o modelo
model.fit(X_train, y_train)

# Fazendo previsões
y_pred = model.predict(X_test)

# Avaliando o modelo
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Absolute Error (MAE): {mae}')
print(f'Mean Squared Error (MSE): {mse}')
print(f'R-squared (R2): {r2}')
