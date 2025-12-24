from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# 1. Cargar Datos
data = fetch_california_housing()
X, y = data.data, data.target

# 2. Dividir
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# 3. Entrenar
model = LinearRegression()
model.fit(X_train, y_train)

# 4. Predecir
predictions = model.predict(X_test[:5])
print("--- PREDICCIONES DE PRECIO (x $100,000) ---")
print(f"Predicho: {predictions}")
print(f"Real:     {y_test[:5]}")
