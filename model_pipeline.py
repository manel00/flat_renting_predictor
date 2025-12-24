from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import numpy as np

def train_and_save():
    print("Cargando datos de California Housing...")
    data = fetch_california_housing()
    X, y = data.data, data.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Creamos un Pipeline: Primero escala datos, luego aplica regresión
    # Esto es vital en producción para escalar los inputs nuevos igual que los de entrenamiento
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('regressor', LinearRegression())
    ])

    print("Entrenando modelo...")
    pipeline.fit(X_train, y_train)

    # Evaluación
    preds = pipeline.predict(X_test)
    mse = mean_squared_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    
    print(f"Modelo Entrenado. MSE: {mse:.4f}, R2 Score: {r2:.4f}")

    # Guardar modelo en disco
    joblib.dump(pipeline, 'house_price_model.pkl')
    print("Modelo guardado en 'house_price_model.pkl'")

if __name__ == "__main__":
    train_and_save()
