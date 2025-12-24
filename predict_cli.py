import joblib
import numpy as np

def predict_price(features):
    # Cargar modelo
    try:
        model = joblib.load('house_price_model.pkl')
    except FileNotFoundError:
        print("Error: Ejecuta primero model_pipeline.py para entrenar el modelo.")
        return

    # Features esperadas: [MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude]
    prediction = model.predict([features])
    price = prediction[0] * 100000 # El dataset está en unidades de 100k
    return price

if __name__ == "__main__":
    print("--- CALCULADORA DE PRECIOS INMOBILIARIOS ---")
    print("Ingresa los datos de la propiedad:")
    
    # Valores por defecto para el ejemplo rápido
    val = [
        float(input("Ingreso Medio Zona (ej. 3.5): ") or 3.5),
        float(input("Edad de la casa (ej. 20): ") or 20),
        float(input("Promedio Habitaciones (ej. 6): ") or 6),
        float(input("Promedio Dormitorios (ej. 1): ") or 1),
        float(input("Población zona (ej. 800): ") or 800),
        float(input("Ocupación promedio (ej. 3): ") or 3),
        float(input("Latitud (ej. 34.2): ") or 34.2),
        float(input("Longitud (ej. -118.4): ") or -118.4)
    ]
    
    price = predict_price(val)
    print(f"\n--------------------------------")
    print(f"PRECIO ESTIMADO: ${price:,.2f}")
    print(f"--------------------------------")
