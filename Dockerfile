FROM python:3.9-slim

# Instalamos scikit-learn y joblib
RUN pip install scikit-learn joblib numpy

WORKDIR /app

COPY . .

# Por defecto dejamos una shell interactiva para que puedas elegir 
# si entrenar (model_pipeline.py) o predecir (predict_cli.py)
CMD ["/bin/bash"]
