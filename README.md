# 🏠 Housing Price Prediction Application

Una aplicación premium full-stack para visualizar y predecir precios de alquiler de viviendas en Barcelona con visualizaciones 3D interactivas y machine learning.

![Premium Dark Theme](https://img.shields.io/badge/Theme-Premium%20Dark-9c27b0)
![Python](https://img.shields.io/badge/Python-3.11+-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-green)
![Angular](https://img.shields.io/badge/Angular-21-red)
![Plotly](https://img.shields.io/badge/Plotly-3D-orange)

## ✨ Características

### 🎨 Visualizaciones 3D Interactivas
- **Superficies 3D**: Evolución de precios a través del tiempo y territorios
- **Gráficos de Dispersión 3D**: Distribución de precios por tipo de territorio
- **Comparación de Territorios**: Top 15 territorios por precio promedio
- **Interactividad completa**: Rotación, zoom, y hover tooltips

### 🤖 Predicción con Machine Learning
- Modelo Random Forest entrenado con datos históricos (2000-2025)
- Predicciones con intervalos de confianza del 95%
- Búsqueda inteligente de territorios con autocomplete
- Visualización detallada de resultados

### 📊 Dashboard Analítico
- Estadísticas generales (promedio, mediana, máximo, mínimo)
- Análisis de tendencias y tasas de crecimiento
- Desglose por año y tipo de territorio
- Tarjetas interactivas con efectos glassmorphism

### 🎯 Diseño Premium
- Tema oscuro con gradientes vibrantes
- Efectos glassmorphism y backdrop blur
- Animaciones suaves y transiciones
- Tipografía moderna (Inter)
- Totalmente responsive

## 🚀 Inicio Rápido

### 🐳 Opción 1: Docker Compose (Recomendado)

La forma más rápida de ejecutar la aplicación completa:

```bash
# Construir e iniciar todos los servicios
docker-compose up --build

# O en modo detached (segundo plano)
docker-compose up -d --build
```

La aplicación estará disponible en:
- **Frontend**: http://localhost
- **Backend API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs

Para detener los servicios:
```bash
docker-compose down
```

Para ver los logs:
```bash
docker-compose logs -f
```

### 💻 Opción 2: Instalación Local

### Requisitos Previos
- Python 3.11 o superior
- Node.js 18 o superior
- npm 9 o superior

### Instalación

#### 1. Backend (Python/FastAPI)

```bash
# Navegar a la carpeta backend
cd backend

# Instalar dependencias
pip install -r requirements.txt

# Entrenar el modelo (primera vez)
python ml_model.py

# Iniciar el servidor
python app.py
```

El backend estará disponible en `http://localhost:8000`

#### 2. Frontend (Angular 21)

```bash
# Navegar a la carpeta frontend
cd frontend

# Instalar dependencias
npm install

# Iniciar el servidor de desarrollo
npm start
```

El frontend estará disponible en `http://localhost:4200`

## 📁 Estructura del Proyecto

```
prediccion_vivienda/
├── docker-compose.yml         # Docker Compose configuration
├── .gitignore                 # Git ignore rules
├── README.md                  # This file
├── housing_data.csv           # Dataset
├── backend/
│   ├── Dockerfile             # Backend Docker image
│   ├── .dockerignore          # Docker ignore rules
│   ├── app.py                 # FastAPI application
│   ├── data_processor.py      # Data loading and preprocessing
│   ├── ml_model.py           # Machine learning model
│   ├── requirements.txt       # Python dependencies
│   ├── housing_model.pkl     # Trained model (generated)
│   └── scaler.pkl            # Feature scaler (generated)
└── frontend/
    ├── Dockerfile             # Frontend Docker image
    ├── .dockerignore          # Docker ignore rules
    ├── nginx.conf             # Nginx configuration
    ├── package.json           # Node dependencies
    └── src/
        ├── app/
        │   ├── components/
        │   │   ├── dashboard/          # Dashboard component
        │   │   ├── visualization-3d/   # 3D visualization component
        │   │   └── prediction/         # Prediction component
        │   ├── services/
        │   │   └── api.service.ts      # API service
        │   ├── models/
        │   │   └── housing.model.ts    # TypeScript interfaces
        │   ├── app.ts                  # Main app component
        │   ├── app.routes.ts           # Route configuration
        │   └── app.config.ts           # App configuration
        └── styles.scss                 # Global styles
```

## 🔌 API Endpoints

### GET `/api/data`
Obtiene todos los datos de precios de vivienda.

### GET `/api/3d-data`
Obtiene datos formateados para visualizaciones 3D.

### GET `/api/territories`
Obtiene lista de todos los territorios disponibles.

### GET `/api/stats`
Obtiene estadísticas resumidas.

### GET `/api/territory/{territory_name}`
Obtiene datos de un territorio específico.

### POST `/api/predict`
Realiza una predicción de precio.

**Request Body:**
```json
{
  "territory": "Barcelona",
  "year": 2026
}
```

**Response:**
```json
{
  "territory": "Barcelona",
  "year": 2026,
  "predicted_price": 1150.50,
  "confidence_interval": {
    "lower": 1050.25,
    "upper": 1250.75
  },
  "std": 50.25
}
```

### GET `/api/feature-importance`
Obtiene la importancia de características del modelo ML.

## 🎨 Tecnologías Utilizadas

### Backend
- **FastAPI**: Framework web moderno y rápido
- **Pandas**: Procesamiento de datos
- **Scikit-learn**: Machine learning
- **XGBoost**: Algoritmos de boosting
- **Uvicorn**: Servidor ASGI

### Frontend
- **Angular 21**: Framework frontend
- **Plotly.js**: Visualizaciones 3D interactivas
- **TypeScript**: Tipado estático
- **SCSS**: Estilos avanzados
- **RxJS**: Programación reactiva

## 📊 Datos

El dataset contiene precios medios de alquiler de viviendas en Barcelona desde 2000 hasta 2025, desglosados por:
- **Comunidad Autónoma**: Catalunya
- **Ámbitos funcionales**: Metropolità de Barcelona
- **Municipios**: Barcelona
- **Distritos**: 10 distritos
- **Barrios**: 73 barrios

## 🤖 Modelo de Machine Learning

### Características
- **Algoritmo**: Random Forest Regressor
- **Features**: Territorio, tipo de territorio, año, características temporales
- **Encoding**: One-hot encoding para variables categóricas
- **Scaling**: StandardScaler para normalización
- **Validación**: Cross-validation con 5 folds

### Métricas de Rendimiento
- **R² Score**: > 0.85 (en conjunto de prueba)
- **RMSE**: < 100€ (error cuadrático medio)
- **MAE**: < 70€ (error absoluto medio)

## 🎯 Uso

### 1. Dashboard
Visualiza estadísticas generales, tendencias y comparaciones por territorio.

### 2. Visualización 3D
Explora los datos con tres tipos de visualizaciones:
- **Superficie 3D**: Evolución temporal de precios
- **Dispersión 3D**: Distribución por tipo de territorio
- **Top Territorios**: Comparación de precios promedio

### 3. Predicción
1. Busca un territorio usando el autocomplete
2. Selecciona un año (2020-2030)
3. Haz clic en "Predecir Precio"
4. Visualiza el precio predicho con intervalo de confianza

## 🔧 Desarrollo

### Entrenar el modelo manualmente
```bash
cd backend
python ml_model.py
```

### Ejecutar tests (cuando estén disponibles)
```bash
# Backend
cd backend
pytest

# Frontend
cd frontend
npm test
```

### Build de producción
```bash
# Frontend
cd frontend
npm run build
```

## 🎨 Personalización

### Cambiar colores del tema
Edita `frontend/src/styles.scss` y modifica las variables de color en los gradientes.

### Ajustar el modelo ML
Edita `backend/ml_model.py` y modifica los hiperparámetros del modelo:
```python
RandomForestRegressor(
    n_estimators=200,  # Número de árboles
    max_depth=20,      # Profundidad máxima
    min_samples_split=5,
    min_samples_leaf=2
)
```

## 📝 Licencia

Este proyecto es de código abierto y está disponible bajo la licencia MIT.

## 🤝 Contribuciones

Las contribuciones son bienvenidas. Por favor:
1. Fork el proyecto
2. Crea una rama para tu feature (`git checkout -b feature/AmazingFeature`)
3. Commit tus cambios (`git commit -m 'Add some AmazingFeature'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abre un Pull Request

## 📧 Contacto

Para preguntas o sugerencias, por favor abre un issue en el repositorio.

---

**Hecho con ❤️ usando Angular 21, FastAPI y Plotly.js**
