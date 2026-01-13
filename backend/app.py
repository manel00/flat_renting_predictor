from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import uvicorn
import json
from data_processor import DataProcessor
from ml_model import HousingPriceModel
import os

# Initialize FastAPI app
app = FastAPI(
    title="Housing Price Prediction API",
    description="API for Barcelona housing rental price predictions and visualizations",
    version="1.0.0"
)

# Configure CORS - Allow all origins for Render deployment
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (Render frontend URL changes)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize data processor and model
DATA_PATH = os.path.join(os.path.dirname(__file__), 'housing_data.csv')
if not os.path.exists(DATA_PATH):
    DATA_PATH = '/app/housing_data.csv'  # Docker path
data_processor = DataProcessor(DATA_PATH)
ml_model = HousingPriceModel(data_processor)

# Try to load existing model, if not available, train it
try:
    ml_model.load_model()
    print("✅ Model loaded successfully")
except FileNotFoundError:
    print("⚠️  No trained model found. Training new model...")
    ml_model.train(model_type='xgboost')
    print("✅ Model trained and saved")

# Pydantic models for request/response
class PredictionRequest(BaseModel):
    territory: str
    year: int

class PredictionResponse(BaseModel):
    territory: str
    year: int
    predicted_price: float
    confidence_interval: Dict[str, float]
    std: float

class Territory(BaseModel):
    name: str
    type: str

class BulkPredictionRequest(BaseModel):
    territories: List[str]
    years: List[int]

# API Endpoints
@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Housing Price Prediction API",
        "version": "1.0.0",
        "endpoints": {
            "data": "/api/data",
            "3d_data": "/api/3d-data",
            "territories": "/api/territories",
            "stats": "/api/stats",
            "insights": "/api/insights",
            "predict": "/api/predict (POST)",
            "feature_importance": "/api/feature-importance",
            "model_metrics": "/api/model/metrics"
        }
    }

@app.get("/api/data")
async def get_data():
    """Get all housing data"""
    try:
        data = data_processor.get_all_data()
        return {
            "success": True,
            "count": len(data),
            "data": data
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/3d-data")
async def get_3d_data():
    """Get data formatted for 3D visualizations"""
    try:
        data = data_processor.get_3d_data()
        return {
            "success": True,
            "data": data
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/territories")
async def get_territories():
    """Get list of all territories"""
    try:
        territories = data_processor.get_territories()
        return {
            "success": True,
            "count": len(territories),
            "territories": territories
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/stats")
async def get_stats():
    """Get statistical summaries"""
    try:
        stats = data_processor.get_stats()
        return {
            "success": True,
            "stats": stats
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/territory/{territory_name}")
async def get_territory_data(territory_name: str):
    """Get data for a specific territory"""
    try:
        data = data_processor.get_territory_data(territory_name)
        if not data:
            raise HTTPException(status_code=404, detail=f"Territory '{territory_name}' not found")
        return {
            "success": True,
            "territory": territory_name,
            "count": len(data),
            "data": data
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/data-by-type/{territory_type}")
async def get_data_by_type(territory_type: str):
    """Get data filtered by territory type (e.g., 'Districte', 'Barri')"""
    try:
        data = data_processor.get_data_by_type(territory_type)
        if not data:
            raise HTTPException(status_code=404, detail=f"No data found for territory type '{territory_type}'")
        return {
            "success": True,
            "territory_type": territory_type,
            "count": len(data),
            "data": data
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/predict", response_model=PredictionResponse)
async def predict_price(request: PredictionRequest):
    """Predict housing price for a territory and year"""
    try:
        prediction = ml_model.predict(request.territory, request.year)
        return prediction
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class BulkPredictionRequest(BaseModel):
    territories: List[str]
    years: List[int]

@app.post("/api/bulk-predict")
async def bulk_predict(request: BulkPredictionRequest):
    """Predict housing prices for multiple territories and years"""
    try:
        predictions = ml_model.bulk_predict(request.territories, request.years)
        return {
            "success": True,
            "count": len(predictions),
            "predictions": predictions
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/feature-importance")
async def get_feature_importance():
    """Get feature importance from the ML model"""
    try:
        importance = ml_model.get_feature_importance()
        return {
            "success": True,
            "features": importance
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/model/metrics")
async def get_model_metrics():
    """Get model performance metrics (R², RMSE, MAE, CV scores)"""
    try:
        metrics = ml_model.get_metrics()
        if not metrics:
            return {
                "success": False,
                "message": "No metrics available. Model may not be trained yet."
            }
        return {
            "success": True,
            "metrics": metrics
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/insights")
async def get_insights():
    """Get comprehensive data analysis insights"""
    try:
        # Try to load insights from file
        insights_path = os.path.join(os.path.dirname(__file__), 'housing_insights.json')
        if not os.path.exists(insights_path):
            insights_path = '/app/housing_insights.json'  # Docker path
        
        if not os.path.exists(insights_path):
            raise HTTPException(
                status_code=404, 
                detail="Insights file not found. Run analysis script first."
            )
        
        with open(insights_path, 'r', encoding='utf-8') as f:
            insights = json.load(f)
        
        return {
            "success": True,
            "insights": insights
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": ml_model.model is not None
    }

if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
