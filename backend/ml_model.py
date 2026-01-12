import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import xgboost as xgb
import joblib
import os
from typing import Dict, Any, List
from data_processor import DataProcessor

class HousingPriceModel:
    def __init__(self, data_processor: DataProcessor):
        self.data_processor = data_processor
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        self.territory_columns = None
        # Create models directory if it doesn't exist
        import os
        os.makedirs('models', exist_ok=True)
        self.model_path = 'models/housing_model.pkl'
        self.scaler_path = 'models/scaler.pkl'
        self.metadata_path = 'models/model_metadata.pkl'
        
    def train(self, model_type='xgboost'):
        """Train the machine learning model with advanced features"""
        print("Preparing data for training...")
        X, y, feature_names, territory_columns = self.data_processor.prepare_ml_data()
        
        if X is None:
            raise ValueError("No data available for training")
        
        self.feature_names = feature_names
        self.territory_columns = territory_columns
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        print(f"Training {model_type} model with advanced features...")
        if model_type == 'xgboost':
            self.model = xgb.XGBRegressor(
                n_estimators=500,
                max_depth=8,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                min_child_weight=3,
                gamma=0.1,
                reg_alpha=0.1,
                reg_lambda=1.0,
                random_state=42,
                n_jobs=-1,
                verbosity=0
            )
        elif model_type == 'random_forest':
            self.model = RandomForestRegressor(
                n_estimators=300,
                max_depth=25,
                min_samples_split=3,
                min_samples_leaf=1,
                max_features='sqrt',
                random_state=42,
                n_jobs=-1
            )
        elif model_type == 'gradient_boosting':
            self.model = GradientBoostingRegressor(
                n_estimators=300,
                max_depth=12,
                learning_rate=0.05,
                subsample=0.8,
                random_state=42
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred_train = self.model.predict(X_train_scaled)
        y_pred_test = self.model.predict(X_test_scaled)
        
        train_r2 = r2_score(y_train, y_pred_train)
        test_r2 = r2_score(y_test, y_pred_test)
        train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
        test_mae = mean_absolute_error(y_test, y_pred_test)
        
        print(f"\n{'='*50}")
        print(f"Model Training Results - {model_type}")
        print(f"{'='*50}")
        print(f"Train R² Score: {train_r2:.4f}")
        print(f"Test R² Score:  {test_r2:.4f}")
        print(f"Train RMSE:     {train_rmse:.2f} €")
        print(f"Test RMSE:      {test_rmse:.2f} €")
        print(f"Test MAE:       {test_mae:.2f} €")
        print(f"{'='*50}\n")
        
        # Cross-validation
        cv_scores = cross_val_score(
            self.model, X_train_scaled, y_train, 
            cv=5, scoring='r2', n_jobs=-1
        )
        print(f"Cross-validation R² scores: {cv_scores}")
        print(f"Mean CV R² Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        # Save model with metrics
        self.save_model({
            'train_r2': train_r2,
            'test_r2': test_r2,
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'test_mae': test_mae,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'model_type': model_type
        })
        
        return {
            'train_r2': train_r2,
            'test_r2': test_r2,
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'test_mae': test_mae,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std()
        }
    
    def save_model(self, metrics: Dict[str, Any] = None):
        """Save trained model and metadata"""
        if self.model is None:
            raise ValueError("No model to save. Train the model first.")
        
        joblib.dump(self.model, self.model_path)
        joblib.dump(self.scaler, self.scaler_path)
        
        metadata = {
            'feature_names': self.feature_names,
            'territory_columns': self.territory_columns
        }
        
        if metrics:
            metadata['metrics'] = metrics
            
        joblib.dump(metadata, self.metadata_path)
        
        print(f"Model saved to {self.model_path}")
        print(f"Scaler saved to {self.scaler_path}")
        print(f"Metadata saved to {self.metadata_path}")
    
    def load_model(self):
        """Load trained model and metadata"""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        
        self.model = joblib.load(self.model_path)
        self.scaler = joblib.load(self.scaler_path)
        metadata = joblib.load(self.metadata_path)
        self.feature_names = metadata['feature_names']
        self.territory_columns = metadata['territory_columns']
        
        print("Model loaded successfully")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get model performance metrics"""
        if not os.path.exists(self.metadata_path):
            return {}
        
        metadata = joblib.load(self.metadata_path)
        return metadata.get('metrics', {})
    
    def predict(self, territory: str, year: int) -> Dict[str, Any]:
        """Make a price prediction for a territory and year"""
        if self.model is None:
            try:
                self.load_model()
            except FileNotFoundError:
                raise ValueError("Model not trained. Train the model first.")
        
        # Get historical data for this territory to calculate trend
        all_data = self.data_processor.get_all_data()
        territory_data = [d for d in all_data if d['territory'] == territory]
        
        if territory_data:
            # Calculate trend from historical data using linear regression
            years = np.array([d['year'] for d in territory_data])
            prices = np.array([d['price'] for d in territory_data])
            
            # Fit linear trend
            if len(years) > 1:
                # Calculate slope and intercept
                slope, intercept = np.polyfit(years, prices, 1)
                
                # Get the last historical year and price
                last_year = years.max()
                last_price = prices[prices.shape[0] - 1]  # Last price in the series
                
                # If predicting future years, use trend
                if year > last_year:
                    # Use linear trend for future predictions
                    trend_prediction = slope * year + intercept
                    
                    # Also get ML model prediction
                    features = self._prepare_prediction_features(territory, year)
                    features_scaled = self.scaler.transform(features)
                    ml_prediction = self.model.predict(features_scaled)[0]
                    
                    # Blend both predictions (70% trend, 30% ML model)
                    prediction = 0.7 * trend_prediction + 0.3 * ml_prediction
                    
                    # Calculate standard deviation based on historical variance
                    residuals = prices - (slope * years + intercept)
                    std = np.std(residuals)
                else:
                    # For historical years, use ML model
                    features = self._prepare_prediction_features(territory, year)
                    features_scaled = self.scaler.transform(features)
                    prediction = self.model.predict(features_scaled)[0]
                    std = np.std(prices) * 0.1
            else:
                # Only one data point, use ML model
                features = self._prepare_prediction_features(territory, year)
                features_scaled = self.scaler.transform(features)
                prediction = self.model.predict(features_scaled)[0]
                std = prediction * 0.1
        else:
            # No historical data, use ML model only
            features = self._prepare_prediction_features(territory, year)
            features_scaled = self.scaler.transform(features)
            prediction = self.model.predict(features_scaled)[0]
            std = prediction * 0.1
        
        # Calculate confidence interval
        confidence_lower = prediction - 1.96 * std
        confidence_upper = prediction + 1.96 * std
        
        return {
            'territory': territory,
            'year': year,
            'predicted_price': float(prediction),
            'confidence_interval': {
                'lower': float(max(0, confidence_lower)),
                'upper': float(confidence_upper)
            },
            'std': float(std)
        }
    
    def bulk_predict(self, territories: List[str], years: List[int]) -> List[Dict[str, Any]]:
        """Make predictions for multiple territories and years"""
        predictions = []
        for territory in territories:
            for year in years:
                try:
                    pred = self.predict(territory, year)
                    predictions.append(pred)
                except Exception as e:
                    print(f"Warning: Could not predict for {territory} ({year}): {e}")
        return predictions
    
    def _prepare_prediction_features(self, territory: str, year: int) -> pd.DataFrame:
        """Prepare features for prediction with advanced features"""
        # Get all data to calculate features
        all_data = self.data_processor.get_all_data()
        df = pd.DataFrame(all_data)
        
        # Get territory-specific historical data
        territory_data = df[df['territory'] == territory].sort_values('year')
        
        year_min = df['year'].min()
        year_max = df['year'].max()
        year_normalized = (year - year_min) / (year_max - year_min)
        year_squared = year_normalized ** 2
        
        # Calculate advanced features based on historical data
        if len(territory_data) > 0:
            # Lag features - use last available prices
            price_lag_1 = territory_data['price'].iloc[-1] if len(territory_data) >= 1 else 0
            price_lag_2 = territory_data['price'].iloc[-2] if len(territory_data) >= 2 else price_lag_1
            price_lag_3 = territory_data['price'].iloc[-3] if len(territory_data) >= 3 else price_lag_2
            
            # Rolling averages
            price_rolling_3 = territory_data['price'].tail(3).mean()
            price_rolling_5 = territory_data['price'].tail(5).mean()
            
            # Growth rates
            if len(territory_data) >= 2:
                price_growth_1y = (territory_data['price'].iloc[-1] - territory_data['price'].iloc[-2]) / territory_data['price'].iloc[-2]
            else:
                price_growth_1y = 0
            
            if len(territory_data) >= 3:
                price_growth_2y = (territory_data['price'].iloc[-1] - territory_data['price'].iloc[-3]) / territory_data['price'].iloc[-3]
            else:
                price_growth_2y = price_growth_1y
            
            # Territory statistics
            territory_mean = territory_data['price'].mean()
            territory_std = territory_data['price'].std() if len(territory_data) > 1 else 0
            territory_min = territory_data['price'].min()
            territory_max = territory_data['price'].max()
        else:
            # Default values if no historical data
            price_lag_1 = price_lag_2 = price_lag_3 = 0
            price_rolling_3 = price_rolling_5 = 0
            price_growth_1y = price_growth_2y = 0
            territory_mean = territory_std = territory_min = territory_max = 0
        
        # Create feature dictionary
        features = {
            'year': year,
            'year_normalized': year_normalized,
            'year_squared': year_squared,
            'price_lag_1': price_lag_1,
            'price_lag_2': price_lag_2,
            'price_lag_3': price_lag_3,
            'price_rolling_3': price_rolling_3,
            'price_rolling_5': price_rolling_5,
            'price_growth_1y': price_growth_1y,
            'price_growth_2y': price_growth_2y,
            'territory_mean': territory_mean,
            'territory_std': territory_std,
            'territory_min': territory_min,
            'territory_max': territory_max
        }
        
        # Add territory encoding
        for col in self.territory_columns:
            territory_name = col.replace('territory_', '')
            features[col] = 1 if territory_name == territory else 0
        
        # Add territory type encoding
        territory_info = next(
            (t for t in self.data_processor.get_territories() if t['name'] == territory),
            None
        )
        
        if territory_info:
            territory_type = territory_info['type']
            type_columns = [col for col in self.feature_names if col.startswith('type_')]
            for col in type_columns:
                type_name = col.replace('type_', '')
                features[col] = 1 if type_name == territory_type else 0
        
        # Create DataFrame with correct column order
        df_features = pd.DataFrame([features])
        
        # Ensure all feature columns are present
        for col in self.feature_names:
            if col not in df_features.columns:
                df_features[col] = 0
        
        # Reorder columns to match training data
        df_features = df_features[self.feature_names]
        
        return df_features
    
    def get_feature_importance(self) -> List[Dict[str, Any]]:
        """Get feature importance from the model"""
        if self.model is None or not hasattr(self.model, 'feature_importances_'):
            return []
        
        importances = self.model.feature_importances_
        feature_importance = [
            {'feature': name, 'importance': float(imp)}
            for name, imp in zip(self.feature_names, importances)
        ]
        
        # Sort by importance
        feature_importance.sort(key=lambda x: x['importance'], reverse=True)
        
        return feature_importance[:20]  # Return top 20 features

if __name__ == "__main__":
    # Train the model
    print("Loading data...")
    data_processor = DataProcessor('/app/housing_data.csv')
    
    print("Initializing model...")
    model = HousingPriceModel(data_processor)
    
    print("\nTraining XGBoost model with advanced features...")
    results = model.train(model_type='xgboost')
    
    print("\n" + "="*50)
    print("Testing predictions...")
    print("="*50)
    
    # Test predictions
    test_cases = [
        ('Barcelona', 2024),
        ('Barcelona', 2026),
        ('Catalunya', 2025),
        ('Eixample', 2024)
    ]
    
    for territory, year in test_cases:
        try:
            prediction = model.predict(territory, year)
            print(f"\n{territory} ({year}):")
            print(f"  Predicted Price: {prediction['predicted_price']:.2f} €")
            print(f"  Confidence Interval: [{prediction['confidence_interval']['lower']:.2f}, {prediction['confidence_interval']['upper']:.2f}] €")
        except Exception as e:
            print(f"\n{territory} ({year}): Error - {e}")
    
    print("\n" + "="*50)
    print("Top 10 Most Important Features:")
    print("="*50)
    for i, feat in enumerate(model.get_feature_importance()[:10], 1):
        print(f"{i}. {feat['feature']}: {feat['importance']:.4f}")

