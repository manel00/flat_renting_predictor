import pandas as pd
import numpy as np
from typing import Dict, List, Any
import json

class DataProcessor:
    def __init__(self, csv_path: str):
        self.csv_path = csv_path
        self.df = None
        self.load_data()
    
    def load_data(self):
        """Load and preprocess the housing data"""
        self.df = pd.read_csv(self.csv_path)
        # Clean column names
        self.df.columns = self.df.columns.str.strip()
        
    def get_all_data(self) -> List[Dict[str, Any]]:
        """Return all data in JSON format"""
        data = []
        
        for _, row in self.df.iterrows():
            territory = row['Territorio']
            territory_type = row['Tipo de territorio']
            
            # Get all year columns (from 2000 to 2025)
            year_columns = [col for col in self.df.columns if col.isdigit()]
            
            for year in year_columns:
                price = row[year]
                # Skip missing values (represented as '-')
                if price != '-' and pd.notna(price):
                    data.append({
                        'territory': territory,
                        'territory_type': territory_type,
                        'year': int(year),
                        'price': float(price)
                    })
        
        return data
    
    def get_3d_data(self) -> Dict[str, Any]:
        """Return data formatted for 3D visualizations"""
        all_data = self.get_all_data()
        
        # Create arrays for 3D plotting
        territories = []
        years = []
        prices = []
        territory_types = []
        
        for item in all_data:
            territories.append(item['territory'])
            years.append(item['year'])
            prices.append(item['price'])
            territory_types.append(item['territory_type'])
        
        return {
            'territories': territories,
            'years': years,
            'prices': prices,
            'territory_types': territory_types
        }
    
    def get_territories(self) -> List[Dict[str, str]]:
        """Return list of unique territories with their types"""
        territories = []
        for _, row in self.df.iterrows():
            territories.append({
                'name': row['Territorio'],
                'type': row['Tipo de territorio']
            })
        return territories
    
    def get_stats(self) -> Dict[str, Any]:
        """Calculate statistical summaries"""
        all_data = self.get_all_data()
        df_data = pd.DataFrame(all_data)
        
        if df_data.empty:
            return {}
        
        # Overall statistics
        stats = {
            'overall': {
                'mean_price': float(df_data['price'].mean()),
                'median_price': float(df_data['price'].median()),
                'min_price': float(df_data['price'].min()),
                'max_price': float(df_data['price'].max()),
                'std_price': float(df_data['price'].std())
            },
            'by_year': {},
            'by_territory_type': {},
            'trends': {}
        }
        
        # Statistics by year
        for year in sorted(df_data['year'].unique()):
            year_data = df_data[df_data['year'] == year]
            stats['by_year'][str(year)] = {
                'mean_price': float(year_data['price'].mean()),
                'count': int(len(year_data))
            }
        
        # Statistics by territory type
        for territory_type in df_data['territory_type'].unique():
            type_data = df_data[df_data['territory_type'] == territory_type]
            stats['by_territory_type'][territory_type] = {
                'mean_price': float(type_data['price'].mean()),
                'count': int(len(type_data))
            }
        
        # Calculate growth trends
        recent_years = df_data[df_data['year'] >= 2020]
        older_years = df_data[df_data['year'] < 2020]
        
        if not recent_years.empty and not older_years.empty:
            recent_mean = recent_years['price'].mean()
            older_mean = older_years['price'].mean()
            growth_rate = ((recent_mean - older_mean) / older_mean) * 100
            
            stats['trends'] = {
                'recent_mean': float(recent_mean),
                'historical_mean': float(older_mean),
                'growth_rate_percent': float(growth_rate)
            }
        
        return stats
    
    def get_territory_data(self, territory_name: str) -> List[Dict[str, Any]]:
        """Get data for a specific territory"""
        all_data = self.get_all_data()
        return [item for item in all_data if item['territory'] == territory_name]
    
    def get_data_by_type(self, territory_type: str) -> List[Dict[str, Any]]:
        """Get data filtered by territory type (e.g., 'Districte', 'Barri')"""
        all_data = self.get_all_data()
        return [item for item in all_data if item['territory_type'] == territory_type]
    
    def prepare_ml_data(self) -> tuple:
        """Prepare data for machine learning with advanced feature engineering"""
        all_data = self.get_all_data()
        df = pd.DataFrame(all_data)
        
        if df.empty:
            return None, None, None, None
        
        # Sort by territory and year for proper lag/rolling calculations
        df = df.sort_values(['territory', 'year']).reset_index(drop=True)
        
        # Advanced Feature Engineering
        features_list = []
        
        for territory in df['territory'].unique():
            territory_df = df[df['territory'] == territory].copy()
            
            # Lag features (previous 1, 2, 3 years)
            territory_df['price_lag_1'] = territory_df['price'].shift(1)
            territory_df['price_lag_2'] = territory_df['price'].shift(2)
            territory_df['price_lag_3'] = territory_df['price'].shift(3)
            
            # Rolling averages (3-year, 5-year windows)
            territory_df['price_rolling_3'] = territory_df['price'].rolling(window=3, min_periods=1).mean()
            territory_df['price_rolling_5'] = territory_df['price'].rolling(window=5, min_periods=1).mean()
            
            # Growth rates
            territory_df['price_growth_1y'] = territory_df['price'].pct_change(1)
            territory_df['price_growth_2y'] = territory_df['price'].pct_change(2)
            
            # Territory-level statistics
            territory_df['territory_mean'] = territory_df['price'].expanding().mean()
            territory_df['territory_std'] = territory_df['price'].expanding().std()
            territory_df['territory_min'] = territory_df['price'].expanding().min()
            territory_df['territory_max'] = territory_df['price'].expanding().max()
            
            features_list.append(territory_df)
        
        # Combine all territories
        df = pd.concat(features_list, ignore_index=True)
        
        # Fill NaN values from lag/rolling operations
        df = df.fillna(method='bfill').fillna(method='ffill').fillna(0)
        
        # Encode territories
        territory_encoded = pd.get_dummies(df['territory'], prefix='territory')
        territory_type_encoded = pd.get_dummies(df['territory_type'], prefix='type')
        
        # Temporal features
        df['year_normalized'] = (df['year'] - df['year'].min()) / (df['year'].max() - df['year'].min())
        df['year_squared'] = df['year_normalized'] ** 2
        
        # Combine all features
        X = pd.concat([
            df[['year', 'year_normalized', 'year_squared',
                'price_lag_1', 'price_lag_2', 'price_lag_3',
                'price_rolling_3', 'price_rolling_5',
                'price_growth_1y', 'price_growth_2y',
                'territory_mean', 'territory_std', 'territory_min', 'territory_max']],
            territory_encoded,
            territory_type_encoded
        ], axis=1)
        
        y = df['price']
        
        # Store feature names for later use
        feature_names = X.columns.tolist()
        territory_columns = territory_encoded.columns.tolist()
        
        return X, y, feature_names, territory_columns
