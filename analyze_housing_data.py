#!/usr/bin/env python3
"""
Barcelona Housing Price Analysis
Comprehensive data analysis and insights generation for frontend integration
"""

import pandas as pd
import numpy as np
import json
from datetime import datetime
from typing import Dict, List, Any

class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for numpy types"""
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

def load_and_clean_data(filepath: str) -> pd.DataFrame:
    """Load CSV and handle missing values"""
    df = pd.read_csv(filepath)
    # Replace '-' with NaN
    df = df.replace('-', np.nan)
    # Convert year columns to numeric
    year_cols = [col for col in df.columns if col.isdigit()]
    for col in year_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

def calculate_statistics(df: pd.DataFrame) -> Dict[str, Any]:
    """Calculate comprehensive statistics"""
    year_cols = [col for col in df.columns if col.isdigit()]
    
    stats = {
        'total_territories': len(df),
        'territory_types': df['Tipo de territorio'].value_counts().to_dict(),
        'year_range': {
            'start': min(year_cols),
            'end': max(year_cols)
        },
        'data_quality': {
            'total_cells': len(df) * len(year_cols),
            'missing_values': df[year_cols].isna().sum().sum(),
            'completeness_pct': round((1 - df[year_cols].isna().sum().sum() / (len(df) * len(year_cols))) * 100, 2)
        }
    }
    
    return stats

def analyze_price_trends(df: pd.DataFrame) -> Dict[str, Any]:
    """Analyze price trends and growth rates"""
    year_cols = [col for col in df.columns if col.isdigit()]
    recent_years = [str(y) for y in range(2020, 2026)]
    
    trends = {}
    
    # Barcelona city analysis
    bcn = df[df['Territorio'] == 'Barcelona'].iloc[0]
    
    trends['barcelona_city'] = {
        'current_price_2025': float(bcn['2025']) if pd.notna(bcn['2025']) else None,
        'price_2020': float(bcn['2020']) if pd.notna(bcn['2020']) else None,
        'price_2015': float(bcn['2015']) if pd.notna(bcn['2015']) else None,
        'growth_5yr_pct': round(((float(bcn['2025']) / float(bcn['2020'])) - 1) * 100, 2) if pd.notna(bcn['2025']) and pd.notna(bcn['2020']) else None,
        'growth_10yr_pct': round(((float(bcn['2025']) / float(bcn['2015'])) - 1) * 100, 2) if pd.notna(bcn['2025']) and pd.notna(bcn['2015']) else None
    }
    
    # District analysis
    districts = df[df['Tipo de territorio'] == 'Districte'].copy()
    district_data = []
    
    for _, row in districts.iterrows():
        if pd.notna(row['2025']) and pd.notna(row['2020']):
            growth_5yr = ((float(row['2025']) / float(row['2020'])) - 1) * 100
            district_data.append({
                'name': row['Territorio'],
                'price_2025': float(row['2025']),
                'price_2020': float(row['2020']),
                'growth_5yr_pct': round(growth_5yr, 2)
            })
    
    # Sort by growth rate
    district_data.sort(key=lambda x: x['growth_5yr_pct'], reverse=True)
    
    trends['top_growing_districts'] = district_data[:5]
    trends['slowest_growing_districts'] = district_data[-5:]
    
    return trends

def find_investment_opportunities(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Identify undervalued neighborhoods with growth potential"""
    neighborhoods = df[df['Tipo de territorio'] == 'Barri'].copy()
    
    opportunities = []
    
    for _, row in neighborhoods.iterrows():
        if pd.notna(row['2025']) and pd.notna(row['2020']) and pd.notna(row['2023']):
            price_2025 = float(row['2025'])
            price_2020 = float(row['2020'])
            price_2023 = float(row['2023'])
            
            # Calculate metrics
            growth_5yr = ((price_2025 / price_2020) - 1) * 100
            growth_2yr = ((price_2025 / price_2023) - 1) * 100
            
            # Calculate volatility (coefficient of variation)
            recent_prices = [float(row[str(y)]) for y in range(2020, 2026) if pd.notna(row[str(y)])]
            volatility = (np.std(recent_prices) / np.mean(recent_prices)) * 100 if recent_prices else 0
            
            # Investment score: high growth, low price, moderate volatility
            barcelona_avg = 1111.06  # Barcelona 2025 price
            price_ratio = price_2025 / barcelona_avg
            
            # Score calculation (0-100)
            growth_score = min(growth_5yr / 2, 50)  # Max 50 points for growth
            value_score = max(0, (1 - price_ratio) * 30)  # Max 30 points for being below avg
            stability_score = max(0, 20 - volatility)  # Max 20 points for stability
            
            investment_score = growth_score + value_score + stability_score
            
            opportunities.append({
                'neighborhood': row['Territorio'],
                'price_2025': round(price_2025, 2),
                'growth_5yr_pct': round(growth_5yr, 2),
                'growth_2yr_pct': round(growth_2yr, 2),
                'volatility_pct': round(volatility, 2),
                'vs_barcelona_avg': round((price_ratio - 1) * 100, 2),
                'investment_score': round(investment_score, 2)
            })
    
    # Sort by investment score
    opportunities.sort(key=lambda x: x['investment_score'], reverse=True)
    
    return opportunities[:15]  # Top 15 opportunities

def predict_2026_prices(df: pd.DataFrame) -> Dict[str, Any]:
    """Simple linear regression predictions for 2026"""
    predictions = {}
    
    # Predict for Barcelona city
    bcn = df[df['Territorio'] == 'Barcelona'].iloc[0]
    recent_years = [2020, 2021, 2022, 2023, 2024, 2025]
    recent_prices = [float(bcn[str(y)]) for y in recent_years if pd.notna(bcn[str(y)])]
    
    if len(recent_prices) >= 4:
        # Simple linear regression
        x = np.array(range(len(recent_prices)))
        y = np.array(recent_prices)
        
        # Calculate slope and intercept
        slope = np.polyfit(x, y, 1)[0]
        intercept = np.polyfit(x, y, 1)[1]
        
        # Predict 2026
        prediction_2026 = slope * len(recent_prices) + intercept
        
        predictions['barcelona_2026'] = {
            'predicted_price': round(prediction_2026, 2),
            'confidence': 'medium',
            'trend': 'increasing' if slope > 0 else 'decreasing',
            'annual_change_avg': round(slope, 2)
        }
    
    # Predict for top districts
    districts = df[df['Tipo de territorio'] == 'Districte'].copy()
    district_predictions = []
    
    for _, row in districts.iterrows():
        recent_prices = [float(row[str(y)]) for y in recent_years if pd.notna(row[str(y)])]
        
        if len(recent_prices) >= 4:
            x = np.array(range(len(recent_prices)))
            y = np.array(recent_prices)
            slope = np.polyfit(x, y, 1)[0]
            intercept = np.polyfit(x, y, 1)[1]
            prediction_2026 = slope * len(recent_prices) + intercept
            
            district_predictions.append({
                'district': row['Territorio'],
                'price_2025': float(row['2025']) if pd.notna(row['2025']) else None,
                'predicted_2026': round(prediction_2026, 2),
                'expected_growth_pct': round(((prediction_2026 / float(row['2025'])) - 1) * 100, 2) if pd.notna(row['2025']) else None
            })
    
    district_predictions.sort(key=lambda x: x['expected_growth_pct'] if x['expected_growth_pct'] else 0, reverse=True)
    predictions['districts_2026'] = district_predictions[:10]
    
    return predictions

def generate_insights(df: pd.DataFrame) -> Dict[str, Any]:
    """Generate comprehensive insights report"""
    
    print("🔍 Analyzing Barcelona Housing Data...")
    
    insights = {
        'generated_at': datetime.now().isoformat(),
        'data_source': 'housing_data.csv',
        'analysis_version': '1.0'
    }
    
    # 1. Basic statistics
    print("📊 Calculating statistics...")
    insights['statistics'] = calculate_statistics(df)
    
    # 2. Price trends
    print("📈 Analyzing price trends...")
    insights['trends'] = analyze_price_trends(df)
    
    # 3. Investment opportunities
    print("💰 Finding investment opportunities...")
    insights['investment_opportunities'] = find_investment_opportunities(df)
    
    # 4. 2026 predictions
    print("🔮 Predicting 2026 prices...")
    insights['predictions_2026'] = predict_2026_prices(df)
    
    # 5. Key findings
    insights['key_findings'] = generate_key_findings(df, insights)
    
    return insights

def generate_key_findings(df: pd.DataFrame, insights: Dict) -> List[str]:
    """Generate human-readable key findings"""
    findings = []
    
    # Data quality
    completeness = insights['statistics']['data_quality']['completeness_pct']
    findings.append(f"Dataset contains {insights['statistics']['total_territories']} territories with {completeness}% data completeness")
    
    # Barcelona trend
    bcn_trend = insights['trends']['barcelona_city']
    if bcn_trend['growth_5yr_pct']:
        findings.append(f"Barcelona city prices grew {bcn_trend['growth_5yr_pct']}% in the last 5 years (2020-2025)")
    
    # Top growing district
    if insights['trends']['top_growing_districts']:
        top_district = insights['trends']['top_growing_districts'][0]
        findings.append(f"{top_district['name']} is the fastest-growing district with {top_district['growth_5yr_pct']}% growth")
    
    # Investment opportunity
    if insights['investment_opportunities']:
        top_opp = insights['investment_opportunities'][0]
        findings.append(f"{top_opp['neighborhood']} shows highest investment potential with score {top_opp['investment_score']}")
    
    # 2026 prediction
    if 'barcelona_2026' in insights['predictions_2026']:
        pred = insights['predictions_2026']['barcelona_2026']
        findings.append(f"Barcelona prices predicted to reach €{pred['predicted_price']}/m² in 2026")
    
    return findings

def main():
    """Main analysis execution"""
    print("=" * 60)
    print("BARCELONA HOUSING PRICE ANALYSIS")
    print("=" * 60)
    print()
    
    # Load data
    df = load_and_clean_data('housing_data.csv')
    print(f"✅ Loaded {len(df)} territories\n")
    
    # Generate insights
    insights = generate_insights(df)
    
    # Save to JSON for frontend
    output_file = 'housing_insights.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(insights, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)
    
    print(f"\n✅ Analysis complete! Results saved to {output_file}")
    print("\n" + "=" * 60)
    print("KEY FINDINGS:")
    print("=" * 60)
    for i, finding in enumerate(insights['key_findings'], 1):
        print(f"{i}. {finding}")
    
    print("\n" + "=" * 60)
    print("TOP 5 INVESTMENT OPPORTUNITIES:")
    print("=" * 60)
    for i, opp in enumerate(insights['investment_opportunities'][:5], 1):
        print(f"{i}. {opp['neighborhood']}")
        print(f"   Price: €{opp['price_2025']}/m² | Growth: {opp['growth_5yr_pct']}% | Score: {opp['investment_score']}")
    
    print("\n✨ Data ready for frontend integration!")

if __name__ == '__main__':
    main()
