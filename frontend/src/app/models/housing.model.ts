export interface HousingData {
    territory: string;
    territory_type: string;
    year: number;
    price: number;
}

export interface Territory {
    name: string;
    type: string;
}

export interface Stats {
    overall: {
        mean_price: number;
        median_price: number;
        min_price: number;
        max_price: number;
        std_price: number;
    };
    by_year: { [key: string]: { mean_price: number; count: number } };
    by_territory_type: { [key: string]: { mean_price: number; count: number } };
    trends: {
        recent_mean: number;
        historical_mean: number;
        growth_rate_percent: number;
    };
}

export interface PredictionRequest {
    territory: string;
    year: number;
}

export interface PredictionResponse {
    territory: string;
    year: number;
    predicted_price: number;
    confidence_interval: {
        lower: number;
        upper: number;
    };
    std: number;
}

export interface ThreeDData {
    territories: string[];
    years: number[];
    prices: number[];
    territory_types: string[];
}
