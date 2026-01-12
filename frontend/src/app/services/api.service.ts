import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';
import { map } from 'rxjs/operators';
import {
    HousingData,
    Territory,
    Stats,
    PredictionRequest,
    PredictionResponse,
    ThreeDData
} from '../models/housing.model';
import { environment } from '../../environments/environment';

@Injectable({
    providedIn: 'root'
})
export class ApiService {
    private baseUrl = environment.apiUrl;

    constructor(private http: HttpClient) { }

    getData(): Observable<HousingData[]> {
        return this.http.get<{ success: boolean; data: HousingData[] }>(`${this.baseUrl}/data`)
            .pipe(map(response => response.data));
    }

    get3DData(): Observable<ThreeDData> {
        return this.http.get<{ success: boolean; data: ThreeDData }>(`${this.baseUrl}/3d-data`)
            .pipe(map(response => response.data));
    }

    getTerritories(): Observable<Territory[]> {
        return this.http.get<{ success: boolean; territories: Territory[] }>(`${this.baseUrl}/territories`)
            .pipe(map(response => response.territories));
    }

    getStats(): Observable<Stats> {
        return this.http.get<{ success: boolean; stats: Stats }>(`${this.baseUrl}/stats`)
            .pipe(map(response => response.stats));
    }

    getTerritoryData(territoryName: string): Observable<HousingData[]> {
        return this.http.get<{ success: boolean; data: HousingData[] }>(`${this.baseUrl}/territory/${territoryName}`)
            .pipe(map(response => response.data));
    }

    predict(request: PredictionRequest): Observable<PredictionResponse> {
        return this.http.post<PredictionResponse>(`${this.baseUrl}/predict`, request);
    }

    getFeatureImportance(): Observable<any[]> {
        return this.http.get<{ success: boolean; features: any[] }>(`${this.baseUrl}/feature-importance`)
            .pipe(map(response => response.features));
    }

    getDataByType(territoryType: string): Observable<HousingData[]> {
        return this.http.get<{ success: boolean; data: HousingData[] }>(`${this.baseUrl}/data-by-type/${territoryType}`)
            .pipe(map(response => response.data));
    }

    bulkPredict(territories: string[], years: number[]): Observable<PredictionResponse[]> {
        return this.http.post<{ success: boolean; predictions: PredictionResponse[] }>(
            `${this.baseUrl}/bulk-predict`,
            { territories, years }
        ).pipe(map(response => response.predictions));
    }

    getModelMetrics(): Observable<any> {
        return this.http.get<{ success: boolean; metrics: any }>(`${this.baseUrl}/model/metrics`)
            .pipe(map(response => response.metrics));
    }
}
