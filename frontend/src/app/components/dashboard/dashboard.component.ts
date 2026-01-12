import { Component, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { ApiService } from '../../services/api.service';
import { Stats, HousingData, PredictionResponse } from '../../models/housing.model';
import * as Plotly from 'plotly.js-dist-min';
import { forkJoin } from 'rxjs';

@Component({
    selector: 'app-dashboard',
    standalone: true,
    imports: [CommonModule],
    templateUrl: './dashboard.component.html',
    styleUrls: ['./dashboard.component.scss']
})
export class DashboardComponent implements OnInit {
    stats: Stats | null = null;
    loading = true;
    error: string | null = null;
    loadingPredictions = false;
    predictionsLoaded = false;
    modelMetrics: any = null;
    private neighborhoodsData: HousingData[] = [];

    constructor(private apiService: ApiService) { }

    ngOnInit() {
        this.loadStats();
        this.loadTerritoryComparison();
        this.loadModelMetrics();
    }

    loadModelMetrics() {
        this.apiService.getModelMetrics().subscribe({
            next: (metrics) => {
                this.modelMetrics = metrics;
                console.log('Model metrics loaded:', metrics);
            },
            error: (err) => {
                console.error('Error loading model metrics:', err);
            }
        });
    }

    loadStats() {
        this.loading = true;
        this.error = null;

        this.apiService.getStats().subscribe({
            next: (stats) => {
                this.stats = stats;
                this.loading = false;
            },
            error: (err) => {
                this.error = 'Error loading statistics: ' + err.message;
                this.loading = false;
                console.error('Error loading stats:', err);
            }
        });
    }


    loadTerritoryComparison() {
        this.loading = true;
        forkJoin({
            districts: this.apiService.getDataByType('Districte'),
            neighborhoods: this.apiService.getDataByType('Barri')
        }).subscribe({
            next: ({ districts, neighborhoods }) => {
                this.loading = false;
                this.neighborhoodsData = neighborhoods;
                setTimeout(() => {
                    this.createDistrictsChart(districts);
                    this.createNeighborhoodsChart(neighborhoods, false);
                }, 100);
            },
            error: (err) => {
                this.loading = false;
                this.error = 'Error loading charts: ' + err.message;
                console.error('Error loading territory comparison:', err);
            }
        });
    }

    loadPredictions() {
        if (this.loadingPredictions || this.predictionsLoaded) return;

        this.loadingPredictions = true;
        this.createNeighborhoodsChart(this.neighborhoodsData, true);
    }

    createDistrictsChart(data: HousingData[]) {
        const plotElement = document.getElementById('districtsChart');
        if (!plotElement) return;

        // Group by territory
        const territoryData: { [key: string]: { years: number[], prices: number[] } } = {};
        data.forEach(item => {
            if (!territoryData[item.territory]) {
                territoryData[item.territory] = { years: [], prices: [] };
            }
            territoryData[item.territory].years.push(item.year);
            territoryData[item.territory].prices.push(item.price);
        });

        // Create traces for each district
        const traces = Object.entries(territoryData).map(([territory, values]) => ({
            x: values.years,
            y: values.prices,
            type: 'scatter',
            mode: 'lines+markers',
            name: territory,
            line: { width: 2 },
            marker: { size: 4 }
        }));

        const layout: any = {
            title: {
                text: 'Evolución de Precios por Distrito',
                font: { size: 18, color: '#ffffff', family: 'Inter, sans-serif' }
            },
            xaxis: {
                title: 'Año',
                titlefont: { color: '#e0e0e0' },
                tickfont: { color: '#e0e0e0' },
                gridcolor: '#444444'
            },
            yaxis: {
                title: 'Precio (€)',
                titlefont: { color: '#e0e0e0' },
                tickfont: { color: '#e0e0e0' },
                gridcolor: '#444444'
            },
            paper_bgcolor: '#0a0a0a',
            plot_bgcolor: '#0a0a0a',
            legend: {
                font: { color: '#e0e0e0' },
                bgcolor: 'rgba(0,0,0,0.5)'
            },
            margin: { l: 60, r: 40, t: 60, b: 60 },
            height: 500
        };

        const config = {
            responsive: true,
            displayModeBar: true,
            displaylogo: false
        };

        Plotly.newPlot('districtsChart', traces, layout, config);
    }

    createNeighborhoodsChart(data: HousingData[], withPredictions: boolean) {
        const plotElement = document.getElementById('neighborhoodsChart');
        if (!plotElement) return;

        // Group by territory
        const territoryData: { [key: string]: { years: number[], prices: number[] } } = {};
        data.forEach(item => {
            if (!territoryData[item.territory]) {
                territoryData[item.territory] = { years: [], prices: [] };
            }
            territoryData[item.territory].years.push(item.year);
            territoryData[item.territory].prices.push(item.price);
        });

        // Show all historical data
        const historicalTraces = Object.entries(territoryData).map(([territory, values]) => ({
            x: values.years,
            y: values.prices,
            type: 'scatter',
            mode: 'lines+markers',
            name: territory,
            line: { width: 2 },
            marker: { size: 4 }
        }));

        if (!withPredictions) {
            const layout: any = {
                title: {
                    text: 'Evolución de Precios por Barrio',
                    font: { size: 18, color: '#ffffff', family: 'Inter, sans-serif' }
                },
                xaxis: {
                    title: 'Año',
                    titlefont: { color: '#e0e0e0' },
                    tickfont: { color: '#e0e0e0' },
                    gridcolor: '#444444'
                },
                yaxis: {
                    title: 'Precio (€)',
                    titlefont: { color: '#e0e0e0' },
                    tickfont: { color: '#e0e0e0' },
                    gridcolor: '#444444'
                },
                paper_bgcolor: '#0a0a0a',
                plot_bgcolor: '#0a0a0a',
                legend: {
                    font: { color: '#e0e0e0' },
                    bgcolor: 'rgba(0,0,0,0.5)'
                },
                margin: { l: 60, r: 40, t: 60, b: 60 },
                height: 500
            };

            const config = {
                responsive: true,
                displayModeBar: true,
                displaylogo: false
            };

            Plotly.newPlot('neighborhoodsChart', historicalTraces, layout, config);
            return;
        }

        // Get all neighborhoods for predictions
        const neighborhoods = Object.keys(territoryData);
        const predictionYears = [2026, 2027, 2028, 2029, 2030];

        // Request predictions
        this.apiService.bulkPredict(neighborhoods, predictionYears).subscribe({
            next: (predictions: PredictionResponse[]) => {
                this.loadingPredictions = false;
                this.predictionsLoaded = true;

                // Group predictions by territory
                const predictionsByTerritory: { [key: string]: { years: number[], prices: number[] } } = {};
                predictions.forEach(pred => {
                    if (!predictionsByTerritory[pred.territory]) {
                        predictionsByTerritory[pred.territory] = { years: [], prices: [] };
                    }
                    predictionsByTerritory[pred.territory].years.push(pred.year);
                    predictionsByTerritory[pred.territory].prices.push(pred.predicted_price);
                });

                // Create prediction traces that connect with historical data
                const predictionTraces = Object.entries(predictionsByTerritory).map(([territory, predValues]) => {
                    // Get the last historical point for this territory
                    const historicalData = territoryData[territory];
                    const lastYear = historicalData.years[historicalData.years.length - 1];
                    const lastPrice = historicalData.prices[historicalData.prices.length - 1];

                    // Prepend the last historical point to create continuity
                    return {
                        x: [lastYear, ...predValues.years],
                        y: [lastPrice, ...predValues.prices],
                        type: 'scatter',
                        mode: 'lines+markers',
                        name: `${territory} (predicción)`,
                        line: { width: 2, dash: 'dash' },
                        marker: { size: 6, symbol: 'diamond' },
                        showlegend: false,
                        connectgaps: true
                    };
                });

                const layout: any = {
                    title: {
                        text: 'Evolución y Predicción de Precios por Barrio',
                        font: { size: 18, color: '#ffffff', family: 'Inter, sans-serif' }
                    },
                    xaxis: {
                        title: 'Año',
                        titlefont: { color: '#e0e0e0' },
                        tickfont: { color: '#e0e0e0' },
                        gridcolor: '#444444'
                    },
                    yaxis: {
                        title: 'Precio (€)',
                        titlefont: { color: '#e0e0e0' },
                        tickfont: { color: '#e0e0e0' },
                        gridcolor: '#444444'
                    },
                    paper_bgcolor: '#0a0a0a',
                    plot_bgcolor: '#0a0a0a',
                    legend: {
                        font: { color: '#e0e0e0' },
                        bgcolor: 'rgba(0,0,0,0.5)'
                    },
                    margin: { l: 60, r: 40, t: 60, b: 60 },
                    height: 500,
                    annotations: [{
                        x: 2025.5,
                        y: 0,
                        xref: 'x',
                        yref: 'paper',
                        text: 'Predicciones →',
                        showarrow: false,
                        font: { color: '#e91e63', size: 14 },
                        xanchor: 'center'
                    }]
                };

                const config = {
                    responsive: true,
                    displayModeBar: true,
                    displaylogo: false
                };

                Plotly.newPlot('neighborhoodsChart', [...historicalTraces, ...predictionTraces], layout, config);
            },
            error: (err) => {
                console.error('Error loading predictions:', err);
                // Still show historical data even if predictions fail
                const traces = Object.entries(territoryData).map(([territory, values]) => ({
                    x: values.years,
                    y: values.prices,
                    type: 'scatter',
                    mode: 'lines+markers',
                    name: territory,
                    line: { width: 2 },
                    marker: { size: 4 }
                }));

                const layout: any = {
                    title: {
                        text: 'Evolución de Precios por Barrio',
                        font: { size: 18, color: '#ffffff', family: 'Inter, sans-serif' }
                    },
                    xaxis: {
                        title: 'Año',
                        titlefont: { color: '#e0e0e0' },
                        tickfont: { color: '#e0e0e0' },
                        gridcolor: '#444444'
                    },
                    yaxis: {
                        title: 'Precio (€)',
                        titlefont: { color: '#e0e0e0' },
                        tickfont: { color: '#e0e0e0' },
                        gridcolor: '#444444'
                    },
                    paper_bgcolor: '#0a0a0a',
                    plot_bgcolor: '#0a0a0a',
                    legend: {
                        font: { color: '#e0e0e0' },
                        bgcolor: 'rgba(0,0,0,0.5)'
                    },
                    margin: { l: 60, r: 40, t: 60, b: 60 },
                    height: 500
                };

                const config = {
                    responsive: true,
                    displayModeBar: true,
                    displaylogo: false
                };

                Plotly.newPlot('neighborhoodsChart', traces, layout, config);
            }
        });
    }
}
