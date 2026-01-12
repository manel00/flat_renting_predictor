import { Component, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { ApiService } from '../../services/api.service';
import { ThreeDData } from '../../models/housing.model';
import * as Plotly from 'plotly.js-dist-min';

@Component({
    selector: 'app-visualization-3d',
    standalone: true,
    imports: [CommonModule],
    templateUrl: './visualization-3d.component.html',
    styleUrls: ['./visualization-3d.component.scss']
})
export class Visualization3dComponent implements OnInit {
    loading = true;
    error: string | null = null;
    currentView: 'surface' | 'scatter' = 'surface';
    private cachedData: ThreeDData | null = null;

    constructor(private apiService: ApiService) { }

    ngOnInit() {
        this.load3DData();
    }

    load3DData() {
        this.loading = true;
        this.error = null;

        this.apiService.get3DData().subscribe({
            next: (data) => {
                // Filter only Barri data
                const barriIndices: number[] = [];
                data.territory_types.forEach((type, index) => {
                    if (type === 'Barri') {
                        barriIndices.push(index);
                    }
                });

                const filteredData: ThreeDData = {
                    territories: barriIndices.map(i => data.territories[i]),
                    years: barriIndices.map(i => data.years[i]),
                    prices: barriIndices.map(i => data.prices[i]),
                    territory_types: barriIndices.map(i => data.territory_types[i])
                };

                this.cachedData = filteredData;
                this.loading = false;
                // Wait for Angular to render the DOM element
                setTimeout(() => {
                    this.createVisualization(filteredData);
                }, 100);
            },
            error: (err) => {
                this.error = 'Error loading data: ' + err.message;
                this.loading = false;
                console.error('Error loading 3D data:', err);
            }
        });
    }

    createVisualization(data: ThreeDData) {
        const plotElement = document.getElementById('plot3d');
        if (!plotElement) {
            console.error('Plot element not found in DOM');
            return;
        }

        if (this.currentView === 'surface') {
            this.createSurfacePlot(data);
        } else if (this.currentView === 'scatter') {
            this.createScatterPlot(data);
        }
    }

    createSurfacePlot(data: ThreeDData) {
        // Prepare data for surface plot
        const uniqueTerritories = [...new Set(data.territories)].slice(0, 20); // Top 20 territories
        const uniqueYears = [...new Set(data.years)].sort();

        // Create a matrix for the surface plot
        const zMatrix: number[][] = [];
        const territoryIndices: { [key: string]: number } = {};

        uniqueTerritories.forEach((territory, idx) => {
            territoryIndices[territory] = idx;
            zMatrix[idx] = new Array(uniqueYears.length).fill(null);
        });

        // Fill the matrix
        data.territories.forEach((territory, idx) => {
            if (uniqueTerritories.includes(territory)) {
                const territoryIdx = territoryIndices[territory];
                const yearIdx = uniqueYears.indexOf(data.years[idx]);
                if (yearIdx !== -1) {
                    zMatrix[territoryIdx][yearIdx] = data.prices[idx];
                }
            }
        });

        const surfaceData: any = {
            type: 'surface',
            x: uniqueYears,
            y: uniqueTerritories.map((_, idx) => idx),
            z: zMatrix,
            colorscale: [
                [0, '#1a0033'],
                [0.2, '#4a148c'],
                [0.4, '#7b1fa2'],
                [0.6, '#9c27b0'],
                [0.8, '#e91e63'],
                [1, '#ff6090']
            ],
            colorbar: {
                title: 'Precio (€)',
                titlefont: { color: '#e0e0e0' },
                tickfont: { color: '#e0e0e0' }
            },
            hovertemplate: 'Año: %{x}<br>Territorio: %{y}<br>Precio: €%{z:.2f}<extra></extra>'
        };

        const layout: any = {
            title: {
                text: 'Evolución de Precios de Alquiler en Barcelona (3D)',
                font: { size: 24, color: '#ffffff', family: 'Inter, sans-serif' }
            },
            scene: {
                xaxis: {
                    title: 'Año',
                    titlefont: { color: '#e0e0e0' },
                    tickfont: { color: '#e0e0e0' },
                    gridcolor: '#444444'
                },
                yaxis: {
                    title: 'Territorio',
                    titlefont: { color: '#e0e0e0' },
                    tickfont: { color: '#e0e0e0' },
                    ticktext: uniqueTerritories.map(t => t.length > 20 ? t.substring(0, 20) + '...' : t),
                    tickvals: uniqueTerritories.map((_, idx) => idx),
                    gridcolor: '#444444'
                },
                zaxis: {
                    title: 'Precio (€)',
                    titlefont: { color: '#e0e0e0' },
                    tickfont: { color: '#e0e0e0' },
                    gridcolor: '#444444'
                },
                bgcolor: '#0a0a0a',
                camera: {
                    eye: { x: 1.5, y: 1.5, z: 1.3 }
                }
            },
            paper_bgcolor: '#0a0a0a',
            plot_bgcolor: '#0a0a0a',
            margin: { l: 0, r: 0, t: 50, b: 0 },
            height: 700
        };

        const config = {
            responsive: true,
            displayModeBar: true,
            displaylogo: false,
            modeBarButtonsToRemove: ['toImage']
        };

        Plotly.newPlot('plot3d', [surfaceData], layout, config);
    }

    createScatterPlot(data: ThreeDData) {
        // Create 3D scatter plot with color by territory type
        const territoryTypes = [...new Set(data.territory_types)];
        const colors = ['#e91e63', '#9c27b0', '#673ab7', '#3f51b5', '#2196f3', '#00bcd4'];

        const traces = territoryTypes.map((type, idx) => {
            const indices = data.territory_types
                .map((t, i) => t === type ? i : -1)
                .filter(i => i !== -1);

            return {
                type: 'scatter3d',
                mode: 'markers',
                name: type,
                x: indices.map(i => data.years[i]),
                y: indices.map(i => data.territories[i]),
                z: indices.map(i => data.prices[i]),
                marker: {
                    size: 4,
                    color: colors[idx % colors.length],
                    opacity: 0.8,
                    line: {
                        color: '#ffffff',
                        width: 0.5
                    }
                },
                hovertemplate: '<b>%{y}</b><br>Año: %{x}<br>Precio: €%{z:.2f}<extra></extra>'
            };
        });

        const layout: any = {
            title: {
                text: 'Distribución de Precios por Tipo de Territorio',
                font: { size: 24, color: '#ffffff', family: 'Inter, sans-serif' }
            },
            scene: {
                xaxis: {
                    title: 'Año',
                    titlefont: { color: '#e0e0e0' },
                    tickfont: { color: '#e0e0e0' },
                    gridcolor: '#444444'
                },
                yaxis: {
                    title: 'Territorio',
                    titlefont: { color: '#e0e0e0' },
                    tickfont: { color: '#e0e0e0' },
                    gridcolor: '#444444'
                },
                zaxis: {
                    title: 'Precio (€)',
                    titlefont: { color: '#e0e0e0' },
                    tickfont: { color: '#e0e0e0' },
                    gridcolor: '#444444'
                },
                bgcolor: '#0a0a0a',
                camera: {
                    eye: { x: 1.8, y: 1.8, z: 1.3 }
                }
            },
            paper_bgcolor: '#0a0a0a',
            plot_bgcolor: '#0a0a0a',
            legend: {
                font: { color: '#e0e0e0' },
                bgcolor: 'rgba(0,0,0,0.5)'
            },
            margin: { l: 0, r: 0, t: 50, b: 0 },
            height: 700
        };

        const config = {
            responsive: true,
            displayModeBar: true,
            displaylogo: false
        };

        Plotly.newPlot('plot3d', traces, layout, config);
    }

    createTerritoryComparison(data: ThreeDData) {
        // Show top territories by average price
        const territoryPrices: { [key: string]: number[] } = {};

        data.territories.forEach((territory, idx) => {
            if (!territoryPrices[territory]) {
                territoryPrices[territory] = [];
            }
            territoryPrices[territory].push(data.prices[idx]);
        });

        const territoryAverages = Object.entries(territoryPrices)
            .map(([territory, prices]) => ({
                territory,
                avgPrice: prices.reduce((a, b) => a + b, 0) / prices.length
            }))
            .sort((a, b) => b.avgPrice - a.avgPrice)
            .slice(0, 15);

        const trace: any = {
            type: 'bar',
            x: territoryAverages.map(t => t.territory),
            y: territoryAverages.map(t => t.avgPrice),
            marker: {
                color: territoryAverages.map((_, idx) => {
                    const ratio = idx / territoryAverages.length;
                    return `rgba(233, 30, 99, ${1 - ratio * 0.5})`;
                }),
                line: {
                    color: '#ffffff',
                    width: 1
                }
            },
            hovertemplate: '<b>%{x}</b><br>Precio Promedio: €%{y:.2f}<extra></extra>'
        };

        const layout: any = {
            title: {
                text: 'Top 15 Territorios por Precio Promedio',
                font: { size: 24, color: '#ffffff', family: 'Inter, sans-serif' }
            },
            xaxis: {
                title: 'Territorio',
                titlefont: { color: '#e0e0e0' },
                tickfont: { color: '#e0e0e0', size: 10 },
                tickangle: -45,
                gridcolor: '#444444'
            },
            yaxis: {
                title: 'Precio Promedio (€)',
                titlefont: { color: '#e0e0e0' },
                tickfont: { color: '#e0e0e0' },
                gridcolor: '#444444'
            },
            paper_bgcolor: '#0a0a0a',
            plot_bgcolor: '#0a0a0a',
            margin: { l: 80, r: 40, t: 80, b: 150 },
            height: 600
        };

        const config = {
            responsive: true,
            displayModeBar: true,
            displaylogo: false
        };

        Plotly.newPlot('plot3d', [trace], layout, config);
    }

    switchView(view: 'surface' | 'scatter') {
        this.currentView = view;
        if (this.cachedData) {
            // Use cached data instead of reloading
            setTimeout(() => {
                this.createVisualization(this.cachedData!);
            }, 100);
        } else {
            this.load3DData();
        }
    }
}
