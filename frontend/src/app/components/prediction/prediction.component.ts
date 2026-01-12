import { Component, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { ApiService } from '../../services/api.service';
import { Territory, PredictionResponse } from '../../models/housing.model';

@Component({
    selector: 'app-prediction',
    standalone: true,
    imports: [CommonModule, FormsModule],
    templateUrl: './prediction.component.html',
    styleUrls: ['./prediction.component.scss']
})
export class PredictionComponent implements OnInit {
    territories: Territory[] = [];
    selectedTerritory: string = '';
    selectedYear: number = new Date().getFullYear() + 1;
    prediction: PredictionResponse | null = null;
    loading = false;
    error: string | null = null;
    filteredTerritories: Territory[] = [];
    searchTerm: string = '';

    constructor(private apiService: ApiService) { }

    ngOnInit() {
        this.loadTerritories();
    }

    loadTerritories() {
        this.apiService.getTerritories().subscribe({
            next: (territories) => {
                this.territories = territories;
                this.filteredTerritories = territories;
            },
            error: (err) => {
                console.error('Error loading territories:', err);
                this.error = 'Error cargando territorios';
            }
        });
    }

    filterTerritories() {
        if (!this.searchTerm) {
            this.filteredTerritories = this.territories;
        } else {
            this.filteredTerritories = this.territories.filter(t =>
                t.name.toLowerCase().includes(this.searchTerm.toLowerCase())
            );
        }
    }

    selectTerritory(territory: Territory) {
        this.selectedTerritory = territory.name;
        this.searchTerm = territory.name;
        this.filteredTerritories = [];
    }

    predict() {
        if (!this.selectedTerritory || !this.selectedYear) {
            this.error = 'Por favor selecciona un territorio y un año';
            return;
        }

        this.loading = true;
        this.error = null;
        this.prediction = null;

        this.apiService.predict({
            territory: this.selectedTerritory,
            year: this.selectedYear
        }).subscribe({
            next: (prediction) => {
                this.loading = false;
                this.prediction = prediction;
            },
            error: (err) => {
                this.loading = false;
                this.error = 'Error al realizar la predicción: ' + err.message;
                console.error('Prediction error:', err);
            }
        });
    }

    getConfidenceWidth(): number {
        if (!this.prediction) return 0;
        const range = this.prediction.confidence_interval.upper - this.prediction.confidence_interval.lower;
        return (this.prediction.std / range) * 100;
    }
}
