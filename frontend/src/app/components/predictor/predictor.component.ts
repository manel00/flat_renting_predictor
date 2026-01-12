import { Component, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { ApiService } from '../../services/api.service';
import { Territory, PredictionResponse } from '../../models/housing.model';

@Component({
    selector: 'app-predictor',
    standalone: true,
    imports: [CommonModule, FormsModule],
    templateUrl: './predictor.component.html',
    styleUrls: ['./predictor.component.scss']
})
export class PredictorComponent implements OnInit {
    territories: Territory[] = [];
    selectedTerritory: string = '';
    selectedYear: number = 2026;
    prediction: PredictionResponse | null = null;
    loading: boolean = false;
    error: string | null = null;

    minYear: number = 2000;
    maxYear: number = 2035;

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
                this.error = 'Error loading territories: ' + err.message;
                console.error('Error loading territories:', err);
            }
        });
    }

    filterTerritories() {
        if (!this.searchTerm) {
            this.filteredTerritories = this.territories;
            return;
        }

        const search = this.searchTerm.toLowerCase();
        this.filteredTerritories = this.territories.filter(t =>
            t.name.toLowerCase().includes(search)
        );
    }

    onTerritorySearch(event: Event) {
        const input = event.target as HTMLInputElement;
        this.searchTerm = input.value;
        this.filterTerritories();
    }

    predict() {
        if (!this.selectedTerritory) {
            this.error = 'Por favor selecciona un territorio';
            return;
        }

        if (this.selectedYear < this.minYear || this.selectedYear > this.maxYear) {
            this.error = `El año debe estar entre ${this.minYear} y ${this.maxYear}`;
            return;
        }

        this.loading = true;
        this.error = null;
        this.prediction = null;

        this.apiService.predict({
            territory: this.selectedTerritory,
            year: this.selectedYear
        }).subscribe({
            next: (result) => {
                this.prediction = result;
                this.loading = false;
            },
            error: (err) => {
                this.error = 'Error al generar predicción: ' + err.message;
                this.loading = false;
                console.error('Prediction error:', err);
            }
        });
    }

    reset() {
        this.selectedTerritory = '';
        this.selectedYear = 2026;
        this.prediction = null;
        this.error = null;
        this.searchTerm = '';
        this.filteredTerritories = this.territories;
    }
}
