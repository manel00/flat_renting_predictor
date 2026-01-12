import { Routes } from '@angular/router';
import { DashboardComponent } from './components/dashboard/dashboard.component';
import { Visualization3dComponent } from './components/visualization-3d/visualization-3d.component';
import { PredictionComponent } from './components/prediction/prediction.component';

export const routes: Routes = [
    { path: '', redirectTo: '/dashboard', pathMatch: 'full' },
    { path: 'dashboard', component: DashboardComponent },
    { path: 'visualization', component: Visualization3dComponent },
    { path: 'prediction', component: PredictionComponent },
    { path: '**', redirectTo: '/dashboard' }
];
