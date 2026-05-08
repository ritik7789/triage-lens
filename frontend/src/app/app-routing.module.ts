import { NgModule } from '@angular/core';
import { RouterModule, Routes } from '@angular/router';

import { AiValidation } from './components/ai-validation/ai-validation';
import { ItDashboard } from './components/it-dashboard/it-dashboard';
import { UserInput } from './components/user-input/user-input';

export const appRoutes: Routes = [
  { path: '', pathMatch: 'full', redirectTo: 'user-input' },
  { path: 'user-input', component: UserInput },
  { path: 'ai-validation', component: AiValidation },
  { path: 'it-dashboard', component: ItDashboard },
  { path: '**', redirectTo: 'user-input' },
];

@NgModule({
  imports: [RouterModule.forRoot(appRoutes)],
  exports: [RouterModule],
})
export class AppRoutingModule {}
