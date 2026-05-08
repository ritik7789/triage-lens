import { HttpClient } from '@angular/common/http';
import { Injectable, inject } from '@angular/core';
import { Observable, delay, of, tap } from 'rxjs';

import { AITriageResult, RawTicket } from '../models/ticket.model';

@Injectable({
  providedIn: 'root',
})
export class TicketService {
  private readonly http = inject(HttpClient);

  private readonly mockTriageResult: AITriageResult = {
    category: 'Network',
    cleanDescription:
      'User cannot connect to the corporate VPN and cannot access internal business applications.',
    label: 'VPN Connectivity Issue',
    priority: 'High',
    explanation:
      'The AI classified this as a network issue because the user reported VPN connectivity failure and loss of access to internal systems. It was marked High priority because the issue blocks core work and may affect time-sensitive operations.',
  };

  private readonly mockDashboardTickets: AITriageResult[] = [
    this.mockTriageResult,
    {
      category: 'Hardware',
      cleanDescription:
        'Laptop camera is not detected by video conferencing applications after a driver update.',
      label: 'Device Support',
      priority: 'Medium',
      explanation:
        'The AI classified this as hardware because the affected capability is a laptop camera. It was marked Medium because meetings are impacted, but the user can continue most other work.',
    },
    {
      category: 'Software',
      cleanDescription:
        'Finance reporting dashboard shows a permissions error for a single employee.',
      label: 'Application Access',
      priority: 'Low',
      explanation:
        'The AI classified this as software access because the problem is isolated to one application permission state. It was marked Low because there is no indication of a system-wide outage.',
    },
  ];

  private currentTriageResult: AITriageResult | null = null;

  submitForTriage(ticket: RawTicket): Observable<AITriageResult> {
    const description = ticket.description.trim();
    const result: AITriageResult = {
      ...this.mockTriageResult,
      cleanDescription: description
        ? `User reports: ${description}`
        : this.mockTriageResult.cleanDescription,
    };

    return of(result).pipe(
      delay(1000),
      tap((triageResult) => {
        this.currentTriageResult = triageResult;
      }),
    );
  }

  confirmTicket(result: AITriageResult): Observable<boolean> {
    this.currentTriageResult = result;

    return of(true).pipe(delay(1000));
  }

  getDashboardTickets(): Observable<AITriageResult[]> {
    const tickets = this.currentTriageResult
      ? [this.currentTriageResult, ...this.mockDashboardTickets.slice(1)]
      : this.mockDashboardTickets;

    return of(tickets).pipe(delay(1000));
  }

  getCurrentTriageResult(): AITriageResult {
    return this.currentTriageResult ?? this.mockTriageResult;
  }
}
