import { HttpClient } from '@angular/common/http';
import { Injectable, inject } from '@angular/core';
import { Observable, delay, map, of, tap } from 'rxjs';

import {
  AITriageResult,
  FinalTriageApiRequest,
  FinalTriageApiResponse,
  LegacyTriageApiResponse,
  QueriesApiResponse,
  QueryRecordApiResponse,
  RawTicket,
} from '../models/ticket.model';

@Injectable({
  providedIn: 'root',
})
export class TicketService {
  private readonly http = inject(HttpClient);
  private readonly legacyTriageApiUrl = 'http://localhost:8000/api/v1/triage';
  private readonly finalTriageApiUrl = 'http://localhost:8000/triage';
  private readonly queriesApiUrl = 'http://localhost:8000/queries';

  private readonly mockTriageResult: AITriageResult = {
    rawQuery:
      'User cannot connect to the corporate VPN and cannot access internal business applications.',
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
      rawQuery:
        'Laptop camera is not detected by video conferencing applications after a driver update.',
      category: 'Hardware',
      cleanDescription:
        'Laptop camera is not detected by video conferencing applications after a driver update.',
      label: 'Device Support',
      priority: 'Medium',
      explanation:
        'The AI classified this as hardware because the affected capability is a laptop camera. It was marked Medium because meetings are impacted, but the user can continue most other work.',
    },
    {
      rawQuery:
        'Finance reporting dashboard shows a permissions error for a single employee.',
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
    return this.http
      .post<LegacyTriageApiResponse>(this.legacyTriageApiUrl, ticket)
      .pipe(
        map((response) => this.mapLegacyTriageResponse(ticket, response)),
        tap((triageResult) => {
          this.currentTriageResult = triageResult;
        }),
      );
  }

  confirmTicket(
    result: AITriageResult,
    userRole: string,
  ): Observable<AITriageResult> {
    const payload: FinalTriageApiRequest = {
      raw_query: result.rawQuery,
      nlp_cleaned_query: result.explanation,
      user_role: userRole,
    };

    return this.http
      .post<FinalTriageApiResponse>(this.finalTriageApiUrl, payload)
      .pipe(
        map((response) => this.mapFinalTriageResponse(result, response)),
        tap((triageResult) => {
          this.currentTriageResult = triageResult;
        }),
      );
  }

  getDashboardTickets(): Observable<AITriageResult[]> {
    const tickets = this.currentTriageResult
      ? [this.currentTriageResult, ...this.mockDashboardTickets.slice(1)]
      : this.mockDashboardTickets;

    return of(tickets).pipe(delay(1000));
  }

  getStoredQueries(): Observable<AITriageResult[]> {
    return this.http.get<QueriesApiResponse>(this.queriesApiUrl).pipe(
      map((response) =>
        response.queries.map((queryRecord) =>
          this.mapQueryRecordResponse(queryRecord),
        ),
      ),
      tap((tickets) => {
        this.currentTriageResult = tickets[0] ?? this.currentTriageResult;
      }),
    );
  }

  getCurrentTriageResult(): AITriageResult {
    return this.currentTriageResult ?? this.mockTriageResult;
  }

  private mapLegacyTriageResponse(
    ticket: RawTicket,
    response: LegacyTriageApiResponse,
  ): AITriageResult {
    return {
      rawQuery: ticket.description.trim() || ticket.subject,
      category: response.category,
      cleanDescription: ticket.description.trim() || ticket.subject,
      label: response.assigned_team,
      priority: this.mapPriority(response.priority),
      explanation: response.explanation,
    };
  }

  private mapFinalTriageResponse(
    result: AITriageResult,
    response: FinalTriageApiResponse,
  ): AITriageResult {
    return {
      ...result,
      category: response.category,
      label: response.assigned_team,
      priority: this.mapPriority(response.severity_level),
      explanation: response.reasoning,
    };
  }

  private mapQueryRecordResponse(
    queryRecord: QueryRecordApiResponse,
  ): AITriageResult {
    return {
      rawQuery: queryRecord.request.raw_query,
      category: queryRecord.triage_result.category,
      cleanDescription:
        queryRecord.request.nlp_cleaned_query || queryRecord.request.raw_query,
      label: queryRecord.triage_result.assigned_team,
      priority: this.mapPriority(queryRecord.triage_result.severity_level),
      explanation: queryRecord.triage_result.reasoning,
    };
  }

  private mapPriority(
    priority:
      | LegacyTriageApiResponse['priority']
      | FinalTriageApiResponse['severity_level'],
  ): AITriageResult['priority'] {
    const priorityMap: Record<
      LegacyTriageApiResponse['priority'] | FinalTriageApiResponse['severity_level'],
      AITriageResult['priority']
    > = {
      P1: 'Critical',
      P2: 'High',
      P3: 'Medium',
      P4: 'Low',
    };

    return priorityMap[priority];
  }
}
