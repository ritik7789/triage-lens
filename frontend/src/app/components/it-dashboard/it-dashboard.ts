import { CommonModule } from '@angular/common';
import { Component, OnInit } from '@angular/core';

import { AITriageResult } from '../../models/ticket.model';
import { TicketService } from '../../services/ticket.service';

@Component({
  selector: 'app-it-dashboard',
  standalone: true,
  imports: [CommonModule],
  templateUrl: './it-dashboard.html',
  styleUrls: ['./it-dashboard.css'],
})
export class ItDashboard implements OnInit {
  tickets: AITriageResult[] = [];
  filteredTickets: AITriageResult[] = [];
  isLoading = true;
  errorMessage = '';
  totalTickets = 0;
  highPriority = 0;
  activeCategory = 'All';
  expandedIndex: number | null = null;

  categories: string[] = ['All'];

  constructor(private readonly ticketService: TicketService) {}

  ngOnInit(): void {
    this.loadStoredQueries();
  }

  applyFilter(category: string): void {
    this.activeCategory = category;
    this.expandedIndex = null;

    if (category === 'All') {
      this.loadStoredQueries();
      return;
    }

    this.filteredTickets = this.tickets.filter(
      (ticket) => ticket.category === category,
    );
  }

  formatCategory(category: string): string {
    return category.replace(/_/g, ' ');
  }

  private loadStoredQueries(): void {
    this.isLoading = true;
    this.errorMessage = '';

    this.ticketService.getStoredQueries().subscribe({
      next: (tickets) => {
        this.setTickets(tickets);
        this.isLoading = false;
      },
      error: () => {
        this.setTickets([]);
        this.errorMessage =
          'We could not load stored queries right now. Please try again.';
        this.isLoading = false;
      },
    });
  }

  private setTickets(tickets: AITriageResult[]): void {
    this.tickets = tickets;
    this.filteredTickets = tickets;
    this.totalTickets = tickets.length;
    this.highPriority = tickets.filter(
      (ticket) => ticket.priority === 'High' || ticket.priority === 'Critical',
    ).length;
    this.categories = [
      'All',
      ...new Set(tickets.map((ticket) => ticket.category)),
    ];
  }

  toggleTicket(index: number): void {
    this.expandedIndex = this.expandedIndex === index ? null : index;
  }

  priorityBadgeClass(priority: AITriageResult['priority']): string {
    const classes: Record<AITriageResult['priority'], string> = {
      Critical: 'bg-rose-100 text-rose-700',
      High: 'bg-rose-100 text-rose-700',
      Medium: 'bg-amber-100 text-amber-700',
      Low: 'bg-emerald-100 text-emerald-700',
    };

    return classes[priority];
  }

  labelBadgeClass(priority: AITriageResult['priority']): string {
    const classes: Record<AITriageResult['priority'], string> = {
      Critical: 'bg-rose-50 text-rose-700 border-rose-100',
      High: 'bg-indigo-50 text-indigo-700 border-indigo-100',
      Medium: 'bg-slate-50 text-slate-600 border-slate-200',
      Low: 'bg-emerald-50 text-emerald-700 border-emerald-100',
    };

    return classes[priority];
  }
}
