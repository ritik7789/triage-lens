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
  totalTickets = 0;
  highPriority = 0;
  activeCategory = 'All';
  expandedIndex: number | null = null;

  readonly categories = ['All', 'Network', 'Hardware', 'Software'];

  constructor(private readonly ticketService: TicketService) {}

  ngOnInit(): void {
    this.ticketService.getDashboardTickets().subscribe({
      next: (tickets) => {
        this.tickets = tickets;
        this.filteredTickets = tickets;
        this.totalTickets = this.tickets.length;
        this.highPriority = this.tickets.filter(
          (ticket) => ticket.priority === 'High' || ticket.priority === 'Critical',
        ).length;
        this.isLoading = false;
      },
      error: () => {
        this.tickets = [];
        this.filteredTickets = [];
        this.totalTickets = 0;
        this.highPriority = 0;
        this.isLoading = false;
      },
    });
  }

  applyFilter(category: string): void {
    this.activeCategory = category;
    this.expandedIndex = null;

    if (category === 'All') {
      this.filteredTickets = this.tickets;
      return;
    }

    this.filteredTickets = this.tickets.filter(
      (ticket) => ticket.category === category,
    );
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
