import { CommonModule } from '@angular/common';
import { Component, OnInit } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { Router } from '@angular/router';

import { AITriageResult } from '../../models/ticket.model';
import { TicketService } from '../../services/ticket.service';

@Component({
  selector: 'app-ai-validation',
  standalone: true,
  imports: [CommonModule, FormsModule],
  templateUrl: './ai-validation.html',
  styleUrls: ['./ai-validation.css'],
})
export class AiValidation implements OnInit {
  triageResult!: AITriageResult;
  isConfirming = false;
  errorMessage = '';

  constructor(
    private readonly ticketService: TicketService,
    private readonly router: Router,
  ) {}

  ngOnInit(): void {
    this.triageResult = { ...this.ticketService.getCurrentTriageResult() };
  }

  confirmTicket(): void {
    if (this.isConfirming) {
      return;
    }

    this.isConfirming = true;
    this.errorMessage = '';

    this.ticketService.confirmTicket(this.triageResult).subscribe({
      next: () => {
        this.isConfirming = false;
        this.router.navigate(['/it-dashboard']);
      },
      error: () => {
        this.isConfirming = false;
        this.errorMessage =
          'We could not submit the ticket right now. Please try again.';
      },
    });
  }

  cancel(): void {
    this.router.navigate(['/user-input']);
  }
}
