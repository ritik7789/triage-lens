import { CommonModule } from '@angular/common';
import { Component } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { Router } from '@angular/router';

import { RawTicket } from '../../models/ticket.model';
import { TicketService } from '../../services/ticket.service';

@Component({
  selector: 'app-user-input',
  standalone: true,
  imports: [CommonModule, FormsModule],
  templateUrl: './user-input.html',
  styleUrls: ['./user-input.css'],
})
export class UserInput {
  ticketDescription = '';
  isSubmitting = false;
  errorMessage = '';

  constructor(
    private readonly ticketService: TicketService,
    private readonly router: Router,
  ) {}

  submitTicket(): void {
    const description = this.ticketDescription.trim();

    if (!description || this.isSubmitting) {
      return;
    }

    this.isSubmitting = true;
    this.errorMessage = '';

    const ticket: RawTicket = {
      id: `TL-${Date.now()}`,
      description,
    };

    this.ticketService.submitForTriage(ticket).subscribe({
      next: () => {
        this.isSubmitting = false;
        this.router.navigate(['/ai-validation']);
      },
      error: () => {
        this.isSubmitting = false;
        this.errorMessage =
          'We could not analyze the ticket right now. Please try again.';
      },
    });
  }
}
