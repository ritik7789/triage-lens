import { CommonModule } from '@angular/common';
import { Component } from '@angular/core';
import { FormsModule } from '@angular/forms';

@Component({
  selector: 'app-user-helpdesk',
  standalone: true,
  imports: [CommonModule, FormsModule],
  templateUrl: './user-helpdesk.html',
  styleUrls: ['./user-helpdesk.css'],
})
export class UserHelpdesk {
  userInput = '';
  cleanedDescription =
    'User is unable to connect to the corporate VPN and cannot access internal systems.';
  isModalOpen = false;

  openModal(): void {
    this.cleanedDescription =
      this.userInput.trim() ||
      'User is unable to connect to the corporate VPN and cannot access internal systems.';
    this.isModalOpen = true;
  }

  closeModal(): void {
    this.isModalOpen = false;
  }

  submitTicket(): void {
    this.isModalOpen = false;
    alert('Ticket Sent to IT');
  }
}
