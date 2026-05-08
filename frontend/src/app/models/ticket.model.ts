export interface RawTicket {
  id: string;
  description: string;
}

export interface AITriageResult {
  category: string;
  cleanDescription: string;
  label: string;
  priority: 'Low' | 'Medium' | 'High' | 'Critical';
  explanation: string;
}
