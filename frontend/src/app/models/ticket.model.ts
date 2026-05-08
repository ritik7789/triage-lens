export interface RawTicket {
  ticket_id: string;
  subject: string;
  submitter_role: string;
  description: string;
}

export interface LegacyTriageApiResponse {
  category: string;
  priority: 'P1' | 'P2' | 'P3' | 'P4';
  assigned_team: string;
  explanation: string;
  confidence_score: number;
}

export interface FinalTriageApiRequest {
  raw_query: string;
  nlp_cleaned_query: string;
  user_role: string;
}

export interface FinalTriageApiResponse {
  category: string;
  assigned_team: string;
  severity_level: 'P1' | 'P2' | 'P3' | 'P4';
  key_entities_identified: string[];
  reasoning: string;
}

export interface QueryRecordApiResponse {
  query_id: string;
  created_at: string;
  status: 'open' | 'resolved';
  resolved_at: string | null;
  request: {
    raw_query: string;
    nlp_cleaned_query: string;
    user_role: string;
    top_k: number;
  };
  triage_result: FinalTriageApiResponse;
}

export interface QueriesApiResponse {
  queries: QueryRecordApiResponse[];
  total: number;
}

export interface AITriageResult {
  rawQuery: string;
  category: string;
  cleanDescription: string;
  label: string;
  priority: 'Low' | 'Medium' | 'High' | 'Critical';
  explanation: string;
}
