# IT Ticket Triage API Reference

Base URL for local development: `http://localhost:8000`

Interactive documentation:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`
- OpenAPI JSON: `http://localhost:8000/openapi.json`

## Endpoints

### `GET /`
Returns a small service index with links to the API docs and main endpoints.

Sample response:

```json
{
  "service": "IT Ticket Triage RAG Service",
  "swagger_docs": "/docs",
  "redoc": "/redoc",
  "openapi_json": "/openapi.json",
  "health_endpoint": "/health",
  "triage_endpoint": "/triage"
}
```

### `GET /health`
Returns a lightweight operational status for the running backend.

Sample response:

```json
{
  "status": "ok",
  "collection_name": "it_ticket_history",
  "indexed_documents": 120,
  "ollama_model": "llama-3.2-3b-it:latest"
}
```

### `POST /triage`
Classifies a ticket into category, assigned team, and priority using retrieval-augmented generation.
Each successful triage request is also saved locally so it can be retrieved later through the query endpoints.

Request body:

```json
{
  "raw_query": "My laptop dock keeps disconnecting ethernet and the external monitor whenever I join a Microsoft Teams call.",
  "nlp_cleaned_query": "USB-C docking station intermittent ethernet and monitor disconnect during Microsoft Teams calls",
  "user_role": "Senior Java Developer",
  "top_k": 4
}
```

Sample success response:

```json
{
  "category": "Hardware_BreakFix",
  "assigned_team": "Workplace End User Computing",
  "severity_level": "P2",
  "key_entities_identified": [
    "USB-C dock",
    "ethernet disconnect",
    "external monitor",
    "Microsoft Teams calls",
    "developer workstation"
  ],
  "reasoning": "The issue matches historical dock and peripheral stability incidents that affected monitors and network connectivity during video calls, so it is best routed to the workplace hardware support queue."
}
```

Alternative request example:

```json
{
  "raw_query": "I joined the platform squad today and still cannot access the production AWS account or assume the deploy role.",
  "nlp_cleaned_query": "New engineer missing AWS production account access and deploy role permissions",
  "user_role": "Platform Engineer",
  "top_k": 4
}
```

Possible error responses:

`422 Unprocessable Entity`

```json
{
  "detail": [
    {
      "type": "string_too_short",
      "loc": ["body", "raw_query"],
      "msg": "String should have at least 10 characters",
      "input": "short",
      "ctx": {
        "min_length": 10
      }
    }
  ]
}
```

`503 Service Unavailable`

```json
{
  "detail": "The Chroma collection is empty. Run backend/ingest.py before using /triage."
}
```

`502 Bad Gateway`

```json
{
  "detail": "The language model returned an invalid triage payload."
}
```

### `GET /queries`
Returns all saved triaged queries.

Sample response:

```json
{
  "queries": [
    {
      "query_id": "QRY-000001",
      "created_at": "2026-05-08T11:30:00Z",
      "status": "open",
      "resolved_at": null,
      "request": {
        "raw_query": "My laptop dock keeps disconnecting ethernet and the external monitor whenever I join a Microsoft Teams call.",
        "nlp_cleaned_query": "USB-C docking station intermittent ethernet and monitor disconnect during Microsoft Teams calls",
        "user_role": "Senior Java Developer",
        "top_k": 4
      },
      "triage_result": {
        "category": "Hardware_BreakFix",
        "assigned_team": "Workplace End User Computing",
        "severity_level": "P2",
        "key_entities_identified": [
          "USB-C dock",
          "ethernet disconnect",
          "external monitor",
          "Microsoft Teams calls",
          "developer workstation"
        ],
        "reasoning": "The issue matches historical dock and peripheral stability incidents that affected monitors and network connectivity during video calls, so it is best routed to the workplace hardware support queue."
      }
    }
  ],
  "total": 1
}
```

### `GET /queries/{query_id}`
Returns one saved query by ID.

Sample response:

```json
{
  "query_id": "QRY-000001",
  "created_at": "2026-05-08T11:30:00Z",
  "status": "open",
  "resolved_at": null,
  "request": {
    "raw_query": "My laptop dock keeps disconnecting ethernet and the external monitor whenever I join a Microsoft Teams call.",
    "nlp_cleaned_query": "USB-C docking station intermittent ethernet and monitor disconnect during Microsoft Teams calls",
    "user_role": "Senior Java Developer",
    "top_k": 4
  },
  "triage_result": {
    "category": "Hardware_BreakFix",
    "assigned_team": "Workplace End User Computing",
    "severity_level": "P2",
    "key_entities_identified": [
      "USB-C dock",
      "ethernet disconnect",
      "external monitor",
      "Microsoft Teams calls",
      "developer workstation"
    ],
    "reasoning": "The issue matches historical dock and peripheral stability incidents that affected monitors and network connectivity during video calls, so it is best routed to the workplace hardware support queue."
  }
}
```

### `DELETE /queries/{query_id}`
Deletes one saved query by ID.

Sample response:

```json
{
  "query_id": "QRY-000001",
  "created_at": "2026-05-08T11:30:00Z",
  "status": "open",
  "resolved_at": null,
  "request": {
    "raw_query": "My laptop dock keeps disconnecting ethernet and the external monitor whenever I join a Microsoft Teams call.",
    "nlp_cleaned_query": "USB-C docking station intermittent ethernet and monitor disconnect during Microsoft Teams calls",
    "user_role": "Senior Java Developer",
    "top_k": 4
  },
  "triage_result": {
    "category": "Hardware_BreakFix",
    "assigned_team": "Workplace End User Computing",
    "severity_level": "P2",
    "key_entities_identified": [
      "USB-C dock",
      "ethernet disconnect",
      "external monitor",
      "Microsoft Teams calls",
      "developer workstation"
    ],
    "reasoning": "The issue matches historical dock and peripheral stability incidents that affected monitors and network connectivity during video calls, so it is best routed to the workplace hardware support queue."
  }
}
```

### `PATCH /tickets/{query_id}/resolve`
Marks a saved query as resolved.

Sample response:

```json
{
  "message": "Ticket marked as resolved.",
  "ticket": {
    "query_id": "QRY-000001",
    "created_at": "2026-05-08T11:30:00Z",
    "status": "resolved",
    "resolved_at": "2026-05-08T12:15:00Z",
    "request": {
      "raw_query": "My laptop dock keeps disconnecting ethernet and the external monitor whenever I join a Microsoft Teams call.",
      "nlp_cleaned_query": "USB-C docking station intermittent ethernet and monitor disconnect during Microsoft Teams calls",
      "user_role": "Senior Java Developer",
      "top_k": 4
    },
    "triage_result": {
      "category": "Hardware_BreakFix",
      "assigned_team": "Workplace End User Computing",
      "severity_level": "P2",
      "key_entities_identified": [
        "USB-C dock",
        "ethernet disconnect",
        "external monitor",
        "Microsoft Teams calls",
        "developer workstation"
      ],
      "reasoning": "The issue matches historical dock and peripheral stability incidents that affected monitors and network connectivity during video calls, so it is best routed to the workplace hardware support queue."
    }
  }
}
```

### `GET /tickets/resolved`
Returns all tickets currently marked as resolved.

Sample response:

```json
{
  "queries": [
    {
      "query_id": "QRY-000001",
      "created_at": "2026-05-08T11:30:00Z",
      "status": "resolved",
      "resolved_at": "2026-05-08T12:15:00Z",
      "request": {
        "raw_query": "My laptop dock keeps disconnecting ethernet and the external monitor whenever I join a Microsoft Teams call.",
        "nlp_cleaned_query": "USB-C docking station intermittent ethernet and monitor disconnect during Microsoft Teams calls",
        "user_role": "Senior Java Developer",
        "top_k": 4
      },
      "triage_result": {
        "category": "Hardware_BreakFix",
        "assigned_team": "Workplace End User Computing",
        "severity_level": "P2",
        "key_entities_identified": [
          "USB-C dock",
          "ethernet disconnect",
          "external monitor",
          "Microsoft Teams calls",
          "developer workstation"
        ],
        "reasoning": "The issue matches historical dock and peripheral stability incidents that affected monitors and network connectivity during video calls, so it is best routed to the workplace hardware support queue."
      }
    }
  ],
  "total": 1
}
```

## Notes

- Run `.\backend\run_ingest.ps1` before calling `POST /triage`.
- The interactive docs always reflect the current code, because they are generated from the FastAPI schema.
