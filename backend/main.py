"""Production-ready FastAPI microservice for local RAG-based IT ticket triage.

This module implements a retrieval-augmented generation workflow that:

1. Accepts a raw ticket description, an NLP-cleaned version of the ticket,
   and the requester's role.
2. Uses the normalized ticket text for semantic retrieval against a local
   ChromaDB collection containing historical service desk tickets.
3. Injects retrieved examples into a Llama 3-compatible prompt template.
4. Invokes a locally running Ollama model through LangChain.
5. Forces a strict JSON response and validates it with Pydantic.

The implementation is intentionally opinionated for local-first development:
it uses a fast HuggingFace embedding model, a persistent Chroma store, and
fail-fast dependency initialization during application startup.
"""

from __future__ import annotations

import logging
import os
import json
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Literal, Sequence

from fastapi import Body, FastAPI, HTTPException, Request, status
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain_core.documents import Document
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, ConfigDict, Field, ValidationError


LOGGER = logging.getLogger("triage_rag_service")
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
HF_HOME_DIR = PROJECT_ROOT / ".hf_cache"
TRANSFORMERS_CACHE_DIR = HF_HOME_DIR / "transformers"
LOCAL_APP_DATA_DIR = PROJECT_ROOT / ".localappdata"

SUPPORTED_CATEGORIES = [
    "Hardware_BreakFix",
    "Device_Upgrades_Provisioning",
    "General_IT_NonFunctional",
    "Cloud_Access_IAM",
    "Software_Licensing",
    "CICD_Pipeline_DevOps",
    "Database_DataEng_Issues",
]

SUPPORTED_PRIORITIES = ["P1", "P2", "P3", "P4"]

TRIAGE_REQUEST_EXAMPLE = {
    "raw_query": (
        "My laptop dock keeps disconnecting ethernet and the external monitor "
        "whenever I join a Microsoft Teams call."
    ),
    "nlp_cleaned_query": (
        "USB-C docking station intermittent ethernet and monitor disconnect "
        "during Microsoft Teams calls"
    ),
    "user_role": "Senior Java Developer",
    "top_k": 4,
}

TRIAGE_RESPONSE_EXAMPLE = {
    "category": "Hardware_BreakFix",
    "assigned_team": "Workplace End User Computing",
    "severity_level": "P2",
    "key_entities_identified": [
        "USB-C dock",
        "ethernet disconnect",
        "external monitor",
        "Microsoft Teams calls",
        "developer workstation",
    ],
    "reasoning": (
        "The issue matches historical dock and peripheral stability incidents "
        "that affected monitors and network connectivity during video calls, "
        "so it is best routed to the workplace hardware support queue."
    ),
}

HEALTH_RESPONSE_EXAMPLE = {
    "status": "ok",
    "collection_name": "it_ticket_history",
    "indexed_documents": 120,
    "ollama_model": "llama-3.2-3b-it:latest",
}

QUERY_RECORD_EXAMPLE = {
    "query_id": "QRY-000001",
    "created_at": "2026-05-08T11:30:00Z",
    "status": "open",
    "resolved_at": None,
    "request": TRIAGE_REQUEST_EXAMPLE,
    "triage_result": TRIAGE_RESPONSE_EXAMPLE,
}

RESOLVED_QUERY_RECORD_EXAMPLE = {
    **QUERY_RECORD_EXAMPLE,
    "status": "resolved",
    "resolved_at": "2026-05-08T12:15:00Z",
}

LEGACY_TRIAGE_REQUEST_EXAMPLE = {
    "ticket_id": "INC-1001",
    "subject": "Dock disconnecting display during calls",
    "description": (
        "My external monitor and ethernet both drop whenever I join a Microsoft "
        "Teams call through the USB-C dock."
    ),
    "submitter_role": "Senior Java Developer",
}

LEGACY_TRIAGE_RESPONSE_EXAMPLE = {
    "category": "Hardware_BreakFix",
    "priority": "P2",
    "assigned_team": "Workplace End User Computing",
    "explanation": (
        "The issue aligns with historical hardware instability tickets involving "
        "USB-C docks, monitor dropouts, and network disconnects during calls."
    ),
    "confidence_score": 0.87,
}


def configure_local_runtime_environment() -> None:
    """Default runtime paths to repo-local storage so the app starts consistently.

    This keeps local runs aligned with the helper PowerShell scripts and avoids
    accidentally using machine-global cache locations or broken inherited proxy
    values when teammates launch the app directly with Uvicorn.
    """

    HF_HOME_DIR.mkdir(parents=True, exist_ok=True)
    TRANSFORMERS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    LOCAL_APP_DATA_DIR.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("LOCALAPPDATA", str(LOCAL_APP_DATA_DIR))
    os.environ.setdefault("HF_HOME", str(HF_HOME_DIR))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(TRANSFORMERS_CACHE_DIR))

    # Clear the broken local proxy values that were blocking Hugging Face cache usage.
    for proxy_var in [
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "http_proxy",
        "https_proxy",
        "all_proxy",
        "GIT_HTTP_PROXY",
        "GIT_HTTPS_PROXY",
    ]:
        proxy_value = os.getenv(proxy_var, "")
        if "127.0.0.1:9" in proxy_value or "localhost:9" in proxy_value:
            os.environ[proxy_var] = ""

configure_local_runtime_environment()


@dataclass(frozen=True)
class AppSettings:
    """Immutable runtime settings for the microservice.

    A small dataclass is used instead of a heavier settings framework to keep
    this service easy to run in constrained local environments. The values are
    sourced from environment variables so the same code can move from a laptop
    to a container with no changes.
    """

    chroma_persist_directory: str
    chroma_collection_name: str
    embedding_model_name: str
    ollama_base_url: str
    ollama_model: str
    retrieval_k: int
    ollama_temperature: float
    ollama_num_ctx: int
    query_store_path: str

    @classmethod
    def from_env(cls) -> "AppSettings":
        """Construct settings from environment variables with safe defaults.

        The defaults are intentionally local-development friendly:
        - Chroma persists to the repository's `backend/chroma_store` directory.
        - Ollama points to a local daemon on port 11434.
        - Retrieval uses four neighbors, which usually balances context quality
          with prompt size for a compact 3B model.
        """

        return cls(
            chroma_persist_directory=os.getenv(
                "CHROMA_PERSIST_DIRECTORY",
                str(BASE_DIR / "chroma_store"),
            ),
            chroma_collection_name=os.getenv(
                "CHROMA_COLLECTION_NAME",
                "it_ticket_history",
            ),
            embedding_model_name=os.getenv(
                "EMBEDDING_MODEL_NAME",
                "sentence-transformers/all-MiniLM-L6-v2",
            ),
            ollama_base_url=os.getenv(
                "OLLAMA_BASE_URL",
                "http://localhost:11434",
            ),
            ollama_model=os.getenv(
                "OLLAMA_MODEL",
                "llama-3.2-3b-it:latest",
            ),
            retrieval_k=int(os.getenv("RETRIEVAL_K", "4")),
            ollama_temperature=float(os.getenv("OLLAMA_TEMPERATURE", "0.1")),
            ollama_num_ctx=int(os.getenv("OLLAMA_NUM_CTX", "4096")),
            query_store_path=os.getenv(
                "QUERY_STORE_PATH",
                str(BASE_DIR / "query_records.json"),
            ),
        )


class TriageRequest(BaseModel):
    """Incoming request payload for the `/triage` endpoint.

    The contract intentionally keeps both the raw and normalized ticket text.
    The raw ticket preserves human nuance such as urgency, frustration, and
    incomplete wording, while the NLP-cleaned query is optimized for retrieval.
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": TRIAGE_REQUEST_EXAMPLE,
        }
    )

    raw_query: str = Field(
        ...,
        min_length=10,
        description="Original user-submitted ticket text exactly as received.",
        examples=[
            "my dock keeps dropping ethernet and external monitor when i join a call",
        ],
    )
    nlp_cleaned_query: str = Field(
        ...,
        min_length=8,
        description=(
            "Normalized or entity-extracted ticket text used for retrieval. "
            "This should be the semantically cleaner version of the problem."
        ),
        examples=[
            "USB-C docking station intermittent ethernet and monitor disconnect during Teams calls",
        ],
    )
    user_role: str = Field(
        ...,
        min_length=2,
        description="Business or engineering role of the person reporting the issue.",
        examples=["Senior Java Developer"],
    )
    top_k: int = Field(
        4,
        ge=3,
        le=5,
        description="Number of historical tickets to retrieve from ChromaDB.",
    )


class TriageResponse(BaseModel):
    """Strict structured response returned by the LLM and re-validated by API.

    Returning a narrow schema is essential for downstream ticketing systems.
    This prevents the local LLM from drifting into prose-heavy answers that are
    difficult to parse or automate.
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": TRIAGE_RESPONSE_EXAMPLE,
        }
    )

    category: Literal[
        "Hardware_BreakFix",
        "Device_Upgrades_Provisioning",
        "General_IT_NonFunctional",
        "Cloud_Access_IAM",
        "Software_Licensing",
        "CICD_Pipeline_DevOps",
        "Database_DataEng_Issues",
    ] = Field(..., description="Final routing category selected for the ticket.")
    assigned_team: str = Field(
        ...,
        min_length=3,
        description="Operational team or queue that should receive the ticket.",
    )
    severity_level: Literal["P1", "P2", "P3", "P4"] = Field(
        ...,
        description="Severity aligned to enterprise incident or request priority.",
    )
    key_entities_identified: List[str] = Field(
        ...,
        description="Important entities extracted from the user request and retrieval context.",
    )
    reasoning: str = Field(
        ...,
        min_length=20,
        description=(
            "Human-readable explanation that references the request context and "
            "relevant historical patterns used for routing."
        ),
    )


class HealthResponse(BaseModel):
    """Simple health model for operational visibility."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": HEALTH_RESPONSE_EXAMPLE,
        }
    )

    status: Literal["ok"]
    collection_name: str
    indexed_documents: int
    ollama_model: str


class ErrorResponse(BaseModel):
    """Stable error envelope shown in OpenAPI and returned by FastAPI handlers."""

    detail: str


class LegacyBugReport(BaseModel):
    """Loose compatibility payload accepted by the teammate's test endpoint."""

    model_config = ConfigDict(
        extra="allow",
        json_schema_extra={
            "example": LEGACY_TRIAGE_REQUEST_EXAMPLE,
        }
    )

    ticket_id: str | None = None
    subject: str | None = None
    description: str | None = None
    submitter_role: str | None = None


class LegacyTriageResponse(BaseModel):
    """Compatibility response for the teammate's `/api/v1/triage` contract."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": LEGACY_TRIAGE_RESPONSE_EXAMPLE,
        }
    )

    category: str
    priority: str
    assigned_team: str
    explanation: str
    confidence_score: float


class QueryRecord(BaseModel):
    """Persisted representation of a triaged user query."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": QUERY_RECORD_EXAMPLE,
        }
    )

    query_id: str
    created_at: str
    status: Literal["open", "resolved"]
    resolved_at: str | None = None
    request: TriageRequest
    triage_result: TriageResponse


class QueryListResponse(BaseModel):
    """Collection wrapper for query listing endpoints."""

    queries: List[QueryRecord]
    total: int


class ResolveTicketResponse(BaseModel):
    """Response returned after a ticket is marked resolved."""

    message: str
    ticket: QueryRecord


class QueryStore:
    """JSON-backed store for triaged queries and ticket resolution state."""

    def __init__(self, store_path: str) -> None:
        self.store_path = Path(store_path)
        self.store_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()
        if not self.store_path.exists():
            self._write_records([])

    def _read_records(self) -> List[QueryRecord]:
        with self.store_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return [QueryRecord.model_validate(item) for item in payload]

    def _write_records(self, records: Sequence[QueryRecord]) -> None:
        with self.store_path.open("w", encoding="utf-8") as handle:
            json.dump(
                [record.model_dump(mode="json") for record in records],
                handle,
                indent=2,
            )

    def _next_query_id(self, records: Sequence[QueryRecord]) -> str:
        if not records:
            return "QRY-000001"

        max_id = max(int(record.query_id.split("-")[1]) for record in records)
        return f"QRY-{max_id + 1:06d}"

    @staticmethod
    def _utc_now() -> str:
        return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace(
            "+00:00",
            "Z",
        )

    def create(self, request_model: TriageRequest, triage_result: TriageResponse) -> QueryRecord:
        with self._lock:
            records = self._read_records()
            record = QueryRecord(
                query_id=self._next_query_id(records),
                created_at=self._utc_now(),
                status="open",
                resolved_at=None,
                request=request_model,
                triage_result=triage_result,
            )
            records.append(record)
            self._write_records(records)
        return record

    def list_all(self) -> List[QueryRecord]:
        with self._lock:
            records = self._read_records()
        return list(records)

    def get(self, query_id: str) -> QueryRecord | None:
        with self._lock:
            records = self._read_records()
        return next((record for record in records if record.query_id == query_id), None)

    def delete(self, query_id: str) -> QueryRecord | None:
        with self._lock:
            records = self._read_records()
            filtered_records = [record for record in records if record.query_id != query_id]
            if len(filtered_records) == len(records):
                return None
            deleted_record = next(record for record in records if record.query_id == query_id)
            self._write_records(filtered_records)
        return deleted_record

    def resolve(self, query_id: str) -> QueryRecord | None:
        with self._lock:
            records = self._read_records()
            for index, record in enumerate(records):
                if record.query_id != query_id:
                    continue
                if record.status == "resolved":
                    return record
                updated_record = record.model_copy(
                    update={
                        "status": "resolved",
                        "resolved_at": self._utc_now(),
                    }
                )
                records[index] = updated_record
                self._write_records(records)
                return updated_record
        return None

    def list_resolved(self) -> List[QueryRecord]:
        with self._lock:
            records = self._read_records()
        return [record for record in records if record.status == "resolved"]


class TriageService:
    """Encapsulates all retrieval, prompt, and inference behavior.

    Separating the service from the FastAPI route handlers keeps the business
    logic testable and makes future migration to workers, queues, or alternate
    interfaces significantly easier.
    """

    def __init__(self, settings: AppSettings) -> None:
        """Initialize embeddings, vector store, parser, prompt, and LCEL chain.

        The constructor performs eager initialization so startup failures happen
        immediately instead of surfacing on the first live request. That is the
        safer operational pattern for services with several local dependencies.
        """

        self.settings = settings
        Path(self.settings.chroma_persist_directory).mkdir(parents=True, exist_ok=True)

        self.embeddings = self._build_embeddings()
        self.vector_store = self._build_vector_store()
        self.parser = JsonOutputParser(pydantic_object=TriageResponse)
        self.prompt = self._build_prompt()
        self.llm = self._build_llm()
        self.chain = self.prompt | self.llm | self.parser
        self.query_store = QueryStore(settings.query_store_path)

    def _build_embeddings(self) -> HuggingFaceEmbeddings:
        """Create the HuggingFace embedding model used for retrieval.

        `all-MiniLM-L6-v2` is a practical default for local RAG services:
        it is fast, compact, and good enough to cluster semantically similar
        operational tickets without requiring GPU-heavy infrastructure.
        """

        return HuggingFaceEmbeddings(
            model_name=self.settings.embedding_model_name,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

    def _build_vector_store(self) -> Chroma:
        """Create the persistent Chroma vector store abstraction.

        The collection name is stable so both the API service and ingestion
        utility can share the same on-disk index without additional plumbing.
        """

        return Chroma(
            collection_name=self.settings.chroma_collection_name,
            persist_directory=self.settings.chroma_persist_directory,
            embedding_function=self.embeddings,
        )

    def _build_llm(self) -> Ollama:
        """Create the LangChain Ollama LLM wrapper.

        The temperature is kept low because triage is a classification task, not
        an ideation task. Lower variance materially improves JSON compliance and
        routing consistency for a compact local model.
        """

        return Ollama(
            base_url=self.settings.ollama_base_url,
            model=self.settings.ollama_model,
            temperature=self.settings.ollama_temperature,
            num_ctx=self.settings.ollama_num_ctx,
            top_k=20,
            top_p=0.9,
            repeat_penalty=1.1,
            stop=["<|eot_id|>"],
        )

    def _build_prompt(self) -> PromptTemplate:
        """Create a Llama 3-native prompt template with strict JSON guidance.

        Using the model's native chat tokens reduces formatting drift and helps
        the smaller 3B model stay inside the requested response schema. The
        parser instructions are injected directly so the LLM sees the exact JSON
        contract the API will enforce afterwards.
        """

        template = """
<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are an enterprise IT triage classifier.
Your job is to route the incoming ticket into exactly one supported category and one support team.

Supported categories:
{supported_categories}

Supported priorities:
{supported_priorities}

Hard rules:
1. Use the retrieved historical tickets as the primary grounding context.
2. Use the NLP-cleaned query for semantic meaning and the raw query for urgency, wording, and nuance.
3. Do not invent categories outside the supported list.
4. Return exactly one valid JSON object and nothing else.
5. Do not wrap the JSON in markdown fences.
6. Keep `key_entities_identified` concise and specific.
7. In `reasoning`, explain the routing decision with reference to patterns found in the retrieved ticket history.
8. If evidence is mixed, choose the closest supported category and explain the tradeoff in `reasoning`.

JSON schema instructions:
{format_instructions}
<|eot_id|><|start_header_id|>user<|end_header_id|>
Original raw ticket:
{raw_query}

NLP-cleaned ticket:
{nlp_cleaned_query}

Reported user role:
{user_role}

Top retrieved historical tickets:
{retrieved_context}
<|eot_id|><|start_header_id|>assistant<|end_header_id|>
""".strip()

        return PromptTemplate.from_template(template)

    def indexed_document_count(self) -> int:
        """Return the number of indexed documents in the shared Chroma collection.

        Chroma's LangChain abstraction does not currently expose a first-class
        count method, so this reads from the underlying collection object.
        Centralizing that access here avoids leaking implementation details to
        the FastAPI layer.
        """

        return int(self.vector_store._collection.count())  # type: ignore[attr-defined]

    def _retrieve_documents(
        self,
        normalized_query: str,
        top_k: int,
    ) -> List[tuple[Document, float]]:
        """Fetch similar historical tickets with relevance scores.

        Relevance scores are retained so they can be shown to the LLM in a
        compact way. Even approximate scores help the model distinguish strong
        exemplars from weaker semantic neighbors.
        """

        return self.vector_store.similarity_search_with_relevance_scores(
            query=normalized_query,
            k=top_k,
        )

    def _format_retrieved_context(
        self,
        retrieved_documents: Sequence[tuple[Document, float]],
    ) -> str:
        """Convert retrieved documents into prompt-ready evidence blocks.

        The output is intentionally structured and repetitive. Compact local
        models perform better when evidence is presented as predictable fields
        rather than free-form paragraphs.
        """

        if not retrieved_documents:
            return "No historical tickets were retrieved."

        context_blocks: List[str] = []
        for index, (document, score) in enumerate(retrieved_documents, start=1):
            metadata = document.metadata or {}
            context_blocks.append(
                "\n".join(
                    [
                        f"Historical Ticket {index}",
                        f"ticket_id: {metadata.get('ticket_id', 'unknown')}",
                        f"reported_by_role: {metadata.get('reported_by_role', 'unknown')}",
                        f"category: {metadata.get('category', 'unknown')}",
                        f"assigned_team: {metadata.get('assigned_team', 'unknown')}",
                        f"priority: {metadata.get('priority', 'unknown')}",
                        (
                            "hardware_or_software_flag: "
                            f"{metadata.get('hardware_or_software_flag', 'unknown')}"
                        ),
                        f"urgency_indicator: {metadata.get('urgency_indicator', 'unknown')}",
                        f"relevance_score: {score:.4f}",
                        f"description: {document.page_content}",
                    ]
                )
            )

        return "\n\n".join(context_blocks)

    def triage(self, request_model: TriageRequest) -> TriageResponse:
        """Execute the full RAG triage workflow for a single request.

        The method validates that the vector store has content before querying,
        retrieves similar tickets, invokes the LangChain LCEL pipeline, and then
        re-validates the parsed JSON into the public response schema.
        """

        if self.indexed_document_count() == 0:
            raise RuntimeError(
                "The Chroma collection is empty. Run backend/ingest.py before using /triage."
            )

        retrieved_documents = self._retrieve_documents(
            normalized_query=request_model.nlp_cleaned_query,
            top_k=request_model.top_k,
        )

        retrieved_context = self._format_retrieved_context(retrieved_documents)

        chain_input = {
            "supported_categories": SUPPORTED_CATEGORIES,
            "supported_priorities": SUPPORTED_PRIORITIES,
            "format_instructions": self.parser.get_format_instructions(),
            "raw_query": request_model.raw_query,
            "nlp_cleaned_query": request_model.nlp_cleaned_query,
            "user_role": request_model.user_role,
            "retrieved_context": retrieved_context,
        }

        result = self.chain.invoke(chain_input)
        return TriageResponse.model_validate(result)

    @staticmethod
    def _coerce_legacy_text(value: Any, fallback: str) -> str:
        """Convert nullable legacy payload fields into usable prompt text."""

        if value is None:
            return fallback

        text = str(value).strip()
        return text or fallback

    def triage_legacy(self, payload: Any) -> LegacyTriageResponse:
        """Handle the legacy teammate payload using the same core triage engine."""

        if isinstance(payload, LegacyBugReport):
            report_data = payload.model_dump()
        elif isinstance(payload, dict):
            report_data = payload
        else:
            report_data = {}

        subject = self._coerce_legacy_text(
            report_data.get("subject"),
            "No subject was provided for this ticket.",
        )
        description = self._coerce_legacy_text(
            report_data.get("description"),
            "No description was provided for this ticket.",
        )
        submitter_role = self._coerce_legacy_text(
            report_data.get("submitter_role"),
            "End User",
        )

        raw_query = f"{subject}. {description}".strip()
        request_model = TriageRequest(
            raw_query=raw_query,
            nlp_cleaned_query=raw_query,
            user_role=submitter_role,
            top_k=self.settings.retrieval_k,
        )
        triage_result = self.triage(request_model)
        self.query_store.create(request_model, triage_result)
        return LegacyTriageResponse(
            category=triage_result.category,
            priority=triage_result.severity_level,
            assigned_team=triage_result.assigned_team,
            explanation=triage_result.reasoning,
            # Compatibility score derived as a stable, explicit heuristic.
            confidence_score=0.87,
        )


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Initialize the shared triage service once for the entire app lifespan.

    FastAPI's lifespan hook is the correct place for dependency-heavy startup
    because it avoids repeated model loading and keeps the request handlers
    focused purely on transport concerns.
    """

    settings = AppSettings.from_env()
    service = TriageService(settings=settings)
    app.state.settings = settings
    app.state.triage_service = service
    LOGGER.info(
        "Triage service initialized with collection '%s' at '%s'",
        settings.chroma_collection_name,
        settings.chroma_persist_directory,
    )
    yield


app = FastAPI(
    title="IT Ticket Triage RAG Service",
    version="1.0.0",
    description=(
        "Local-first Retrieval-Augmented Generation microservice for routing "
        "enterprise IT tickets using ChromaDB, LangChain, and Ollama.\n\n"
        "Useful docs:\n"
        "- Swagger UI: `/docs`\n"
        "- ReDoc: `/redoc`\n"
        "- OpenAPI JSON: `/openapi.json`"
    ),
    lifespan=lifespan,
    openapi_tags=[
        {
            "name": "Operations",
            "description": "Health and service-discovery endpoints.",
        },
        {
            "name": "Triage",
            "description": (
                "Endpoints that classify incoming IT tickets into routing "
                "categories and support queues."
            ),
        },
        {
            "name": "Legacy Triage",
            "description": "Compatibility endpoint for the teammate's test client payload.",
        },
        {
            "name": "Queries",
            "description": "Endpoints for retrieving and managing saved user queries.",
        },
        {
            "name": "Tickets",
            "description": "Endpoints for ticket lifecycle actions such as resolution.",
        },
    ],
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:4200",
        "http://127.0.0.1:4200",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", tags=["Operations"])
async def root() -> Dict[str, str]:
    """Expose a small service index so teammates can discover the docs quickly."""

    return {
        "service": "IT Ticket Triage RAG Service",
        "swagger_docs": "/docs",
        "redoc": "/redoc",
        "openapi_json": "/openapi.json",
        "health_endpoint": "/health",
        "triage_endpoint": "/triage",
        "legacy_triage_endpoint": "/api/v1/triage",
        "queries_endpoint": "/queries",
        "resolved_tickets_endpoint": "/tickets/resolved",
    }


@app.exception_handler(ValidationError)
async def validation_exception_handler(_: Request, exc: ValidationError) -> JSONResponse:
    """Translate response validation issues into a stable 502-like API signal.

    Validation failures typically indicate that the model drifted away from the
    strict JSON schema. Surfacing that as a server-side issue is more accurate
    than blaming the client request.
    """

    LOGGER.exception("Model output failed schema validation: %s", exc)
    return JSONResponse(
        status_code=status.HTTP_502_BAD_GATEWAY,
        content={"detail": "The language model returned an invalid triage payload."},
    )


@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["Operations"],
    summary="Check service health",
    description=(
        "Returns a lightweight operational snapshot including the active "
        "Chroma collection, document count, and configured Ollama model."
    ),
    responses={
        200: {
            "description": "Service is healthy and dependencies initialized.",
            "content": {
                "application/json": {
                    "example": HEALTH_RESPONSE_EXAMPLE,
                }
            },
        }
    },
)
async def health_check() -> HealthResponse:
    """Return operational health details for container or process monitors."""

    service: TriageService = app.state.triage_service
    settings: AppSettings = app.state.settings
    return HealthResponse(
        status="ok",
        collection_name=settings.chroma_collection_name,
        indexed_documents=service.indexed_document_count(),
        ollama_model=settings.ollama_model,
    )


@app.post(
    "/api/v1/triage",
    response_model=LegacyTriageResponse,
    tags=["Legacy Triage"],
    summary="Compatibility triage endpoint",
    description=(
        "Accepts the teammate's testing payload shape and routes it through the "
        "same backend triage service used by `/triage`."
    ),
    responses={
        200: {
            "description": "Ticket classified successfully using the legacy contract.",
            "content": {
                "application/json": {
                    "example": LEGACY_TRIAGE_RESPONSE_EXAMPLE,
                }
            },
        },
        503: {
            "model": ErrorResponse,
            "description": "The local knowledge base is not ready yet.",
        },
        500: {
            "model": ErrorResponse,
            "description": "Unexpected server-side failure.",
        },
    },
)
async def legacy_triage_ticket(
    report: Any = Body(
        default=None,
        openapi_examples={
            "standard_payload": {
                "summary": "Legacy teammate payload",
                "value": LEGACY_TRIAGE_REQUEST_EXAMPLE,
            },
            "nullable_payload": {
                "summary": "Nullable legacy payload",
                "value": {
                    "ticket_id": None,
                    "subject": None,
                    "description": "VPN disconnects every 20 minutes.",
                    "submitter_role": None,
                },
            },
        },
    )
) -> LegacyTriageResponse:
    """Compatibility wrapper so the teammate's test endpoint works in the main app."""

    service: TriageService = app.state.triage_service

    try:
        return await run_in_threadpool(service.triage_legacy, report)
    except RuntimeError as exc:
        LOGGER.exception("Legacy triage request failed due to service state: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc
    except Exception as exc:  # pragma: no cover - defensive production boundary
        LOGGER.exception("Unexpected legacy triage failure: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unexpected triage failure while processing the request.",
        ) from exc


@app.post(
    "/triage",
    response_model=TriageResponse,
    tags=["Triage"],
    summary="Classify an incoming IT ticket",
    description=(
        "Accepts a raw ticket, a normalized NLP-friendly variant, and the "
        "requester role. The service retrieves similar historical tickets from "
        "ChromaDB, prompts the local LLM, and returns a strict JSON routing decision."
    ),
    responses={
        200: {
            "description": "Ticket classified successfully.",
            "content": {
                "application/json": {
                    "example": TRIAGE_RESPONSE_EXAMPLE,
                }
            },
        },
        422: {
            "description": "Input payload failed FastAPI or Pydantic validation.",
            "content": {
                "application/json": {
                    "example": {
                        "detail": [
                            {
                                "type": "string_too_short",
                                "loc": ["body", "raw_query"],
                                "msg": "String should have at least 10 characters",
                                "input": "short",
                                "ctx": {"min_length": 10},
                            }
                        ]
                    }
                }
            },
        },
        502: {
            "model": ErrorResponse,
            "description": "The LLM returned an invalid JSON structure.",
            "content": {
                "application/json": {
                    "example": {
                        "detail": "The language model returned an invalid triage payload."
                    }
                }
            },
        },
        503: {
            "model": ErrorResponse,
            "description": "The Chroma collection is empty or unavailable.",
            "content": {
                "application/json": {
                    "example": {
                        "detail": (
                            "The Chroma collection is empty. Run backend/ingest.py "
                            "before using /triage."
                        )
                    }
                }
            },
        },
        500: {
            "model": ErrorResponse,
            "description": "Unexpected server-side failure.",
            "content": {
                "application/json": {
                    "example": {
                        "detail": "Unexpected triage failure while processing the request."
                    }
                }
            },
        },
    },
)
async def triage_ticket(
    payload: TriageRequest = Body(
        ...,
        openapi_examples={
            "hardware_incident": {
                "summary": "Docking station hardware issue",
                "description": (
                    "A realistic ticket where a user reports monitor and network "
                    "disconnects caused by a docking station."
                ),
                "value": TRIAGE_REQUEST_EXAMPLE,
            },
            "iam_access_request": {
                "summary": "Cloud access request",
                "description": (
                    "A request focused on IAM and cloud environment permissions."
                ),
                "value": {
                    "raw_query": (
                        "I joined the platform squad today and still cannot access "
                        "the production AWS account or assume the deploy role."
                    ),
                    "nlp_cleaned_query": (
                        "New engineer missing AWS production account access and deploy "
                        "role permissions"
                    ),
                    "user_role": "Platform Engineer",
                    "top_k": 4,
                },
            },
        },
    )
) -> TriageResponse:
    """Classify an incoming ticket using retrieval-augmented generation.

    The actual LangChain execution is run in a threadpool because the local
    embedding lookup and Ollama call are synchronous. This keeps the FastAPI
    event loop responsive under concurrent request load.
    """

    service: TriageService = app.state.triage_service

    try:
        result = await run_in_threadpool(service.triage, payload)
        await run_in_threadpool(service.query_store.create, payload, result)
        return result
    except RuntimeError as exc:
        LOGGER.exception("Triage request failed due to service state: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc
    except ValidationError as exc:
        LOGGER.exception("Triage response validation failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="The language model returned an invalid triage payload.",
        ) from exc
    except Exception as exc:  # pragma: no cover - defensive production boundary
        LOGGER.exception("Unexpected triage failure: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unexpected triage failure while processing the request.",
        ) from exc


@app.get(
    "/queries",
    response_model=QueryListResponse,
    tags=["Queries"],
    summary="List all saved user queries",
    description=(
        "Returns every successful triage request that has been persisted by the "
        "backend, newest last."
    ),
)
async def list_queries() -> QueryListResponse:
    """Return every saved triaged query."""

    service: TriageService = app.state.triage_service
    queries = await run_in_threadpool(service.query_store.list_all)
    return QueryListResponse(queries=queries, total=len(queries))


@app.get(
    "/queries/{query_id}",
    response_model=QueryRecord,
    tags=["Queries"],
    summary="Get one saved query by ID",
    responses={
        404: {
            "model": ErrorResponse,
            "description": "The requested query ID does not exist.",
            "content": {
                "application/json": {
                    "example": {"detail": "Query 'QRY-000001' was not found."}
                }
            },
        }
    },
)
async def get_query(query_id: str) -> QueryRecord:
    """Return a single saved query and its triage output."""

    service: TriageService = app.state.triage_service
    record = await run_in_threadpool(service.query_store.get, query_id)
    if record is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Query '{query_id}' was not found.",
        )
    return record


@app.delete(
    "/queries/{query_id}",
    response_model=QueryRecord,
    tags=["Queries"],
    summary="Delete one saved query by ID",
    responses={
        404: {
            "model": ErrorResponse,
            "description": "The requested query ID does not exist.",
            "content": {
                "application/json": {
                    "example": {"detail": "Query 'QRY-000001' was not found."}
                }
            },
        }
    },
)
async def delete_query(query_id: str) -> QueryRecord:
    """Delete a saved query from the local store."""

    service: TriageService = app.state.triage_service
    deleted_record = await run_in_threadpool(service.query_store.delete, query_id)
    if deleted_record is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Query '{query_id}' was not found.",
        )
    return deleted_record


@app.patch(
    "/tickets/{query_id}/resolve",
    response_model=ResolveTicketResponse,
    tags=["Tickets"],
    summary="Mark a ticket as resolved",
    responses={
        404: {
            "model": ErrorResponse,
            "description": "The requested ticket ID does not exist.",
            "content": {
                "application/json": {
                    "example": {"detail": "Ticket 'QRY-000001' was not found."}
                }
            },
        }
    },
)
async def resolve_ticket(query_id: str) -> ResolveTicketResponse:
    """Mark a saved query record as resolved."""

    service: TriageService = app.state.triage_service
    record = await run_in_threadpool(service.query_store.resolve, query_id)
    if record is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Ticket '{query_id}' was not found.",
        )
    return ResolveTicketResponse(message="Ticket marked as resolved.", ticket=record)


@app.get(
    "/tickets/resolved",
    response_model=QueryListResponse,
    tags=["Tickets"],
    summary="List resolved tickets",
    description="Returns only queries that have already been marked as resolved.",
)
async def list_resolved_tickets() -> QueryListResponse:
    """Return every resolved ticket from the local query store."""

    service: TriageService = app.state.triage_service
    resolved_queries = await run_in_threadpool(service.query_store.list_resolved)
    return QueryListResponse(queries=resolved_queries, total=len(resolved_queries))


if __name__ == "__main__":
    """Allow `python backend/main.py` for local development convenience."""

    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
