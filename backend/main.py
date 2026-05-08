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
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal, Sequence

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain_core.documents import Document
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field, ValidationError


LOGGER = logging.getLogger("triage_rag_service")
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent

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
        )


class TriageRequest(BaseModel):
    """Incoming request payload for the `/triage` endpoint.

    The contract intentionally keeps both the raw and normalized ticket text.
    The raw ticket preserves human nuance such as urgency, frustration, and
    incomplete wording, while the NLP-cleaned query is optimized for retrieval.
    """

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

    status: Literal["ok"]
    collection_name: str
    indexed_documents: int
    ollama_model: str


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
        "enterprise IT tickets using ChromaDB, LangChain, and Ollama."
    ),
    lifespan=lifespan,
)


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


@app.get("/health", response_model=HealthResponse, tags=["Operations"])
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


@app.post("/triage", response_model=TriageResponse, tags=["Triage"])
async def triage_ticket(payload: TriageRequest) -> TriageResponse:
    """Classify an incoming ticket using retrieval-augmented generation.

    The actual LangChain execution is run in a threadpool because the local
    embedding lookup and Ollama call are synchronous. This keeps the FastAPI
    event loop responsive under concurrent request load.
    """

    service: TriageService = app.state.triage_service

    try:
        return await run_in_threadpool(service.triage, payload)
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


if __name__ == "__main__":
    """Allow `python backend/main.py` for local development convenience."""

    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
