"""Utility script to ingest historical IT tickets into a persistent ChromaDB.

The ingestion strategy intentionally embeds only the descriptive ticket text
while storing routing and audit information as metadata. This keeps the vector
space focused on semantic issue similarity and preserves structured fields for
filtering, analytics, and downstream review.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

from chromadb.errors import NotFoundError
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document


LOGGER = logging.getLogger("triage_ingest")
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent


@dataclass(frozen=True)
class IngestSettings:
    """Minimal configuration for loading tickets into a local Chroma store."""

    input_file: str
    chroma_persist_directory: str
    chroma_collection_name: str
    embedding_model_name: str
    reset_collection: bool


def parse_args() -> IngestSettings:
    """Parse CLI arguments for the ingestion workflow.

    A dedicated CLI keeps the ingestion process explicit and repeatable, which
    is particularly useful when experimenting with embeddings or collection
    schemas during RAG iteration.
    """

    parser = argparse.ArgumentParser(
        description="Ingest historical IT tickets into local ChromaDB.",
    )
    parser.add_argument(
        "--input-file",
        default=str(PROJECT_ROOT / "data" / "it_service_tickets.json"),
        help="Path to the JSON file containing historical tickets.",
    )
    parser.add_argument(
        "--persist-directory",
        default=os.getenv(
            "CHROMA_PERSIST_DIRECTORY",
            str(BASE_DIR / "chroma_store"),
        ),
        help="Directory where Chroma should persist the local collection.",
    )
    parser.add_argument(
        "--collection-name",
        default=os.getenv("CHROMA_COLLECTION_NAME", "it_ticket_history"),
        help="Name of the Chroma collection to create or update.",
    )
    parser.add_argument(
        "--embedding-model",
        default=os.getenv(
            "EMBEDDING_MODEL_NAME",
            "sentence-transformers/all-MiniLM-L6-v2",
        ),
        help="HuggingFace sentence-transformer model name used for embeddings.",
    )
    parser.add_argument(
        "--reset-collection",
        action="store_true",
        help="Delete the existing Chroma collection before ingesting fresh data.",
    )
    args = parser.parse_args()

    return IngestSettings(
        input_file=args.input_file,
        chroma_persist_directory=args.persist_directory,
        chroma_collection_name=args.collection_name,
        embedding_model_name=args.embedding_model,
        reset_collection=args.reset_collection,
    )


def load_tickets(input_file: str) -> List[Dict[str, Any]]:
    """Load and validate the historical ticket JSON file from disk.

    The script keeps validation lightweight by checking the top-level structure
    and a few essential fields. Full schema validation could be added later, but
    this strikes a good balance for ingestion tooling.
    """

    file_path = Path(input_file)
    if not file_path.exists():
        raise FileNotFoundError(f"Input file not found: {file_path}")

    with file_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    if not isinstance(payload, list):
        raise ValueError("The ticket ingestion file must contain a top-level JSON array.")

    required_fields = {
        "ticket_id",
        "raw_description",
        "reported_by_role",
        "created_date_time",
        "metadata",
        "routing_classification",
    }

    for index, ticket in enumerate(payload, start=1):
        if not isinstance(ticket, dict):
            raise ValueError(f"Ticket at position {index} is not a JSON object.")
        missing_fields = required_fields.difference(ticket.keys())
        if missing_fields:
            raise ValueError(
                f"Ticket at position {index} is missing required fields: {sorted(missing_fields)}"
            )

    return payload


def build_documents(tickets: List[Dict[str, Any]]) -> tuple[List[Document], List[str]]:
    """Convert ticket records into LangChain documents and stable Chroma IDs.

    Only the natural-language description is embedded as `page_content`. Routing
    metadata is stored separately so future retrieval pipelines can filter on it
    without polluting the semantic representation.
    """

    documents: List[Document] = []
    ids: List[str] = []

    for ticket in tickets:
        routing = ticket.get("routing_classification", {})
        metadata = ticket.get("metadata", {})
        ticket_id = str(ticket["ticket_id"])

        documents.append(
            Document(
                page_content=str(ticket["raw_description"]),
                metadata={
                    "ticket_id": ticket_id,
                    "reported_by_role": str(ticket["reported_by_role"]),
                    "created_date_time": str(ticket["created_date_time"]),
                    "category": str(routing.get("category", "")),
                    "assigned_team": str(routing.get("assigned_team", "")),
                    "priority": str(routing.get("priority", "")),
                    "hardware_or_software_flag": str(
                        metadata.get("hardware_or_software_flag", "")
                    ),
                    "urgency_indicator": str(metadata.get("urgency_indicator", "")),
                },
            )
        )
        ids.append(ticket_id)

    return documents, ids


def build_embeddings(model_name: str) -> HuggingFaceEmbeddings:
    """Create the shared embedding model used during Chroma ingestion."""

    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def reset_collection_if_requested(
    vector_store: Chroma,
    collection_name: str,
    reset_collection: bool,
) -> None:
    """Delete the existing collection when the operator requests a clean ingest.

    This explicit reset path is safer than silently overwriting data because it
    makes reindexing a deliberate operational action.
    """

    if not reset_collection:
        return

    try:
        vector_store._client.delete_collection(name=collection_name)  # type: ignore[attr-defined]
        LOGGER.info("Deleted existing Chroma collection '%s'.", collection_name)
    except NotFoundError:
        LOGGER.info("Collection '%s' did not exist, so no reset was needed.", collection_name)


def ingest_documents(settings: IngestSettings) -> int:
    """Load ticket records and upsert them into persistent Chroma storage.

    Existing IDs are deleted before insertion so rerunning the script remains
    deterministic even when the source JSON is updated incrementally.
    """

    Path(settings.chroma_persist_directory).mkdir(parents=True, exist_ok=True)
    tickets = load_tickets(settings.input_file)
    documents, ids = build_documents(tickets)

    vector_store = Chroma(
        collection_name=settings.chroma_collection_name,
        persist_directory=settings.chroma_persist_directory,
        embedding_function=build_embeddings(settings.embedding_model_name),
    )

    reset_collection_if_requested(
        vector_store=vector_store,
        collection_name=settings.chroma_collection_name,
        reset_collection=settings.reset_collection,
    )

    if settings.reset_collection:
        vector_store = Chroma(
            collection_name=settings.chroma_collection_name,
            persist_directory=settings.chroma_persist_directory,
            embedding_function=build_embeddings(settings.embedding_model_name),
        )

    try:
        vector_store.delete(ids=ids)
    except Exception:
        LOGGER.info("Some document IDs did not already exist; continuing with fresh insert.")

    vector_store.add_documents(documents=documents, ids=ids)
    LOGGER.info(
        "Ingested %s tickets into collection '%s' at '%s'.",
        len(documents),
        settings.chroma_collection_name,
        settings.chroma_persist_directory,
    )
    return len(documents)


def main() -> None:
    """CLI entrypoint for local ChromaDB ingestion."""

    settings = parse_args()
    count = ingest_documents(settings)
    print(
        json.dumps(
            {
                "status": "success",
                "ingested_documents": count,
                "collection_name": settings.chroma_collection_name,
                "persist_directory": settings.chroma_persist_directory,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
