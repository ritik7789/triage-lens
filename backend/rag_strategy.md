# RAG Improvement Strategy For Local IT Ticket Triage

## Overview

This backend is intentionally optimized for a local-first deployment model:
FastAPI handles API transport, LangChain orchestrates prompt and parsing logic,
ChromaDB provides persistent dense retrieval, HuggingFace embeddings power the
semantic index, and Ollama runs `llama-3.2-3b-it:latest` entirely on local
infrastructure.

That architecture is strong enough for a hackathon or proof-of-value, but the
triage domain has unusual retrieval requirements:

- User phrasing is noisy, emotional, and inconsistent.
- Routing labels are operationally strict.
- Many tickets differ only by a few technical entities such as service name,
  cloud platform, device type, cluster name, or office location.
- Small local models can become brittle when irrelevant examples enter context.

The rest of this document focuses on improving that specific setup without
abandoning the local RAG design.

## 1. Pre-Retrieval Optimization

### 1.1 Why Pre-Retrieval Matters

Dense vector search works well when the corpus is semantically coherent, but IT
triage data spans radically different domains:

- Physical hardware break/fix
- Software licensing requests
- IAM and cloud access failures
- CI/CD platform errors
- Data engineering and database incidents

If the retrieval layer searches the full corpus every time, the model can
receive partially similar but operationally wrong examples. For example, a VPN
issue may retrieve generic "network not working" office tickets, or a Redshift
timeout may retrieve unrelated IAM failures because both mention AWS.

The cheapest improvement is to narrow the search space before vector retrieval.

### 1.2 Use The NLP Pipeline To Produce Retrieval Filters

The upstream NLP normalization step should not only produce `nlp_cleaned_query`.
It should also emit structured signals that can become Chroma metadata filters.

Recommended NLP outputs:

- `problem_domain`: hardware, office, cloud, devops, data, licensing
- `platform`: windows, macos, aws, azure, gcp, snowflake, github, jenkins
- `location`: office floor, conference room, region, VPC, subnet, cluster
- `asset_type`: laptop, monitor, dock, printer, role, license, pipeline
- `urgency_hint`: blocked, degraded, request, outage
- `service_name`: redshift, glue, argo, buildkite, docker desktop, intellij

These signals should be mapped into the historical ticket metadata model at
ingestion time. Once available, the `/triage` flow can:

1. Use NLP to predict likely domains or services.
2. Apply metadata filtering in Chroma.
3. Run semantic search only inside the filtered subset.
4. Fall back to a wider search when the filtered query returns too few results.

### 1.3 Example Metadata Filtering Pattern

For a cleaned query like:

`aws glue partition projection failure on s3-backed table after schema change`

The NLP pipeline might extract:

- `problem_domain=data`
- `platform=aws`
- `service_name=glue`

The retrieval layer can then filter on:

```python
filter_payload = {
    "$and": [
        {"category": "Database_DataEng_Issues"},
        {"hardware_or_software_flag": "Software"}
    ]
}
```

Or, if service metadata is indexed in the future:

```python
filter_payload = {
    "$and": [
        {"problem_domain": "data"},
        {"platform": "aws"},
        {"service_name": "glue"}
    ]
}
```

### 1.4 Recommended Retrieval Cascade

A strong local strategy is a cascading retrieval plan:

1. Attempt highly specific filtered retrieval.
2. If fewer than `k_min` documents are found, relax service-level filters.
3. If still sparse, relax to domain-level filters only.
4. If still sparse, perform full-corpus vector retrieval.
5. Record which fallback level was used for observability.

This pattern reduces irrelevant context without sacrificing recall.

## 2. Advanced Retrieval Strategy

### 2.1 Why Dense Search Alone Is Not Enough

Dense embeddings capture semantic similarity well, but IT tickets often contain
rare identifiers that matter disproportionately:

- Error codes
- Hostnames
- Role names
- Application names
- Product SKUs
- Ticket system abbreviations
- Cluster and environment names

Dense retrieval may miss exact token matches like `prod-observability-readonly`,
`payments-staging`, or `configure-aws-credentials`, especially when a compact
embedding model is used for speed.

### 2.2 Hybrid Search

Hybrid search combines:

- Dense retrieval for semantic understanding
- Sparse or keyword retrieval for exact lexical matching

For this ticket domain, BM25 or another sparse retriever is especially useful
for technical fragments such as:

- CLI commands
- Error strings
- Role names
- Cloud resource identifiers
- Room names
- Device models

Practical implementation options:

- Use a local BM25 index over `raw_description`.
- Use Chroma for dense search and a parallel sparse retriever for keyword hits.
- Merge the ranked results before prompt assembly.

### 2.3 Reciprocal Rank Fusion (RRF)

Reciprocal Rank Fusion is a simple and robust way to merge dense and sparse
retrieval lists without tuning complicated weights.

RRF score formula:

`RRF(d) = sum(1 / (k + rank_i(d)))`

Where:

- `d` is a candidate document
- `rank_i(d)` is the rank assigned by retriever `i`
- `k` is a smoothing constant, often around 60

Why RRF fits this use case:

- It is resilient when dense and sparse retrievers disagree.
- It does not require score normalization across systems.
- It rewards documents that rank reasonably well across both retrievers.
- It is easy to implement and debug in a local service.

### 2.4 Recommended Hybrid Pipeline

For each incoming triage request:

1. Run dense retrieval in Chroma using the cleaned query.
2. Run BM25 or keyword retrieval using the raw query and cleaned query.
3. Normalize each result set to ranked lists.
4. Fuse them with RRF.
5. Take the top 8-10 fused candidates.
6. Pass those candidates into a re-ranker before final prompt assembly.

This gives the local LLM better evidence without increasing its reasoning load.

## 3. Post-Retrieval Optimization

### 3.1 Add A Cross-Encoder Re-Ranker

Once dense or hybrid retrieval produces a candidate pool, use a cross-encoder
re-ranker such as `BAAI/bge-reranker-base`.

Why this matters:

- Chroma retrieves fast but compares vectors independently.
- A cross-encoder jointly scores `(query, document)` pairs.
- This usually produces much better relevance ordering for short candidate sets.

That matters especially when:

- Several tickets share the same broad category
- The right routing depends on subtle wording
- The local Llama model has a small context window budget

### 3.2 Recommended Re-Ranking Flow

1. Retrieve top 10-20 candidates using dense or hybrid retrieval.
2. Score each `(cleaned_query, ticket_description)` pair with the re-ranker.
3. Sort candidates by cross-encoder score.
4. Keep only the top 3-5 tickets for prompt injection.
5. Optionally append the re-ranker score to the prompt as a confidence signal.

### 3.3 Why Re-Ranking Is Better Than Larger `top_k`

Simply increasing `top_k` is not ideal for a 3B local model because:

- More context increases prompt noise.
- Long prompts reduce response consistency.
- Small models are more likely to anchor on irrelevant examples.
- Latency rises even if retrieval remains fast.

Re-ranking improves relevance quality without forcing the LLM to inspect a
large block of low-signal history.

### 3.4 Practical Notes For Local Deployment

Cross-encoders are heavier than embedding models, so plan for:

- CPU-only mode for small throughput deployments
- Optional GPU acceleration when available
- Batching candidate scoring
- Separate timeout budgets for retrieval and re-ranking

If latency becomes an issue, keep re-ranking optional and enable it only when:

- Retrieval confidence is low
- Categories are ambiguous
- The top dense results span multiple routing domains

## 4. Agentic Workflows With LangGraph

### 4.1 Why Move Beyond A Single LCEL Chain

The current LCEL chain is appropriate for a first production version because it
is:

- Simple
- Deterministic
- Easy to observe
- Easy to harden with a JSON parser

However, compact local models can struggle when:

- The retrieved context includes multiple plausible categories
- The ticket mixes request and incident language
- The model must validate output against strict business rules
- The context window becomes crowded by examples and formatting instructions

In those cases, a multi-step agentic workflow can outperform a single prompt.

### 4.2 Recommended LangGraph Decomposition

Transition the pipeline into specialized nodes:

1. `NormalizeIntent`
2. `GenerateRetrievalFilters`
3. `DenseRetriever`
4. `SparseRetriever`
5. `RankFusion`
6. `CrossEncoderReranker`
7. `Classifier`
8. `PolicyValidator`
9. `ConfidenceGate`
10. `FallbackEscalation`

### 4.3 Suggested Responsibilities

`NormalizeIntent`

- Standardize the query
- Extract entities
- Infer likely routing domain

`GenerateRetrievalFilters`

- Produce metadata filters
- Decide whether broad or narrow retrieval is appropriate

`DenseRetriever` and `SparseRetriever`

- Retrieve separate candidate pools optimized for semantics and keyword overlap

`RankFusion`

- Merge candidates with RRF

`CrossEncoderReranker`

- Produce the final high-precision evidence set

`Classifier`

- Generate the routing JSON

`PolicyValidator`

- Verify that:
  - category is in the approved list
  - severity is one of `P1-P4`
  - assigned team is valid
  - reasoning references evidence

`ConfidenceGate`

- Check whether retrieval scores, validation signals, and classifier certainty
  are strong enough for auto-routing

`FallbackEscalation`

- Route low-confidence cases to a human queue
- Optionally produce a "needs review" annotation

### 4.4 Benefits Of The Agentic Pattern

- Easier debugging of retrieval failures vs classification failures
- Better control over context budgets
- More explicit confidence management
- Easier insertion of rule-based business logic
- Better support for human-in-the-loop workflows

### 4.5 When To Keep The Simpler LCEL Version

Stay with the current single-chain design when:

- The corpus is small
- The routing categories are stable
- Latency is critical
- The system is still being evaluated
- Manual review is available for uncertain outputs

Adopt LangGraph when:

- Ticket volume increases
- Routing mistakes become operationally expensive
- You need richer validation and fallback handling
- Multiple retrieval and ranking stages must be coordinated explicitly

## 5. Additional Local-First Hardening Recommendations

### 5.1 Introduce Confidence Signals

Store and log:

- dense retrieval scores
- sparse retrieval ranks
- fused ranks
- re-ranker scores
- final model output category
- parser success or failure

These signals make it much easier to identify whether bad routing comes from
retrieval quality, poor prompt alignment, or model instability.

### 5.2 Maintain A Golden Evaluation Set

Create a curated validation set of real or synthetic tickets covering:

- one-shot obvious cases
- ambiguous cross-domain cases
- typo-heavy user phrasing
- high urgency incidents
- low urgency requests
- cloud tickets with exact technical identifiers
- office and hardware tickets with physical symptoms

Run this set after every prompt, retriever, or embedding change.

### 5.3 Version The Prompt And Embeddings

Persist:

- prompt version
- embedding model version
- collection build timestamp
- ingestion source file hash

Without versioning, regression analysis becomes difficult when retrieval quality
changes over time.

### 5.4 Consider Category-Specific Sub-Collections

If the corpus grows substantially, you can create:

- one global collection
- several domain-specific collections

Then route retrieval dynamically:

- first choose a likely domain
- then search the domain-specific collection
- optionally fall back to the global collection

This can improve both latency and precision for local deployments.

## 6. Recommended Roadmap

### Phase 1: Immediate Improvements

- Add metadata fields during ingestion for domain, platform, service, and asset
- Add Chroma metadata filtering before semantic retrieval
- Log retrieval scores and chosen examples

### Phase 2: Retrieval Quality

- Add BM25 sparse retrieval
- Fuse dense and sparse ranks with RRF
- Introduce cross-encoder re-ranking

### Phase 3: Decision Reliability

- Add business-rule validation
- Add confidence thresholds
- Add manual-review fallback paths

### Phase 4: Workflow Maturity

- Move to LangGraph
- Split retrieval, ranking, classification, and validation into graph nodes
- Add observability and benchmark-driven tuning

## Conclusion

The current FastAPI + LangChain + Chroma + Ollama design is a solid local RAG
foundation for IT ticket triage. The biggest gains will come from improving the
retrieval stage rather than immediately changing the language model. In this
domain, better filtering, hybrid retrieval, re-ranking, and validation will
usually outperform a naive "larger prompt with more examples" strategy.

For a compact local model like `llama-3.2-3b-it`, the winning pattern is:

1. retrieve fewer but better examples
2. validate the model output aggressively
3. escalate uncertain cases instead of forcing overconfident auto-routing
