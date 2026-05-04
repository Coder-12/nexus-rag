# Nexus RAG Interview Coach Deployment

This app is ready to deploy as a Streamlit public app backed by the existing Nexus RAG pipeline.

## Recommended Path: Streamlit Community Cloud

1. Push this repo to GitHub.
2. In Streamlit Community Cloud, create a new app from the repo.
3. Set the app entrypoint to:

```text
app.py
```

4. Add secrets in Streamlit Cloud using the format from `.streamlit/secrets.example.toml`.

Required:

```toml
OPENAI_API_KEY = "sk-..."
PINECONE_API_KEY = "pcsk_..."
PINECONE_INDEX_NAME = "nexus-rag"
```

Recommended:

```toml
COHERE_API_KEY = "..."
```

Optional private debugging:

```toml
NEXUS_SHOW_DEBUG = "false"
```

Keep `NEXUS_SHOW_DEBUG` false for public deployment.

## Pre-Deploy Checklist

- Pinecone index exists and contains the `tier1_v1` namespace.
- `cache/bm25_index.pkl` is included in the deployment for full hybrid retrieval. If it is missing, the app stays online with vector-only sparse fallback, but retrieval quality can be lower.
- OpenAI key has enough quota for embeddings and generation.
- Cohere key is optional; if rate-limited, the app falls back to local cross-encoder reranking.
- Run the app locally and test:
  - `Final Smoke Test`
  - `Core Interview`
  - `Safety & Refusal`
  - `Out-of-Domain Graceful`

## Local Production Smoke Test

```bash
source venv/bin/activate
streamlit run app.py
```

Use the sidebar `Reload Pipeline` button after code or secret changes.

## Public App Behavior

- In-domain AI/ML interview questions are answered with citations and trust signals.
- Out-of-domain questions are handled gracefully instead of guessed.
- Raw JSON is hidden by default.
- Trace summaries remain available for transparency.
- The app exposes confidence, latency, answer mode, refusal status, citations, and evidence context.

## Operational Notes

- Watch p95/p99 latency, provider failures, timeout failures, and fallback rate.
- Cohere quota exhaustion should not break answers because the local reranker fallback is enabled.
- Local reranker fallback is slower, so hosted CPU resources may affect latency.
- For higher-traffic public launch, consider moving the pipeline behind FastAPI with request-level rate limiting and persistent trace storage.
