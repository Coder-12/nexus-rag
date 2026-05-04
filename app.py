import json
import os
import copy
import time
from datetime import datetime, timezone
from html import escape
from pathlib import Path
from typing import Any, Dict, List, Optional

import streamlit as st

from src.app_factory import build_pipeline
from src.logging_config import setup_logging


APP_NAME = "Nexus RAG Interview Coach"
APP_SUBTITLE = (
    "Grounded AI/ML interview preparation for RAG, embeddings, vector search, "
    "LLMs, alignment, evaluation, and production readiness."
)

EXAMPLE_PROMPTS = [
    "Give me the interview answer: What is RAG?",
    "Explain how embeddings, vector databases, and rerankers work together.",
    "How do context windows affect RAG answer quality?",
    "How should a RAG system defend against prompt injection?",
    "What should be monitored during traffic spikes in production RAG?",
    "How do recall@k and answer faithfulness differ in evaluation?",
]

FOLLOW_UP_PROMPTS = [
    "Give me the interview version",
    "Make it more production-focused",
    "Ask me a follow-up question",
    "Show common mistakes",
    "Explain like beginner",
]

REQUIRED_ENV_KEYS = (
    "OPENAI_API_KEY",
    "PINECONE_API_KEY",
    "PINECONE_INDEX_NAME",
)

OPTIONAL_SECRET_KEYS = (
    "COHERE_API_KEY",
    "COHERE_RERANK_ENABLED",
    "LANGSMITH_API_KEY",
    "LANGSMITH_TRACING",
    "LANGSMITH_PROJECT",
    "NEXUS_SHOW_DEBUG",
    "NEXUS_PUBLIC_MAX_QUERY_CHARS",
    "NEXUS_PUBLIC_MAX_TURNS",
    "NEXUS_SHOW_SOURCE_SNIPPETS",
)

FEEDBACK_PATH = Path(
    os.getenv("NEXUS_FEEDBACK_PATH", "data/feedback/public_feedback.jsonl")
)


def env_int(key: str, default: int) -> int:
    try:
        return int(os.getenv(key, str(default)))
    except (TypeError, ValueError):
        return default


def max_query_chars() -> int:
    return env_int("NEXUS_PUBLIC_MAX_QUERY_CHARS", 900)


def max_turns_per_session() -> int:
    return env_int("NEXUS_PUBLIC_MAX_TURNS", 40)


def show_source_snippets_enabled() -> bool:
    return os.getenv("NEXUS_SHOW_SOURCE_SNIPPETS", "").lower() in {
        "1",
        "true",
        "yes",
    }


def load_css() -> None:
    st.markdown(
        """
        <style>
        .block-container {
            max-width: 1180px;
            padding-top: 1.4rem;
            padding-bottom: 7rem;
        }

        [data-testid="stSidebar"] {
            background: #f7f9fc;
            border-right: 1px solid #e5eaf2;
        }

        /* Hide Streamlit top-right toolbar controls (including edit/deploy affordances) */
        [data-testid="stHeader"],
        [data-testid="stToolbar"],
        .stToolbar {
            display: none !important;
            visibility: hidden !important;
            height: 0 !important;
        }

        .nexus-hero {
            background: #0f172a;
            color: #ffffff;
            border-radius: 8px;
            padding: 28px 32px;
            margin-bottom: 22px;
            box-shadow: 0 12px 36px rgba(15, 23, 42, 0.18);
        }

        .nexus-hero h1 {
            margin: 0 0 10px 0;
            font-size: 34px;
            line-height: 1.1;
            letter-spacing: 0;
        }

        .nexus-hero p {
            margin: 0;
            color: #cbd5e1;
            font-size: 17px;
            line-height: 1.55;
        }

        .nexus-info-card {
            border: 1px solid #dbe7f7;
            background: #f8fbff;
            border-radius: 8px;
            padding: 16px 18px;
            margin-bottom: 20px;
            color: #1e293b;
            font-size: 15.5px;
            line-height: 1.5;
        }

        .section-label {
            color: #64748b;
            font-size: 12px;
            font-weight: 800;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            margin-bottom: 8px;
        }

        .answer-card {
            background: #ffffff;
            border: 1px solid #e5e7eb;
            border-radius: 8px;
            padding: 18px 20px;
            margin-bottom: 8px;
            box-shadow: 0 2px 12px rgba(15, 23, 42, 0.04);
        }

        .answer-card p {
            line-height: 1.62;
        }

        .trust-row {
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            margin-top: 12px;
            margin-bottom: 4px;
        }

        .trust-badge {
            border-radius: 999px;
            padding: 5px 10px;
            font-size: 12px;
            font-weight: 700;
            border: 1px solid #dbe7f7;
            background: #f8fafc;
            color: #334155;
            white-space: nowrap;
        }

        .trust-high {
            background: #ecfdf5;
            color: #047857;
            border-color: #a7f3d0;
        }

        .trust-medium {
            background: #fffbeb;
            color: #92400e;
            border-color: #fde68a;
        }

        .trust-low {
            background: #fef2f2;
            color: #991b1b;
            border-color: #fecaca;
        }

        .source-card {
            border: 1px solid #e2e8f0;
            background: #ffffff;
            border-radius: 8px;
            padding: 12px 14px;
            margin: 8px 0;
        }

        .source-title {
            font-weight: 800;
            color: #0f172a;
            margin-bottom: 4px;
        }

        .source-section {
            color: #64748b;
            font-size: 13px;
        }

        .small-muted {
            color: #64748b;
            font-size: 13px;
        }

        .mode-chip {
            display: inline-block;
            border: 1px solid #dbe7f7;
            background: #f8fafc;
            color: #334155;
            border-radius: 999px;
            padding: 5px 10px;
            margin: 2px 4px 6px 0;
            font-size: 12px;
            font-weight: 700;
        }

        div[data-testid="stChatInput"] {
            background: #ffffff;
        }

        .stChatMessage {
            border-radius: 8px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def load_streamlit_secrets_to_env() -> None:
    """
    Streamlit Community Cloud exposes secrets through st.secrets.
    The backend pipeline reads os.environ, so copy known keys once.
    """
    for key in REQUIRED_ENV_KEYS + OPTIONAL_SECRET_KEYS:
        if os.getenv(key):
            continue

        try:
            value = st.secrets.get(key)
        except Exception:
            value = None

        if value:
            os.environ[key] = str(value)


def missing_required_config() -> List[str]:
    return [key for key in REQUIRED_ENV_KEYS if not os.getenv(key)]


def show_debug_enabled() -> bool:
    return os.getenv("NEXUS_SHOW_DEBUG", "").lower() in {"1", "true", "yes"}


@st.cache_resource(show_spinner="Initializing Nexus RAG pipeline...")
def get_pipeline():
    load_streamlit_secrets_to_env()

    missing = missing_required_config()
    if missing:
        raise RuntimeError(
            "Missing required deployment secrets: " + ", ".join(missing)
        )

    setup_logging()
    return build_pipeline()


@st.cache_data(show_spinner=False, ttl=3600, max_entries=256)
def run_cached_query(prompt: str) -> Dict[str, Any]:
    pipeline = get_pipeline()
    return pipeline.run(prompt)


def init_page() -> None:
    st.set_page_config(
        page_title=APP_NAME,
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    load_css()


def init_state() -> None:
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("pending_prompt", None)
    st.session_state.setdefault("turn_count", 0)
    st.session_state.setdefault("chat_title", "New interview session")


def answer_text(response: Dict[str, Any]) -> str:
    refusal = response.get("refusal")
    answer = response.get("answer")

    if isinstance(answer, dict):
        text = str(answer.get("text", "")).strip()
        if text:
            return text

    if isinstance(refusal, dict):
        text = str(refusal.get("message", "")).strip()
        if text:
            return text

    return ""


def trace_summary(response: Dict[str, Any]) -> Dict[str, Any]:
    meta = response.get("meta", {}) or {}
    query_plan = meta.get("query_plan", {}) or {}
    evidence_audit = meta.get("evidence_audit", {}) or {}
    reflexion = meta.get("reflexion", {}) or {}

    return {
        "detected_intent": query_plan.get("intent"),
        "rewritten_query": meta.get("rewritten_query"),
        "retrieval_strategy": meta.get("strategy"),
        "answer_mode": meta.get("answer_mode"),
        "evidence_sufficient": evidence_audit.get("sufficient"),
        "evidence_coverage": evidence_audit.get("coverage_score"),
        "evidence_recovered": meta.get("evidence_recovered"),
        "support_recovered": meta.get("support_recovered"),
        "reflexion_repaired": meta.get("reflexion_repaired"),
        "regeneration_reason": meta.get("regeneration_reason"),
        "critic_needs_repair": reflexion.get("needs_repair"),
        "critic_missing_elements": reflexion.get("missing_elements", []),
        "critic_unsupported_claims": reflexion.get("unsupported_claims", []),
        "retrieved_docs": meta.get("retrieved_doc_ids", []),
        "retrieved_chunks": meta.get("retrieved_chunk_ids", []),
        "used_chunks": meta.get("used_chunk_ids", []),
        "latency_ms": meta.get("latency_ms"),
        "stage_latency_ms": meta.get("stage_latency_ms", {}),
    }


def confidence_class(level: str) -> str:
    level = str(level or "").lower()
    if level == "high":
        return "trust-high"
    if level == "medium":
        return "trust-medium"
    return "trust-low"


def render_trust_badges(response: Dict[str, Any]) -> None:
    confidence = response.get("confidence", {}) or {}
    trust = response.get("trust_signals", {}) or {}
    meta = response.get("meta", {}) or {}
    refusal = response.get("refusal")

    level = str(confidence.get("level", "unknown"))
    score = confidence.get("score", "n/a")
    latency = meta.get("latency_ms")
    answer_mode = meta.get("answer_mode") or "n/a"
    strategy = meta.get("strategy") or "n/a"

    if isinstance(latency, (int, float)):
        latency_text = f"{latency:.0f} ms"
    else:
        latency_text = "n/a"

    evidence_supported = trust.get("evidence_supported")
    if evidence_supported is True:
        evidence_text = "Evidence: supported"
        evidence_class = "trust-high"
    elif evidence_supported is False:
        evidence_text = "Evidence: limited"
        evidence_class = "trust-medium"
    else:
        evidence_text = "Evidence: n/a"
        evidence_class = ""

    st.markdown(
        f"""
        <div class="trust-row">
            <span class="trust-badge {confidence_class(level)}">
                Confidence: {escape(level.title())} · {escape(str(score))}
            </span>
            <span class="trust-badge {evidence_class}">
                {escape(evidence_text)}
            </span>
            <span class="trust-badge">
                Mode: {escape(str(answer_mode))}
            </span>
            <span class="trust-badge">
                Strategy: {escape(str(strategy))}
            </span>
            <span class="trust-badge">
                Latency: {escape(latency_text)}
            </span>
            <span class="trust-badge {'trust-low' if refusal else 'trust-high'}">
                Refused: {"yes" if refusal else "no"}
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    explanation = confidence.get("explanation")
    if explanation:
        st.caption(str(explanation))


def render_citations(response: Dict[str, Any]) -> None:
    citations = response.get("citations", []) or []

    if not citations:
        st.info("No citations returned.")
        return

    for idx, citation in enumerate(citations, start=1):
        doc_id = citation.get("doc_id", "unknown")
        section = citation.get("section", "unknown")
        used_for = citation.get("used_for", "supporting evidence")

        st.markdown(
            f"""
            <div class="source-card">
                <div class="source-title">Source {idx}: {escape(str(doc_id))}</div>
                <div class="source-section">{escape(str(section))}</div>
                <div class="small-muted">Used for: {escape(str(used_for))}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_evidence_context(response: Dict[str, Any]) -> None:
    meta = response.get("meta", {}) or {}
    evidence_context = meta.get("evidence_context", []) or []

    if not evidence_context:
        st.info("No evidence context returned.")
        return

    for item in evidence_context[:6]:
        doc_id = item.get("doc_id", "unknown")
        section = item.get("section", "unknown")
        text = str(item.get("text", ""))[:700]

        st.markdown(
            f"""
            <div class="source-card">
                <div class="source-title">{escape(str(doc_id))}</div>
                <div class="source-section">{escape(str(section))}</div>
                <p>{escape(text)}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_raw(response: Dict[str, Any]) -> None:
    st.code(json.dumps(response, indent=2, ensure_ascii=False), language="json")


def render_trace(response: Dict[str, Any]) -> None:
    st.json(trace_summary(response), expanded=False)


def save_feedback(
    *,
    query: str,
    response: Dict[str, Any],
    rating: str,
    reason: Optional[str] = None,
) -> None:
    FEEDBACK_PATH.parent.mkdir(parents=True, exist_ok=True)

    meta = response.get("meta", {}) or {}
    confidence = response.get("confidence", {}) or {}

    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "query": query,
        "rating": rating,
        "reason": reason,
        "answer_mode": meta.get("answer_mode"),
        "strategy": meta.get("strategy"),
        "confidence_level": confidence.get("level"),
        "confidence_score": confidence.get("score"),
        "refused": bool(response.get("refusal")),
        "latency_ms": meta.get("latency_ms"),
    }

    with FEEDBACK_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def render_feedback_controls(
    *,
    query: str,
    response: Dict[str, Any],
    message_index: int,
) -> None:
    cols = st.columns([1.1, 1.3, 1.0, 5])

    with cols[0]:
        if st.button("👍 Helpful", key=f"helpful_{message_index}"):
            save_feedback(query=query, response=response, rating="helpful")
            st.toast("Feedback saved.")

    with cols[1]:
        if st.button("👎 Not helpful", key=f"not_helpful_{message_index}"):
            save_feedback(query=query, response=response, rating="not_helpful")
            st.toast("Feedback saved.")

    with cols[2]:
        if st.button("💾 Save", key=f"save_{message_index}"):
            save_feedback(query=query, response=response, rating="saved")
            st.toast("Saved.")


def render_assistant_message(
    response: Dict[str, Any],
    *,
    query: str = "",
    message_index: int = 0,
) -> None:
    text = answer_text(response) or "I could not produce a reliable answer for that question."
    safe_text = escape(text).replace("\n", "<br>")

    st.markdown(
        f"""
        <div class="answer-card">
            {safe_text}
        </div>
        """,
        unsafe_allow_html=True,
    )

    render_trust_badges(response)

    citations = response.get("citations", []) or []
    if citations:
        with st.expander(f"Sources ({len(citations)})", expanded=False):
            render_citations(response)

    with st.expander("Trust signals", expanded=False):
        render_trace(response)

    if show_source_snippets_enabled():
        with st.expander("Evidence snippets", expanded=False):
            render_evidence_context(response)

    if show_debug_enabled():
        with st.expander("Raw JSON", expanded=False):
            render_raw(response)

    if query:
        render_feedback_controls(
            query=query,
            response=response,
            message_index=message_index,
        )


def validate_prompt(prompt: str) -> Optional[str]:
    if not prompt or not prompt.strip():
        return "Please enter a question."

    query_limit = max_query_chars()
    if len(prompt) > query_limit:
        return f"Please keep your question under {query_limit} characters."

    if st.session_state["turn_count"] >= max_turns_per_session():
        return "This session has reached the public turn limit. Start a new chat to continue."

    return None


def normalize_prompt(prompt: str) -> str:
    return " ".join(prompt.strip().split())


def render_sidebar() -> None:
    with st.sidebar:
        st.markdown("## Nexus RAG")
        st.caption("Domain-specific AI/ML interview preparation.")

        if st.button("➕ New chat", use_container_width=True):
            st.session_state["messages"] = []
            st.session_state["pending_prompt"] = None
            st.session_state["turn_count"] = 0
            st.session_state["chat_title"] = "New interview session"
            st.rerun()

        st.divider()

        st.markdown("### Best for")
        st.markdown(
            """
            - RAG architecture
            - embeddings and vector search
            - reranking and evaluation
            - LLM safety and alignment
            - production readiness
            """
        )

        st.divider()

        with st.expander("Advanced"):
            st.caption("Use after changing secrets or deploying new code.")

            if st.button("Reload pipeline", use_container_width=True):
                get_pipeline.clear()
                run_cached_query.clear()
                st.session_state["messages"] = []
                st.session_state["turn_count"] = 0
                st.success("Pipeline cache cleared.")
                st.rerun()

            st.markdown("#### Runtime")
            st.write(f"Debug mode: `{show_debug_enabled()}`")
            st.write(f"Source snippets: `{show_source_snippets_enabled()}`")
            st.write(f"Max query chars: `{max_query_chars()}`")
            st.write(f"Max turns/session: `{max_turns_per_session()}`")


def render_landing() -> None:
    st.markdown(
        f"""
        <div class="nexus-hero">
            <h1>{escape(APP_NAME)}</h1>
            <p>{escape(APP_SUBTITLE)}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="nexus-info-card">
            Practice real AI engineer interview questions. Answers are grounded in the curated corpus,
            with sources and trust signals available under each response.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="section-label">Try a prompt</div>', unsafe_allow_html=True)

    cols = st.columns(2)
    for idx, prompt in enumerate(EXAMPLE_PROMPTS):
        with cols[idx % 2]:
            if st.button(prompt, key=f"example_{idx}", use_container_width=True):
                st.session_state["pending_prompt"] = prompt
                st.rerun()


def render_chat_history() -> None:
    for idx, message in enumerate(st.session_state["messages"]):
        role = message.get("role")

        with st.chat_message(role):
            if role == "assistant" and isinstance(message.get("response"), dict):
                render_assistant_message(
                    message["response"],
                    query=message.get("query", ""),
                    message_index=idx,
                )
            else:
                st.markdown(escape(str(message.get("content", ""))))


def render_followup_chips() -> None:
    user_messages = [
        m.get("content", "")
        for m in st.session_state["messages"]
        if m.get("role") == "user"
    ]

    if not user_messages:
        return

    last_query = user_messages[-1]

    st.caption("Continue practicing:")
    cols = st.columns(len(FOLLOW_UP_PROMPTS))

    for idx, label in enumerate(FOLLOW_UP_PROMPTS):
        with cols[idx]:
            if st.button(label, key=f"followup_{idx}", use_container_width=True):
                st.session_state["pending_prompt"] = f"{label}: {last_query}"
                st.rerun()


def run_query(prompt: str) -> None:
    raw_prompt = prompt
    error = validate_prompt(raw_prompt)
    if error:
        st.warning(error)
        return

    prompt = normalize_prompt(raw_prompt)

    st.session_state["messages"].append(
        {
            "role": "user",
            "content": raw_prompt,
            "created_at": time.time(),
        }
    )

    with st.chat_message("user"):
        st.markdown(escape(raw_prompt))

    with st.chat_message("assistant"):
        try:
            request_start = time.perf_counter()
            with st.spinner("Retrieving evidence and generating grounded answer..."):
                response = copy.deepcopy(run_cached_query(prompt))
            request_latency_ms = round((time.perf_counter() - request_start) * 1000, 2)
            meta = response.setdefault("meta", {})
            if "latency_ms" in meta:
                meta.setdefault("pipeline_latency_ms", meta.get("latency_ms"))
            meta["latency_ms"] = request_latency_ms

        except Exception as exc:
            response = {
                "answer": {
                    "text": (
                        "I could not process that request because the app backend "
                        "is temporarily unavailable. Please try again."
                    )
                },
                "confidence": {
                    "level": "low",
                    "score": 0.0,
                    "explanation": "Temporary application error.",
                },
                "citations": [],
                "trust_signals": {
                    "evidence_supported": False,
                    "multi_source": False,
                    "contradiction_checked": False,
                    "retrieval_agreement": "weak",
                    "attribution_quality": "weak",
                },
                "refusal": None,
                "meta": {
                    "answer_mode": "application_error",
                    "strategy": "none",
                    "latency_ms": None,
                    "error_type": exc.__class__.__name__,
                },
            }

        render_assistant_message(
            response,
            query=prompt,
            message_index=len(st.session_state["messages"]),
        )

        if show_debug_enabled():
            with st.expander("Technical exception", expanded=False):
                st.code(str(exc) if "exc" in locals() else "No exception.")

    st.session_state["messages"].append(
        {
            "role": "assistant",
            "content": answer_text(response),
            "query": raw_prompt,
            "response": response,
            "created_at": time.time(),
        }
    )

    st.session_state["turn_count"] += 1

    if st.session_state["chat_title"] == "New interview session":
        st.session_state["chat_title"] = prompt[:56]


def main() -> None:
    init_page()
    load_streamlit_secrets_to_env()
    init_state()

    missing = missing_required_config()
    if missing:
        st.error(
            "The app is missing required configuration: "
            + ", ".join(missing)
            + ". Add these values in Streamlit secrets or environment variables."
        )
        st.stop()

    render_sidebar()

    if not st.session_state["messages"]:
        render_landing()
    else:
        st.markdown(f"### {escape(st.session_state['chat_title'])}")
        render_chat_history()
        render_followup_chips()

    pending_prompt = st.session_state.pop("pending_prompt", None)
    typed_prompt = st.chat_input("Ask an AI/ML interview question...")

    prompt = typed_prompt or pending_prompt

    if prompt:
        run_query(prompt)


if __name__ == "__main__":
    main()
