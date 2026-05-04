import json
import os
from html import escape

import streamlit as st

from main import build_pipeline
from src.logging_config import setup_logging


st.set_page_config(
    page_title="Nexus RAG Interview Coach",
    page_icon="N",
    layout="wide",
    initial_sidebar_state="expanded",
)


EXAMPLE_PROMPTS = [
    "Explain RAG like I am in an AI engineer interview.",
    "What are the main components of a production RAG pipeline?",
    "How do embeddings, vector databases, and rerankers work together?",
    "Why does RAG reduce hallucination but not eliminate it?",
    "How should a RAG system handle prompt injection?",
    "What metrics prove a RAG system is ready to ship?",
]


def _load_css() -> None:
    st.markdown(
        """
        <style>
        .block-container {
            padding-top: 1.25rem;
            padding-bottom: 2rem;
            max-width: 980px;
        }
        .header-band {
            background: #0f172a;
            border: 1px solid #243044;
            border-radius: 8px;
            padding: 1.35rem 1.25rem 1.1rem 1.25rem;
            margin-bottom: 1.05rem;
        }
        .app-title {
            color: #f8fafc;
            font-size: 1.65rem;
            line-height: 1.35;
            font-weight: 700;
            letter-spacing: 0;
            margin: 0 0 0.28rem 0;
        }
        .app-subtitle {
            color: #cbd5e1;
            font-size: 0.95rem;
            line-height: 1.45;
            margin: 0;
            max-width: 920px;
        }
        .scope-note {
            border: 1px solid #d9e2f3;
            background: #f8fbff;
            color: #233047;
            border-radius: 8px;
            padding: 0.75rem 0.9rem;
            margin-bottom: 1rem;
            font-size: 0.9rem;
            line-height: 1.45;
        }
        .example-grid {
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 0.55rem;
            margin: 0.85rem 0 1.1rem 0;
        }
        .example-card {
            border: 1px solid #d9e2f3;
            border-radius: 8px;
            padding: 0.75rem 0.85rem;
            background: #ffffff;
            color: #172033;
            font-size: 0.9rem;
            line-height: 1.35;
        }
        .assistant-card {
            border: 1px solid #d9dee7;
            border-radius: 8px;
            padding: 0.85rem 0.95rem;
            background: #f8fafc;
            color: #172033;
            line-height: 1.55;
            font-size: 0.96rem;
            white-space: pre-wrap;
            overflow-wrap: anywhere;
        }
        .trace-box {
            border: 1px solid #d9dee7;
            border-radius: 8px;
            padding: 0.8rem 0.9rem;
            background: #fbfcfe;
            color: #172033;
            font-size: 0.9rem;
        }
        .small-label {
            color: #5b6472;
            font-size: 0.82rem;
            text-transform: uppercase;
            letter-spacing: 0.04em;
            margin-bottom: 0.25rem;
            font-weight: 700;
        }
        .citation {
            border-left: 3px solid #4472c4;
            padding: 0.45rem 0.7rem;
            margin-bottom: 0.5rem;
            background: #f7f9fc;
            color: #172033;
            border-radius: 4px;
            overflow-wrap: anywhere;
        }
        .citation strong {
            color: #172033;
        }
        .meta-row {
            color: #64748b;
            font-size: 0.82rem;
            margin-top: 0.45rem;
        }
        .stChatMessage {
            border-radius: 8px;
        }
        @media (max-width: 760px) {
            .example-grid {
                grid-template-columns: 1fr;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _load_streamlit_secrets_to_env() -> None:
    """
    Streamlit Community Cloud exposes secrets through st.secrets, while the
    backend pipeline reads os.environ. Copy known deployment secrets once.
    """
    for key in (
        "OPENAI_API_KEY",
        "PINECONE_API_KEY",
        "PINECONE_INDEX_NAME",
        "COHERE_API_KEY",
        "LANGSMITH_API_KEY",
        "NEXUS_SHOW_DEBUG",
    ):
        if os.getenv(key):
            continue
        try:
            value = st.secrets.get(key)
        except Exception:
            value = None
        if value:
            os.environ[key] = str(value)


def _missing_required_config() -> list[str]:
    required = ("OPENAI_API_KEY", "PINECONE_API_KEY", "PINECONE_INDEX_NAME")
    return [key for key in required if not os.getenv(key)]


@st.cache_resource(show_spinner="Initializing Nexus RAG pipeline...")
def get_pipeline():
    _load_streamlit_secrets_to_env()
    missing = _missing_required_config()
    if missing:
        raise RuntimeError(
            "Missing required deployment secrets: " + ", ".join(missing)
        )
    setup_logging()
    return build_pipeline()


def answer_text(response: dict) -> str:
    answer = response.get("answer")
    if isinstance(answer, dict):
        return str(answer.get("text", "")).strip()
    if response.get("refusal"):
        return str(response["refusal"].get("message", "")).strip()
    return ""


def trace_summary(response: dict) -> dict:
    meta = response.get("meta", {}) or {}
    query_plan = meta.get("query_plan", {}) or {}
    evidence_audit = meta.get("evidence_audit", {}) or {}
    reflexion = meta.get("reflexion", {}) or {}
    return {
        "detected_intent": query_plan.get("intent"),
        "rewritten_query": meta.get("rewritten_query"),
        "retrieval_strategy": meta.get("strategy"),
        "answer_mode": meta.get("answer_mode"),
        "reflexion_repaired": meta.get("reflexion_repaired"),
        "regeneration_reason": meta.get("regeneration_reason"),
        "critic_needs_repair": reflexion.get("needs_repair"),
        "critic_missing_elements": reflexion.get("missing_elements", []),
        "critic_unsupported_claims": reflexion.get("unsupported_claims", []),
        "evidence_sufficient": evidence_audit.get("sufficient"),
        "evidence_coverage": evidence_audit.get("coverage_score"),
        "retrieved_docs": meta.get("retrieved_doc_ids", []),
        "retrieved_chunks": meta.get("retrieved_chunk_ids", []),
        "used_chunks": meta.get("used_chunk_ids", []),
        "latency_ms": meta.get("latency_ms"),
    }


def render_answer(response: dict) -> None:
    confidence = response.get("confidence", {}) or {}
    refusal = response.get("refusal")
    meta = response.get("meta", {}) or {}

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Confidence", confidence.get("level", "unknown"), confidence.get("score"))
    m2.metric("Latency", f"{meta.get('latency_ms', 0):.0f} ms" if isinstance(meta.get("latency_ms"), (int, float)) else "n/a")
    m3.metric("Answer Mode", meta.get("answer_mode") or "n/a")
    m4.metric("Refused", "yes" if refusal else "no")

    st.markdown('<div class="small-label">Answer</div>', unsafe_allow_html=True)
    text = answer_text(response) or "No answer returned."
    escaped_text = escape(text).replace("\n", "<br>")
    if text == "No answer returned.":
        escaped_text = f'<span class="empty-answer">{escaped_text}</span>'
    st.markdown(f'<div class="answer-box">{escaped_text}</div>', unsafe_allow_html=True)

    if confidence.get("explanation"):
        st.caption(confidence["explanation"])


def render_chat_assistant_message(response: dict) -> None:
    confidence = response.get("confidence", {}) or {}
    meta = response.get("meta", {}) or {}
    text = answer_text(response) or "I could not produce a reliable answer for that question."

    st.markdown(
        f'<div class="assistant-card">{escape(text).replace(chr(10), "<br>")}</div>',
        unsafe_allow_html=True,
    )
    meta_bits = []
    if confidence.get("level"):
        meta_bits.append(f"confidence: {confidence.get('level')}")
    if isinstance(confidence.get("score"), (int, float)):
        meta_bits.append(f"score: {confidence.get('score')}")
    if meta.get("answer_mode"):
        meta_bits.append(f"mode: {meta.get('answer_mode')}")
    if isinstance(meta.get("latency_ms"), (int, float)):
        meta_bits.append(f"latency: {meta.get('latency_ms'):.0f} ms")
    if meta_bits:
        st.markdown(
            f'<div class="meta-row">{" · ".join(escape(str(bit)) for bit in meta_bits)}</div>',
            unsafe_allow_html=True,
        )

    citations = response.get("citations", []) or []
    if citations:
        with st.expander("Sources", expanded=False):
            render_citations(response)

    with st.expander("Trust signals", expanded=False):
        trace = trace_summary(response)
        compact_trace = {
            "intent": trace.get("detected_intent"),
            "answer_mode": trace.get("answer_mode"),
            "evidence_sufficient": trace.get("evidence_sufficient"),
            "evidence_coverage": trace.get("evidence_coverage"),
            "reflexion_repaired": trace.get("reflexion_repaired"),
            "retrieved_docs": trace.get("retrieved_docs"),
        }
        st.json(compact_trace, expanded=False)


def render_citations(response: dict) -> None:
    citations = response.get("citations", []) or []
    if not citations:
        st.info("No citations returned.")
        return

    for citation in citations:
        doc_id = citation.get("doc_id", "unknown")
        section = citation.get("section", "unknown")
        st.markdown(
            f'<div class="citation"><strong>{escape(str(doc_id))}</strong><br>{escape(str(section))}</div>',
            unsafe_allow_html=True,
        )


def render_trace(response: dict) -> None:
    summary = trace_summary(response)
    st.markdown('<div class="small-label">Trace Summary</div>', unsafe_allow_html=True)
    st.json(summary, expanded=True)

    meta = response.get("meta", {}) or {}
    evidence_context = meta.get("evidence_context", []) or []
    with st.expander("Evidence Context", expanded=False):
        for item in evidence_context:
            st.markdown(
                f"**{item.get('doc_id', 'unknown')}** · {item.get('section', 'unknown')}"
            )
            st.write(item.get("text", ""))


def render_raw(response: dict) -> None:
    st.code(json.dumps(response, indent=2, ensure_ascii=False), language="json")


def main() -> None:
    _load_streamlit_secrets_to_env()
    _load_css()

    st.markdown(
        """
        <div class="header-band">
            <div class="app-title">Nexus RAG Interview Coach</div>
            <div class="app-subtitle">Ask AI engineering interview questions and get grounded, source-backed answers on RAG, embeddings, vector search, LLMs, alignment, evaluation, and production readiness.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    missing_config = _missing_required_config()
    if missing_config:
        st.error(
            "The app is missing required configuration: "
            + ", ".join(missing_config)
            + ". Add these values in Streamlit secrets or environment variables."
        )
        st.stop()

    with st.sidebar:
        st.header("Nexus RAG")
        st.caption("Domain-specific AI/ML interview preparation.")
        st.markdown(
            """
            **Best for**
            - RAG architecture
            - embeddings and vector search
            - reranking and evaluation
            - LLM safety and alignment
            - production readiness
            """
        )
        st.divider()
        if st.button("New chat", use_container_width=True):
            st.session_state["messages"] = []
            st.rerun()

        with st.expander("Advanced"):
            st.caption("Use after changing secrets or deploying new code.")
            if st.button("Reload Pipeline", use_container_width=True):
                get_pipeline.clear()
                st.session_state["messages"] = []
                st.success("Pipeline cache cleared.")

    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    if not st.session_state["messages"]:
        st.markdown(
            """
            <div class="scope-note">
                Practice real AI engineer interview questions. Answers are grounded in the curated corpus, with sources and trust signals available under each response.
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown('<div class="small-label">Try a prompt</div>', unsafe_allow_html=True)
        cols = st.columns(2)
        for idx, prompt in enumerate(EXAMPLE_PROMPTS):
            with cols[idx % 2]:
                if st.button(prompt, key=f"example_{idx}", use_container_width=True):
                    st.session_state["pending_prompt"] = prompt
                    st.rerun()

    for message in st.session_state["messages"]:
        with st.chat_message(message["role"]):
            if message["role"] == "assistant" and isinstance(message.get("response"), dict):
                render_chat_assistant_message(message["response"])
                if os.getenv("NEXUS_SHOW_DEBUG", "").lower() in {"1", "true", "yes"}:
                    with st.expander("Raw JSON", expanded=False):
                        render_raw(message["response"])
            else:
                st.markdown(escape(str(message.get("content", ""))))

    prompt = st.session_state.pop("pending_prompt", None)
    typed_prompt = st.chat_input("Ask an AI/ML interview question...")
    if typed_prompt:
        prompt = typed_prompt

    if prompt:
        st.session_state["messages"].append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(escape(prompt))

        with st.chat_message("assistant"):
            try:
                pipeline = get_pipeline()
                with st.spinner("Thinking with retrieved evidence..."):
                    response = pipeline.run(prompt)
                render_chat_assistant_message(response)
                if os.getenv("NEXUS_SHOW_DEBUG", "").lower() in {"1", "true", "yes"}:
                    with st.expander("Raw JSON", expanded=False):
                        render_raw(response)
            except Exception as exc:
                response = {
                    "answer": {
                        "text": "I could not process that request because the app backend is temporarily unavailable.",
                    },
                    "confidence": {"level": "low", "score": 0.0},
                    "citations": [],
                    "refusal": None,
                    "meta": {},
                }
                render_chat_assistant_message(response)
                if os.getenv("NEXUS_SHOW_DEBUG", "").lower() in {"1", "true", "yes"}:
                    with st.expander("Technical details"):
                        st.code(str(exc))

        st.session_state["messages"].append(
            {"role": "assistant", "content": answer_text(response), "response": response}
        )


if __name__ == "__main__":
    main()
