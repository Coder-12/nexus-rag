# Nexus RAG

> Intelligent Self-Improving RAG System with Agentic Routing and Continuous Learning


**Status**: 🚧 Active Development 
**Version**: 0.1.0  
**Author**: Aklesh Mishra  
**Started**: December 2025

## Overview

Nexus RAG is a cutting-edge Retrieval-Augmented Generation system featuring:
- 🧠 Intelligent agentic routing
- 🔄 Multi-strategy retrieval (vector + BM25 + graph)
- 📈 Continuous learning from user feedback
- 🏭 Production-grade error handling and monitoring

## Project Classification

**Industry Position**: Top 10-15% of production RAG systems (Dec 2025)

**Core Differentiators**:
- ML-based query routing (not rule-based)
- Multi-strategy parallel retrieval with intelligent fusion
- Closed-loop continuous improvement
- Production observability and A/B testing

## Quick Start

### Prerequisites
- Python 3.9+
- OpenAI API key
- Pinecone account

### Installation
```bash
# Clone repository
git clone https://github.com/yourusername/nexus-rag.git
cd nexus-rag

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your API keys
```

### Usage
```bash
# Run API server
uvicorn src.api:app --reload

# Run Streamlit UI (separate terminal)
streamlit run src/app.py
```

## Architecture
```
Query → Analysis → Routing → Multi-Strategy Retrieval → 
Fusion → Reranking → Generation → Response + Citations
```

## Development Roadmap

- [x] Week 1: Foundation + Basic RAG
- [ ] Week 2: Multi-Strategy + Intelligent Routing
- [ ] Week 3: Continuous Learning + Production Polish
- [ ] Week 4: Advanced Features (Optional)

## Technology Stack

- **Framework**: LangChain
- **LLM**: GPT-4 Turbo / Claude Sonnet 4
- **Embeddings**: OpenAI text-embedding-3-large
- **Vector DB**: Pinecone
- **API**: FastAPI
- **UI**: Streamlit
- **Monitoring**: LangSmith + Prometheus

## Project Structure
```
nexus-rag/
├── src/              # Source code
├── data/             # Documents and datasets
├── tests/            # Test suite
├── notebooks/        # Experiments
├── docs/             # Documentation
└── logs/             # Application logs
```

## Contributing

This is a portfolio project. Contributions welcome after v1.0 release.

## License

MIT License

## Contact

Aklesh Mishra - akleshmishra7@gmail.com  
Portfolio: https://github.com/Coder-12
LinkedIn: https://linkedin.com/in/akleshmishra

---