# RAG LangChain

A local retrieval-augmented generation (RAG) app built with **LangChain** and **Ollama** — the framework-based counterpart to [local-rag](https://github.com/veerverma828/local-rag), which implements the same RAG pipeline from scratch in plain Python (manual chunking, manual cosine similarity, manual prompt construction).

Building both versions was deliberate: [local-rag](https://github.com/veerverma828/local-rag) proves understanding of how RAG actually works under the hood, and this repo shows using the industry-standard framework to build the same thing faster and with more production-ready features (persistent vector storage via Chroma, LangChain's document loaders, etc.).

## Status

🚧 In progress — scaffolding created, implementation next.

## Planned stack

- **LangChain** — document loading, text splitting, retrieval chains
- **Chroma** (via `langchain-chroma`) — persistent vector store
- **Ollama** (via `langchain-ollama`) — local embeddings (`nomic-embed-text`) and generation (`qwen2.5:7b` or similar)
- **Streamlit** — same chat-style UI as the manual version, for direct comparison

## Setup

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
ollama pull nomic-embed-text
ollama pull qwen2.5:7b
```
