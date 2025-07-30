# RAG-Boost Template

Minimal Retrieval-Augmented Generation (RAG) API using FastAPI, OpenAI, and FAISS (in-memory).

## Features
- Embed and store documents
- Query using semantic search
- GPT-based responses based on retrieved context
- Auto-generated OpenAPI docs

## Setup

```bash
git clone https://github.com/YOUR_USERNAME/rag-boost-template.git
cd rag-boost-template
python -m venv env
source env/bin/activate  # or env\Scripts\activate on Windows
pip install -r requirements.txt
export OPENAI_API_KEY=your-api-key
uvicorn main:app --reload
