# 🚀 AWS Enterprise AI Risk Intelligence Copilot

## 📌 Elevator Pitch
An AI risk intelligence system built **entirely within AWS Free Tier constraints (1GB RAM)** that processes enterprise signals, implements RAG with external LLM APIs, and demonstrates production-ready optimization techniques.

## 🎯 Key Differentiators
- **RAG Implementation**: Evidence-based explanations using FAISS vector search
- **External LLM APIs**: Google Gemini + Hugging Face instead of local models
- **Memory Optimization**: Techniques for 1GB RAM constraint (quantization, caching, streaming)
- **AWS Cloud Skills**: First AWS deployment with cost optimization
- **Production MLOps**: Despite free tier, includes monitoring, CI/CD, containerization

## 🏗️ Tech Stack (All AWS Free Tier Compatible)
| Component | Technology | Why Chosen |
|-----------|------------|------------|
| **Compute** | EC2 t3.micro (1GB RAM) | Free Tier, memory constraint challenge |
| **Vector DB** | FAISS (in-memory) | Lightweight, efficient for 1GB RAM |
| **LLM** | Google Gemini API + Hugging Face | Free tiers available |
| **Backend** | FastAPI + async/await | Memory efficient, fast |
| **Frontend** | Streamlit | Lightweight, Python-native |
| **Database** | PostgreSQL on RDS (db.t3.micro) | Free Tier, relational data |
| **Cache** | Redis Streams | Lightweight real-time processing |

## 📊 7-Day Implementation Plan
- **Day 1**: AWS Free Tier Setup & Architecture ✓
- **Day 2**: Memory-Efficient Vector Search Setup (FAISS)
- **Day 3**: External LLM API Integration
- **Day 4**: Constraint-Aware RAG Pipeline
- **Day 5**: Real-time Features within 1GB RAM
- **Day 6**: AWS Deployment & Cost Optimization
- **Day 7**: Dashboard & Production Polish

## 🚀 Quick Start
```bash
# 1. Clone repository
git clone https://github.com/YOUR_USERNAME/aws-enterprise-risk-copilot.git
cd aws-enterprise-risk-copilot

# 2. Setup environment
cp .env.example .env
# Edit .env with your credentials

# 3. Run with Docker
docker-compose up --build

## Operational Smoke Checks
```bash
# Backend health
curl -fsS http://localhost:8000/health

# Frontend health
curl -fsS http://localhost:8501/_stcore/health

# AI endpoint (Gemini if configured, degraded fallback otherwise)
curl -sS -X POST http://localhost:8000/ai/copilot/advanced \
  -H "Content-Type: application/json" \
  -d '{"query":"Analyze Apple cybersecurity risk"}'
```

## EC2 Docker Disk Cleanup
Run this when EC2 disk usage is high after repeated image builds.
```bash
chmod +x scripts/ec2_docker_cleanup.sh
./scripts/ec2_docker_cleanup.sh
```

Recommended cadence:
- Weekly on dev EC2 boxes
- After major rebuilds
- Before large deploy tests

## Render Deployment Configuration
- This repo now includes `render.yaml` for service definitions.
- Set sensitive values in Render dashboard env vars (never commit secrets):
  - `GOOGLE_API_KEY`
  - `NEWSAPI_KEY`
  - `FINNHUB_API_KEY`
  - `SEC_EDGAR_EMAIL`
