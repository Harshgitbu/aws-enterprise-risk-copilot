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
