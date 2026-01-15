# Memory Optimization Techniques for 1GB RAM

## Overview
This document details the optimization techniques used to run AWS Risk Copilot within 1GB RAM on EC2 t3.micro while targeting $0/month cost.

## Core Techniques

### 1. Lazy Loading
- **Embedding Model**: SentenceTransformer loaded only when needed
- **LLM Clients**: Gemini/HuggingFace initialized on first use
- **Database Connections**: Connection pool initialized only if RDS configured

### 2. Efficient Data Structures
- **FAISS Index**: Flat L2 index (most memory efficient)
- **Redis Cache**: In-memory caching with TTL and size limits
- **Pandas**: Use appropriate dtypes, avoid copying

### 3. Memory Monitoring & Limits
- **Real-time monitoring**: `/memory` endpoint with RSS tracking
- **Circuit Breakers**: Prevent memory leaks from API failures
- **Auto-shutdown**: Shut down during idle periods

### 4. Streaming & Chunking
- **Document Processing**: 500 character chunks with 50 overlap
- **Redis Streams**: Fixed length (1000 messages) with auto-trim
- **WebSocket**: Message queues with size limits

## Service Memory Allocation

| Service | Max Memory | Target Memory | Optimization |
|---------|------------|---------------|--------------|
| FastAPI Backend | 512MB | ~80MB | Async, lazy loading |
| Streamlit Frontend | 256MB | ~150MB | Pagination, caching |
| Redis | 128MB | ~10MB | Data persistence only |
| **Total** | **896MB** | **~240MB** | **56% buffer** |

## Cost Optimization ($0/month)

### AWS Free Tier Utilization
- **EC2 t3.micro**: 750 hours/month (use 240h = 8h/day)
- **ECR**: 500MB-month storage
- **S3**: 5GB storage
- **CloudWatch**: 10 metrics, 1M API requests
- **RDS**: 750 hours/month (optional)

### Auto-shutdown Strategy
- **Idle timeout**: 30 minutes
- **Daily limit**: 8 hours uptime
- **Monthly target**: 240 hours (well under 750h Free Tier)

## Performance Metrics
- **Current memory**: ~80MB backend + ~150MB frontend = ~230MB
- **Response time**: < 2s for RAG queries
- **Concurrent users**: 50 WebSocket connections max
- **Data retention**: 1000 messages in Redis streams

## Tools & Libraries
- `psutil` - Memory monitoring
- `async/await` - Non-blocking I/O
- `FAISS` - Efficient vector search
- `Redis` - In-memory caching
- `Streamlit` - Lightweight dashboard

## Monitoring & Alerts
1. **Memory alerts**: CloudWatch > 80% utilization
2. **Cost alerts**: > $0.50 monthly estimate
3. **Health checks**: Every 30 seconds
4. **Auto-recovery**: Circuit breakers and restarts
