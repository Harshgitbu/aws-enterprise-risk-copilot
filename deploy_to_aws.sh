#!/bin/bash

echo "🚀 AWS Risk Copilot Deployment Script"
echo "📅 Day 4: Complete Backend Stack"
echo ""

# Check for required files
echo "📋 Checking project structure..."
required_files=("src/backend/main.py" "src/rag/vector_store.py" "src/llm/gemini_client.py" "Dockerfile.backend" "docker-compose.full.yml")
for file in "${required_files[@]}"; do
    if [ -f "$file" ]; then
        echo "✅ $file"
    else
        echo "❌ Missing: $file"
        exit 1
    fi
done

echo ""
echo "🔑 Checking environment variables..."
if [ -z "$GOOGLE_API_KEY" ]; then
    echo "⚠️  GOOGLE_API_KEY not set in environment"
    read -p "Enter Google API Key: " GOOGLE_API_KEY
    export GOOGLE_API_KEY
fi

if [ -z "$HUGGINGFACE_TOKEN" ]; then
    echo "⚠️  HUGGINGFACE_TOKEN not set in environment"
    read -p "Enter HuggingFace Token: " HUGGINGFACE_TOKEN
    export HUGGINGFACE_TOKEN
fi

echo ""
echo "🐳 Building Docker images..."
docker-compose -f docker-compose.full.yml build

echo ""
echo "📊 Resource Summary:"
echo "   - Backend: 800MB memory limit"
echo "   - Redis: 50MB memory limit"
echo "   - PostgreSQL: 100MB (approx)"
echo "   - Total: ~950MB (within 1GB EC2 limit)"
echo "   - Ports: 8000 (API), 8501 (Frontend)"
echo ""
echo "🚀 To deploy on AWS EC2:"
echo "   1. Copy this project to EC2 instance"
echo "   2. Run: ./deploy_to_aws.sh"
echo "   3. Access API: http://<ec2-public-ip>:8000"
echo "   4. API Docs: http://<ec2-public-ip>:8000/docs"
echo ""
echo "📝 Sample curl commands:"
echo '  # Health check'
echo '  curl http://localhost:8000/health'
echo ''
echo '  # Risk analysis'
echo '  curl -X POST http://localhost:8000/analyze-risk \'
echo '    -H "Content-Type: application/json" \'
echo '    -d '\''{"query": "AWS security risks", "top_k": 3}'\'''
echo ''
echo '  # Memory stats'
echo '  curl http://localhost:8000/memory'
echo ''
echo "✅ Day 4 deployment ready!"
