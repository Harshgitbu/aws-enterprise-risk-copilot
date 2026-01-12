#!/bin/bash

# AWS Risk Copilot - EC2 Deployment Script
# Deploys Days 1-4 to EC2 t3.micro (1GB RAM)

set -e  # Exit on error

echo "🚀 AWS Risk Copilot - EC2 Deployment"
echo "📅 Days 1-4: Complete Stack"
echo ""

# Configuration
EC2_USER="ubuntu"
EC2_IP="107.23.177.65"  
GITHUB_REPO="https://github.com/Harshgitbu/aws-enterprise-risk-copilot.git"
APP_PORT="8000"

# Check for EC2 IP
if [ -z "$EC2_IP" ]; then
    echo "❌ Please set EC2_IP in this script (line 10)"
    echo ""
    echo "To get your EC2 IP:"
    echo "1. Go to AWS Console → EC2"
    echo "2. Find your t3.micro instance"
    echo "3. Copy 'Public IPv4 address'"
    exit 1
fi

echo "📊 Deployment Configuration:"
echo "   EC2 IP: $EC2_IP"
echo "   User: $EC2_USER"
echo "   Repo: $GITHUB_REPO"
echo "   Port: $APP_PORT"
echo ""

# Check SSH access
echo "🔐 Testing SSH connection..."
ssh -o ConnectTimeout=5 -o StrictHostKeyChecking=no ${EC2_USER}@${EC2_IP} "echo '✅ SSH connection successful'" || {
    echo "❌ SSH failed. Check:"
    echo "   1. EC2 instance is running"
    echo "   2. Security group allows SSH (port 22)"
    echo "   3. You have the correct .pem key"
    exit 1
}

echo ""
echo "📦 Installing dependencies on EC2..."
ssh ${EC2_USER}@${EC2_IP} << 'SSHCOMMANDS'
    set -e
    
    echo "1. Updating system..."
    sudo apt-get update -y
    
    echo "2. Installing Docker and Docker Compose..."
    sudo apt-get install -y docker.io docker-compose
    
    echo "3. Adding user to docker group..."
    sudo usermod -aG docker $USER
    
    echo "4. Installing Git..."
    sudo apt-get install -y git
    
    echo "5. Installing Python (for scripts)..."
    sudo apt-get install -y python3 python3-pip python3-venv
    
    echo "✅ Dependencies installed"
SSHCOMMANDS

echo ""
echo "📥 Cloning repository..."
ssh ${EC2_USER}@${EC2_IP} << 'SSHCOMMANDS'
    set -e
    
    echo "Cloning repo..."
    if [ -d "aws-risk-copilot" ]; then
        echo "Repository exists, pulling updates..."
        cd aws-risk-copilot
        git pull origin main
    else
        git clone https://github.com/Harshgitbu/aws-enterprise-risk-copilot.git
        cd aws-risk-copilot
    fi
    
    echo "✅ Repository cloned/updated"
SSHCOMMANDS

echo ""
echo "⚙️  Setting up environment..."
ssh ${EC2_USER}@${EC2_IP} << 'SSHCOMMANDS'
    set -e
    cd aws-risk-copilot
    
    echo "1. Creating .env file from example..."
    if [ ! -f .env ]; then
        cp .env.example .env
        echo "⚠️  Please edit .env file with your API keys:"
        echo "   - GOOGLE_API_KEY"
        echo "   - HUGGINGFACE_TOKEN"
        echo ""
        echo "Run: nano .env"
    else
        echo "✅ .env file already exists"
    fi
    
    echo "2. Creating necessary directories..."
    mkdir -p documents data/vector_store data/cache
    
    echo "3. Setting permissions..."
    chmod +x *.sh 2>/dev/null || true
    
    echo "✅ Environment setup complete"
SSHCOMMANDS

echo ""
echo "🐳 Building Docker containers..."
ssh ${EC2_USER}@${EC2_IP} << 'SSHCOMMANDS'
    set -e
    cd aws-risk-copilot
    
    echo "Building Docker images (this may take a few minutes)..."
    docker-compose -f docker-compose.full.yml build --no-cache
    
    echo "✅ Docker build complete"
SSHCOMMANDS

echo ""
echo "🔧 Configuring firewall..."
ssh ${EC2_USER}@${EC2_IP} << 'SSHCOMMANDS'
    set -e
    
    echo "Opening ports in firewall..."
    # Allow API port
    sudo ufw allow 8000/tcp comment "Risk Copilot API"
    
    # Allow frontend port (for Day 7)
    sudo ufw allow 8501/tcp comment "Risk Copilot Frontend"
    
    # Enable firewall if not enabled
    echo "y" | sudo ufw enable 2>/dev/null || true
    
    echo "✅ Firewall configured"
SSHCOMMANDS

echo ""
echo "🚀 Starting application..."
ssh ${EC2_USER}@${EC2_IP} << 'SSHCOMMANDS'
    set -e
    cd aws-risk-copilot
    
    echo "Starting containers in detached mode..."
    docker-compose -f docker-compose.full.yml up -d
    
    echo "Waiting for services to start..."
    sleep 10
    
    echo "Checking container status..."
    docker-compose -f docker-compose.full.yml ps
    
    echo "✅ Application started"
SSHCOMMANDS

echo ""
echo "🧪 Testing deployment..."
ssh ${EC2_USER}@${EC2_IP} << 'SSHCOMMANDS'
    set -e
    cd aws-risk-copilot
    
    echo "1. Testing API health..."
    curl -f http://localhost:8000/health || {
        echo "❌ Health check failed"
        docker-compose -f docker-compose.full.yml logs backend
        exit 1
    }
    
    echo "2. Testing risk analysis..."
    curl -X POST http://localhost:8000/analyze-risk \
        -H "Content-Type: application/json" \
        -d '{"query": "AWS security test", "top_k": 1}' \
        -s | grep -q "error" || echo "✅ Risk analysis working"
    
    echo "3. Checking memory..."
    curl -s http://localhost:8000/memory | python3 -c "
import json, sys
data = json.load(sys.stdin)
mem = data.get('rss_mb', 0)
print(f'   Memory usage: {mem}MB')
if mem < 1024:
    print('   ✅ Within 1GB limit')
else:
    print('   ⚠️  Above 1GB limit')
"
    
    echo "✅ Deployment tests passed"
SSHCOMMANDS

echo ""
echo "🎉 DEPLOYMENT SUCCESSFUL!"
echo ""
echo "🌐 Your Risk Copilot is now running at:"
echo "   API: http://${EC2_IP}:8000"
echo "   Docs: http://${EC2_IP}:8000/docs"
echo "   Health: http://${EC2_IP}:8000/health"
echo ""
echo "📋 Quick test commands:"
echo "   curl http://${EC2_IP}:8000/health"
echo "   curl -X POST http://${EC2_IP}:8000/analyze-risk \\"
echo "     -H 'Content-Type: application/json' \\"
echo "     -d '{\"query\":\"AWS risks\",\"top_k\":2}'"
echo ""
echo "📝 To view logs:"
echo "   ssh ${EC2_USER}@${EC2_IP}"
echo "   cd aws-risk-copilot"
echo "   docker-compose -f docker-compose.full.yml logs -f"
echo ""
echo "🛑 To stop:"
echo "   docker-compose -f docker-compose.full.yml down"
