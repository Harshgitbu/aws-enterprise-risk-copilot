#!/bin/bash
echo "🔧 AWS RISK COPILOT - EC2 BACKUP SCRIPT"
echo "========================================"
echo "Backup Date: $(date)"
echo ""

# Create backup directory
BACKUP_DIR="/home/ubuntu/ec2-backup"
mkdir -p $BACKUP_DIR

# Backup critical files
echo "📁 Backing up project files..."
cp -r ~/aws-enterprise-risk-copilot $BACKUP_DIR/project_backup_$(date +%Y%m%d_%H%M%S)

# Backup Docker compose files
echo "🐳 Backing up Docker configuration..."
cp docker-compose.full.yml $BACKUP_DIR/
cp Dockerfile.frontend $BACKUP_DIR/

# Backup requirements
echo "📦 Backing up requirements..."
cp requirements.txt $BACKUP_DIR/

# Backup environment variables (if exists)
if [ -f .env ]; then
    echo "🔐 Backing up environment variables..."
    cp .env $BACKUP_DIR/
fi

# Backup test scripts
echo "🧪 Backing up test scripts..."
cp test_comprehensive_day1_7.sh $BACKUP_DIR/

# Create restore script
echo "⚡ Creating restore script..."
cat > $BACKUP_DIR/restore_project.sh << 'RESTORE'
#!/bin/bash
echo "🔄 AWS RISK COPILOT - RESTORE SCRIPT"
echo "====================================="
echo ""

if [ ! -d "$BACKUP_DIR" ]; then
    echo "❌ Backup directory not found: $BACKUP_DIR"
    exit 1
fi

# Find latest backup
LATEST_BACKUP=$(ls -td $BACKUP_DIR/project_backup_* | head -1)

if [ -z "$LATEST_BACKUP" ]; then
    echo "❌ No backup found"
    exit 1
fi

echo "📁 Restoring from: $LATEST_BACKUP"

# Restore project
rm -rf ~/aws-enterprise-risk-copilot
cp -r $LATEST_BACKUP ~/aws-enterprise-risk-copilot

# Restore critical files
if [ -f "$BACKUP_DIR/.env" ]; then
    cp $BACKUP_DIR/.env ~/aws-enterprise-risk-copilot/
fi

cp $BACKUP_DIR/docker-compose.full.yml ~/aws-enterprise-risk-copilot/
cp $BACKUP_DIR/Dockerfile.frontend ~/aws-enterprise-risk-copilot/
cp $BACKUP_DIR/requirements.txt ~/aws-enterprise-risk-copilot/
cp $BACKUP_DIR/test_comprehensive_day1_7.sh ~/aws-enterprise-risk-copilot/

echo "✅ Restoration complete!"
echo ""
echo "To start the project:"
echo "cd ~/aws-enterprise-risk-copilot"
echo "sudo docker-compose -f docker-compose.full.yml up -d --build"
echo ""
echo "To test:"
echo "./test_comprehensive_day1_7.sh"
RESTORE

chmod +x $BACKUP_DIR/restore_project.sh

# Create quick status script
cat > $BACKUP_DIR/check_status.sh << 'STATUS'
#!/bin/bash
echo "📊 AWS RISK COPILOT - QUICK STATUS"
echo "==================================="
echo ""

# Check Docker
echo "🐳 Docker Services:"
sudo docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"

echo ""
echo "🌐 Backend Health:"
curl -s http://localhost:8000/health | python3 -c "
import json, sys
data = json.load(sys.stdin)
print(f'Status: {data.get(\"status\")}')
print(f'Memory: {data.get(\"memory_mb\", \"N/A\")}MB')
print(f'Redis: {data.get(\"redis\")}')
"

echo ""
echo "💾 Memory Usage:"
free -m | awk 'NR==2{printf "Total: %sMB, Used: %sMB, Free: %sMB\n", $2, $3, $4}'

echo ""
echo "🔗 Access URLs:"
echo "Dashboard: http://$(curl -s ifconfig.me):8501"
echo "Backend API: http://$(curl -s ifconfig.me):8000"
STATUS

chmod +x $BACKUP_DIR/check_status.sh

# Summary
echo ""
echo "✅ BACKUP COMPLETE!"
echo "Backup location: $BACKUP_DIR"
echo ""
echo "📁 Backup Contents:"
ls -la $BACKUP_DIR/
echo ""
echo "⚡ Quick commands:"
echo "  Check status: $BACKUP_DIR/check_status.sh"
echo "  Restore: $BACKUP_DIR/restore_project.sh"
echo "  Run tests: ./test_comprehensive_day1_7.sh"
