#!/bin/bash

echo "🚀 AWS Risk Copilot - EC2 State Backup"
echo "======================================="
echo ""
echo "Backup started at: $(date)"
echo "EC2 Instance: $(curl -s http://169.254.169.254/latest/meta-data/instance-id)"
echo "Public IP: $(curl -s http://169.254.169.254/latest/meta-data/public-ipv4)"
echo ""

# Create backup directory with timestamp
BACKUP_DIR="/home/ubuntu/ec2-backup-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$BACKUP_DIR"

echo "📁 Creating backup in: $BACKUP_DIR"
echo ""

# 1. Backup project code
echo "1. Backing up project code..."
cp -r /home/ubuntu/aws-enterprise-risk-copilot "$BACKUP_DIR/project-code"

# 2. Backup Docker state
echo "2. Backing up Docker state..."
docker ps -a > "$BACKUP_DIR/docker-ps.txt"
docker images > "$BACKUP_DIR/docker-images.txt"
docker volume ls > "$BACKUP_DIR/docker-volumes.txt"

# 3. Backup environment files (without exposing keys)
echo "3. Backing up environment configuration..."
if [ -f "/home/ubuntu/aws-enterprise-risk-copilot/.env" ]; then
    # Create a safe version without actual keys
    grep -E "^(#|$)" /home/ubuntu/aws-enterprise-risk-copilot/.env > "$BACKUP_DIR/env-template.txt"
    echo "" >> "$BACKUP_DIR/env-template.txt"
    echo "# ACTUAL KEYS REMOVED FOR SECURITY" >> "$BACKUP_DIR/env-template.txt"
    echo "# Backup contains only template structure" >> "$BACKUP_DIR/env-template.txt"
fi

# 4. Backup data directories (cache, etc.)
echo "4. Backing up data directories..."
if [ -d "/home/ubuntu/aws-enterprise-risk-copilot/data" ]; then
    cp -r /home/ubuntu/aws-enterprise-risk-copilot/data "$BACKUP_DIR/data-backup"
fi

# 5. Backup service logs
echo "5. Backing up service logs..."
docker logs risk-copilot-backend > "$BACKUP_DIR/logs-backend.txt" 2>/dev/null || true
docker logs risk-copilot-frontend > "$BACKUP_DIR/logs-frontend.txt" 2>/dev/null || true
docker logs risk-copilot-redis > "$BACKUP_DIR/logs-redis.txt" 2>/dev/null || true

# 6. Backup system info
echo "6. Backing up system information..."
df -h > "$BACKUP_DIR/disk-usage.txt"
free -h > "$BACKUP_DIR/memory-usage.txt"
docker system df > "$BACKUP_DIR/docker-disk.txt"

# 7. Create a restore script
echo "7. Creating restore script..."
cat > "$BACKUP_DIR/restore-ec2.sh" << 'RESTORE_EOF'
#!/bin/bash

echo "🔧 EC2 State Restore Script"
echo "==========================="
echo ""
echo "This script helps restore your EC2 state from backup."
echo ""
echo "Steps to restore:"
echo "1. Copy backup directory to EC2: scp -r backup-dir/ ubuntu@ec2-ip:/home/ubuntu/"
echo "2. Run: chmod +x restore-ec2.sh"
echo "3. Run: ./restore-ec2.sh"
echo ""
echo "Note: API keys need to be re-added to .env file"
echo ""

if [ -d "project-code" ]; then
    echo "To restore project code:"
    echo "  cp -r project-code /home/ubuntu/aws-enterprise-risk-copilot"
    echo ""
fi

if [ -d "data-backup" ]; then
    echo "To restore data:"
    echo "  cp -r data-backup /home/ubuntu/aws-enterprise-risk-copilot/data"
    echo ""
fi

echo "To restart services:"
echo "  cd /home/ubuntu/aws-enterprise-risk-copilot"
echo "  docker-compose down"
echo "  docker-compose up -d"
echo ""
echo "To check services:"
echo "  docker-compose ps"
echo "  curl http://localhost:8000/health"
echo ""
echo "✅ Restore instructions generated"
RESTORE_EOF

chmod +x "$BACKUP_DIR/restore-ec2.sh"

# 8. Create README with current state
echo "8. Documenting current state..."
cat > "$BACKUP_DIR/README.md" << 'README_EOF'
# AWS Risk Copilot - EC2 State Backup

## 📅 Backup Information
- **Date**: $(date)
- **EC2 Instance**: $(curl -s http://169.254.169.254/latest/meta-data/instance-id)
- **Public IP**: $(curl -s http://169.254.169.254/latest/meta-data/public-ipv4)
- **Region**: $(curl -s http://169.254.169.254/latest/meta-data/placement/region)

## 🏗️ Project State
- **Repository**: https://github.com/Harshgitbu/aws-enterprise-risk-copilot
- **Render Deployment**: https://risk-copilot-frontend.onrender.com
- **Local Dashboard**: http://$(curl -s http://169.254.169.254/latest/meta-data/public-ipv4):8501
- **Local API**: http://$(curl -s http://169.254.169.254/latest/meta-data/public-ipv4):8000

## 📊 Current Services
$(docker-compose ps)

## 🎯 What's Working
- ✅ Backend API (FastAPI)
- ✅ Frontend Dashboard (Streamlit)
- ✅ Redis Cache
- ✅ Gemini AI Integration
- ✅ News API Integration
- ✅ Company Search
- ✅ Risk Analysis

## 🔧 How to Use This Backup
1. **Emergency Restore**: If EC2 instance is lost
2. **Development**: Reference for local setup
3. **Documentation**: Current working state

## 📁 Backup Contents
- `project-code/` - Complete project source
- `data-backup/` - Cached data and files
- `*.txt` - Logs and system information
- `restore-ec2.sh` - Restoration script

## ⚠️ Important Notes
- API keys are NOT included in backup
- Docker images need to be rebuilt
- Environment variables must be re-configured
README_EOF

# 9. Create archive
echo "9. Creating compressed archive..."
cd "$BACKUP_DIR/.."
tar -czf "$(basename $BACKUP_DIR).tar.gz" "$(basename $BACKUP_DIR)"
rm -rf "$BACKUP_DIR"

echo ""
echo "✅ BACKUP COMPLETE!"
echo ""
echo "📦 Backup archive created: $(basename $BACKUP_DIR).tar.gz"
echo "📁 Size: $(du -h "$(basename $BACKUP_DIR).tar.gz" | cut -f1)"
echo ""
echo "🔒 To download backup to your local machine:"
echo "   scp ubuntu@$(curl -s http://169.254.169.254/latest/meta-data/public-ipv4):$(basename $BACKUP_DIR).tar.gz ./"
echo ""
echo "💾 To backup to S3 (optional):"
echo "   aws s3 cp $(basename $BACKUP_DIR).tar.gz s3://your-bucket/backups/"
echo ""
echo "🔄 Current services status:"
docker-compose ps
