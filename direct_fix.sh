#!/bin/bash
echo "🔧 DIRECT FIX IN CONTAINER"

# Backup current rag_integration.py
sudo docker exec risk-copilot-backend cp /app/src/backend/rag_integration.py /app/src/backend/rag_integration.py.backup

# Add get_rag_pipeline function
sudo docker exec risk-copilot-backend bash -c '
cd /app/src/backend

# Check if function already exists
if ! grep -q "def get_rag_pipeline" rag_integration.py; then
    echo "Adding get_rag_pipeline function..."
    
    # Read file and add function
    python3 -c "
import sys

with open(\"rag_integration.py\", \"r\") as f:
    lines = f.readlines()

# Find rag_pipeline instance
for i, line in enumerate(lines):
    if \"rag_pipeline = RAGPipeline()\" in line:
        # Insert get_rag_pipeline function after
        lines.insert(i+1, \"\\n\")
        lines.insert(i+2, \"def get_rag_pipeline():\\n\")
        lines.insert(i+3, \"    \\\"\\\"\\\"Get the global RAG pipeline instance\\\"\\\"\\\"\\n\")
        lines.insert(i+4, \"    return rag_pipeline\\n\")
        break

with open(\"rag_integration.py\", \"w\") as f:
    f.writelines(lines)
    
print(\"✅ Added get_rag_pipeline function\")
"
else
    echo "get_rag_pipeline function already exists"
fi

# Verify
echo "Verifying fix..."
python3 -c "
try:
    from rag_integration import get_rag_pipeline
    print(\"✅ get_rag_pipeline import successful\")
    
    # Test it
    pipeline = get_rag_pipeline()
    print(f\"✅ get_rag_pipeline() returned: {type(pipeline)}\")
except Exception as e:
    print(f\"❌ Import failed: {e}\")
    import traceback
    traceback.print_exc()
"
'

# Clean cache
sudo docker exec risk-copilot-backend find /app/src -name "*.pyc" -delete
sudo docker exec risk-copilot-backend find /app/src -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true

# Restart
sudo docker restart risk-copilot-backend
sleep 5
sudo docker logs risk-copilot-backend --tail 10
