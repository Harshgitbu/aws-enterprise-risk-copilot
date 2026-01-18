#!/bin/bash
echo "Testing AWS Risk Copilot..."
echo ""

echo "1. Health check:"
curl -s http://localhost:8000/health | grep -E "status|memory_mb"

echo ""
echo "2. SEC stats:"
curl -s http://localhost:8000/sec/stats | grep -E "status|total_entries"

echo ""
echo "3. Production stats:"
curl -s http://localhost:8000/production/stats | grep -E "status|loaded|total_companies"

echo ""
echo "4. Dashboard:"
curl -s -o /dev/null -w "Status: %{http_code}\n" http://localhost:8501

echo ""
echo "✅ System test complete"
