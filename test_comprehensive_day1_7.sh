#!/bin/bash
echo "🚀 AWS RISK COPILOT - COMPREHENSIVE DAY 1-7 TEST"
echo "================================================"
echo "Timestamp: $(date)"
echo ""

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Test counters
PASSED=0
FAILED=0
SKIPPED=0

# Function to run test
run_test() {
    local name="$1"
    local command="$2"
    local critical="$3"
    
    echo -n "Testing: $name... "
    
    if eval "$command" > /tmp/test_output.log 2>&1; then
        echo -e "${GREEN}✅ PASSED${NC}"
        ((PASSED++))
    else
        echo -e "${RED}❌ FAILED${NC}"
        if [ "$critical" = "critical" ]; then
            echo "CRITICAL FAILURE - Output:"
            cat /tmp/test_output.log
            exit 1
        else
            echo "Output:"
            cat /tmp/test_output.log
        fi
        ((FAILED++))
    fi
}

# Test 1: Docker services
echo "1. Docker Services Test"
run_test "Docker Compose" "sudo docker-compose -f docker-compose.full.yml ps | grep -q 'Up'"
run_test "Backend Container" "sudo docker ps | grep -q 'risk-copilot-backend'"
run_test "Frontend Container" "sudo docker ps | grep -q 'risk-copilot-frontend'"
run_test "Redis Container" "sudo docker ps | grep -q 'risk-copilot-redis'"

# Test 2: Backend API
echo ""
echo "2. Backend API Test"
run_test "Health Endpoint" "curl -s http://localhost:8000/health | grep -q 'healthy'"
run_test "Memory Endpoint" "curl -s http://localhost:8000/memory | grep -q 'memory_mb'"
run_test "Status Endpoint" "curl -s http://localhost:8000/status | grep -q 'rag_available'"

# Test 3: SEC Integration
echo ""
echo "3. SEC Integration Test"
run_test "SEC Stats" "curl -s http://localhost:8000/sec/stats | grep -q 'companies_count'"
run_test "Load SEC to RAG" "curl -s -X POST http://localhost:8000/sec/load-to-rag | grep -q 'Loaded'"

# Test 4: News Integration
echo ""
echo "4. News Integration Test"
run_test "News Stats" "curl -s http://localhost:8000/news/stats | grep -q 'news_api'"
run_test "Fetch News" "curl -s -X POST http://localhost:8000/news/fetch?max_companies=2 | grep -q 'Fetched news'"

# Test 5: Risk Analysis
echo ""
echo "5. Risk Analysis Test"
run_test "Risk Categories" "curl -s http://localhost:8000/analysis/categories | grep -q 'cybersecurity'"

# Create test data for company analysis
cat > /tmp/test_risk.json << 'TESTDATA'
{
  "company_name": "Test Company",
  "risk_text": "Cybersecurity threats and data breaches are significant risks. Competition is intense. Regulatory investigations could result in fines.",
  "ticker": "TEST"
}
TESTDATA

run_test "Company Risk Analysis" "curl -s -X POST http://localhost:8000/analyze/company-risk -H 'Content-Type: application/json' -d @/tmp/test_risk.json | grep -q 'risk_level'"

# Test 6: Frontend Dashboard
echo ""
echo "6. Frontend Dashboard Test"
run_test "Streamlit Dashboard" "curl -s http://localhost:8501 | grep -q 'Streamlit'"

# Test 7: Memory Constraint
echo ""
echo "7. Memory Constraint Test"
MEMORY_MB=$(curl -s http://localhost:8000/memory | python3 -c "import json,sys; data=json.load(sys.stdin); print(data.get('memory_mb', 0))")
if (( $(echo "$MEMORY_MB < 1024" | bc -l) )); then
    echo -e "Memory Usage: ${GREEN}${MEMORY_MB} MB ✅ (Under 1GB limit)${NC}"
    ((PASSED++))
else
    echo -e "Memory Usage: ${RED}${MEMORY_MB} MB ❌ (Over 1GB limit)${NC}"
    ((FAILED++))
fi

# Test 8: Cost Analysis
echo ""
echo "8. Cost Analysis Test"
run_test "Cost Estimate" "curl -s http://localhost:8000/cost/estimate | grep -q '\"total_cost\": 0'"

# Summary
echo ""
echo "================================================"
echo "TEST SUMMARY"
echo "================================================"
echo -e "Total Tests: $((PASSED + FAILED + SKIPPED))"
echo -e "${GREEN}Passed: $PASSED${NC}"
echo -e "${RED}Failed: $FAILED${NC}"
echo -e "${YELLOW}Skipped: $SKIPPED${NC}"

if [ $FAILED -eq 0 ]; then
    echo -e "${GREEN}🎉 ALL TESTS PASSED! System is fully operational.${NC}"
    echo ""
    echo "✅ Docker Services: Running"
    echo "✅ Backend API: 10+ endpoints working"
    echo "✅ SEC Integration: Real risk data loaded"
    echo "✅ News Integration: Real-time news working"
    echo "✅ AI Risk Analysis: Scoring working"
    echo "✅ Frontend Dashboard: Accessible"
    echo "✅ Memory Constraint: Under 1GB RAM"
    echo "✅ Cost Target: $0/month achieved"
else
    echo -e "${YELLOW}⚠️  Some tests failed. Check logs above.${NC}"
    exit 1
fi

# Final verification
echo ""
echo "Final Verification:"
echo "1. Dashboard URL: http://$(curl -s ifconfig.me):8501"
echo "2. Backend API: http://$(curl -s ifconfig.me):8000"
echo "3. Memory Usage: ${MEMORY_MB} MB / 1024 MB"
echo "4. Services: Backend, Frontend, Redis"
echo ""
echo "🚀 AWS RISK COPILOT - DAY 1-7 COMPLETE! 🎉"
