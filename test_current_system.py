#!/usr/bin/env python3
"""
Test Current Working System (without heavy ML)
"""
import requests
import json

BASE_URL = "http://localhost:8000"

def test_endpoint(endpoint, method="GET", data=None):
    """Test an API endpoint"""
    try:
        if method == "GET":
            response = requests.get(f"{BASE_URL}{endpoint}", timeout=10)
        else:
            response = requests.post(f"{BASE_URL}{endpoint}", json=data, timeout=10)
        
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"Status {response.status_code}", "text": response.text}
    except Exception as e:
        return {"error": str(e)}

print("🧪 Testing Current AWS Risk Copilot System")
print("=" * 50)

# Test 1: Basic health
print("\n1. Testing System Health...")
health = test_endpoint("/health")
if "status" in health and health["status"] == "healthy":
    print(f"   ✅ Healthy: {health['memory_mb']}MB memory, Redis: {health['redis']}")
else:
    print(f"   ❌ Health check failed: {health.get('error', 'Unknown')}")

# Test 2: SEC Integration
print("\n2. Testing SEC Integration...")
sec_stats = test_endpoint("/sec/stats")
if sec_stats.get("status") == "success":
    print(f"   ✅ SEC Data: {sec_stats.get('total_entries', 0)} entries")
    print(f"   Companies: {sec_stats.get('companies_count', 0)}")
else:
    print(f"   ❌ SEC failed: {sec_stats.get('error', 'Unknown')}")

# Test 3: Production Data
print("\n3. Testing Production Data...")
prod_stats = test_endpoint("/production/stats")
if prod_stats.get("status") == "success":
    if prod_stats.get("loaded"):
        print(f"   ✅ Loaded: {prod_stats.get('total_companies', 0)} companies")
        print(f"   Sectors: {prod_stats.get('sectors_covered', 0)}")
    else:
        print(f"   ℹ️ Not loaded yet")
else:
    print(f"   ❌ Production failed: {prod_stats.get('error', 'Unknown')}")

# Test 4: AI Capabilities
print("\n4. Testing AI Capabilities...")
ai_caps = test_endpoint("/ai/capabilities")
if ai_caps.get("status") == "success":
    caps = ai_caps.get("ai_capabilities", {})
    print(f"   ✅ AI Features Available:")
    print(f"   - Advanced Copilot: {caps.get('advanced_copilot', {}).get('available', False)}")
    print(f"   - Sentiment Analysis: {caps.get('advanced_sentiment', {}).get('available', False)}")
else:
    print(f"   ❌ AI check failed: {ai_caps.get('error', 'Unknown')}")

# Test 5: Test AI Copilot (with fallback)
print("\n5. Testing AI Copilot (with fallback)...")
copilot_response = test_endpoint("/ai/copilot/advanced", "POST", {
    "query": "What are the main cybersecurity risks?",
    "context_type": "risk_analysis"
})
if copilot_response and "response" in copilot_response:
    resp = copilot_response["response"]
    print(f"   ✅ Response received ({resp.get('llm_used', 'unknown')})")
    print(f"   Query: {resp.get('query', '')[:50]}...")
    print(f"   Answer preview: {resp.get('answer', '')[:100]}...")
else:
    print(f"   ❌ Copilot failed: {copilot_response.get('error', 'Unknown')}")

# Test 6: Test Sentiment
print("\n6. Testing Sentiment Analysis...")
sentiment = test_endpoint("/ai/sentiment/advanced", "POST", {
    "text": "Apple reports strong earnings with record profits",
    "include_financial_context": True
})
if sentiment and "analysis" in sentiment:
    analysis = sentiment["analysis"]
    print(f"   ✅ Sentiment: {analysis.get('label', 'unknown')}")
    print(f"   Score: {analysis.get('score', 0):.2f}")
    print(f"   Method: {analysis.get('method', 'unknown')}")
else:
    print(f"   ❌ Sentiment failed: {sentiment.get('error', 'Unknown')}")

# Test 7: Test Dashboard
print("\n7. Testing Dashboard Access...")
try:
    dash_response = requests.get("http://localhost:8501", timeout=5)
    if dash_response.status_code in [200, 302]:
        print(f"   ✅ Dashboard accessible (Status: {dash_response.status_code})")
    else:
        print(f"   ⚠️ Dashboard status: {dash_response.status_code}")
except Exception as e:
    print(f"   ❌ Dashboard error: {e}")

print("\n" + "=" * 50)
print("🎉 System Test Complete!")
print("\nSummary:")
print(f"- Health: {'✅' if health.get('status') == 'healthy' else '❌'}")
print(f"- SEC Data: {'✅' if sec_stats.get('status') == 'success' else '❌'}")
print(f"- Production: {'✅' if prod_stats.get('loaded') else '⚠️'}")
print(f"- AI Copilot: {'✅' if copilot_response and 'response' in copilot_response else '❌'}")
print(f"- Dashboard: {'✅' if 'dash_response' in locals() and dash_response.status_code in [200, 302] else '❌'}")
print(f"\nYour system is {'OPERATIONAL' if health.get('status') == 'healthy' else 'NEEDS ATTENTION'}")
