#!/usr/bin/env python3
"""
Test Advanced AI Features
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from analysis.advanced_copilot import get_advanced_copilot
from analysis.advanced_sentiment import get_advanced_sentiment
import json

def test_advanced_ai():
    print("🧪 Testing Advanced AI Features...")
    
    # Test 1: Advanced Sentiment Analysis
    print("\n1. Testing Advanced Sentiment Analysis...")
    sentiment_analyzer = get_advanced_sentiment()
    
    test_texts = [
        "Apple reported strong quarterly earnings with record profits and growth in all segments.",
        "Microsoft faces cybersecurity investigation after major data breach affecting millions.",
        "Amazon stock remains stable despite market volatility and supply chain challenges."
    ]
    
    for i, text in enumerate(test_texts):
        analysis = sentiment_analyzer.analyze_sentiment(text)
        print(f"   Text {i+1}:")
        print(f"     Sentiment: {analysis['label']} (score: {analysis['score']:.2f})")
        print(f"     Method: {analysis.get('method', 'N/A')}")
        print(f"     Preview: {text[:60]}...")
    
    # Test 2: Batch Sentiment Analysis
    print("\n2. Testing Batch Sentiment Analysis...")
    batch_analyses = sentiment_analyzer.analyze_batch(test_texts)
    summary = sentiment_analyzer.get_sentiment_summary(batch_analyses)
    
    print(f"   Total texts: {summary['total']}")
    print(f"   Positive: {summary['positive']}, Negative: {summary['negative']}, Neutral: {summary['neutral']}")
    print(f"   Average score: {summary['average_score']:.2f}")
    
    # Test 3: Advanced AI Copilot (simulated - no LLM/RAG)
    print("\n3. Testing Advanced AI Copilot (simulated)...")
    copilot = get_advanced_copilot()
    
    test_queries = [
        "What are the cybersecurity risks for Apple?",
        "Compare risk between Microsoft and Amazon",
        "How can we mitigate cloud security risks?"
    ]
    
    for query in test_queries:
        print(f"\n   Query: {query}")
        # Note: Without LLM service, this will use fallback
        # In production, you'd have LLM connected
        print(f"   (LLM service would be connected in production)")
    
    # Test 4: Save sample data
    print("\n4. Creating sample AI test data...")
    
    sample_data = {
        "sentiment_test": {
            "texts": test_texts,
            "analyses": batch_analyses,
            "summary": summary
        },
        "ai_capabilities": {
            "advanced_sentiment": {
                "model": sentiment_analyzer.model_name,
                "loaded": sentiment_analyzer.is_loaded
            },
            "advanced_copilot": {
                "available": True,
                "requires_llm": True
            }
        }
    }
    
    with open("test_advanced_ai.json", "w") as f:
        json.dump(sample_data, f, indent=2, default=str)
    
    print("   ✅ Sample data saved to test_advanced_ai.json")
    
    # Test 5: Memory check for ML models
    print("\n5. Memory Efficiency Check...")
    import psutil
    process = psutil.Process()
    memory_mb = process.memory_info().rss / (1024 * 1024)
    
    print(f"   Current memory: {memory_mb:.1f} MB")
    print(f"   Sentiment model loaded: {sentiment_analyzer.is_loaded}")
    
    if memory_mb < 700:
        print("   ✅ Memory is within 1GB limit for ML models")
    else:
        print("   ⚠️  Memory usage high - consider lighter models")
    
    print("\n✅ Advanced AI Features test complete!")
    print("\n📋 Next steps for production:")
    print("   1. Add your Gemini/HuggingFace API keys to .env")
    print("   2. Test with real LLM: curl -X POST http://localhost:8000/ai/copilot/advanced")
    print("   3. Load production data: curl -X POST http://localhost:8000/production/load-companies")

if __name__ == "__main__":
    test_advanced_ai()
