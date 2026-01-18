#!/usr/bin/env python3
"""
Test Production Scaling Features
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data.production_loader import get_production_loader
import json

def test_production_scaling():
    print("🧪 Testing Production Scaling...")
    
    # Test 1: Production Loader
    print("\n1. Testing Production Data Loader...")
    loader = get_production_loader()
    
    print(f"   Max Companies: {loader.max_companies}")
    print(f"   Cache Directory: {loader.cache_dir}")
    
    # Test 2: Load company data
    print("\n2. Loading production company data...")
    companies_data = loader.load_company_data(batch_size=25)
    
    print(f"   Loaded {len(companies_data)} companies")
    
    if companies_data:
        print(f"   Sample company:")
        sample = companies_data[0]
        print(f"   Name: {sample['name']}")
        print(f"   Ticker: {sample['ticker']}")
        print(f"   Sector: {sample['sector']}")
        print(f"   Risk Score: {sample['risk_score']} ({sample['risk_level']})")
        print(f"   Risk Categories: {', '.join(sample.get('risk_categories', []))}")
    
    # Test 3: Sector Analysis
    print("\n3. Testing Sector Analysis...")
    sector_analysis = loader.get_sector_analysis(companies_data)
    
    print(f"   Sectors analyzed: {len(sector_analysis)}")
    
    for sector, data in list(sector_analysis.items())[:3]:  # First 3 sectors
        print(f"   {sector}:")
        print(f"     Companies: {data['company_count']}")
        print(f"     Avg Risk Score: {data['average_risk_score']}")
        print(f"     High Risk %: {data['high_risk_percentage']}%")
        print(f"     Sample: {', '.join(data['sample_companies'][:3])}")
    
    # Test 4: High Risk Companies
    print("\n4. Testing High Risk Detection...")
    high_risk = loader.get_high_risk_companies(companies_data, top_n=10)
    
    print(f"   Found {len(high_risk)} high-risk companies")
    
    if high_risk:
        print(f"   Top 3 high-risk companies:")
        for i, company in enumerate(high_risk[:3]):
            print(f"   {i+1}. {company['name']} ({company['ticker']}): "
                  f"{company['risk_score']} - {company['sector']}")
    
    # Test 5: Cache operations
    print("\n5. Testing Cache Operations...")
    
    # Save to cache
    loader._save_to_cache(companies_data)
    print(f"   Data saved to compressed cache")
    
    # Load from cache
    cached_data = loader.load_from_cache()
    print(f"   Loaded {len(cached_data) if cached_data else 0} companies from cache")
    
    # Test 6: Save sample data for API testing
    print("\n6. Creating sample data for API testing...")
    
    sample_output = {
        "production_test": {
            "total_companies": len(companies_data),
            "sectors_covered": len(sector_analysis),
            "high_risk_count": len(high_risk),
            "sample_sector_analysis": {k: v for k, v in list(sector_analysis.items())[:2]},
            "top_high_risk": [
                {
                    "name": c["name"],
                    "ticker": c["ticker"],
                    "sector": c["sector"],
                    "risk_score": c["risk_score"]
                }
                for c in high_risk[:5]
            ]
        }
    }
    
    with open("test_production_data.json", "w") as f:
        json.dump(sample_output, f, indent=2, default=str)
    
    print("   ✅ Sample data saved to test_production_data.json")
    
    # Test 7: Memory efficiency check
    print("\n7. Memory Efficiency Check...")
    import psutil
    process = psutil.Process()
    memory_mb = process.memory_info().rss / (1024 * 1024)
    
    print(f"   Current memory usage: {memory_mb:.1f} MB")
    print(f"   Memory limit: 1024 MB (1GB)")
    print(f"   Usage percentage: {(memory_mb / 1024) * 100:.1f}%")
    
    if memory_mb < 800:
        print("   ✅ Memory usage is within safe limits")
    else:
        print("   ⚠️ Memory usage is getting high")
    
    print("\n✅ Production Scaling test complete!")

if __name__ == "__main__":
    test_production_scaling()
