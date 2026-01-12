import requests

# Test the actual API directly
import os
from dotenv import load_dotenv

load_dotenv()
token = os.getenv("HUGGINGFACE_TOKEN")

if token:
    headers = {"Authorization": f"Bearer {token}"}
    
    # Test different endpoints
    endpoints = [
        "https://api-inference.huggingface.co/models/sentence-transformers/all-MiniLM-L6-v2",
        "https://router.huggingface.co/models/sentence-transformers/all-MiniLM-L6-v2"
    ]
    
    for endpoint in endpoints:
        try:
            response = requests.post(
                endpoint,
                headers=headers,
                json={"inputs": ["test embedding"]},
                timeout=10
            )
            print(f"{endpoint}: Status {response.status_code}")
            if response.status_code == 200:
                print(f"  Works! Response: {type(response.json())}")
            else:
                print(f"  Error: {response.text[:100]}")
        except Exception as e:
            print(f"{endpoint}: Exception - {e}")
