"""
Test Pinecone connection and verify index setup.
Run this to confirm everything is working.
"""

import os
from dotenv import load_dotenv
from pinecone import Pinecone

# Load environment variables
load_dotenv()


def test_connection() -> bool:
    """Test Pinecone connection"""
    print("🔧 Testing Pinecone Connection...\n")
    
    # Initialize Pinecone
    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        print("❌ PINECONE_API_KEY not found in .env")
        return False
    
    try:
        pc = Pinecone(api_key=api_key)
        print("✅ Pinecone client initialized")
        
        # List all indexes
        indexes = pc.list_indexes()
        print(f"\n📋 Available indexes: {indexes.names()}")
        
        # Connect to nexus-rag index
        index_name = os.getenv("PINECONE_INDEX_NAME", "nexus-rag")
        
        if index_name in indexes.names():
            index = pc.Index(index_name)
            print(f"\n✅ Connected to index: '{index_name}'")
            
            # Get index stats
            stats = index.describe_index_stats()
            print(f"\n📊 Index Statistics:")
            print(f"   - Dimension: {stats.get('dimension')}")
            print(f"   - Total vectors: {stats.get('total_vector_count', 0)}")
            print(f"   - Namespaces: {list(stats.get('namespaces', {}).keys())}")
            
            # Get index description
            index_info = pc.describe_index(index_name)
            print(f"\n🔍 Index Configuration:")
            print(f"   - Metric: {index_info.metric}")
            print(f"   - Cloud: {index_info.spec.serverless.cloud}")
            print(f"   - Region: {index_info.spec.serverless.region}")
            
            print("\n✅ All checks passed! Pinecone is ready to use.")
            return True
        else:
            print(f"\n❌ Index '{index_name}' not found!")
            return False
            
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return False


if __name__ == "__main__":
    success = test_connection()
    assert success is True
    exit(0 if success else 1)
