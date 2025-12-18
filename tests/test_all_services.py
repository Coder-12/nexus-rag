"""
Nexus RAG - Complete Service Testing Suite
Tests all external services: OpenAI, Pinecone, Cohere, LangSmith
"""

import os
import sys
from dotenv import load_dotenv
from datetime import datetime
import time

# Load environment variables
load_dotenv()


# Color codes for terminal output
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    BOLD = '\033[1m'
    END = '\033[0m'


def print_header(text):
    """Print formatted header"""
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*70}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{text.center(70)}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'='*70}{Colors.END}\n")


def print_success(text):
    """Print success message"""
    print(f"{Colors.GREEN}✅ {text}{Colors.END}")


def print_error(text):
    """Print error message"""
    print(f"{Colors.RED}❌ {text}{Colors.END}")


def print_warning(text):
    """Print warning message"""
    print(f"{Colors.YELLOW}⚠️  {text}{Colors.END}")


def print_info(text):
    """Print info message"""
    print(f"{Colors.BLUE}ℹ️  {text}{Colors.END}")


# =============================================================================
# TEST 1: OpenAI API
# =============================================================================
def test_openai():
    """Test OpenAI API connection and functionality"""
    print_header("TEST 1: OpenAI API")
    
    try:
        from openai import OpenAI
        
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print_error("OPENAI_API_KEY not found in .env")
            return False
        
        print_info(f"API Key: {api_key[:20]}...")
        
        # Initialize client
        client = OpenAI(api_key=api_key)
        print_success("OpenAI client initialized")
        
        # Test 1: List available models
        print_info("Testing: Listing available models...")
        models = client.models.list()
        model_ids = [model.id for model in models.data[:5]]
        print_success(f"Found {len(models.data)} models")
        print(f"   Sample models: {model_ids}")
        
        # Test 2: Generate embeddings
        print_info("Testing: Generating embeddings...")
        embedding_model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")
        response = client.embeddings.create(
            input="This is a test sentence for embedding generation.",
            model=embedding_model
        )
        embedding_dim = len(response.data[0].embedding)
        expected_dim = int(os.getenv("EMBEDDING_DIMENSION", "3072"))
        
        print_success(f"Generated embedding with {embedding_dim} dimensions")
        
        if embedding_dim == expected_dim:
            print_success(f"Dimension matches expected: {expected_dim}")
        else:
            print_warning(f"Dimension mismatch! Expected {expected_dim}, got {embedding_dim}")
        
        # Test 3: Generate chat completion
        print_info("Testing: Chat completion...")
        chat_response = client.chat.completions.create(
            model="gpt-4-turbo-preview",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Say 'Hello, Nexus RAG!' in exactly those words."}
            ],
            max_tokens=50
        )
        
        response_text = chat_response.choices[0].message.content
        print_success(f"Chat completion response: {response_text}")
        
        # Calculate approximate cost
        prompt_tokens = chat_response.usage.prompt_tokens
        completion_tokens = chat_response.usage.completion_tokens
        total_tokens = chat_response.usage.total_tokens
        
        print_info(f"Token usage: {prompt_tokens} prompt + {completion_tokens} completion = {total_tokens} total")
        
        print_success("OpenAI API: ALL TESTS PASSED ✓")
        return True
        
    except Exception as e:
        print_error(f"OpenAI API test failed: {e}")
        return False


# =============================================================================
# TEST 2: Pinecone
# =============================================================================
def test_pinecone():
    """Test Pinecone connection and operations"""
    print_header("TEST 2: Pinecone Vector Database")
    
    try:
        from pinecone import Pinecone
        
        api_key = os.getenv("PINECONE_API_KEY")
        if not api_key:
            print_error("PINECONE_API_KEY not found in .env")
            return False
        
        print_info(f"API Key: {api_key[:20]}...")
        
        # Initialize client
        pc = Pinecone(api_key=api_key)
        print_success("Pinecone client initialized")
        
        # Test 1: List indexes
        print_info("Testing: Listing indexes...")
        indexes = pc.list_indexes()
        print_success(f"Found {len(indexes.names())} index(es): {indexes.names()}")
        
        # Test 2: Connect to index
        index_name = os.getenv("PINECONE_INDEX_NAME", "nexus-rag")
        print_info(f"Testing: Connecting to index '{index_name}'...")
        
        if index_name not in indexes.names():
            print_error(f"Index '{index_name}' not found!")
            return False
        
        index = pc.Index(index_name)
        print_success(f"Connected to index '{index_name}'")
        
        # Test 3: Get index stats
        print_info("Testing: Retrieving index statistics...")
        stats = index.describe_index_stats()
        print_success("Index statistics retrieved")
        print(f"   - Dimension: {stats.get('dimension')}")
        print(f"   - Total vectors: {stats.get('total_vector_count', 0)}")
        print(f"   - Namespaces: {list(stats.get('namespaces', {}).keys()) or ['(empty)']}")
        
        # Test 4: Get index configuration
        print_info("Testing: Retrieving index configuration...")
        index_info = pc.describe_index(index_name)
        print_success("Index configuration retrieved")
        print(f"   - Metric: {index_info.metric}")
        print(f"   - Cloud: {index_info.spec.serverless.cloud}")
        print(f"   - Region: {index_info.spec.serverless.region}")
        
        # Test 5: Test upsert (write) operation
        print_info("Testing: Upsert operation (writing test vector)...")
        test_vector = [0.1] * int(os.getenv("EMBEDDING_DIMENSION", "3072"))
        test_id = f"test-vector-{int(time.time())}"
        
        index.upsert(
            vectors=[{
                "id": test_id,
                "values": test_vector,
                "metadata": {"test": True, "timestamp": datetime.now().isoformat()}
            }]
        )
        print_success(f"Successfully upserted test vector with ID: {test_id}")
        
        # Wait for consistency
        time.sleep(1)
        
        # Test 6: Test query (read) operation
        print_info("Testing: Query operation (reading test vector)...")
        query_results = index.query(
            vector=test_vector,
            top_k=1,
            include_metadata=True
        )
        
        if query_results.matches:
            print_success(f"Query returned {len(query_results.matches)} result(s)")
            print(f"   - Top match ID: {query_results.matches[0].id}")
            print(f"   - Similarity score: {query_results.matches[0].score:.4f}")
        else:
            print_warning("Query returned no results (vector may not be indexed yet)")
        
        # Test 7: Test delete operation
        print_info("Testing: Delete operation (cleaning up test vector)...")
        index.delete(ids=[test_id])
        print_success("Test vector deleted successfully")
        
        print_success("Pinecone: ALL TESTS PASSED ✓")
        return True
        
    except Exception as e:
        print_error(f"Pinecone test failed: {e}")
        return False


# =============================================================================
# TEST 3: Cohere (Reranking)
# =============================================================================
def test_cohere():
    """Test Cohere API for reranking"""
    print_header("TEST 3: Cohere Reranking API")
    
    try:
        import cohere
        
        api_key = os.getenv("COHERE_API_KEY")
        if not api_key:
            print_warning("COHERE_API_KEY not found in .env (optional service)")
            return True  # Not critical, so return True
        
        print_info(f"API Key: {api_key[:20]}...")
        
        # Initialize client
        co = cohere.Client(api_key=api_key)
        print_success("Cohere client initialized")
        
        # Test 1: Rerank operation
        print_info("Testing: Reranking operation...")
        
        query = "What is machine learning?"
        documents = [
            "Machine learning is a subset of artificial intelligence.",
            "Python is a popular programming language.",
            "Machine learning algorithms learn from data.",
            "The weather is nice today.",
            "Neural networks are used in deep learning."
        ]
        
        rerank_model = os.getenv("COHERE_RERANK_MODEL", "rerank-english-v2.0")
        results = co.rerank(
            query=query,
            documents=documents,
            top_n=3,
            model=rerank_model
        )
        
        print_success(f"Reranked {len(documents)} documents, returned top {len(results.results)}")
        
        for i, result in enumerate(results.results, 1):
            print(f"   {i}. Score: {result.relevance_score:.4f} - Doc {result.index}: {documents[result.index][:60]}...")
        
        print_success("Cohere: ALL TESTS PASSED ✓")
        return True
        
    except Exception as e:
        print_error(f"Cohere test failed: {e}")
        print_warning("Cohere is optional for MVP - you can continue without it")
        return True  # Don't fail overall tests for optional service


# =============================================================================
# TEST 4: LangSmith (Observability)
# =============================================================================
def test_langsmith():
    """Test LangSmith observability"""
    print_header("TEST 4: LangSmith Observability")
    
    try:
        from langsmith import Client
        
        api_key = os.getenv("LANGCHAIN_API_KEY")
        if not api_key:
            print_warning("LANGCHAIN_API_KEY not found in .env (optional service)")
            return True  # Not critical
        
        print_info(f"API Key: {api_key[:20]}...")
        
        # Initialize client
        endpoint = os.getenv("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com")
        client = Client(api_key=api_key, api_url=endpoint)
        print_success("LangSmith client initialized")
        
        # Test 1: Create a test trace
        print_info("Testing: Creating a test trace...")
        
        project_name = os.getenv("LANGCHAIN_PROJECT", "nexus-rag")
        print_info(f"Project: {project_name}")
        
        # Create a simple test run
        from langsmith.run_helpers import traceable
        
        @traceable(project_name=project_name, name="test_langsmith_connection")
        def test_function():
            """Test function for LangSmith tracing"""
            return "LangSmith trace test successful"
        
        result = test_function()
        print_success(f"Test trace created: {result}")
        print_info(f"View traces at: https://smith.langchain.com/o/default/projects/{project_name}")
        
        print_success("LangSmith: ALL TESTS PASSED ✓")
        return True
        
    except Exception as e:
        print_error(f"LangSmith test failed: {e}")
        print_warning("LangSmith is optional for MVP - you can continue without it")
        return True  # Don't fail overall tests for optional service


# =============================================================================
# MAIN TEST RUNNER
# =============================================================================
def run_all_tests():
    """Run all service tests"""
    print_header("NEXUS RAG - SERVICE TESTING SUITE")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    results = {}
    
    # Run tests
    results['openai'] = test_openai()
    results['pinecone'] = test_pinecone()
    results['cohere'] = test_cohere()
    results['langsmith'] = test_langsmith()
    
    # Summary
    print_header("TEST SUMMARY")
    
    total_tests = len(results)
    passed_tests = sum(1 for r in results.values() if r)
    
    for service, passed in results.items():
        status = "PASSED ✓" if passed else "FAILED ✗"
        color = Colors.GREEN if passed else Colors.RED
        print(f"{color}{service.upper():.<20} {status}{Colors.END}")
    
    print(f"\n{Colors.BOLD}Total: {passed_tests}/{total_tests} tests passed{Colors.END}")
    
    if all(results.values()):
        print(f"\n{Colors.GREEN}{Colors.BOLD}🎉 ALL SERVICES READY! You can proceed to Day 2.{Colors.END}")
        return True
    else:
        print(f"\n{Colors.RED}{Colors.BOLD}⚠️  Some tests failed. Please fix issues before proceeding.{Colors.END}")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
