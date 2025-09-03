"""
Test the actual FastAPI endpoints to identify potential API-level issues.
"""
import pytest
import json
from fastapi.testclient import TestClient
from unittest.mock import patch, Mock
import sys
import os

# Add the backend directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + '/..')

from app import app


@pytest.fixture
def client():
    """Create a test client for the FastAPI app"""
    return TestClient(app)


def test_query_endpoint_success(client):
    """Test successful query to the API endpoint"""
    print("\n=== TESTING QUERY ENDPOINT ===")
    
    # Test data
    query_data = {
        "query": "What is Python?",
        "session_id": None
    }
    
    try:
        response = client.post("/api/query", json=query_data)
        
        print(f"Status Code: {response.status_code}")
        print(f"Response Headers: {dict(response.headers)}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"✓ Query successful")
            print(f"Answer length: {len(data.get('answer', ''))}")
            print(f"Sources count: {len(data.get('sources', []))}")
            print(f"Session ID: {data.get('session_id', 'None')}")
            print(f"Answer preview: {data.get('answer', '')[:100]}...")
            
        else:
            print(f"✗ Query failed with status {response.status_code}")
            print(f"Response body: {response.text}")
            
            # This might be where we find the "query failed" error
            if "query failed" in response.text.lower():
                print("🎯 FOUND IT: This is where 'query failed' comes from!")
                
    except Exception as e:
        print(f"✗ Exception during API call: {e}")
        import traceback
        traceback.print_exc()


def test_query_endpoint_with_invalid_data(client):
    """Test query endpoint with invalid data to reproduce errors"""
    print("\n=== TESTING QUERY ENDPOINT WITH INVALID DATA ===")
    
    # Test cases that might cause "query failed"
    test_cases = [
        {"description": "Empty query", "data": {"query": ""}},
        {"description": "No query field", "data": {}},
        {"description": "Invalid session_id", "data": {"query": "test", "session_id": "invalid"}},
        {"description": "Null query", "data": {"query": None}},
    ]
    
    for test_case in test_cases:
        print(f"\n--- Testing: {test_case['description']} ---")
        
        try:
            response = client.post("/api/query", json=test_case['data'])
            print(f"Status Code: {response.status_code}")
            
            if response.status_code != 200:
                print(f"Response: {response.text}")
                if "query failed" in response.text.lower():
                    print("🎯 FOUND 'query failed' error source!")
                    
        except Exception as e:
            print(f"Exception: {e}")


def test_query_endpoint_with_api_error():
    """Test query endpoint when internal components fail"""
    print("\n=== TESTING WITH MOCKED COMPONENT FAILURES ===")
    
    # Test with mocked RAG system that fails
    with patch('app.rag_system') as mock_rag:
        mock_rag.query.side_effect = Exception("Internal error")
        
        client = TestClient(app)
        response = client.post("/api/query", json={"query": "test"})
        
        print(f"Status Code with internal error: {response.status_code}")
        print(f"Response: {response.text}")
        
        if "query failed" in response.text.lower():
            print("🎯 FOUND IT: Internal errors cause 'query failed'")


def test_courses_endpoint(client):
    """Test the courses endpoint"""
    print("\n=== TESTING COURSES ENDPOINT ===")
    
    try:
        response = client.get("/api/courses")
        
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"✓ Courses endpoint successful")
            print(f"Total courses: {data.get('total_courses', 0)}")
            print(f"Course titles: {data.get('course_titles', [])}")
            
        else:
            print(f"✗ Courses endpoint failed: {response.text}")
            
    except Exception as e:
        print(f"✗ Exception: {e}")


def test_session_endpoint_if_exists(client):
    """Test session clearing endpoint if it exists"""
    print("\n=== TESTING SESSION ENDPOINT ===")
    
    try:
        # Try to clear a session
        response = client.delete("/api/session/test_session_123")
        
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.text}")
        
        if response.status_code == 200:
            print("✓ Session endpoint working")
        else:
            print("ℹ️  Session endpoint might not be implemented or failing")
            
    except Exception as e:
        print(f"Exception: {e}")


def test_root_endpoint(client):
    """Test if the root endpoint serves the frontend"""
    print("\n=== TESTING ROOT ENDPOINT ===")
    
    try:
        response = client.get("/")
        
        print(f"Status Code: {response.status_code}")
        print(f"Content Type: {response.headers.get('content-type', 'Unknown')}")
        
        if response.status_code == 200:
            print("✓ Root endpoint working (serves frontend)")
        else:
            print(f"✗ Root endpoint failed: {response.text}")
            
    except Exception as e:
        print(f"Exception: {e}")


if __name__ == "__main__":
    client = TestClient(app)
    
    test_query_endpoint_success(client)
    test_query_endpoint_with_invalid_data(client)
    test_query_endpoint_with_api_error()
    test_courses_endpoint(client)
    test_session_endpoint_if_exists(client)
    test_root_endpoint(client)