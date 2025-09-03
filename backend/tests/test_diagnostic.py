"""
Diagnostic test to identify the actual "query failed" issue in the RAG system.

This test attempts to reproduce the actual error reported by the user.
"""
import pytest
import os
import sys
from unittest.mock import patch
from config import Config
from rag_system import RAGSystem


def test_actual_rag_system_query():
    """
    Test the actual RAG system to reproduce the 'query failed' error.
    """
    print("\n=== DIAGNOSTIC TEST: Actual RAG System Query ===")
    
    try:
        # Initialize with actual config (but potentially missing API key)
        config = Config()
        print(f"Config loaded - API Key present: {'Yes' if config.ANTHROPIC_API_KEY else 'No'}")
        print(f"ChromaDB Path: {config.CHROMA_PATH}")
        print(f"Embedding Model: {config.EMBEDDING_MODEL}")
        
        # Try to initialize RAG system
        print("\n--- Initializing RAG System ---")
        rag_system = RAGSystem(config)
        print("✓ RAG System initialized successfully")
        print(f"Tool Manager tools: {list(rag_system.tool_manager.tools.keys())}")
        
        # Check if there's any course data
        print("\n--- Checking Course Data ---")
        try:
            analytics = rag_system.get_course_analytics()
            print(f"Total courses: {analytics['total_courses']}")
            print(f"Course titles: {analytics['course_titles']}")
            
            if analytics['total_courses'] == 0:
                print("⚠️  WARNING: No course data found in vector store")
                print("This could be the cause of 'query failed' errors")
                
        except Exception as e:
            print(f"✗ Error getting course analytics: {e}")
        
        # Test vector store directly
        print("\n--- Testing Vector Store ---")
        try:
            existing_titles = rag_system.vector_store.get_existing_course_titles()
            course_count = rag_system.vector_store.get_course_count()
            print(f"Vector Store - Course count: {course_count}")
            print(f"Vector Store - Existing titles: {existing_titles}")
            
            if course_count == 0:
                print("⚠️  ISSUE IDENTIFIED: Vector store is empty!")
                print("The system has no course data to search against")
                
        except Exception as e:
            print(f"✗ Error accessing vector store: {e}")
        
        # Test search tool directly
        print("\n--- Testing Search Tool ---")
        try:
            search_result = rag_system.search_tool.execute("What is Python?")
            print(f"Search tool result: {search_result[:100]}...")
            
            if "No relevant content found" in search_result:
                print("⚠️  ISSUE IDENTIFIED: Search tool returns no content")
                
        except Exception as e:
            print(f"✗ Error executing search tool: {e}")
            print("This might be the source of 'query failed' errors")
        
        # Test AI Generator (might fail due to API key)
        print("\n--- Testing AI Generator ---")
        try:
            # This will likely fail if no API key is set
            tools = rag_system.tool_manager.get_tool_definitions()
            print(f"Tool definitions available: {len(tools)}")
            
            if not config.ANTHROPIC_API_KEY:
                print("⚠️  WARNING: No Anthropic API key configured")
                print("Set ANTHROPIC_API_KEY in .env file")
            else:
                print("✓ API key is configured")
                
        except Exception as e:
            print(f"✗ Error with AI Generator setup: {e}")
            
        # Try a full query (this will likely reproduce the error)
        print("\n--- Testing Full Query (Expected to reproduce 'query failed') ---")
        try:
            response, sources = rag_system.query("What is Python?")
            print(f"✓ Query succeeded: {response[:100]}...")
            print(f"Sources returned: {len(sources)}")
            
        except Exception as e:
            print(f"✗ FOUND THE ISSUE: Query failed with error: {e}")
            print(f"Error type: {type(e)}")
            import traceback
            traceback.print_exc()
            
        print("\n=== DIAGNOSTIC COMPLETE ===")
        
    except Exception as e:
        print(f"✗ Failed to initialize RAG system: {e}")
        import traceback
        traceback.print_exc()


def test_check_dependencies():
    """Check if all required dependencies are available"""
    print("\n=== DEPENDENCY CHECK ===")
    
    try:
        import chromadb
        print("✓ ChromaDB available")
    except ImportError as e:
        print(f"✗ ChromaDB missing: {e}")
        
    try:
        import anthropic
        print("✓ Anthropic available")
    except ImportError as e:
        print(f"✗ Anthropic missing: {e}")
        
    try:
        import sentence_transformers
        print("✓ SentenceTransformers available")
    except ImportError as e:
        print(f"✗ SentenceTransformers missing: {e}")
        
    try:
        from sentence_transformers import SentenceTransformer
        # Try to load the model
        model = SentenceTransformer('all-MiniLM-L6-v2')
        print("✓ Embedding model loaded successfully")
    except Exception as e:
        print(f"✗ Embedding model failed to load: {e}")


def test_check_data_directory():
    """Check if docs directory exists and has files"""
    print("\n=== DATA DIRECTORY CHECK ===")
    
    docs_path = "../docs"
    if os.path.exists(docs_path):
        print(f"✓ Docs directory exists: {docs_path}")
        files = [f for f in os.listdir(docs_path) 
                if f.lower().endswith(('.pdf', '.docx', '.txt'))]
        print(f"Document files found: {len(files)}")
        for file in files:
            print(f"  - {file}")
            
        if len(files) == 0:
            print("⚠️  WARNING: No document files found in docs directory")
            print("This could explain why there's no course data")
            
    else:
        print(f"✗ Docs directory not found: {docs_path}")
        print("This explains why there's no course data to search")


if __name__ == "__main__":
    test_check_dependencies()
    test_check_data_directory()
    test_actual_rag_system_query()