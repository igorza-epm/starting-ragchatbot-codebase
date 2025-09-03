"""
Pytest configuration and fixtures for RAG system tests.
"""
import pytest
import sys
import os
from unittest.mock import Mock, MagicMock, patch
from typing import List, Dict, Any

# Add the backend directory to Python path so we can import modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from config import Config
from models import Course, Lesson, CourseChunk
from vector_store import VectorStore, SearchResults
from search_tools import CourseSearchTool, ToolManager
from ai_generator import AIGenerator
from rag_system import RAGSystem
from .fixtures import MockTestData, MockChromaResponse, MockAnthropicResponse


@pytest.fixture
def test_config():
    """Test configuration with safe defaults"""
    config = Config()
    config.ANTHROPIC_API_KEY = "test-api-key"
    config.CHROMA_PATH = ":memory:"  # In-memory database for tests
    config.MAX_RESULTS = 3
    return config


@pytest.fixture
def sample_courses():
    """Sample course data for testing"""
    return MockTestData.get_sample_courses()


@pytest.fixture
def sample_course_chunks():
    """Sample course chunks for testing"""
    return MockTestData.get_sample_course_chunks()


@pytest.fixture
def sample_search_results():
    """Sample search results for testing"""
    return MockTestData.get_sample_search_results()


@pytest.fixture
def mock_vector_store():
    """Mock VectorStore for unit testing"""
    mock_store = Mock(spec=VectorStore)
    
    # Configure default behavior
    mock_store.search.return_value = MockTestData.get_sample_search_results()
    mock_store.get_lesson_link.return_value = "https://example.com/lesson3"
    mock_store.get_existing_course_titles.return_value = ["Advanced Python Programming", "Machine Learning Fundamentals"]
    
    return mock_store


@pytest.fixture
def mock_vector_store_empty():
    """Mock VectorStore that returns empty results"""
    mock_store = Mock(spec=VectorStore)
    mock_store.search.return_value = MockTestData.get_empty_search_results()
    return mock_store


@pytest.fixture
def mock_vector_store_error():
    """Mock VectorStore that returns error results"""
    mock_store = Mock(spec=VectorStore)
    mock_store.search.return_value = MockTestData.get_error_search_results()
    return mock_store


@pytest.fixture
def mock_chroma_client():
    """Mock ChromaDB client"""
    mock_client = MagicMock()
    
    # Mock collections
    mock_course_catalog = MagicMock()
    mock_course_content = MagicMock()
    
    # Configure collection responses
    mock_course_catalog.query.return_value = MockChromaResponse.course_catalog_response()
    mock_course_content.query.return_value = MockChromaResponse.successful_query_response()
    
    # Configure client to return mock collections
    mock_client.get_or_create_collection.side_effect = lambda name, **kwargs: (
        mock_course_catalog if name == "course_catalog" else mock_course_content
    )
    
    return mock_client


@pytest.fixture
def mock_anthropic_client():
    """Mock Anthropic client"""
    mock_client = Mock()
    
    # Configure messages.create to return different responses based on context
    mock_client.messages.create.return_value = MockAnthropicResponse.direct_response()
    
    return mock_client


@pytest.fixture
def course_search_tool(mock_vector_store):
    """CourseSearchTool with mock vector store"""
    return CourseSearchTool(mock_vector_store)


@pytest.fixture
def course_search_tool_empty(mock_vector_store_empty):
    """CourseSearchTool with mock vector store that returns empty results"""
    return CourseSearchTool(mock_vector_store_empty)


@pytest.fixture
def course_search_tool_error(mock_vector_store_error):
    """CourseSearchTool with mock vector store that returns errors"""
    return CourseSearchTool(mock_vector_store_error)


@pytest.fixture
def tool_manager(course_search_tool):
    """ToolManager with registered CourseSearchTool"""
    manager = ToolManager()
    manager.register_tool(course_search_tool)
    return manager


@pytest.fixture
def mock_ai_generator(mock_anthropic_client):
    """Mock AIGenerator for testing"""
    with patch('ai_generator.anthropic.Anthropic', return_value=mock_anthropic_client):
        generator = AIGenerator("test-api-key", "claude-3-sonnet")
        return generator


@pytest.fixture
def mock_rag_system(test_config, mock_vector_store, mock_ai_generator):
    """Mock RAGSystem for integration testing"""
    with patch('rag_system.VectorStore', return_value=mock_vector_store), \
         patch('rag_system.AIGenerator', return_value=mock_ai_generator):
        system = RAGSystem(test_config)
        return system


# Test data constants
PYTHON_DECORATORS_QUERY = "What are Python decorators?"
EMPTY_QUERY = ""
COMPLEX_QUERY = "Explain the relationship between decorators and closures in Python lesson 3"
COURSE_FILTER_QUERY = "machine learning basics"
INVALID_COURSE_QUERY = "nonexistent course content"


@pytest.fixture(autouse=True)
def setup_test_environment():
    """Setup test environment for each test"""
    # Clear any existing environment variables that might interfere
    if 'ANTHROPIC_API_KEY' in os.environ:
        original_key = os.environ['ANTHROPIC_API_KEY']
    else:
        original_key = None
        
    os.environ['ANTHROPIC_API_KEY'] = 'test-key'
    
    yield
    
    # Restore original environment
    if original_key:
        os.environ['ANTHROPIC_API_KEY'] = original_key
    else:
        os.environ.pop('ANTHROPIC_API_KEY', None)