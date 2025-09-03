"""
Unit tests for VectorStore functionality.

These tests validate the ChromaDB integration, search functionality,
and data management operations of the VectorStore component.
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
from vector_store import VectorStore, SearchResults
from .fixtures import MockTestData, MockChromaResponse
import json


class TestSearchResults:
    """Test SearchResults data structure"""
    
    def test_search_results_creation(self):
        """Test SearchResults creation with data"""
        results = SearchResults(
            documents=["doc1", "doc2"],
            metadata=[{"course": "test1"}, {"course": "test2"}],
            distances=[0.1, 0.2]
        )
        
        assert results.documents == ["doc1", "doc2"]
        assert results.metadata == [{"course": "test1"}, {"course": "test2"}]
        assert results.distances == [0.1, 0.2]
        assert results.error is None
        assert not results.is_empty()

    def test_search_results_from_chroma(self):
        """Test creating SearchResults from ChromaDB response"""
        chroma_response = MockChromaResponse.successful_query_response()
        
        results = SearchResults.from_chroma(chroma_response)
        
        assert len(results.documents) == 2
        assert len(results.metadata) == 2
        assert len(results.distances) == 2
        assert results.error is None

    def test_search_results_empty(self):
        """Test empty SearchResults"""
        results = SearchResults.empty("No results found")
        
        assert results.is_empty()
        assert results.error == "No results found"
        assert results.documents == []
        assert results.metadata == []
        assert results.distances == []

    def test_search_results_from_empty_chroma(self):
        """Test creating SearchResults from empty ChromaDB response"""
        chroma_response = MockChromaResponse.empty_query_response()
        
        results = SearchResults.from_chroma(chroma_response)
        
        assert results.is_empty()
        assert len(results.documents) == 0


class TestVectorStoreInitialization:
    """Test VectorStore initialization"""
    
    @patch('chromadb.PersistentClient')
    @patch('chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction')
    def test_vector_store_initialization(self, mock_embedding_func, mock_client_class):
        """Test proper VectorStore initialization"""
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        
        mock_collection = Mock()
        mock_client.get_or_create_collection.return_value = mock_collection
        
        vector_store = VectorStore("/test/path", "test-model", max_results=10)
        
        # Verify client initialization
        mock_client_class.assert_called_once()
        call_args = mock_client_class.call_args
        assert call_args[1]['path'] == "/test/path"
        
        # Verify embedding function setup
        mock_embedding_func.assert_called_once_with(model_name="test-model")
        
        # Verify collections were created
        assert mock_client.get_or_create_collection.call_count == 2
        
        # Verify configuration
        assert vector_store.max_results == 10


class TestVectorStoreSearch:
    """Test VectorStore search functionality"""
    
    @patch('chromadb.PersistentClient')
    @patch('chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction')
    def test_search_without_filters(self, mock_embedding_func, mock_client_class):
        """Test basic search without any filters"""
        # Setup mocks
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        
        mock_catalog = Mock()
        mock_content = Mock()
        mock_content.query.return_value = MockChromaResponse.successful_query_response()
        
        mock_client.get_or_create_collection.side_effect = [mock_catalog, mock_content]
        
        vector_store = VectorStore("/test/path", "test-model")
        
        # Perform search
        results = vector_store.search("test query")
        
        # Verify search was called correctly
        mock_content.query.assert_called_once_with(
            query_texts=["test query"],
            n_results=5,  # default max_results
            where=None
        )
        
        # Verify results
        assert not results.is_empty()
        assert len(results.documents) == 2

    @patch('chromadb.PersistentClient')
    @patch('chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction')
    def test_search_with_course_name_filter(self, mock_embedding_func, mock_client_class):
        """Test search with course name filtering"""
        # Setup mocks
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        
        mock_catalog = Mock()
        mock_catalog.query.return_value = MockChromaResponse.course_catalog_response()
        
        mock_content = Mock()
        mock_content.query.return_value = MockChromaResponse.successful_query_response()
        
        mock_client.get_or_create_collection.side_effect = [mock_catalog, mock_content]
        
        vector_store = VectorStore("/test/path", "test-model")
        
        # Perform search with course filter
        results = vector_store.search("decorators", course_name="python")
        
        # Verify course catalog was queried for resolution
        mock_catalog.query.assert_called_once_with(
            query_texts=["python"],
            n_results=1
        )
        
        # Verify content search with filter
        mock_content.query.assert_called_once()
        call_args = mock_content.query.call_args
        assert call_args[1]['where'] == {"course_title": "Advanced Python Programming"}

    @patch('chromadb.PersistentClient')
    @patch('chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction')
    def test_search_with_lesson_number_filter(self, mock_embedding_func, mock_client_class):
        """Test search with lesson number filtering"""
        # Setup mocks
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        
        mock_catalog = Mock()
        mock_content = Mock()
        mock_content.query.return_value = MockChromaResponse.successful_query_response()
        
        mock_client.get_or_create_collection.side_effect = [mock_catalog, mock_content]
        
        vector_store = VectorStore("/test/path", "test-model")
        
        # Perform search with lesson filter
        results = vector_store.search("decorators", lesson_number=3)
        
        # Verify content search with lesson filter
        mock_content.query.assert_called_once()
        call_args = mock_content.query.call_args
        assert call_args[1]['where'] == {"lesson_number": 3}

    @patch('chromadb.PersistentClient')
    @patch('chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction')
    def test_search_with_combined_filters(self, mock_embedding_func, mock_client_class):
        """Test search with both course name and lesson number filters"""
        # Setup mocks
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        
        mock_catalog = Mock()
        mock_catalog.query.return_value = MockChromaResponse.course_catalog_response()
        
        mock_content = Mock()
        mock_content.query.return_value = MockChromaResponse.successful_query_response()
        
        mock_client.get_or_create_collection.side_effect = [mock_catalog, mock_content]
        
        vector_store = VectorStore("/test/path", "test-model")
        
        # Perform search with combined filters
        results = vector_store.search("decorators", course_name="python", lesson_number=3)
        
        # Verify content search with combined filter
        mock_content.query.assert_called_once()
        call_args = mock_content.query.call_args
        expected_filter = {"$and": [
            {"course_title": "Advanced Python Programming"},
            {"lesson_number": 3}
        ]}
        assert call_args[1]['where'] == expected_filter

    @patch('chromadb.PersistentClient')
    @patch('chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction')
    def test_search_course_not_found(self, mock_embedding_func, mock_client_class):
        """Test search when course name cannot be resolved"""
        # Setup mocks
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        
        mock_catalog = Mock()
        mock_catalog.query.return_value = MockChromaResponse.empty_query_response()
        
        mock_content = Mock()
        
        mock_client.get_or_create_collection.side_effect = [mock_catalog, mock_content]
        
        vector_store = VectorStore("/test/path", "test-model")
        
        # Perform search with non-existent course
        results = vector_store.search("decorators", course_name="nonexistent")
        
        # Should return error results
        assert not results.is_empty() or results.error is not None
        if results.error:
            assert "No course found matching 'nonexistent'" in results.error

    @patch('chromadb.PersistentClient')
    @patch('chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction')
    def test_search_with_custom_limit(self, mock_embedding_func, mock_client_class):
        """Test search with custom result limit"""
        # Setup mocks
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        
        mock_catalog = Mock()
        mock_content = Mock()
        mock_content.query.return_value = MockChromaResponse.successful_query_response()
        
        mock_client.get_or_create_collection.side_effect = [mock_catalog, mock_content]
        
        vector_store = VectorStore("/test/path", "test-model", max_results=5)
        
        # Perform search with custom limit
        results = vector_store.search("test query", limit=10)
        
        # Verify custom limit was used
        mock_content.query.assert_called_once()
        call_args = mock_content.query.call_args
        assert call_args[1]['n_results'] == 10

    @patch('chromadb.PersistentClient')
    @patch('chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction')
    def test_search_error_handling(self, mock_embedding_func, mock_client_class):
        """Test search error handling"""
        # Setup mocks
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        
        mock_catalog = Mock()
        mock_content = Mock()
        mock_content.query.side_effect = Exception("Database connection failed")
        
        mock_client.get_or_create_collection.side_effect = [mock_catalog, mock_content]
        
        vector_store = VectorStore("/test/path", "test-model")
        
        # Perform search that will fail
        results = vector_store.search("test query")
        
        # Should return error results
        assert results.error is not None
        assert "Search error: Database connection failed" in results.error


class TestVectorStoreDataManagement:
    """Test data addition and management in VectorStore"""
    
    @patch('chromadb.PersistentClient')
    @patch('chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction')
    def test_add_course_metadata(self, mock_embedding_func, mock_client_class):
        """Test adding course metadata to catalog"""
        # Setup mocks
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        
        mock_catalog = Mock()
        mock_content = Mock()
        
        mock_client.get_or_create_collection.side_effect = [mock_catalog, mock_content]
        
        vector_store = VectorStore("/test/path", "test-model")
        
        # Add course metadata
        sample_course = MockTestData.get_sample_courses()[0]
        vector_store.add_course_metadata(sample_course)
        
        # Verify add was called correctly
        mock_catalog.add.assert_called_once()
        call_args = mock_catalog.add.call_args
        
        assert call_args[1]['documents'] == [sample_course.title]
        assert call_args[1]['ids'] == [sample_course.title]
        
        metadata = call_args[1]['metadatas'][0]
        assert metadata['title'] == sample_course.title
        assert metadata['instructor'] == sample_course.instructor
        assert 'lessons_json' in metadata

    @patch('chromadb.PersistentClient')
    @patch('chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction')
    def test_add_course_content(self, mock_embedding_func, mock_client_class):
        """Test adding course content chunks"""
        # Setup mocks
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        
        mock_catalog = Mock()
        mock_content = Mock()
        
        mock_client.get_or_create_collection.side_effect = [mock_catalog, mock_content]
        
        vector_store = VectorStore("/test/path", "test-model")
        
        # Add course content
        sample_chunks = MockTestData.get_sample_course_chunks()[:2]
        vector_store.add_course_content(sample_chunks)
        
        # Verify add was called correctly
        mock_content.add.assert_called_once()
        call_args = mock_content.add.call_args
        
        assert len(call_args[1]['documents']) == 2
        assert len(call_args[1]['metadatas']) == 2
        assert len(call_args[1]['ids']) == 2
        
        # Check metadata structure
        metadata = call_args[1]['metadatas'][0]
        assert 'course_title' in metadata
        assert 'lesson_number' in metadata
        assert 'chunk_index' in metadata

    @patch('chromadb.PersistentClient')
    @patch('chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction')
    def test_add_empty_course_content(self, mock_embedding_func, mock_client_class):
        """Test adding empty course content list"""
        # Setup mocks
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        
        mock_catalog = Mock()
        mock_content = Mock()
        
        mock_client.get_or_create_collection.side_effect = [mock_catalog, mock_content]
        
        vector_store = VectorStore("/test/path", "test-model")
        
        # Add empty content
        vector_store.add_course_content([])
        
        # Should not call add on content collection
        mock_content.add.assert_not_called()

    @patch('chromadb.PersistentClient')
    @patch('chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction')
    def test_clear_all_data(self, mock_embedding_func, mock_client_class):
        """Test clearing all data from collections"""
        # Setup mocks
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        
        mock_catalog = Mock()
        mock_content = Mock()
        
        mock_client.get_or_create_collection.side_effect = [mock_catalog, mock_content, mock_catalog, mock_content]
        
        vector_store = VectorStore("/test/path", "test-model")
        
        # Clear all data
        vector_store.clear_all_data()
        
        # Verify collections were deleted and recreated
        assert mock_client.delete_collection.call_count == 2
        delete_calls = [call[0][0] for call in mock_client.delete_collection.call_args_list]
        assert "course_catalog" in delete_calls
        assert "course_content" in delete_calls


class TestVectorStoreUtilityMethods:
    """Test utility methods of VectorStore"""
    
    @patch('chromadb.PersistentClient')
    @patch('chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction')
    def test_get_existing_course_titles(self, mock_embedding_func, mock_client_class):
        """Test retrieving existing course titles"""
        # Setup mocks
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        
        mock_catalog = Mock()
        mock_catalog.get.return_value = {
            'ids': ['Course 1', 'Course 2', 'Course 3']
        }
        mock_content = Mock()
        
        mock_client.get_or_create_collection.side_effect = [mock_catalog, mock_content]
        
        vector_store = VectorStore("/test/path", "test-model")
        
        # Get existing titles
        titles = vector_store.get_existing_course_titles()
        
        assert titles == ['Course 1', 'Course 2', 'Course 3']
        mock_catalog.get.assert_called_once()

    @patch('chromadb.PersistentClient')
    @patch('chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction')
    def test_get_course_count(self, mock_embedding_func, mock_client_class):
        """Test getting course count"""
        # Setup mocks
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        
        mock_catalog = Mock()
        mock_catalog.get.return_value = {
            'ids': ['Course 1', 'Course 2']
        }
        mock_content = Mock()
        
        mock_client.get_or_create_collection.side_effect = [mock_catalog, mock_content]
        
        vector_store = VectorStore("/test/path", "test-model")
        
        # Get course count
        count = vector_store.get_course_count()
        
        assert count == 2
        mock_catalog.get.assert_called_once()

    @patch('chromadb.PersistentClient')
    @patch('chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction')
    def test_get_lesson_link(self, mock_embedding_func, mock_client_class):
        """Test retrieving lesson link for specific course and lesson"""
        # Setup mocks
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        
        mock_catalog = Mock()
        lessons_data = [
            {"lesson_number": 1, "title": "Intro", "lesson_link": "https://example.com/lesson1"},
            {"lesson_number": 3, "title": "Advanced", "lesson_link": "https://example.com/lesson3"}
        ]
        mock_catalog.get.return_value = {
            'metadatas': [{
                'lessons_json': json.dumps(lessons_data)
            }]
        }
        mock_content = Mock()
        
        mock_client.get_or_create_collection.side_effect = [mock_catalog, mock_content]
        
        vector_store = VectorStore("/test/path", "test-model")
        
        # Get lesson link
        link = vector_store.get_lesson_link("Test Course", 3)
        
        assert link == "https://example.com/lesson3"
        mock_catalog.get.assert_called_once_with(ids=["Test Course"])

    @patch('chromadb.PersistentClient')
    @patch('chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction')
    def test_get_lesson_link_not_found(self, mock_embedding_func, mock_client_class):
        """Test retrieving lesson link when lesson doesn't exist"""
        # Setup mocks
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        
        mock_catalog = Mock()
        lessons_data = [
            {"lesson_number": 1, "title": "Intro", "lesson_link": "https://example.com/lesson1"}
        ]
        mock_catalog.get.return_value = {
            'metadatas': [{
                'lessons_json': json.dumps(lessons_data)
            }]
        }
        mock_content = Mock()
        
        mock_client.get_or_create_collection.side_effect = [mock_catalog, mock_content]
        
        vector_store = VectorStore("/test/path", "test-model")
        
        # Get link for non-existent lesson
        link = vector_store.get_lesson_link("Test Course", 99)
        
        assert link is None