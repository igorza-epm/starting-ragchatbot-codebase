"""
Integration tests for the RAG system.

These tests validate the end-to-end functionality of the RAG system, including
the interaction between components and overall query processing workflow.
"""
import pytest
import os
import tempfile
from unittest.mock import Mock, patch, MagicMock
from rag_system import RAGSystem
from config import Config
from .fixtures import MockTestData, MockAnthropicResponse


class TestRAGSystemInitialization:
    """Test RAG system initialization and component setup"""
    
    @patch('rag_system.VectorStore')
    @patch('rag_system.AIGenerator') 
    @patch('rag_system.DocumentProcessor')
    def test_rag_system_initialization(self, mock_doc_processor, mock_ai_gen, mock_vector_store):
        """Test that RAG system initializes all components correctly"""
        config = Config()
        config.ANTHROPIC_API_KEY = "test-key"
        
        rag_system = RAGSystem(config)
        
        # Verify all components were initialized
        mock_vector_store.assert_called_once_with(
            config.CHROMA_PATH, 
            config.EMBEDDING_MODEL, 
            config.MAX_RESULTS
        )
        mock_ai_gen.assert_called_once_with(
            config.ANTHROPIC_API_KEY, 
            config.ANTHROPIC_MODEL
        )
        mock_doc_processor.assert_called_once_with(
            config.CHUNK_SIZE, 
            config.CHUNK_OVERLAP
        )
        
        # Verify tool setup
        assert rag_system.tool_manager is not None
        assert rag_system.search_tool is not None
        assert len(rag_system.tool_manager.tools) == 1
        assert "search_course_content" in rag_system.tool_manager.tools

    def test_tool_registration(self):
        """Test that search tool is properly registered with tool manager"""
        config = Config()
        config.ANTHROPIC_API_KEY = "test-key"
        
        with patch('rag_system.VectorStore'), \
             patch('rag_system.AIGenerator'), \
             patch('rag_system.DocumentProcessor'):
            
            rag_system = RAGSystem(config)
            
            tool_definitions = rag_system.tool_manager.get_tool_definitions()
            assert len(tool_definitions) == 1
            assert tool_definitions[0]["name"] == "search_course_content"


class TestQueryProcessing:
    """Test end-to-end query processing"""
    
    @patch('rag_system.VectorStore')
    @patch('rag_system.AIGenerator')
    @patch('rag_system.DocumentProcessor')
    def test_successful_query_processing(self, mock_doc_processor, mock_ai_gen_class, mock_vector_store_class):
        """Test successful end-to-end query processing"""
        # Setup mocks
        mock_vector_store = Mock()
        mock_vector_store.search.return_value = MockTestData.get_sample_search_results()
        mock_vector_store.get_lesson_link.return_value = "https://example.com/lesson3"
        mock_vector_store_class.return_value = mock_vector_store
        
        mock_ai_gen = Mock()
        mock_ai_gen.generate_response.return_value = "Python decorators are powerful features for function enhancement."
        mock_ai_gen_class.return_value = mock_ai_gen
        
        config = Config()
        config.ANTHROPIC_API_KEY = "test-key"
        
        rag_system = RAGSystem(config)
        
        # Process query
        response, sources = rag_system.query("What are Python decorators?")
        
        # Verify response
        assert isinstance(response, str)
        assert "Python decorators are powerful features" in response
        
        # Verify AI generator was called with correct parameters
        mock_ai_gen.generate_response.assert_called_once()
        call_args = mock_ai_gen.generate_response.call_args
        
        assert "What are Python decorators?" in call_args[1]["query"]
        assert call_args[1]["tools"] is not None
        assert call_args[1]["tool_manager"] is not None
        
        # Verify sources were returned
        assert isinstance(sources, list)

    @patch('rag_system.VectorStore')
    @patch('rag_system.AIGenerator')
    @patch('rag_system.DocumentProcessor')
    def test_query_with_session_management(self, mock_doc_processor, mock_ai_gen_class, mock_vector_store_class):
        """Test query processing with session management"""
        # Setup mocks
        mock_vector_store = Mock()
        mock_vector_store.search.return_value = MockTestData.get_sample_search_results()
        mock_vector_store_class.return_value = mock_vector_store
        
        mock_ai_gen = Mock()
        mock_ai_gen.generate_response.return_value = "Response with session context"
        mock_ai_gen_class.return_value = mock_ai_gen
        
        config = Config()
        config.ANTHROPIC_API_KEY = "test-key"
        
        rag_system = RAGSystem(config)
        
        # Create a session
        session_id = "test_session_123"
        
        # First query
        response1, _ = rag_system.query("What are decorators?", session_id=session_id)
        
        # Second query in same session
        response2, _ = rag_system.query("Can you give me an example?", session_id=session_id)
        
        # Verify both queries were processed
        assert mock_ai_gen.generate_response.call_count == 2
        
        # Verify second call included conversation history
        second_call_args = mock_ai_gen.generate_response.call_args_list[1]
        assert second_call_args[1]["conversation_history"] is not None

    @patch('rag_system.VectorStore')
    @patch('rag_system.AIGenerator')
    @patch('rag_system.DocumentProcessor')
    def test_query_processing_error_handling(self, mock_doc_processor, mock_ai_gen_class, mock_vector_store_class):
        """Test error handling in query processing"""
        # Setup mocks to simulate AI generator error
        mock_vector_store = Mock()
        mock_vector_store_class.return_value = mock_vector_store
        
        mock_ai_gen = Mock()
        mock_ai_gen.generate_response.side_effect = Exception("AI generation failed")
        mock_ai_gen_class.return_value = mock_ai_gen
        
        config = Config()
        config.ANTHROPIC_API_KEY = "test-key"
        
        rag_system = RAGSystem(config)
        
        # This should raise the exception
        with pytest.raises(Exception, match="AI generation failed"):
            rag_system.query("Test query")

    @patch('rag_system.VectorStore')
    @patch('rag_system.AIGenerator')
    @patch('rag_system.DocumentProcessor')
    def test_source_cleanup_after_query(self, mock_doc_processor, mock_ai_gen_class, mock_vector_store_class):
        """Test that sources are properly cleaned up after query"""
        # Setup mocks
        mock_vector_store = Mock()
        mock_vector_store.search.return_value = MockTestData.get_sample_search_results()
        mock_vector_store_class.return_value = mock_vector_store
        
        mock_ai_gen = Mock()
        mock_ai_gen.generate_response.return_value = "Test response"
        mock_ai_gen_class.return_value = mock_ai_gen
        
        config = Config()
        config.ANTHROPIC_API_KEY = "test-key"
        
        rag_system = RAGSystem(config)
        
        # Setup initial sources in tool manager
        rag_system.search_tool.last_sources = [{"display": "Test", "link": None}]
        
        # Process query
        rag_system.query("Test query")
        
        # Verify sources were reset after query
        assert rag_system.search_tool.last_sources == []


class TestDocumentManagement:
    """Test document loading and management functionality"""
    
    def test_add_course_document_success(self):
        """Test successful course document addition"""
        with patch('rag_system.VectorStore') as mock_vector_store_class, \
             patch('rag_system.AIGenerator'), \
             patch('rag_system.DocumentProcessor') as mock_doc_processor_class:
            
            # Setup mocks
            mock_vector_store = Mock()
            mock_vector_store_class.return_value = mock_vector_store
            
            mock_doc_processor = Mock()
            sample_course = MockTestData.get_sample_courses()[0]
            sample_chunks = MockTestData.get_sample_course_chunks()[:2]
            mock_doc_processor.process_course_document.return_value = (sample_course, sample_chunks)
            mock_doc_processor_class.return_value = mock_doc_processor
            
            config = Config()
            config.ANTHROPIC_API_KEY = "test-key"
            
            rag_system = RAGSystem(config)
            
            # Add document
            course, chunk_count = rag_system.add_course_document("/fake/path/course.pdf")
            
            # Verify results
            assert course == sample_course
            assert chunk_count == 2
            
            # Verify vector store operations
            mock_vector_store.add_course_metadata.assert_called_once_with(sample_course)
            mock_vector_store.add_course_content.assert_called_once_with(sample_chunks)

    def test_add_course_document_error_handling(self):
        """Test error handling when document processing fails"""
        with patch('rag_system.VectorStore') as mock_vector_store_class, \
             patch('rag_system.AIGenerator'), \
             patch('rag_system.DocumentProcessor') as mock_doc_processor_class:
            
            # Setup mocks to simulate error
            mock_vector_store_class.return_value = Mock()
            
            mock_doc_processor = Mock()
            mock_doc_processor.process_course_document.side_effect = Exception("Processing failed")
            mock_doc_processor_class.return_value = mock_doc_processor
            
            config = Config()
            config.ANTHROPIC_API_KEY = "test-key"
            
            rag_system = RAGSystem(config)
            
            # Add document should handle error gracefully
            course, chunk_count = rag_system.add_course_document("/fake/path/course.pdf")
            
            assert course is None
            assert chunk_count == 0

    @patch('os.path.exists')
    @patch('os.listdir')
    def test_add_course_folder_success(self, mock_listdir, mock_exists):
        """Test successful folder processing"""
        with patch('rag_system.VectorStore') as mock_vector_store_class, \
             patch('rag_system.AIGenerator'), \
             patch('rag_system.DocumentProcessor') as mock_doc_processor_class:
            
            # Setup filesystem mocks
            mock_exists.return_value = True
            mock_listdir.return_value = ["course1.pdf", "course2.txt", "not_a_course.log"]
            
            # Setup component mocks
            mock_vector_store = Mock()
            mock_vector_store.get_existing_course_titles.return_value = []
            mock_vector_store_class.return_value = mock_vector_store
            
            mock_doc_processor = Mock()
            sample_courses = MockTestData.get_sample_courses()
            sample_chunks = MockTestData.get_sample_course_chunks()
            
            # Return different courses for different files
            mock_doc_processor.process_course_document.side_effect = [
                (sample_courses[0], sample_chunks[:2]),  # course1.pdf
                (sample_courses[1], sample_chunks[2:])   # course2.txt
            ]
            mock_doc_processor_class.return_value = mock_doc_processor
            
            config = Config()
            config.ANTHROPIC_API_KEY = "test-key"
            
            rag_system = RAGSystem(config)
            
            # Process folder
            total_courses, total_chunks = rag_system.add_course_folder("/fake/folder")
            
            # Verify results
            assert total_courses == 2
            assert total_chunks == 4  # 2 chunks per course
            
            # Verify both valid files were processed
            assert mock_doc_processor.process_course_document.call_count == 2

    @patch('os.path.exists')
    def test_add_course_folder_nonexistent_path(self, mock_exists):
        """Test handling of non-existent folder path"""
        mock_exists.return_value = False
        
        with patch('rag_system.VectorStore'), \
             patch('rag_system.AIGenerator'), \
             patch('rag_system.DocumentProcessor'):
            
            config = Config()
            config.ANTHROPIC_API_KEY = "test-key"
            
            rag_system = RAGSystem(config)
            
            total_courses, total_chunks = rag_system.add_course_folder("/nonexistent/folder")
            
            assert total_courses == 0
            assert total_chunks == 0


class TestCourseAnalytics:
    """Test course analytics functionality"""
    
    @patch('rag_system.VectorStore')
    @patch('rag_system.AIGenerator')
    @patch('rag_system.DocumentProcessor')
    def test_get_course_analytics(self, mock_doc_processor, mock_ai_gen, mock_vector_store_class):
        """Test course analytics retrieval"""
        mock_vector_store = Mock()
        mock_vector_store.get_course_count.return_value = 3
        mock_vector_store.get_existing_course_titles.return_value = [
            "Course 1", "Course 2", "Course 3"
        ]
        mock_vector_store_class.return_value = mock_vector_store
        
        config = Config()
        config.ANTHROPIC_API_KEY = "test-key"
        
        rag_system = RAGSystem(config)
        
        analytics = rag_system.get_course_analytics()
        
        assert analytics["total_courses"] == 3
        assert analytics["course_titles"] == ["Course 1", "Course 2", "Course 3"]
        
        # Verify vector store methods were called
        mock_vector_store.get_course_count.assert_called_once()
        mock_vector_store.get_existing_course_titles.assert_called_once()


class TestIntegrationWithRealComponents:
    """Integration tests with minimal mocking"""
    
    def test_tool_manager_integration(self):
        """Test integration between RAG system and tool manager"""
        with patch('rag_system.VectorStore') as mock_vector_store_class, \
             patch('rag_system.AIGenerator') as mock_ai_gen_class, \
             patch('rag_system.DocumentProcessor'):
            
            # Use real tool manager but mock other components
            mock_vector_store = Mock()
            mock_vector_store.search.return_value = MockTestData.get_sample_search_results()
            mock_vector_store_class.return_value = mock_vector_store
            
            mock_ai_gen_class.return_value = Mock()
            
            config = Config()
            config.ANTHROPIC_API_KEY = "test-key"
            
            rag_system = RAGSystem(config)
            
            # Test tool execution through tool manager
            result = rag_system.tool_manager.execute_tool(
                "search_course_content",
                query="test query"
            )
            
            # Should execute successfully
            assert isinstance(result, str)
            mock_vector_store.search.assert_called_once()

    def test_session_manager_integration(self):
        """Test integration with session manager"""
        with patch('rag_system.VectorStore'), \
             patch('rag_system.AIGenerator') as mock_ai_gen_class, \
             patch('rag_system.DocumentProcessor'):
            
            mock_ai_gen = Mock()
            mock_ai_gen.generate_response.return_value = "Test response"
            mock_ai_gen_class.return_value = mock_ai_gen
            
            config = Config()
            config.ANTHROPIC_API_KEY = "test-key"
            config.MAX_HISTORY = 2
            
            rag_system = RAGSystem(config)
            
            session_id = "test_session"
            
            # First exchange
            rag_system.query("First question", session_id)
            
            # Second exchange  
            rag_system.query("Second question", session_id)
            
            # Third exchange - should have conversation history
            rag_system.query("Third question", session_id)
            
            # Verify session manager stored the exchanges
            history = rag_system.session_manager.get_conversation_history(session_id)
            assert history is not None
            assert "First question" in history
            assert "Second question" in history