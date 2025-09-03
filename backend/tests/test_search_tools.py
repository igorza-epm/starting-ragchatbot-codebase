"""
Unit tests for CourseSearchTool and ToolManager.

These tests validate the search functionality, error handling, and result formatting
of the course search tools.
"""
import pytest
from unittest.mock import Mock, patch
from search_tools import CourseSearchTool, ToolManager, Tool
from vector_store import SearchResults
from .fixtures import MockTestData


class TestCourseSearchTool:
    """Test cases for CourseSearchTool"""

    def test_get_tool_definition(self, course_search_tool):
        """Test that tool definition is properly structured"""
        definition = course_search_tool.get_tool_definition()
        
        assert definition["name"] == "search_course_content"
        assert "description" in definition
        assert "input_schema" in definition
        assert definition["input_schema"]["type"] == "object"
        
        properties = definition["input_schema"]["properties"]
        assert "query" in properties
        assert "course_name" in properties  
        assert "lesson_number" in properties
        
        required_fields = definition["input_schema"]["required"]
        assert "query" in required_fields
        assert "course_name" not in required_fields  # Optional
        assert "lesson_number" not in required_fields  # Optional

    def test_execute_with_valid_query(self, course_search_tool, mock_vector_store):
        """Test successful search execution with valid query"""
        mock_vector_store.search.return_value = MockTestData.get_sample_search_results()
        
        result = course_search_tool.execute("python decorators")
        
        # Verify vector store was called with correct parameters
        mock_vector_store.search.assert_called_once_with(
            query="python decorators",
            course_name=None,
            lesson_number=None
        )
        
        # Verify result formatting
        assert isinstance(result, str)
        assert "[Advanced Python Programming - Lesson 3]" in result
        assert "Python decorators are a powerful feature" in result
        assert "closure is a function" in result

    def test_execute_with_course_filter(self, course_search_tool, mock_vector_store):
        """Test search execution with course name filter"""
        mock_vector_store.search.return_value = MockTestData.get_sample_search_results()
        
        result = course_search_tool.execute("decorators", course_name="python")
        
        mock_vector_store.search.assert_called_once_with(
            query="decorators",
            course_name="python", 
            lesson_number=None
        )
        
        assert isinstance(result, str)
        assert "Advanced Python Programming" in result

    def test_execute_with_lesson_filter(self, course_search_tool, mock_vector_store):
        """Test search execution with lesson number filter"""
        mock_vector_store.search.return_value = MockTestData.get_sample_search_results()
        
        result = course_search_tool.execute("decorators", lesson_number=3)
        
        mock_vector_store.search.assert_called_once_with(
            query="decorators",
            course_name=None,
            lesson_number=3
        )
        
        assert isinstance(result, str)
        assert "Lesson 3" in result

    def test_execute_with_combined_filters(self, course_search_tool, mock_vector_store):
        """Test search execution with both course and lesson filters"""
        mock_vector_store.search.return_value = MockTestData.get_sample_search_results()
        
        result = course_search_tool.execute("decorators", course_name="python", lesson_number=3)
        
        mock_vector_store.search.assert_called_once_with(
            query="decorators",
            course_name="python",
            lesson_number=3
        )
        
        assert isinstance(result, str)
        assert "Advanced Python Programming - Lesson 3" in result

    def test_execute_with_empty_results(self, course_search_tool_empty, mock_vector_store_empty):
        """Test handling of empty search results"""
        result = course_search_tool_empty.execute("nonexistent content")
        
        assert result == "No relevant content found."
        
        # Test with filters
        result_with_course = course_search_tool_empty.execute("test", course_name="missing")
        assert "No relevant content found in course 'missing'." == result_with_course
        
        result_with_lesson = course_search_tool_empty.execute("test", lesson_number=99)
        assert "No relevant content found in lesson 99." == result_with_lesson

    def test_execute_with_search_error(self, course_search_tool_error, mock_vector_store_error):
        """Test handling of search errors"""
        result = course_search_tool_error.execute("test query")
        
        assert "Database connection failed" in result

    def test_source_tracking(self, course_search_tool, mock_vector_store):
        """Test that sources are properly tracked for UI display"""
        mock_vector_store.search.return_value = MockTestData.get_sample_search_results()
        mock_vector_store.get_lesson_link.return_value = "https://example.com/lesson3"
        
        # Execute search
        course_search_tool.execute("python decorators")
        
        # Check that sources were stored
        assert len(course_search_tool.last_sources) == 2
        
        source1 = course_search_tool.last_sources[0]
        assert source1["display"] == "Advanced Python Programming - Lesson 3"
        assert source1["link"] == "https://example.com/lesson3"
        
        source2 = course_search_tool.last_sources[1] 
        assert source2["display"] == "Advanced Python Programming - Lesson 3"
        assert source2["link"] == "https://example.com/lesson3"

    def test_source_tracking_without_lesson_numbers(self, course_search_tool, mock_vector_store):
        """Test source tracking when metadata doesn't include lesson numbers"""
        # Create results without lesson numbers
        results = SearchResults(
            documents=["Some course content"],
            metadata=[{"course_title": "Test Course", "chunk_index": 0}],
            distances=[0.1]
        )
        mock_vector_store.search.return_value = results
        
        course_search_tool.execute("test content")
        
        assert len(course_search_tool.last_sources) == 1
        source = course_search_tool.last_sources[0]
        assert source["display"] == "Test Course"
        assert source["link"] is None

    def test_empty_query_handling(self, course_search_tool, mock_vector_store):
        """Test handling of empty or whitespace queries"""
        # This should still call the vector store, as empty queries might be valid
        mock_vector_store.search.return_value = MockTestData.get_empty_search_results()
        
        result = course_search_tool.execute("")
        
        mock_vector_store.search.assert_called_once_with(
            query="",
            course_name=None,
            lesson_number=None  
        )
        
        assert "No relevant content found." in result


class TestToolManager:
    """Test cases for ToolManager"""

    def test_register_tool(self):
        """Test tool registration"""
        manager = ToolManager()
        mock_tool = Mock(spec=Tool)
        mock_tool.get_tool_definition.return_value = {"name": "test_tool"}
        
        manager.register_tool(mock_tool)
        
        assert "test_tool" in manager.tools
        assert manager.tools["test_tool"] == mock_tool

    def test_register_tool_without_name(self):
        """Test error handling when tool definition lacks name"""
        manager = ToolManager()
        mock_tool = Mock(spec=Tool)
        mock_tool.get_tool_definition.return_value = {"description": "No name"}
        
        with pytest.raises(ValueError, match="Tool must have a 'name'"):
            manager.register_tool(mock_tool)

    def test_get_tool_definitions(self, tool_manager):
        """Test retrieving all tool definitions"""
        definitions = tool_manager.get_tool_definitions()
        
        assert isinstance(definitions, list)
        assert len(definitions) == 1
        assert definitions[0]["name"] == "search_course_content"

    def test_execute_tool_success(self, tool_manager):
        """Test successful tool execution"""
        # Mock the tool's execute method
        tool = tool_manager.tools["search_course_content"]
        tool.execute.return_value = "Search completed successfully"
        
        result = tool_manager.execute_tool("search_course_content", query="test")
        
        assert result == "Search completed successfully"
        tool.execute.assert_called_once_with(query="test")

    def test_execute_nonexistent_tool(self, tool_manager):
        """Test execution of non-existent tool"""
        result = tool_manager.execute_tool("nonexistent_tool", query="test")
        
        assert result == "Tool 'nonexistent_tool' not found"

    def test_get_last_sources(self, tool_manager):
        """Test retrieving sources from last search operation"""
        # Setup tool with sources
        tool = tool_manager.tools["search_course_content"]
        tool.last_sources = [
            {"display": "Test Course - Lesson 1", "link": "https://example.com/lesson1"}
        ]
        
        sources = tool_manager.get_last_sources()
        
        assert len(sources) == 1
        assert sources[0]["display"] == "Test Course - Lesson 1"

    def test_get_last_sources_no_sources(self, tool_manager):
        """Test retrieving sources when none exist"""
        # Ensure tool has no sources
        tool = tool_manager.tools["search_course_content"]
        tool.last_sources = []
        
        sources = tool_manager.get_last_sources()
        
        assert sources == []

    def test_reset_sources(self, tool_manager):
        """Test resetting sources from all tools"""
        # Setup tool with sources
        tool = tool_manager.tools["search_course_content"] 
        tool.last_sources = [{"display": "Test", "link": None}]
        
        tool_manager.reset_sources()
        
        assert tool.last_sources == []

    def test_multiple_tools_registration(self):
        """Test registering multiple tools"""
        manager = ToolManager()
        
        # Create two mock tools
        tool1 = Mock(spec=Tool)
        tool1.get_tool_definition.return_value = {"name": "tool1"}
        
        tool2 = Mock(spec=Tool) 
        tool2.get_tool_definition.return_value = {"name": "tool2"}
        
        manager.register_tool(tool1)
        manager.register_tool(tool2)
        
        assert len(manager.tools) == 2
        assert "tool1" in manager.tools
        assert "tool2" in manager.tools
        
        definitions = manager.get_tool_definitions()
        assert len(definitions) == 2


class TestEdgeCases:
    """Test edge cases and error conditions"""

    def test_malformed_metadata(self, mock_vector_store):
        """Test handling of malformed metadata from vector store"""
        # Create results with malformed metadata
        results = SearchResults(
            documents=["Test content"],
            metadata=[{"invalid": "metadata"}],  # Missing expected fields
            distances=[0.1]
        )
        mock_vector_store.search.return_value = results
        
        tool = CourseSearchTool(mock_vector_store)
        result = tool.execute("test query")
        
        # Should handle gracefully
        assert isinstance(result, str)
        assert "[unknown" in result  # Default course title

    def test_mismatched_results_lengths(self, mock_vector_store):
        """Test handling when documents and metadata have different lengths"""
        # Create mismatched results
        results = SearchResults(
            documents=["Doc1", "Doc2"],
            metadata=[{"course_title": "Course1", "lesson_number": 1}],  # Only one metadata
            distances=[0.1, 0.2]
        )
        mock_vector_store.search.return_value = results
        
        tool = CourseSearchTool(mock_vector_store)
        result = tool.execute("test query")
        
        # Should handle gracefully without crashing
        assert isinstance(result, str)