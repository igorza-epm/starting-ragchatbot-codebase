"""
Test fixtures and sample data for RAG system tests.
"""
import json
from typing import List, Dict, Any
from models import Course, Lesson, CourseChunk
from vector_store import SearchResults


class MockTestData:
    """Container for mock test data"""
    
    @staticmethod
    def get_sample_courses() -> List[Course]:
        """Get sample course data for testing"""
        return [
            Course(
                title="Advanced Python Programming",
                instructor="Sarah Chen", 
                course_link="https://example.com/python-course",
                lessons=[
                    Lesson(lesson_number=1, title="Introduction to Python", lesson_link="https://example.com/lesson1"),
                    Lesson(lesson_number=2, title="Data Structures", lesson_link="https://example.com/lesson2"),
                    Lesson(lesson_number=3, title="Decorators and Closures", lesson_link="https://example.com/lesson3"),
                ]
            ),
            Course(
                title="Machine Learning Fundamentals",
                instructor="Dr. Alex Rodriguez",
                course_link="https://example.com/ml-course", 
                lessons=[
                    Lesson(lesson_number=1, title="Introduction to ML", lesson_link="https://example.com/ml-lesson1"),
                    Lesson(lesson_number=2, title="Linear Regression", lesson_link="https://example.com/ml-lesson2"),
                ]
            )
        ]
    
    @staticmethod
    def get_sample_course_chunks() -> List[CourseChunk]:
        """Get sample course chunks for testing"""
        return [
            CourseChunk(
                content="Python decorators are a powerful feature that allows you to modify or enhance functions without changing their code directly.",
                course_title="Advanced Python Programming", 
                lesson_number=3,
                chunk_index=0
            ),
            CourseChunk(
                content="A closure is a function that captures variables from its enclosing scope, allowing access to those variables even after the outer function returns.",
                course_title="Advanced Python Programming",
                lesson_number=3, 
                chunk_index=1
            ),
            CourseChunk(
                content="Machine learning is a subset of artificial intelligence that focuses on algorithms that can learn from and make predictions on data.",
                course_title="Machine Learning Fundamentals",
                lesson_number=1,
                chunk_index=0
            ),
            CourseChunk(
                content="Linear regression is a fundamental algorithm used to model the relationship between a dependent variable and independent variables.",
                course_title="Machine Learning Fundamentals",
                lesson_number=2,
                chunk_index=0
            )
        ]
    
    @staticmethod
    def get_sample_search_results() -> SearchResults:
        """Get sample search results for testing"""
        return SearchResults(
            documents=[
                "Python decorators are a powerful feature that allows you to modify or enhance functions without changing their code directly.",
                "A closure is a function that captures variables from its enclosing scope, allowing access to those variables even after the outer function returns."
            ],
            metadata=[
                {"course_title": "Advanced Python Programming", "lesson_number": 3, "chunk_index": 0},
                {"course_title": "Advanced Python Programming", "lesson_number": 3, "chunk_index": 1}
            ],
            distances=[0.1, 0.2]
        )
    
    @staticmethod
    def get_empty_search_results() -> SearchResults:
        """Get empty search results for testing"""
        return SearchResults(documents=[], metadata=[], distances=[])
    
    @staticmethod
    def get_error_search_results() -> SearchResults:
        """Get error search results for testing"""
        return SearchResults.empty("Database connection failed")


class MockChromaResponse:
    """Mock ChromaDB response data"""
    
    @staticmethod
    def successful_query_response() -> Dict[str, Any]:
        """Mock successful ChromaDB query response"""
        return {
            'documents': [[
                "Python decorators are a powerful feature that allows you to modify or enhance functions without changing their code directly.",
                "A closure is a function that captures variables from its enclosing scope."
            ]],
            'metadatas': [[
                {"course_title": "Advanced Python Programming", "lesson_number": 3, "chunk_index": 0},
                {"course_title": "Advanced Python Programming", "lesson_number": 3, "chunk_index": 1}
            ]],
            'distances': [[0.1, 0.2]]
        }
    
    @staticmethod  
    def empty_query_response() -> Dict[str, Any]:
        """Mock empty ChromaDB query response"""
        return {
            'documents': [[]],
            'metadatas': [[]],
            'distances': [[]]
        }
    
    @staticmethod
    def course_catalog_response() -> Dict[str, Any]:
        """Mock course catalog ChromaDB response"""
        return {
            'documents': [["Advanced Python Programming"]],
            'metadatas': [[{
                "title": "Advanced Python Programming",
                "instructor": "Sarah Chen",
                "course_link": "https://example.com/python-course",
                "lessons_json": json.dumps([
                    {"lesson_number": 1, "title": "Introduction to Python", "lesson_link": "https://example.com/lesson1"},
                    {"lesson_number": 3, "title": "Decorators and Closures", "lesson_link": "https://example.com/lesson3"}
                ]),
                "lesson_count": 3
            }]],
            'distances': [[0.0]]
        }


class MockAnthropicResponse:
    """Mock Anthropic API response data"""
    
    @staticmethod
    def direct_response():
        """Mock direct response without tool use"""
        class MockResponse:
            def __init__(self):
                self.content = [MockContent("This is a direct response to your query about Python programming.")]
                self.stop_reason = "end_turn"
        
        return MockResponse()
    
    @staticmethod
    def tool_use_response():
        """Mock response with tool use"""
        class MockResponse:
            def __init__(self):
                self.content = [MockToolUse()]
                self.stop_reason = "tool_use"
        
        return MockResponse()
    
    @staticmethod  
    def final_response_after_tool():
        """Mock final response after tool execution"""
        class MockResponse:
            def __init__(self):
                self.content = [MockContent("Based on the search results, Python decorators are a powerful feature for enhancing functions.")]
                self.stop_reason = "end_turn"
        
        return MockResponse()


class MockContent:
    """Mock content block for Anthropic responses"""
    def __init__(self, text: str):
        self.text = text


class MockToolUse:
    """Mock tool use block for Anthropic responses"""
    def __init__(self):
        self.type = "tool_use"
        self.id = "tool_123456"
        self.name = "search_course_content"
        self.input = {"query": "python decorators", "course_name": "python"}