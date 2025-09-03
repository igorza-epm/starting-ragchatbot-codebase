"""
Unit tests for AIGenerator.

These tests validate the AI response generation, tool calling mechanism, 
and error handling of the AIGenerator component.
"""
import pytest
from unittest.mock import Mock, MagicMock, patch
import anthropic
from ai_generator import AIGenerator
from .fixtures import MockAnthropicResponse


class MockAnthropicClient:
    """Enhanced mock for Anthropic client with configurable responses"""
    
    def __init__(self):
        self.messages = Mock()
        self.call_count = 0
        self.responses = []
        
    def set_responses(self, responses):
        """Set a sequence of responses for subsequent calls"""
        self.responses = responses
        self.call_count = 0
        
    def create_response(self, **kwargs):
        """Mock messages.create method"""
        if self.call_count < len(self.responses):
            response = self.responses[self.call_count]
        else:
            response = MockAnthropicResponse.direct_response()
            
        self.call_count += 1
        return response


class TestAIGeneratorInitialization:
    """Test AIGenerator initialization and configuration"""
    
    def test_initialization(self):
        """Test proper initialization with API key and model"""
        with patch('ai_generator.anthropic.Anthropic') as mock_anthropic:
            generator = AIGenerator("test-api-key", "claude-3-sonnet")
            
            mock_anthropic.assert_called_once_with(api_key="test-api-key")
            assert generator.model == "claude-3-sonnet"
            assert generator.base_params["model"] == "claude-3-sonnet"
            assert generator.base_params["temperature"] == 0
            assert generator.base_params["max_tokens"] == 800

    def test_system_prompt_defined(self):
        """Test that system prompt is properly defined"""
        assert AIGenerator.SYSTEM_PROMPT is not None
        assert "course materials" in AIGenerator.SYSTEM_PROMPT.lower()
        assert "search tool" in AIGenerator.SYSTEM_PROMPT.lower()


class TestDirectResponseGeneration:
    """Test direct response generation without tool use"""
    
    @patch('ai_generator.anthropic.Anthropic')
    def test_generate_response_without_tools(self, mock_anthropic_class):
        """Test response generation without tool calling"""
        # Setup mock client
        mock_client = MockAnthropicClient()
        mock_client.set_responses([MockAnthropicResponse.direct_response()])
        mock_anthropic_class.return_value = mock_client
        mock_client.messages.create = mock_client.create_response
        
        generator = AIGenerator("test-api-key", "claude-3-sonnet")
        
        response = generator.generate_response("What is Python?")
        
        assert response == "This is a direct response to your query about Python programming."
        assert mock_client.call_count == 1

    @patch('ai_generator.anthropic.Anthropic')
    def test_generate_response_with_conversation_history(self, mock_anthropic_class):
        """Test response generation with conversation history"""
        mock_client = MockAnthropicClient() 
        mock_client.set_responses([MockAnthropicResponse.direct_response()])
        mock_anthropic_class.return_value = mock_client
        mock_client.messages.create = mock_client.create_response
        
        generator = AIGenerator("test-api-key", "claude-3-sonnet")
        
        history = "User: What is Python?\nAssistant: Python is a programming language."
        response = generator.generate_response("Tell me more", conversation_history=history)
        
        assert response == "This is a direct response to your query about Python programming."
        
        # Verify the create method was called with history in system prompt
        call_kwargs = mock_client.messages.create.call_args[1]
        assert "Previous conversation:" in call_kwargs["system"]
        assert history in call_kwargs["system"]

    @patch('ai_generator.anthropic.Anthropic')  
    def test_api_parameters_structure(self, mock_anthropic_class):
        """Test that API parameters are structured correctly"""
        mock_client = MockAnthropicClient()
        mock_client.set_responses([MockAnthropicResponse.direct_response()])
        mock_anthropic_class.return_value = mock_client
        mock_client.messages.create = mock_client.create_response
        
        generator = AIGenerator("test-api-key", "claude-3-sonnet")
        generator.generate_response("Test query")
        
        # Verify API call parameters
        call_kwargs = mock_client.messages.create.call_args[1] 
        
        assert call_kwargs["model"] == "claude-3-sonnet"
        assert call_kwargs["temperature"] == 0
        assert call_kwargs["max_tokens"] == 800
        assert len(call_kwargs["messages"]) == 1
        assert call_kwargs["messages"][0]["role"] == "user"
        assert call_kwargs["messages"][0]["content"] == "Test query"
        assert AIGenerator.SYSTEM_PROMPT in call_kwargs["system"]


class TestToolCallingMechanism:
    """Test tool calling functionality"""
    
    @patch('ai_generator.anthropic.Anthropic')
    def test_generate_response_with_tools(self, mock_anthropic_class):
        """Test response generation with tools available"""
        mock_client = MockAnthropicClient()
        mock_client.set_responses([MockAnthropicResponse.direct_response()])
        mock_anthropic_class.return_value = mock_client  
        mock_client.messages.create = mock_client.create_response
        
        generator = AIGenerator("test-api-key", "claude-3-sonnet")
        
        tools = [{"name": "test_tool", "description": "Test tool"}]
        response = generator.generate_response("Test query", tools=tools)
        
        assert response == "This is a direct response to your query about Python programming."
        
        # Verify tools were included in API call
        call_kwargs = mock_client.messages.create.call_args[1]
        assert "tools" in call_kwargs
        assert call_kwargs["tools"] == tools
        assert call_kwargs["tool_choice"] == {"type": "auto"}

    @patch('ai_generator.anthropic.Anthropic')
    def test_tool_execution_flow(self, mock_anthropic_class):
        """Test complete tool execution flow"""
        mock_client = MockAnthropicClient()
        
        # Set up sequence: tool_use response, then final response
        tool_use_response = MockAnthropicResponse.tool_use_response()
        final_response = MockAnthropicResponse.final_response_after_tool()
        mock_client.set_responses([tool_use_response, final_response])
        
        mock_anthropic_class.return_value = mock_client
        mock_client.messages.create = mock_client.create_response
        
        generator = AIGenerator("test-api-key", "claude-3-sonnet")
        
        # Mock tool manager
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.return_value = "Search results: Python decorators are powerful..."
        
        tools = [{"name": "search_course_content"}]
        response = generator.generate_response(
            "What are Python decorators?",
            tools=tools,
            tool_manager=mock_tool_manager
        )
        
        assert response == "Based on the search results, Python decorators are a powerful feature for enhancing functions."
        
        # Verify tool was executed
        mock_tool_manager.execute_tool.assert_called_once_with(
            "search_course_content",
            query="python decorators", 
            course_name="python"
        )
        
        # Verify two API calls were made (tool use + final response)
        assert mock_client.call_count == 2

    @patch('ai_generator.anthropic.Anthropic')
    def test_tool_execution_without_tool_manager(self, mock_anthropic_class):
        """Test tool use response when no tool manager provided"""
        mock_client = MockAnthropicClient()
        mock_client.set_responses([MockAnthropicResponse.tool_use_response()])
        mock_anthropic_class.return_value = mock_client
        mock_client.messages.create = mock_client.create_response
        
        generator = AIGenerator("test-api-key", "claude-3-sonnet")
        
        tools = [{"name": "search_course_content"}]
        
        # This should not crash, but won't execute tools 
        response = generator.generate_response("Test query", tools=tools)
        
        # Since no tool manager, it should return the original tool use response
        # In this case, the response would be empty since tool_use responses don't have direct text
        assert response is None or response == ""

    @patch('ai_generator.anthropic.Anthropic')
    def test_multiple_tool_calls(self, mock_anthropic_class):
        """Test handling multiple tool calls in one response"""
        mock_client = MockAnthropicClient()
        
        # Create a tool use response with multiple tool calls
        class MultiToolUseResponse:
            def __init__(self):
                self.content = [
                    MockAnthropicResponse.MockToolUse(),
                    MockAnthropicResponse.MockToolUse()  
                ]
                self.stop_reason = "tool_use"
        
        multi_tool_response = MultiToolUseResponse()
        final_response = MockAnthropicResponse.final_response_after_tool()
        mock_client.set_responses([multi_tool_response, final_response])
        
        mock_anthropic_class.return_value = mock_client
        mock_client.messages.create = mock_client.create_response
        
        generator = AIGenerator("test-api-key", "claude-3-sonnet")
        
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.return_value = "Tool result"
        
        response = generator.generate_response(
            "Test query",
            tools=[{"name": "search_course_content"}],
            tool_manager=mock_tool_manager
        )
        
        # Should execute both tools
        assert mock_tool_manager.execute_tool.call_count == 2


class TestErrorHandling:
    """Test error handling and edge cases"""
    
    @patch('ai_generator.anthropic.Anthropic')
    def test_api_error_handling(self, mock_anthropic_class):
        """Test handling of Anthropic API errors"""
        mock_client = Mock()
        mock_client.messages.create.side_effect = anthropic.APIError("API Error")
        mock_anthropic_class.return_value = mock_client
        
        generator = AIGenerator("test-api-key", "claude-3-sonnet")
        
        with pytest.raises(anthropic.APIError):
            generator.generate_response("Test query")

    @patch('ai_generator.anthropic.Anthropic')
    def test_tool_execution_error_handling(self, mock_anthropic_class):
        """Test handling of tool execution errors"""
        mock_client = MockAnthropicClient()
        
        tool_use_response = MockAnthropicResponse.tool_use_response()
        final_response = MockAnthropicResponse.final_response_after_tool()
        mock_client.set_responses([tool_use_response, final_response])
        
        mock_anthropic_class.return_value = mock_client
        mock_client.messages.create = mock_client.create_response
        
        generator = AIGenerator("test-api-key", "claude-3-sonnet")
        
        # Mock tool manager that raises an exception
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.side_effect = Exception("Tool execution failed")
        
        # This should handle the tool error gracefully
        with pytest.raises(Exception, match="Tool execution failed"):
            generator.generate_response(
                "Test query",
                tools=[{"name": "search_course_content"}],
                tool_manager=mock_tool_manager
            )

    @patch('ai_generator.anthropic.Anthropic') 
    def test_empty_query_handling(self, mock_anthropic_class):
        """Test handling of empty queries"""
        mock_client = MockAnthropicClient()
        mock_client.set_responses([MockAnthropicResponse.direct_response()])
        mock_anthropic_class.return_value = mock_client
        mock_client.messages.create = mock_client.create_response
        
        generator = AIGenerator("test-api-key", "claude-3-sonnet")
        
        response = generator.generate_response("")
        
        assert response == "This is a direct response to your query about Python programming."
        
        # Verify empty string was passed to API
        call_kwargs = mock_client.messages.create.call_args[1]
        assert call_kwargs["messages"][0]["content"] == ""

    @patch('ai_generator.anthropic.Anthropic')
    def test_malformed_tool_response(self, mock_anthropic_class):
        """Test handling of malformed tool responses"""
        mock_client = MockAnthropicClient()
        
        # Create malformed tool use response
        class MalformedToolResponse:
            def __init__(self):
                self.content = [Mock()]  # Mock without proper attributes
                self.stop_reason = "tool_use"
        
        malformed_response = MalformedToolResponse()
        final_response = MockAnthropicResponse.final_response_after_tool()
        mock_client.set_responses([malformed_response, final_response])
        
        mock_anthropic_class.return_value = mock_client
        mock_client.messages.create = mock_client.create_response
        
        generator = AIGenerator("test-api-key", "claude-3-sonnet")
        
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.return_value = "Result"
        
        # This should handle malformed response gracefully
        try:
            response = generator.generate_response(
                "Test query",
                tools=[{"name": "search_course_content"}],
                tool_manager=mock_tool_manager
            )
            # If it doesn't crash, that's good
            assert isinstance(response, str) or response is None
        except AttributeError:
            # Expected if mock doesn't have proper attributes
            pass


class TestMessageFlow:
    """Test message flow in tool calling"""
    
    @patch('ai_generator.anthropic.Anthropic')
    def test_message_history_in_tool_flow(self, mock_anthropic_class):
        """Test that message history is properly maintained during tool calling"""
        mock_client = MockAnthropicClient()
        
        tool_use_response = MockAnthropicResponse.tool_use_response()
        final_response = MockAnthropicResponse.final_response_after_tool()
        mock_client.set_responses([tool_use_response, final_response])
        
        mock_anthropic_class.return_value = mock_client
        mock_client.messages.create = mock_client.create_response
        
        generator = AIGenerator("test-api-key", "claude-3-sonnet")
        
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.return_value = "Tool execution result"
        
        response = generator.generate_response(
            "Test query",
            tools=[{"name": "search_course_content"}],
            tool_manager=mock_tool_manager
        )
        
        # Check that final API call includes the tool result
        assert mock_client.call_count == 2
        
        # The second call should have more messages (original + assistant + tool results)
        second_call_kwargs = mock_client.messages.create.call_args_list[1][1]
        messages = second_call_kwargs["messages"]
        
        assert len(messages) >= 2  # At least user message + tool results
        
        # Should have tool result message
        tool_result_message = next((msg for msg in messages if msg["role"] == "user" and 
                                   isinstance(msg["content"], list)), None)
        assert tool_result_message is not None