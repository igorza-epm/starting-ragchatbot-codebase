import anthropic
from typing import List, Optional, Dict, Any

class AIGenerator:
    """Handles interactions with Anthropic's Claude API for generating responses"""
    
    # Static system prompt to avoid rebuilding on each call
    SYSTEM_PROMPT = """ You are an AI assistant specialized in course materials and educational content with access to a comprehensive search tool for course information.

Search Tool Usage:
- Use the search tool **only** for questions about specific course content or detailed educational materials
- **Up to 2 searches per query** - Use sequential searches for complex queries requiring multiple information sources
- For multi-part questions, search for each component separately and synthesize results
- For comparisons, search each subject separately then compare
- Synthesize search results into accurate, fact-based responses
- If search yields no results, state this clearly without offering alternatives

Response Protocol:
- **General knowledge questions**: Answer using existing knowledge without searching
- **Course-specific questions**: Search first, then answer
- **Complex queries**: Use multiple searches to gather comprehensive information
- **No meta-commentary**:
 - Provide direct answers only — no reasoning process, search explanations, or question-type analysis
 - Do not mention "based on the search results"

All responses must be:
1. **Brief, Concise and focused** - Get to the point quickly
2. **Educational** - Maintain instructional value
3. **Clear** - Use accessible language
4. **Example-supported** - Include relevant examples when they aid understanding
Provide only the direct answer to what was asked.
"""
    
    def __init__(self, api_key: str, model: str):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        
        # Pre-build base API parameters
        self.base_params = {
            "model": self.model,
            "temperature": 0,
            "max_tokens": 800
        }
    
    def generate_response(self, query: str,
                         conversation_history: Optional[str] = None,
                         tools: Optional[List] = None,
                         tool_manager=None) -> str:
        """
        Generate AI response with optional tool usage and conversation context.
        Supports up to 2 sequential tool calling rounds.
        
        Args:
            query: The user's question or request
            conversation_history: Previous messages for context
            tools: Available tools the AI can use
            tool_manager: Manager to execute tools
            
        Returns:
            Generated response as string
        """
        
        # Build system content efficiently - avoid string ops when possible
        system_content = (
            f"{self.SYSTEM_PROMPT}\n\nPrevious conversation:\n{conversation_history}"
            if conversation_history 
            else self.SYSTEM_PROMPT
        )
        
        # Initialize message history for this query
        messages = [{"role": "user", "content": query}]
        max_rounds = 2
        current_round = 0
        
        while current_round < max_rounds:
            # Prepare API call parameters
            api_params = {
                **self.base_params,
                "messages": messages,
                "system": system_content
            }
            
            # Add tools if available
            if tools:
                api_params["tools"] = tools
                api_params["tool_choice"] = {"type": "auto"}
            
            try:
                # Get response from Claude
                response = self.client.messages.create(**api_params)
                
                # If no tool use, return the response
                if response.stop_reason != "tool_use":
                    return response.content[0].text
                
                # If tool use but no tool manager, return empty response
                if not tool_manager:
                    return ""
                
                # Add assistant's response to message history
                messages.append({"role": "assistant", "content": response.content})
                
                # Execute tools and add results to message history
                tool_results = []
                try:
                    for content_block in response.content:
                        if content_block.type == "tool_use":
                            tool_result = tool_manager.execute_tool(
                                content_block.name, 
                                **content_block.input
                            )
                            
                            tool_results.append({
                                "type": "tool_result",
                                "tool_use_id": content_block.id,
                                "content": tool_result
                            })
                    
                    # Add tool results to message history
                    if tool_results:
                        messages.append({"role": "user", "content": tool_results})
                    
                except Exception as e:
                    # Handle tool execution errors gracefully
                    error_message = f"Tool execution failed: {str(e)}"
                    messages.append({
                        "role": "user", 
                        "content": [{"type": "tool_result", "tool_use_id": "error", "content": error_message}]
                    })
                
                current_round += 1
                
            except Exception as e:
                # Handle API errors
                if current_round == 0:
                    raise e
                # If we've made progress, return what we have so far
                return "An error occurred while processing your request."
        
        # If we've reached max rounds, make one final call without tools to get the answer
        final_params = {
            **self.base_params,
            "messages": messages,
            "system": system_content
        }
        
        try:
            final_response = self.client.messages.create(**final_params)
            return final_response.content[0].text
        except Exception:
            return "I was unable to complete your request due to technical issues."
    
