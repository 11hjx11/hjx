import json
from typing import List, Dict, Any
from .tools import TOOLS, get_tool_by_name, get_tools_schema
from .llm_service import LLMService

class RecommendationAgent:
    def __init__(self, llm_provider: str = "mock", api_key: str = None, model: str = None):
        self.llm = LLMService(provider=llm_provider, api_key=api_key, model=model)
        self.conversation_history: List[Dict[str, str]] = []
        self.tools = TOOLS
        self.tools_schema = get_tools_schema()
        self.max_iterations = 5
        
        # System prompt that defines the agent's role and capabilities
        self.system_prompt = """你是一个智能餐饮推荐助手，具有以下能力：
1. 基于用户历史行为进行协同过滤推荐
2. 基于用户偏好标签进行个性化推荐
3. 基于热度和评分进行内容推荐
4. 查询业务数据和统计信息
5. 分析用户偏好和区域分布

请根据用户输入，选择最合适的工具来完成任务。如果一个工具无法完成任务，可以组合使用多个工具。
最终请用自然语言总结结果，让用户易于理解。"""

    def run(self, user_input: str) -> Dict[str, Any]:
        """
        Main agent loop implementing ReAct (Thought-Action-Observation) pattern
        """
        # Add user message to history
        self.conversation_history.append({"role": "user", "content": user_input})
        
        # Initialize trace for step-by-step tracking
        trace = []
        final_response = None
        iteration = 0
        
        # Build messages for LLM
        messages = self._build_messages(user_input)
        
        while iteration < self.max_iterations:
            iteration += 1
            
            # Step 1: Thought - LLM decides what to do
            thought_step = {
                "iteration": iteration,
                "phase": "thought",
                "description": f"分析用户输入: {user_input[:100]}...",
                "messages_count": len(messages)
            }
            trace.append(thought_step)
            
            # Call LLM with function calling
            llm_response = self.llm.chat(messages, tools=self.tools_schema)
            
            # Step 2: Action - Execute tool if LLM requests it
            if "tool_calls" in llm_response and llm_response["tool_calls"]:
                tool_calls = llm_response["tool_calls"]
                
                # Add assistant message with tool calls
                assistant_message = {
                    "role": "assistant",
                    "content": llm_response.get("content"),
                    "tool_calls": tool_calls
                }
                messages.append(assistant_message)
                
                # Execute each tool call
                for tool_call in tool_calls:
                    tool_name = tool_call["function"]["name"]
                    try:
                        tool_args = json.loads(tool_call["function"]["arguments"])
                    except (json.JSONDecodeError, KeyError, TypeError) as e:
                        # LLM 返回的 arguments 非法，记录错误并跳过该调用
                        trace.append({
                            "iteration": iteration,
                            "phase": "action",
                            "tool_name": tool_name,
                            "error": f"参数解析失败: {e}"
                        })
                        continue

                    # Log the action
                    action_step = {
                        "iteration": iteration,
                        "phase": "action",
                        "tool_name": tool_name,
                        "tool_args": tool_args
                    }
                    trace.append(action_step)
                    
                    # Execute the tool
                    tool_result = self._execute_tool(tool_name, tool_args)
                    
                    # Step 3: Observation - Add tool result to messages
                    observation_message = {
                        "role": "tool",
                        "tool_call_id": tool_call["id"],
                        "content": json.dumps(tool_result, ensure_ascii=False)
                    }
                    messages.append(observation_message)
                    
                    # Log the observation
                    observation_step = {
                        "iteration": iteration,
                        "phase": "observation",
                        "tool_name": tool_name,
                        "result_summary": self._summarize_result(tool_result)
                    }
                    trace.append(observation_step)
                
            else:
                # LLM returned final text response
                final_response = llm_response.get("content", "")
                
                # Add assistant message to history
                self.conversation_history.append({
                    "role": "assistant",
                    "content": final_response
                })
                
                # Log final response
                final_step = {
                    "iteration": iteration,
                    "phase": "final",
                    "response": final_response
                }
                trace.append(final_step)
                
                break
        
        # If max iterations reached without final response
        if final_response is None:
            final_response = "抱歉，我正在思考中，请稍后再试。您也可以尝试更具体的描述。"
            self.conversation_history.append({
                "role": "assistant",
                "content": final_response
            })
        
        return {
            "response": final_response,
            "trace": trace,
            "iterations": iteration,
            "tools_used": list(set(step.get("tool_name", "") for step in trace if step.get("tool_name")))
        }

    def _build_messages(self, user_input: str) -> List[Dict[str, Any]]:
        """
        Build message list for LLM including system prompt and conversation history
        """
        messages = [
            {"role": "system", "content": self.system_prompt}
        ]
        
        # Add conversation history (exclude the last user message which is handled separately)
        for msg in self.conversation_history[:-1]:
            messages.append(msg)
        
        # Add current user message
        messages.append({"role": "user", "content": user_input})
        
        return messages

    def _execute_tool(self, tool_name: str, tool_args: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a tool by name with given arguments
        """
        tool = get_tool_by_name(tool_name)
        if tool:
            try:
                result = tool.execute(**tool_args)
                return {
                    "status": "success",
                    "data": result,
                    "summary": self._summarize_result(result)
                }
            except Exception as e:
                return {
                    "status": "error",
                    "message": str(e)
                }
        else:
            return {
                "status": "error",
                "message": f"Tool '{tool_name}' not found. Available tools: {[t.name for t in self.tools]}"
            }

    def _summarize_result(self, result: Dict[str, Any]) -> str:
        """
        Summarize tool result for display
        """
        if isinstance(result, dict):
            if "status" in result:
                if result["status"] == "success":
                    data = result.get("data", {})
                    if isinstance(data, list):
                        return f"返回 {len(data)} 条记录"
                    elif isinstance(data, dict):
                        keys = list(data.keys())[:5]
                        return f"返回数据: {', '.join(keys)}"
                else:
                    return f"错误: {result.get('message', '未知错误')}"
        return str(result)[:200]

    def get_available_tools(self) -> List[Dict[str, str]]:
        """
        Get list of available tools and their descriptions
        """
        return [{
            "name": tool.name,
            "description": tool.description
        } for tool in self.tools]

    def clear_memory(self):
        """
        Clear conversation history
        """
        self.conversation_history = []
        self.llm.clear_history()

    def get_stats(self) -> Dict[str, Any]:
        """
        Get agent statistics
        """
        return {
            "available_tools": len(self.tools),
            "conversation_length": len(self.conversation_history),
            "tools_list": [t.name for t in self.tools]
        }