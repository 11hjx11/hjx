import json
import re
import requests
from typing import Optional, List, Dict, Any

# 可选可观测性上报（无 langfuse / 无配置时自动 no-op）
try:
    from .observability import record_generation
except ImportError:  # pragma: no cover
    def record_generation(*args, **kwargs):
        pass

# Qwen API 配置（从 config 读取，避免重复定义；config 不可用时用默认值）
try:
    from config import QWEN_CONFIG as _CFG_QWEN
    QWEN_API_URL = _CFG_QWEN.get("api_url", "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions")
    QWEN_MODEL = _CFG_QWEN.get("model", "qwen-max")
except ImportError:
    QWEN_API_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
    QWEN_MODEL = "qwen-max"

class LLMService:
    def __init__(self, provider: str = "mock", api_key: str = None, model: str = None):
        self.provider = provider
        self.api_key = api_key
        self.model = model or QWEN_MODEL
        self.conversation_history: List[Dict[str, str]] = []
    
    def chat(self, messages: List[Dict[str, str]], tools: List[Dict] = None) -> Dict[str, Any]:
        """
        Send messages to LLM and get response with optional function calling
        支持多种provider: mock, qwen, openai等
        """
        if self.provider == "mock":
            return self._mock_llm_response(messages, tools)
        else:
            # qwen, openai, 或其他真实LLM
            return self._call_real_llm(messages, tools)
    
    def _mock_llm_response(self, messages: List[Dict[str, str]], tools: List[Dict] = None) -> Dict[str, Any]:
        """
        Mock LLM that simulates function calling behavior
        In production, this would call a real LLM API
        """
        # Check if there's an observation (tool result) in the messages
        # If so, LLM should analyze the result and decide next action
        has_observation = any(msg.get("role") == "tool" for msg in messages)
        
        if has_observation:
            # This is a multi-step scenario: analyze observation and decide next step
            return self._handle_observation_response(messages)
        
        # First step: analyze user input and select tool
        user_message = messages[-1]["content"] if messages else ""
        tool_call = self._analyze_and_select_tool(user_message, tools)
        
        if tool_call:
            return {
                "role": "assistant",
                "content": None,
                "tool_calls": [tool_call]
            }
        else:
            return {
                "role": "assistant",
                "content": self._generate_natural_response(user_message)
            }
    
    def _handle_observation_response(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        """
        Handle observation from previous tool call
        LLM analyzes the result and decides next step or final answer
        """
        # Find the most recent tool message (observation)
        last_tool_message = None
        for msg in reversed(messages):
            if msg.get("role") == "tool":
                last_tool_message = msg
                break
        
        if not last_tool_message:
            return {
                "role": "assistant",
                "content": "抱歉，我遇到了一些问题。"
            }
        
        # Parse the observation result
        try:
            observation_data = json.loads(last_tool_message["content"])
        except Exception:
            observation_data = {"status": "unknown"}
        
        # Check if this is a user check that found a new user
        # In a multi-step scenario, Agent would check user first, then decide recommendation strategy
        if observation_data.get("tool") == "check_user":
            if observation_data.get("user_found"):
                # User exists, proceed with personalized recommendation
                user_id = observation_data.get("user_info", {}).get("user_id", "default_user")
                return {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{
                        "id": f"call_multi_{len(messages)}",
                        "type": "function",
                        "function": {
                            "name": "collaborative_filtering",
                            "arguments": json.dumps({"user_id": user_id})
                        }
                    }]
                }
            else:
                # New user, ask for preferences
                return {
                    "role": "assistant",
                    "content": "您好！我注意到您是新用户。请告诉我您喜欢什么菜系（如川菜、粤菜）和预算范围（经济型、中档、高档），我会为您推荐合适的餐厅。"
                }
        
        # If the observation contains recommendations, format final response
        if observation_data.get("result_type") == "recommendation":
            items = observation_data.get("items", [])
            if items:
                response = "为您推荐以下餐厅：\n"
                for i, item in enumerate(items[:5], 1):
                    reason = item.get("recommendation_reason", "")
                    response += f"{i}. {item.get('merchant_name', '未知餐厅')} - 评分: {item.get('score', '0')}分\n"
                response += "\n您可以告诉我更多偏好，我会继续为您调整推荐。"
                return {
                    "role": "assistant",
                    "content": response
                }
        
        # For other types of observations (statistics, analysis, etc.)
        if observation_data.get("status") == "success":
            # Return a natural language summary
            summary = self._summarize_observation(observation_data)
            return {
                "role": "assistant",
                "content": summary
            }
        
        # Default: return observation as text
        return {
            "role": "assistant",
            "content": f"已完成查询，结果如下：\n{json.dumps(observation_data, ensure_ascii=False, indent=2)[:500]}"
        }
    
    def _summarize_observation(self, data: Dict) -> str:
        """
        Summarize observation data into natural language
        """
        result_type = data.get("result_type", "")
        
        if result_type == "statistics":
            d = data.get("data", {})
            return (f"业务概览：\n"
                    f"• 总用户数：{d.get('total_users', 0)}\n"
                    f"• 总商户数：{d.get('total_merchants', 0)}\n"
                    f"• 总订单数：{d.get('total_orders', 0)}\n"
                    f"• 总金额：¥{d.get('total_amount', 0):.2f}\n"
                    f"• 平均订单金额：¥{d.get('avg_order_amount', 0):.2f}")
        
        elif result_type == "ranking":
            ranking = data.get("ranking", [])
            response = "热门餐厅排行榜：\n"
            for item in ranking[:5]:
                response += f"{item.get('rank')}. {item.get('merchant_name')} - 评分: {item.get('score')}分\n"
            return response
        
        elif result_type == "area_analysis":
            areas = data.get("areas", [])
            response = "区域商业分析：\n"
            for area in areas[:3]:
                response += f"• {area.get('area')}: {area.get('total_merchants')}家商户, {area.get('total_orders')}笔订单\n"
            return response
        
        elif result_type == "preference_analysis":
            cuisine = data.get("cuisine_preferences", [])
            price = data.get("price_preferences", [])
            response = "用户偏好分析：\n"
            response += "菜系偏好：\n"
            for item in cuisine[:3]:
                response += f"• {item.get('cuisine')}: {item.get('user_count')}人偏好\n"
            response += "价格偏好：\n"
            for item in price[:3]:
                response += f"• {item.get('price_range')}: {item.get('user_count')}人选择\n"
            return response
        
        return f"查询完成，共获取 {data.get('total_found', '未知')} 条结果。"
    
    def _analyze_and_select_tool(self, user_message: str, tools: List[Dict] = None) -> Optional[Dict]:
        """
        Simulate LLM function calling - analyze user message and select tool
        """
        if not tools:
            return None
        
        # Tool selection logic (simulating LLM reasoning)
        tool_selection_map = {
            "collaborative_filtering": {
                "keywords": ["推荐", "想吃", "商户", "餐厅"],
                "params_extractor": self._extract_user_id
            },
            "tag_based_recommendation": {
                "keywords": ["喜欢", "偏好", "菜系", "口味", "价格", "预算"],
                "params_extractor": self._extract_tag_params
            },
            "content_based_recommendation": {
                "keywords": ["热门", "排行", "最好", "随便", "不知道"]
            },
            "get_business_overview": {
                "keywords": ["概览", "总体", "总用户", "总订单", "业务"]
            },
            "get_area_analysis": {
                "keywords": ["区域", "地区", "分布"]
            },
            "get_top_merchants": {
                "keywords": ["top", "热门", "排行", "最好"]
            },
            "get_user_preferences": {
                "keywords": ["偏好", "喜欢什么", "消费习惯", "口味"]
            },
            "check_user": {
                "keywords": ["检查", "用户信息", "我的ID"]
            }
        }
        
        best_match = None
        best_score = 0
        
        for tool_name, config in tool_selection_map.items():
            score = 0
            for keyword in config["keywords"]:
                if keyword.lower() in user_message.lower():
                    score += 1
            
            if score > best_score:
                best_score = score
                best_match = tool_name
        
        if best_match and best_score > 0:
            tool_params = {}
            config = tool_selection_map.get(best_match, {})
            if "params_extractor" in config:
                tool_params = config["params_extractor"](user_message)
            
            return {
                "id": f"call_{hash(user_message) % 10000}",
                "type": "function",
                "function": {
                    "name": best_match,
                    "arguments": json.dumps(tool_params)
                }
            }
        
        return None
    
    def _extract_user_id(self, message: str) -> Dict[str, str]:
        patterns = [
            r'(?:用户ID|user_id|ID|用户id)\s*[:：]\s*(\w+)',
            r'(?:我是|我的)(\w+)(?:用户|账号)'
        ]
        for pattern in patterns:
            match = re.search(pattern, message, re.IGNORECASE)
            if match:
                return {"user_id": match.group(1)}
        return {"user_id": "default_user"}
    
    def _extract_tag_params(self, message: str) -> Dict[str, str]:
        params = {}
        
        cuisine_map = {
            '湘菜': ['湘菜', '湖南'],
            '川菜': ['川菜', '四川', '辣'],
            '粤菜': ['粤菜', '广东', '茶餐厅'],
            '鲁菜': ['鲁菜', '山东'],
            '苏菜': ['苏菜', '江苏', '淮扬'],
            '火锅': ['火锅', '麻辣'],
            '烧烤': ['烧烤', '烤肉', '串'],
            '西餐': ['西餐', '牛排', '法餐'],
            '日料': ['日料', '寿司', '拉面', '日本'],
            '韩料': ['韩料', '韩式', '烤肉', '泡菜'],
        }
        
        for cuisine, keywords in cuisine_map.items():
            for kw in keywords:
                if kw in message:
                    params["cuisine"] = cuisine
                    break
            if "cuisine" in params:
                break
        
        price_map = {
            "经济型": ["便宜", "实惠", "经济型", "平价"],
            "中档": ["适中", "中档", "中等"],
            "高档": ["贵", "高档", "豪华", "高端"]
        }
        
        for price_range, keywords in price_map.items():
            for kw in keywords:
                if kw in message:
                    params["price_range"] = price_range
                    break
            if "price_range" in params:
                break
        
        if not params:
            params = {"cuisine": "川菜", "price_range": "中档"}
        
        return params
    
    def _generate_natural_response(self, user_message: str) -> str:
        responses = [
            "我可以帮您推荐餐厅、查询业务数据、分析用户偏好等。请告诉我您需要什么帮助？",
            "您好！我是智能推荐助手，可以为您提供餐饮推荐服务。",
            "如果您想找餐厅，可以告诉我您的口味偏好和预算。",
            "我可以帮您查看热门商户、分析业务数据，或者为您推荐合适的餐厅。"
        ]
        return responses[0]
    
    def _call_real_llm(self, messages: List[Dict[str, str]], tools: List[Dict] = None) -> Dict[str, Any]:
        """
        Call real Qwen LLM API with function calling support
        使用阿里云通义千问API (OpenAI兼容模式)
        """
        if not self.api_key:
            print("Warning: No API key provided, falling back to mock LLM")
            return self._mock_llm_response(messages, tools)
        
        try:
            # 构建请求体
            payload = {
                "model": self.model,
                "messages": messages,
                "temperature": 0.7,
                "max_tokens": 2000
            }
            
            # 添加工具定义（Function Calling）
            if tools:
                payload["tools"] = tools
                payload["tool_choice"] = "auto"
            
            # 发送请求
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            print(f"[Qwen API] Calling model: {self.model}")
            print(f"[Qwen API] Messages count: {len(messages)}")
            if tools:
                print(f"[Qwen API] Tools provided: {len(tools)}")
            
            response = requests.post(
                QWEN_API_URL,
                headers=headers,
                json=payload,
                timeout=60
            )
            
            # 检查响应状态
            if response.status_code != 200:
                print(f"[Qwen API] Error: {response.status_code} - {response.text}")
                # 如果API调用失败，回退到mock模式
                print("[Qwen API] Falling back to mock LLM due to API error")
                return self._mock_llm_response(messages, tools)
            
            # 解析响应
            result = response.json()
            print(f"[Qwen API] Response received successfully")
            
            # 提取assistant消息
            choice = result.get("choices", [{}])[0]
            message = choice.get("message", {})
            
            # 构建返回格式（与mock格式一致）
            response_message = {
                "role": "assistant",
                "content": message.get("content")
            }
            
            # 如果有工具调用
            if "tool_calls" in message and message["tool_calls"]:
                response_message["tool_calls"] = message["tool_calls"]
                print(f"[Qwen API] Tool calls: {[tc['function']['name'] for tc in message['tool_calls']]}")
            
            # 记录token使用情况
            usage = result.get("usage", {})
            print(f"[Qwen API] Tokens used: prompt={usage.get('prompt_tokens', 0)}, completion={usage.get('completion_tokens', 0)}")

            # 上报到 LangFuse（无配置时自动跳过）
            try:
                record_generation(
                    name=f"llm_chat_{self.model}",
                    input=messages,
                    output=response_message,
                    model=self.model,
                    metadata={"provider": self.provider, "tools_count": len(tools) if tools else 0},
                    usage={
                        "input": usage.get("prompt_tokens", 0),
                        "output": usage.get("completion_tokens", 0),
                        "unit": "TOKENS",
                    } if usage else None,
                )
            except Exception:
                pass

            return response_message
            
        except requests.exceptions.Timeout:
            print("[Qwen API] Request timeout, falling back to mock LLM")
            return self._mock_llm_response(messages, tools)
            
        except requests.exceptions.RequestException as e:
            print(f"[Qwen API] Request error: {e}, falling back to mock LLM")
            return self._mock_llm_response(messages, tools)
            
        except Exception as e:
            print(f"[Qwen API] Unexpected error: {e}, falling back to mock LLM")
            return self._mock_llm_response(messages, tools)
    
    def clear_history(self):
        self.conversation_history = []

    # ============================================================
    # 流式输出 (SSE)
    # ============================================================
    def stream_chat(self, messages: List[Dict[str, str]], tools: List[Dict] = None):
        """
        流式调用 LLM，逐 token yield 内容。

        - 真实 provider: 调用 Qwen stream API，yield delta content
        - mock provider: 把完整回复按字/词切片 yield（模拟流式）

        Yields:
            str: 内容片段（增量）
        """
        if self.provider == "mock" or not self.api_key:
            # mock 模式：生成完整回复后按片 yield
            full = self.chat(messages, tools).get("content", "")
            if not full:
                return
            # 按句子/标点切片，模拟流式体验
            import re as _re
            chunks = _re.split(r'([。！？\n；;,，])', full)
            buf = ""
            for c in chunks:
                buf += c
                if len(buf) >= 4 or c in "。！？\n；;":
                    yield buf
                    buf = ""
            if buf:
                yield buf
            return

        # 真实流式：Qwen OpenAI 兼容 stream API
        try:
            payload = {
                "model": self.model,
                "messages": messages,
                "temperature": 0.7,
                "max_tokens": 2000,
                "stream": True,
            }
            if tools:
                payload["tools"] = tools
                payload["tool_choice"] = "auto"

            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }

            response = requests.post(
                QWEN_API_URL,
                headers=headers,
                json=payload,
                timeout=60,
                stream=True,
            )

            if response.status_code != 200:
                print(f"[Qwen Stream] Error {response.status_code}, falling back to non-stream")
                full = self._call_real_llm(messages, tools).get("content", "")
                yield full
                return

            # 解析 SSE 流
            for line in response.iter_lines(decode_unicode=True):
                if not line:
                    continue
                if line.startswith("data: "):
                    data = line[6:]
                elif line.startswith("data:"):
                    data = line[5:]
                else:
                    continue
                if data.strip() == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                    choices = chunk.get("choices", [])
                    if choices:
                        delta = choices[0].get("delta", {})
                        content = delta.get("content")
                        if content:
                            yield content
                except json.JSONDecodeError:
                    continue

        except requests.exceptions.Timeout:
            print("[Qwen Stream] Timeout, falling back to non-stream")
            full = self._call_real_llm(messages, tools).get("content", "")
            yield full
        except Exception as e:
            print(f"[Qwen Stream] Error: {e}, falling back to non-stream")
            full = self._call_real_llm(messages, tools).get("content", "")
            yield full