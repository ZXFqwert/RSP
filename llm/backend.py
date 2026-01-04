"""
LLM Backend for RSP experiments (Robust Version)
Handles Azure Content Filters by returning a failure signal instead of crashing.
"""

import os
import time
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class LLMBackend:
    def __init__(
        self,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        model: str = "gpt-4.1",
        timeout: int = 120,
    ):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.api_base = api_base or os.getenv("OPENAI_API_BASE")
        self.model = model
        self.timeout = timeout

        if not self.api_key:
            raise ValueError("API key not found. Set OPENAI_API_KEY environment variable.")

        try:
            from openai import OpenAI
            # 兼容处理 base_url (Azure 必需)
            if self.api_base:
                self.client = OpenAI(api_key=self.api_key, base_url=self.api_base, timeout=self.timeout)
            else:
                self.client = OpenAI(api_key=self.api_key, timeout=self.timeout)
        except ImportError:
            raise ImportError("openai package not installed.")

    def chat(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        retries: int = 3,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Send chat completion request with error handling for Azure Content Filters
        """
        model = model or self.model
        import openai

        for attempt in range(retries):
            try:
                response = self.client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    **kwargs,
                )

                content = response.choices[0].message.content
                usage = response.usage

                return {
                    "content": content,
                    "total_tokens": usage.total_tokens if usage else 0,
                    "prompt_tokens": usage.prompt_tokens if usage else 0,
                    "completion_tokens": usage.completion_tokens if usage else 0,
                    "model": model,
                }

            except openai.BadRequestError as e:
                # === 关键修改：捕获 Azure 内容过滤错误 ===
                err_msg = str(e)
                # Azure 的错误信息通常包含 content_filter 或 content management policy
                if "content_filter" in err_msg or "content management policy" in err_msg:
                    print(f"  [WARNING] Azure Safety Filter Triggered! (Meta-RSP blocked)")
                    # 返回一个特殊的标记，告诉 Agent 发生了什么，而不是让程序崩溃
                    # Agent 的解析逻辑（regex）会失败，从而触发 fallback 机制（Finish[Error]）
                    return {
                        "content": "The response was blocked by the safety filter. [BLOCKED_BY_SAFETY_FILTER]",
                        "total_tokens": 0,
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                        "model": model,
                    }
                else:
                    # 其他真正的请求错误（如 Context Length Exceeded），还是打印出来
                    print(f"  [ERROR] Bad Request: {e}")
                    # 如果是最后一次尝试，返回错误信息而不是崩飞
                    if attempt == retries - 1:
                        return {"content": "Error: Bad Request", "total_tokens": 0}

            except openai.RateLimitError:
                wait_time = 2 ** attempt  # 指数退避 (1s, 2s, 4s...)
                if attempt < retries - 1:
                    print(f"  [WARNING] Rate limit hit. Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    print("  [ERROR] Rate limit exceeded max retries.")
            
            except Exception as e:
                print(f"  [ERROR] Unexpected LLM Error: {e}")
                if attempt == retries - 1:
                    return {"content": "Error: Unexpected", "total_tokens": 0}
                time.sleep(1)
        
        return {"content": "Error: Max retries exceeded.", "total_tokens": 0}

# Global instance for convenience
_default_backend: Optional[LLMBackend] = None

def get_llm_backend(**kwargs) -> LLMBackend:
    global _default_backend
    if _default_backend is None or kwargs:
        _default_backend = LLMBackend(**kwargs)
    return _default_backend