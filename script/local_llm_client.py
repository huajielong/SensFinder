import requests
import logging
import time
import backoff
from typing import List, Dict, Optional, Any

# 使用包导入方式
from config.config import (
    LOCAL_LLM_URL,
    LOCAL_LLM_MODEL,
    TEMPERATURE,
    LOCAL_LLM_MAX_TOKENS,
    MAX_RETRY_COUNT,
    INITIAL_RETRY_INTERVAL,
    RETRY_INTERVAL_MULTIPLIER,
    API_TIMEOUT
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LocalLLMClient:
    """Client class for interacting with LLM service"""
    def __init__(self, url: str = None):
        """Initialize LocalLLMClient instance
        Args:
            url: LLM service endpoint URL (optional, defaults to config value)
        """
        self.url = url or LOCAL_LLM_URL
        self.headers = {"Content-Type": "application/json"}
        self.context: List[Dict[str, str]] = []
        logger.info("LocalLLMClient instance initialized with URL: %s", self.url)
        
    @backoff.on_exception(
        backoff.expo,
        (requests.RequestException, requests.Timeout),
        max_tries=MAX_RETRY_COUNT + 1,
        initial_interval=INITIAL_RETRY_INTERVAL,
        factor=RETRY_INTERVAL_MULTIPLIER
    )
    def _send_request(self, messages: List[Dict[str, str]]) -> str:
        """Send request to LLM service with retry mechanism
        
        Args:
            messages: List of message dictionaries
            
        Returns:
            Response text from LLM
            
        Raises:
            Exception: If request fails after retries
        """
        data = {
            "model": LOCAL_LLM_MODEL,
            "messages": messages,
            "temperature": TEMPERATURE,
            "top_p": 0.8,  # 核采样参数，控制生成的多样性（0.7-0.9效果较好）
            "repetition_penalty": 1.05,  # 重复惩罚系数，减少重复内容生成（1.0-1.2）
            "max_tokens": LOCAL_LLM_MAX_TOKENS
        }
        
        try:
            logger.debug("Sending request to LLM service with data: %s", data)
            start_time = time.time()
            response = requests.post(self.url, headers=self.headers, json=data, timeout=API_TIMEOUT)
            response.raise_for_status()
            end_time = time.time()
            process_time = end_time - start_time
            logger.info("LLM request completed in %.2f seconds", process_time)
            
            # Validate response format
            response_json = response.json()
            if not isinstance(response_json, dict):
                raise ValueError("Invalid response format: not a dictionary")
            if "choices" not in response_json:
                raise ValueError("Invalid response format: 'choices' key missing")
            if not response_json["choices"]:
                raise ValueError("Invalid response format: 'choices' is empty")
            if "message" not in response_json["choices"][0]:
                raise ValueError("Invalid response format: 'message' key missing")
            if "content" not in response_json["choices"][0]["message"]:
                raise ValueError("Invalid response format: 'content' key missing")
            
            result = response_json["choices"][0]["message"]["content"]
            logger.debug("Received valid response from LLM service")
            return result
        except requests.RequestException as e:
            logger.error("HTTP error in LLM request: %s", str(e))
            raise
        except ValueError as e:
            logger.error("Response validation error: %s", str(e))
            raise
        except Exception as e:
            logger.error("Unexpected error in LLM request: %s", str(e))
            raise

    def chat(self, user_content: str, assistant_content: Optional[str] = None, system_content: Optional[str] = None) -> Dict[str, Any]:
        """Single-turn conversation mode (no context)
        
        Args:
            user_content: User input content
            assistant_content: Optional assistant prompt
            system_content: Optional system prompt
            
        Returns:
            Response dictionary with format similar to OpenAI API response
        """
        messages = []
        if system_content:
            messages.append({"role": "system", "content": system_content})
        if assistant_content:
            messages.append({"role": "assistant", "content": assistant_content})    
        messages.append({"role": "user", "content": user_content})
        
        try:
            # Get raw response text
            response_text = self._send_request(messages)
            
            # Return OpenAI-like response structure as dictionary
            return {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": response_text
                        }
                    }
                ]
            }
        except Exception as e:
            logger.error("Error in chat function: %s", str(e))
            # Return error response in similar format
            return {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": f"Error: {str(e)}"
                        }
                    }
                ]
            }
    
if __name__ == "__main__":
    """Test cases for LocalLLMClient class"""
    
    # Create instance
    llm = LocalLLMClient()
    
    response = llm.chat("你好")
    print(response['choices'][0]['message']['content'])

    response = llm.chat("你好", "你是一个AI助手")
    print(response['choices'][0]['message']['content'])

    response = llm.chat("你好", "你是一个AI助手", "你只能回答与EDA相关的问题")
    print(response['choices'][0]['message']['content'])
          
    