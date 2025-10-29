import requests
import logging
from typing import List, Dict, Optional

# 使用包导入方式
from config import (
    LOCAL_LLM_URL,
    LOCAL_LLM_MODEL,
    LLM_TEMPERATURE,
    LOCAL_LLM_MAX_TOKENS
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LocalLLM:
    """Client class for interacting with LLM service"""
    def __init__(self, url: str = None):
        """Initialize LocalLLM instance
        
        Args:
            url: LLM service endpoint URL (optional, defaults to config value)
        """
        self.url = url or LOCAL_LLM_URL
        self.headers = {"Content-Type": "application/json"}
        self.context: List[Dict[str, str]] = []
        logger.info("LocalLLM instance initialized with URL: %s", self.url)
        

    def _send_request(self, messages: List[Dict[str, str]]) -> str:
        """Send request to LLM service
        
        Args:
            messages: List of message dictionaries
            
        Returns:
            Response text from LLM
        """
        data = {
            "model": LOCAL_LLM_MODEL,
            "messages": messages,
            "temperature": LLM_TEMPERATURE,
            "top_p": 0.8,  # 核采样参数，控制生成的多样性（0.7-0.9效果较好）
            "repetition_penalty": 1.05,  # 重复惩罚系数，减少重复内容生成（1.0-1.2）
            "max_tokens": LOCAL_LLM_MAX_TOKENS
        }
        try:
            logger.debug("Sending request to LLM service with data: %s", data)
            response = requests.post(self.url, headers=self.headers, json=data)
            response.raise_for_status()
            result = response.json()["choices"][0]["message"]["content"]
            logger.debug("Received response from LLM service: %s", result)
            return result
        except Exception as e:
            logger.error("Error in LLM request: %s", str(e))
            return f"Error: {str(e)}"

    def chat(self, user_content: str, assistant_content: Optional[str] = None, system_content: Optional[str] = None):
        """Single-turn conversation mode (no context)
        
        Args:
            user_content: User input content
            assistant_content: Optional assistant prompt
            system_content: Optional system prompt
            
        Returns:
            Response object with format similar to OpenAI API response
        """
        messages = []
        if system_content:
            messages.append({"role": "system", "content": system_content})
        if assistant_content:
            messages.append({"role": "assistant", "content": assistant_content})    
        messages.append({"role": "user", "content": user_content})
        
        # Get raw response text
        response_text = self._send_request(messages)
        
        # Create response object with OpenAI-like structure
        class MockMessage:
            def __init__(self, content):
                self.content = content
        
        class MockChoice:
            def __init__(self, message):
                self.message = message
        
        class MockResponse:
            def __init__(self, choices):
                self.choices = choices
        
        return MockResponse([MockChoice(MockMessage(response_text))])
    
if __name__ == "__main__":
    """Test cases for LocalLLM class"""
    
    # Create instance
    llm = LocalLLM()
    
    response = llm.chat("你好")
    print(response.choices[0].message.content)

    response = llm.chat("你好", "你是一个AI助手")
    print(response.choices[0].message.content)

    response = llm.chat("你好", "你是一个AI助手", "你只能回答与EDA相关的问题")
    print(response.choices[0].message.content)
          
    