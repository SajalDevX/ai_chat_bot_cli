# src/ai_chat_bot/clients/groq_client.py
"""Groq API client - FREE and OpenAI-compatible!

Groq provides free access to models like:
- llama-3.3-70b-versatile (best)
- llama-3.1-8b-instant (fastest)
- mixtral-8x7b-32768
- gemma2-9b-it

API format is identical to OpenAI, just different URL!
"""

import json
from typing import Generator

import httpx

from ai_chat_bot.config import Settings, get_settings
from ai_chat_bot.clients.base import BaseClient
from ai_chat_bot.models import Conversation, ChatResponse, TokenUsage, Role
from ai_chat_bot.utils.exceptions import (
    APIError,
    AuthenticationError,
    APIConnectionError,
    ConfigurationError,
    RateLimitError,
)


class GroqClient(BaseClient):
    """Groq API client - FREE tier available!
    
    Groq is OpenAI-compatible, so the code is almost identical
    to OpenAIClient. Only differences:
    - Different BASE_URL
    - Different models
    - It's FREE!
    
    Usage:
        with GroqClient() as client:
            response = client.chat(conversation)
            print(response.content)
    """
    
    PROVIDER_NAME = "groq"
    BASE_URL = "https://api.groq.com/openai/v1"  # OpenAI-compatible endpoint!
    
    def __init__(self, settings: Settings | None = None) -> None:
        """Initialize Groq client."""
        settings = settings or get_settings()
        super().__init__(settings)
        
        if not self.settings.groq_api_key:
            raise ConfigurationError(
                "GROQ_API_KEY not set in environment",
                provider=self.PROVIDER_NAME
            )
        
        # Same auth style as OpenAI - Bearer token
        self.client = httpx.Client(
            timeout=httpx.Timeout(self.settings.timeout),
            headers={
                "Authorization": f"Bearer {self.settings.groq_api_key}",
                "Content-Type": "application/json",
            }
        )
        
        self._chat_url = f"{self.BASE_URL}/chat/completions"
        self.last_stream_usage : TokenUsage | None = None

    @property
    def model_name(self) -> str:
        """Return the current Groq model name."""
        return self.settings.groq_model

    def _build_payload(self, conversation: Conversation) -> dict:
        """Convert conversation to OpenAI/Groq format.
        
        Identical to OpenAI format!
        """
        messages = []
        
        if conversation.system_prompt:
            messages.append({
                "role": "system",
                "content": conversation.system_prompt
            })
        
        for msg in conversation.messages:
            messages.append({
                "role": msg.role.value,
                "content": msg.content
            })
        
        return {
            "model": self.settings.groq_model,
            "messages": messages,
            "max_tokens": self.settings.max_tokens,
            "temperature": self.settings.temperature,
        }
    
    def _parse_response(self, response_data: dict) -> ChatResponse:
        """Parse Groq response - identical to OpenAI format."""
        choices = response_data.get("choices", [])
        if not choices:
            raise APIError("No choices in response", provider=self.PROVIDER_NAME)
        
        content = choices[0].get("message", {}).get("content", "")
        if not content:
            raise APIError("Empty response from API", provider=self.PROVIDER_NAME)
        
        usage_data = response_data.get("usage", {})
        usage = TokenUsage(
            prompt_tokens=usage_data.get("prompt_tokens", 0),
            completion_tokens=usage_data.get("completion_tokens", 0),
        )
        
        finish_reason = choices[0].get("finish_reason", "")
        
        return ChatResponse(
            content=content,
            model=self.settings.groq_model,
            provider=self.PROVIDER_NAME,
            usage=usage,
            finish_reason=finish_reason,
        )
    
    def _handle_error(self, response: httpx.Response) -> None:
        """Handle errors - same as OpenAI."""
        if response.status_code == 200:
            return
        
        try:
            error_data = response.json()
            error_info = error_data.get("error", {})
            message = error_info.get("message", response.text)
        except Exception:
            message = response.text
        
        status = response.status_code
        
        if status == 401:
            raise AuthenticationError(message, provider=self.PROVIDER_NAME)
        elif status == 429:
            raise RateLimitError(message, provider=self.PROVIDER_NAME)
        elif status == 400:
            raise APIError(f"Bad request: {message}", provider=self.PROVIDER_NAME, status_code=status)
        elif status >= 500:
            raise APIConnectionError(f"Server error: {message}", provider=self.PROVIDER_NAME)
        else:
            raise APIError(f"API error ({status}): {message}", provider=self.PROVIDER_NAME, status_code=status)
    
    def chat(self, conversation: Conversation) -> ChatResponse:
        """Send conversation and get response."""
        payload = self._build_payload(conversation)
        
        try:
            response = self.client.post(self._chat_url, json=payload)
        except httpx.ConnectError as e:
            raise APIConnectionError(f"Failed to connect: {e}", provider=self.PROVIDER_NAME)
        except httpx.TimeoutException as e:
            raise APIConnectionError(f"Request timed out: {e}", provider=self.PROVIDER_NAME)
        
        self._handle_error(response)
        return self._parse_response(response.json())
    
    def chat_stream(self, conversation: Conversation) -> Generator[str, None, None]:
        """Stream response - same as OpenAI.

        Uses stream_options to include usage statistics in the final chunk.
        Token usage is stored in self.last_stream_usage after streaming.
        """
        payload = self._build_payload(conversation)
        payload["stream"] = True
        # Request usage stats in streaming mode (OpenAI-compatible feature)
        # Without this, we can't get token counts during streaming
        payload["stream_options"] = {"include_usage": True}
        
        try:
            with self.client.stream("POST", self._chat_url, json=payload) as response:
                if response.status_code != 200:
                    response.read()
                    self._handle_error(response)
                    
                for chunk in self._parse_stream(response):
                    yield chunk
                        
        except httpx.ConnectError as e:
            raise APIConnectionError(f"Failed to connect: {e}", provider=self.PROVIDER_NAME)
        except httpx.TimeoutException as e:
            raise APIConnectionError(f"Request timed out: {e}", provider=self.PROVIDER_NAME)
    
    

    def _parse_stream(self, response: dict) -> Generator[str, None, None]:
         
        self.last_stream_usage = None 
        # Iterate over lines in the stream
        for line in response.iter_lines():
            # Skip empty lines
            if not line:
                continue
            
            # SSE format: lines start with "data: "
            if line.startswith("data: "):
                json_str = line[6:]  # Remove "data: " prefix
                
                # Skip [DONE] marker
                if json_str.strip() == "[DONE]":
                    break
                
                try:
                    data = json.loads(json_str)
                    choices = data.get("choices",[])
                    if choices:
                            delta = choices[0].get("delta",{})
                            content = delta.get("content","")
                            
                            if content:
                                yield content
                
                    usage_data = data.get("usage") 
                    if usage_data:
                        self.last_stream_usage = TokenUsage(
                            prompt_tokens= usage_data.get("prompt_tokens",0),
                            completion_tokens= usage_data.get("completion_tokens",0)
                        )               
                except json.JSONDecodeError:
                    continue
            
    def close(self) -> None:
        """Close the HTTP client."""
        self.client.close()