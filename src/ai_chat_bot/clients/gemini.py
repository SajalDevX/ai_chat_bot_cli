import httpx
from ai_chat_bot.config import Settings, get_settings
from ai_chat_bot.utils.exceptions import (
    APIConnectionError,
    APIError,
    AuthenticationError,
    RateLimitError,
    ConfigurationError
)
import json
from ai_chat_bot.models import Conversation, ChatResponse, Role, TokenUsage
from typing import Generator
from ai_chat_bot.clients.base import BaseClient


class GeminiClient(BaseClient):
    "Client for Google's gemini Api"

    PROVIDER_NAME = "gemini"
    BASE_URL = "https://generativelanguage.googleapis.com/v1beta"

    def __init__(self, settings: Settings | None = None) -> None:
        
        settings = settings or get_settings()
    
        super().__init__(settings)
        if not settings.gemini_api_key:
            raise ConfigurationError("Gemini API key is not configured.")
        self.client = httpx.Client(
            timeout=httpx.Timeout(self.settings.timeout)
        )
        self._chat_url = (
            f"{self.BASE_URL}/models/{self.settings.gemini_model}:generateContent"
        )
        self._stream_url = (
            f"{self.BASE_URL}/models/{self.settings.gemini_model}:streamGenerateContent"
        )
        self.last_stream_usage: TokenUsage | None = None

    @property
    def model_name(self) -> str:
        """Return the current Gemini model name."""
        return self.settings.gemini_model

    def _build_payload(self,conversation:Conversation)-> dict:
        payload = {
            "generationConfig":{
                "maxOutputTokens":self.settings.max_tokens,
                "temperature": self.settings.temperature,
            }
        }
        if conversation.system_prompt:
            payload["system_instruction"] = {
                "parts": [{"text": conversation.system_prompt}]
            }
            
        contents =[]
        for msg in conversation.messages:
            role = "model" if msg.role == Role.ASSISTANT else "user"
            contents.append({
                "role":role,
                "parts":[{"text":msg.content}]
            })
        payload["contents"] = contents        
        return payload
        
    def chat(self,conversation:Conversation)->str:
        
        payload = self._build_payload(conversation)
        try:
            response = self.client.post(
                self._chat_url,
                json = payload,
                params={"key": self.settings.gemini_api_key}
            )
        except httpx.ConnectError as e:
            raise APIConnectionError(f"Failed to connect: {e}")
        except httpx.TimeoutException as e:
            raise APIConnectionError(f"Request timed out: {e}")
        
        self._handle_response_errors(response)
        return self._parse_response(response.json())
        
    def _handle_response_errors(self, response: httpx.Response) -> None:
        """Check for HTTP errors and raise appropriate exceptions.
        
        Args:
            response: The HTTP response to check
            
        Raises:
            AuthenticationError: For 401/403 responses
            RateLimitError: For 429 responses
            APIError: For other error responses
        """
        # Success - no error
        if response.status_code == 200:
            return
        
        # Get error message from response
        try:
            error_data = response.json()
            error_message = error_data.get("error", {}).get("message", response.text)
        except Exception:
            error_message = response.text
        
        # Map status codes to exceptions
        status = response.status_code
        
        if status == 400:
            raise APIError(f"Bad request: {error_message}", status_code=status)
        
        if status in (401, 403):
            raise AuthenticationError(f"Authentication failed: {error_message}")
        
        if status == 429:
            raise RateLimitError(f"Rate limited: {error_message}")
        
        if status >= 500:
            raise APIConnectionError(f"Server error ({status}): {error_message}")
        
        # Generic error for other status codes
        raise APIError(f"API error ({status}): {error_message}", status_code=status)
    
    def _parse_response(self,response=dict)-> ChatResponse:
        try:
            data = response
            candidates = data.get("candidates",[])
            if not candidates:
                raise APIError("No response generated")
            content = candidates[0].get("content",{})
            parts = content.get("parts",[])
            
            if not parts:
                raise APIError("Empty response from api")
            
            content = "".join([part.get("text","") for part in parts])
            if not content:
                raise APIError("Empty response from api",provider=self.PROVIDER_NAME)
            
            usage_data = data.get("usageMetadata",{})
            usage = TokenUsage(
            prompt_tokens=usage_data.get("promptTokenCount", 0),
            completion_tokens=usage_data.get("candidatesTokenCount", 0),
            )
            finish_reason = candidates[0].get("finishReason","").lower()
            
            return ChatResponse(
                content=content,
                model = self.settings.gemini_model,
                provider=self.PROVIDER_NAME,
                usage=usage,
                finish_reason=finish_reason
            )
        except KeyError as e:
            raise APIError(f"Unexpected response format: missing {e}")
        except Exception as e:
            if isinstance(e, APIError):
                raise
            raise APIError(f"Failed to parse response: {e}")
    
        
    def chat_stream(self, conversation: Conversation) -> Generator[str, None, None]:
        
        payload = self._build_payload(conversation)
        
        try:
            # Use stream=True for streaming response
            with self.client.stream(
                "POST",
                self._stream_url,
                json=payload,
                params={"key": self.settings.gemini_api_key, "alt": "sse"},
            ) as response:
                
                # Check for errors before streaming
                if response.status_code != 200:
                    # Read full response for error message
                    response.read()
                    self._handle_response_errors(response)
                
                # Stream the response line by line
                for chunk in self._parse_stream(response):
                    yield chunk
                    
        except httpx.ConnectError as e:
            raise APIConnectionError(f"Failed to connect: {e}")
        except httpx.TimeoutException as e:
            raise APIConnectionError(f"Request timed out: {e}")
    
    def _parse_stream(self, response: dict) -> Generator[str, None, None]:
        """Parse streaming response and yield text chunks.
        
        Gemini streaming uses Server-Sent Events (SSE) format:
        data: {"candidates": [{"content": {"parts": [{"text": "..."}]}}]}
        
        Args:
            response: The streaming HTTP response
            
        Yields:
            Text chunks extracted from SSE data
        """        
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
                    
                    candidates = data.get("candidates", [])
                    if candidates:
                        content = candidates[0].get("content", {})
                        parts = content.get("parts", [])
                        
                        for part in parts:
                            text = part.get("text", "")
                            if text:
                                yield text
                
                    usage_data = data.get("usageMetadata")  # Note: lowercase 'd'  
                    if usage_data:
                        self.last_stream_usage = TokenUsage(
                            prompt_tokens = usage_data.get("promptTokenCount",0),
                            completion_tokens = usage_data.get("candidatesTokenCount",0)
                        )              
                except json.JSONDecodeError:
                    continue
                    
    def close(self) -> None:
        self.client.close()    
        
    def __enter__(self) -> "GeminiClient":
        """Enter context manager - returns self."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit context manager - closes client."""
        self.close()    