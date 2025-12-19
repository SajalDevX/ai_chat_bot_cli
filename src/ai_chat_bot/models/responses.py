from pydantic import BaseModel, Field, computed_field

class TokenUsage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    
    @computed_field
    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens
    
    def __str__(self):
        return f"in={self.prompt_tokens}, out={self.completion_tokens}, total={self.total_tokens}"
    
class ChatResponse(BaseModel):
    
    content: str
    model: str
    provider: str
    usage: TokenUsage = Field(default_factory=TokenUsage)
    finish_reason: str | None = None
    
    _PRICING: dict[str, dict[str, float]] = {
        # OpenAI
        "gpt-4o": {"input": 2.50, "output": 10.00},
        "gpt-4o-mini": {"input": 0.15, "output": 0.60},
        "gpt-4-turbo": {"input": 10.00, "output": 30.00},
        "gpt-3.5-turbo": {"input": 0.50, "output": 1.50},
        # Anthropic
        "claude-3-5-sonnet-20241022": {"input": 3.00, "output": 15.00},
        "claude-3-opus-20240229": {"input": 15.00, "output": 75.00},
        "claude-3-haiku-20240307": {"input": 0.25, "output": 1.25},
        # Gemini
        "gemini-1.5-flash": {"input": 0.075, "output": 0.30},
        "gemini-1.5-pro": {"input": 1.25, "output": 5.00},
        "gemini-2.0-flash-exp": {"input": 0.0, "output": 0.0},
    }
    
    @computed_field
    @property
    def total_tokens(self) -> int:
        return self.usage.total_tokens
    
    @computed_field
    @property
    def cost(self) -> float:
        pricing = self._PRICING.get(self.model)
        if not pricing:
            return 0.0
        input_cost = (self.usage.prompt_tokens / 1_000_000) * pricing["input"]
        output_cost = (self.usage.completion_tokens / 1_000_000) * pricing["output"]
        return input_cost + output_cost
    
    def __str__(self) -> str:
        preview = self.content[:80] + "..." if len(self.content) > 80 else self.content
        return f"[{self.provider}/{self.model}] {preview}"