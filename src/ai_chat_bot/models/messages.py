from enum import Enum
from pydantic import BaseModel, Field


class Role(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


class Message(BaseModel):
    role: Role 
    content: str 
    
    def __str__(self) -> str:
        preview = self.content[:5] if len(self.content) > 50 else self.content
        return f"{self.role.value}: {preview}..."

class Conversation(BaseModel):
    
    messages: list[Message] = Field(
        default_factory=list, 
    )    
    system_prompt: str | None = None
    
    def add_message(self, role: Role, content: str) -> Message:
        message = Message(role=role, content=content)
        self.messages.append(message)
        return message
    
    def add_user_message(self, content: str) -> Message:
        return self.add_message(Role.USER, content)
    
    def get_messages_for_api(self) -> list[Message]:
        messages = []
        
        if self.system_prompt:
            messages.append(Message(role=Role.SYSTEM, content=self.system_prompt))
        
        messages.extend(self.messages)
        return messages

    def add_assistant_message(self, content: str) -> Message:
        return self.add_message(Role.ASSISTANT, content)
        
    def set_system_prompt(self, prompt: str) -> None:
        self.system_prompt = prompt
    
    def get_last_message(self) -> Message | None:
        if self.messages:
            return self.messages[-1]
        return None
    
    def clear_all(self) -> None:
        self.messages.clear()
        self.system_prompt = None
    
    def to_api_format(self) -> list[dict]:
        return [msg.to_api_format() for msg in self.messages]
    
    def clear(self) -> None:
        self.messages.clear()
    
    def __len__(self) -> int:
        return len(self.messages)