from enum import Enum
from pydantic import BaseModel,Field

class Role(BaseModel):
    USER="user"
    MODEL="model"

class Message(BaseModel):
    
    role:Role=Field(
        ...,
        "Owner of the message"
        ),
    content:str=Field(
        ...,
        min_length=1,
        description="Message content"
    )
    def to_api_format(self) -> dict:
        return {
            "role": self.role.value, 
            "parts": [{"text": self.content}]
        }
class Conversation(BaseModel):
    
    messages: list[Message] = Field(
        default_factory=list, 
        description="List of messages in the conversation",
    )    
    
    def add_message(self, role: Role, content: str) -> Message:
        message = Message(role=role, content=content)
        self.messages.append(message)
        return message
    
    def add_user_message(self, content: str) -> Message:
        return self.add_message(Role.USER, content)
    
    def add_model_message(self, content: str) -> Message:
        return self.add_message(Role.MODEL, content)
    
    def get_last_message(self) -> Message | None:
        if self.messages:
            return self.messages[-1]
        return None
    
    def to_api_format(self) -> list[dict]:
        return [msg.to_api_format() for msg in self.messages]
    
    def clear(self) -> None:
        self.messages.clear()
    
    def __len__(self) -> int:
        return len(self.messages)