from pydantic import BaseModel, Field
from typing import List, Optional

class ChatRequest(BaseModel):
    prompt: str = Field(min_length=1)
    system: Optional[str] = None
    max_tokens: int = 512

class ChatResponse(BaseModel):
    text: str
    input_tokens: int
    output_tokens: int
    model: str

class EmbedRequest(BaseModel):
    texts: List[str]

class EmbedResponse(BaseModel):
    vectors: List[list[float]]
    dim: int
    model: str

class RootModel(BaseModel):
    status: str
    message: str

class HealthModel(BaseModel):
    status: str
