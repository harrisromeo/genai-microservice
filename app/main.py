from fastapi import FastAPI, Response
from prometheus_client import Counter, generate_latest
from .schemas import ChatRequest, ChatResponse, EmbedRequest, EmbedResponse
from .service import generate_chat, embed_texts

app = FastAPI(title="GenAI Microservice")

chat_requests = Counter("chat_requests_total", "Total chat requests")
embed_requests = Counter("embed_requests_total", "Total embed requests")

@app.post("/chat", response_model=ChatResponse)
async def chat(body: ChatRequest) -> ChatResponse:
    chat_requests.inc()
    text, in_tok, out_tok, model = generate_chat(body.prompt, body.system, body.max_tokens)
    return ChatResponse(text=text, input_tokens=in_tok, output_tokens=out_tok, model=model)

@app.post("/embed", response_model=EmbedResponse)
async def embed(body: EmbedRequest) -> EmbedResponse:
    embed_requests.inc()
    vecs, dim, model = embed_texts(body.texts)
    return EmbedResponse(vectors=vecs, dim=dim, model=model)

@app.get("/metrics")
async def metrics() -> Response:
    return Response(generate_latest(), media_type="text/plain")
