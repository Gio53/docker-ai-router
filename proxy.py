"""Local OpenAI-compatible proxy that routes chat requests by complexity.

Laya classifies the latest user message. Simple requests go to FreeLLMAPI.
Complex requests go to Z.ai. The full message body is forwarded either way.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
from contextlib import asynccontextmanager
from typing import Any

import httpx
import laya
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

log = logging.getLogger("docker-ai-router")

# Force Z.ai for game-panel server provisioning. Leave Lidarr/media and other
# short Docker ops on FreeLLMAPI — Laya + score>=2 still handles real complexity.
DEFAULT_FORCE_COMPLEX = (
    "pelican,wings,pelican panel,game server,minecraft server,"
    "create a server,create new server,new game server,provision server"
)

# The English checkpoint fits 512 tokens, including the question text.
# Classification sees only this excerpt. The upstream call gets the full body.
CLASSIFY_HEAD_CHARS = 800
CLASSIFY_TAIL_CHARS = 400

HOP_BY_HOP = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailers",
    "transfer-encoding",
    "upgrade",
    "content-encoding",
    "content-length",
    "content-type",
}

REQUIRED_ENV = ("FREE_LLM_API_KEY", "FREE_LLM_BASE_URL", "ZAI_API_KEY")


def _parse_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise SystemExit(f"Invalid {name}: {exc}") from exc


def _parse_keywords(raw: str | None) -> tuple[str, ...]:
    source = DEFAULT_FORCE_COMPLEX if raw is None else raw
    return tuple(part.strip().lower() for part in source.split(",") if part.strip())


class Settings:
    def __init__(self) -> None:
        missing = [name for name in REQUIRED_ENV if not os.environ.get(name)]
        if missing:
            raise SystemExit(
                "Missing required environment variables: " + ", ".join(missing)
            )

        self.listen_host = os.environ.get("LISTEN_HOST", "127.0.0.1")
        self.listen_port = int(os.environ.get("LISTEN_PORT", "8080"))
        self.free_llm_api_key = os.environ["FREE_LLM_API_KEY"]
        self.free_llm_base_url = os.environ["FREE_LLM_BASE_URL"]
        self.zai_api_key = os.environ["ZAI_API_KEY"]
        # Coding Plan quota uses /api/coding/paas/v4, not the general /api/paas/v4.
        self.zai_base_url = os.environ.get(
            "ZAI_BASE_URL", "https://api.z.ai/api/coding/paas/v4"
        )
        self.zai_model = os.environ.get("ZAI_MODEL", "glm-5.3-flash")
        # Hermes/OpenAI clients expect delta.content; thinking streams often look empty.
        self.zai_thinking = os.environ.get("ZAI_THINKING", "disabled").lower()
        # GLM-5.3 defaults to max effort, which can take many minutes. Prefer faster.
        self.zai_reasoning_effort = os.environ.get("ZAI_REASONING_EFFORT", "low")
        max_tokens_raw = os.environ.get("ZAI_MAX_TOKENS", "4096")
        try:
            self.zai_max_tokens = int(max_tokens_raw) if max_tokens_raw else None
        except ValueError as exc:
            raise SystemExit(f"Invalid ZAI_MAX_TOKENS: {exc}") from exc

        # Laya score 0 trivial, 1 easy, 2 moderate, 3 hard.
        self.complex_score_at = _parse_float("ROUTER_COMPLEX_SCORE_AT", 2.0)
        self.complex_tools_at = _parse_float("ROUTER_COMPLEX_TOOLS_AT", 0.5)
        # Empty string disables keyword overrides; unset uses DEFAULT_FORCE_COMPLEX.
        self.force_complex_keywords = _parse_keywords(
            os.environ.get("ROUTER_FORCE_COMPLEX_KEYWORDS")
        )

settings: Settings | None = None
agent: Any = None
questions: dict | None = None
http_client: httpx.AsyncClient | None = None
classify_lock: asyncio.Lock | None = None


def chat_completions_url(base_url: str) -> str:
    base = base_url.rstrip("/")
    if base.endswith("/chat/completions"):
        return base
    return f"{base}/chat/completions"


def excerpt_for_classifier(text: str) -> str:
    limit = CLASSIFY_HEAD_CHARS + CLASSIFY_TAIL_CHARS
    if len(text) <= limit:
        return text
    head = text[:CLASSIFY_HEAD_CHARS]
    tail = text[-CLASSIFY_TAIL_CHARS:]
    return f"{head}\n...\n{tail}"


def message_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict):
                text = part.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "\n".join(parts)
    if content is None:
        return ""
    return str(content)


def latest_user_text(messages: list) -> str:
    for message in reversed(messages):
        if isinstance(message, dict) and message.get("role") == "user":
            return message_text(message.get("content"))
    last = messages[-1]
    if isinstance(last, dict):
        return message_text(last.get("content"))
    return ""


def force_complex_match(text: str) -> str | None:
    """Return the matched keyword if ops/setup text should always use Z.ai."""
    assert settings is not None
    lowered = text.lower()
    for keyword in settings.force_complex_keywords:
        if keyword in lowered:
            return keyword
    return None


def classify(text: str) -> str:
    """Return 'simple' or 'complex'. Failures route to the complex backend."""
    assert settings is not None
    matched = force_complex_match(text)
    if matched:
        log.info("classified tier=complex reason=keyword match=%r chars=%d", matched, len(text))
        return "complex"

    excerpt = excerpt_for_classifier(text)
    try:
        result = agent.predict({"request": excerpt}, questions)
        answers = result.get("answers") or {}
        difficulty = answers.get("difficulty") or {}
        needs_tools = answers.get("needs_tools") or {}
        score = difficulty.get("score")
        if score is None:
            log.warning("Laya returned no difficulty score; routing to complex")
            return "complex"
        tools_score = float(needs_tools.get("noul") or 0.0)
        score_value = float(score)
        if (
            score_value >= settings.complex_score_at
            or tools_score >= settings.complex_tools_at
        ):
            tier = "complex"
        else:
            tier = "simple"
        log.info(
            "classified tier=%s difficulty=%.2f needs_tools=%.2f threshold=%.2f chars=%d",
            tier,
            score_value,
            tools_score,
            settings.complex_score_at,
            len(text),
        )
        return tier
    except Exception:
        log.exception("Laya classification failed; routing to complex")
        return "complex"


async def classify_tier(text: str) -> str:
    assert classify_lock is not None
    async with classify_lock:
        return await asyncio.to_thread(classify, text)


def backend_for(tier: str) -> tuple[str, str, str]:
    assert settings is not None
    if tier == "simple":
        return (
            chat_completions_url(settings.free_llm_base_url),
            settings.free_llm_api_key,
            "auto",
        )
    return (
        chat_completions_url(settings.zai_base_url),
        settings.zai_api_key,
        settings.zai_model,
    )


def error_body(message: str, error_type: str) -> dict:
    return {"error": {"message": message, "type": error_type}}


def response_headers(upstream: httpx.Response, tier: str) -> dict[str, str]:
    headers = {
        key: value
        for key, value in upstream.headers.items()
        if key.lower() not in HOP_BY_HOP
    }
    headers["X-Router-Tier"] = tier
    return headers


def prepare_zai_payload(payload: dict) -> dict:
    """Normalize Z.ai payload for OpenAI clients like Hermes."""
    assert settings is not None
    out = dict(payload)
    model = str(out.get("model") or settings.zai_model).lower()
    # GLM-5.3 family uses forced thinking and rejects type=disabled.
    forced_thinking = model.startswith("glm-5.3")
    if not isinstance(out.get("thinking"), dict):
        if forced_thinking:
            out["thinking"] = {"type": "enabled", "clear_thinking": False}
        elif settings.zai_thinking in {"1", "true", "yes", "enabled", "on"}:
            out["thinking"] = {"type": "enabled"}
        else:
            out["thinking"] = {"type": "disabled"}

    # Cap runaway generations unless the client already set a limit.
    if settings.zai_max_tokens and not out.get("max_tokens") and not out.get("max_completion_tokens"):
        out["max_tokens"] = settings.zai_max_tokens

    # Only applies when thinking is enabled (forced on 5.3).
    thinking = out.get("thinking")
    if (
        isinstance(thinking, dict)
        and thinking.get("type") == "enabled"
        and settings.zai_reasoning_effort
        and "reasoning_effort" not in out
    ):
        out["reasoning_effort"] = settings.zai_reasoning_effort
    return out


def normalize_zai_sse_line(line: str) -> str:
    """Map reasoning_content into content so OpenAI clients see visible text."""
    if not line.startswith("data:"):
        return line
    data = line[5:].lstrip()
    if not data or data == "[DONE]":
        return line
    try:
        event = json.loads(data)
    except Exception:
        return line

    changed = False
    for choice in event.get("choices") or []:
        delta = choice.get("delta")
        if not isinstance(delta, dict):
            continue
        content = delta.get("content")
        reasoning = delta.get("reasoning_content")
        if (not content) and isinstance(reasoning, str) and reasoning:
            delta["content"] = reasoning
            changed = True
        message = choice.get("message")
        if isinstance(message, dict):
            msg_content = message.get("content")
            msg_reasoning = message.get("reasoning_content")
            if (not msg_content) and isinstance(msg_reasoning, str) and msg_reasoning:
                message["content"] = msg_reasoning
                changed = True

    if not changed:
        return line
    return "data: " + json.dumps(event, ensure_ascii=False)


async def forward_complete(url: str, headers: dict, payload: dict, tier: str) -> Response:
    assert http_client is not None
    try:
        upstream = await http_client.post(url, headers=headers, json=payload)
    except httpx.HTTPError as exc:
        log.exception("Upstream request failed")
        return JSONResponse(error_body(f"Upstream request failed: {exc}", "api_error"), status_code=502)

    if tier == "complex" and upstream.is_success:
        try:
            data = upstream.json()
            for choice in data.get("choices") or []:
                message = choice.get("message")
                if not isinstance(message, dict):
                    continue
                if not message.get("content") and isinstance(
                    message.get("reasoning_content"), str
                ):
                    message["content"] = message["reasoning_content"]
            body = json.dumps(data, ensure_ascii=False).encode("utf-8")
            return Response(
                content=body,
                status_code=upstream.status_code,
                media_type="application/json",
                headers=response_headers(upstream, tier),
            )
        except Exception:
            log.exception("Failed to normalize Z.ai JSON response")

    content_type = upstream.headers.get("content-type", "application/json")
    return Response(
        content=upstream.content,
        status_code=upstream.status_code,
        media_type=content_type.split(";")[0].strip(),
        headers=response_headers(upstream, tier),
    )


async def forward_stream(url: str, headers: dict, payload: dict, tier: str) -> Response:
    assert http_client is not None
    request = http_client.build_request("POST", url, headers=headers, json=payload)
    try:
        upstream = await http_client.send(request, stream=True)
    except httpx.HTTPError as exc:
        log.exception("Upstream stream failed")
        return JSONResponse(error_body(f"Upstream request failed: {exc}", "api_error"), status_code=502)

    content_type = upstream.headers.get("content-type", "")
    if upstream.status_code >= 400 or (
        "text/event-stream" not in content_type and "json" in content_type
    ):
        error_bytes = await upstream.aread()
        await upstream.aclose()
        log.error(
            "Upstream stream error status=%s body=%s",
            upstream.status_code,
            error_bytes[:500],
        )
        return Response(
            content=error_bytes,
            status_code=upstream.status_code,
            media_type=(content_type.split(";")[0].strip() or "application/json"),
            headers=response_headers(upstream, tier),
        )

    async def body():
        buffer = ""
        try:
            async for chunk in upstream.aiter_text():
                if tier != "complex":
                    yield chunk.encode("utf-8")
                    continue
                buffer += chunk
                while "\n" in buffer:
                    line, buffer = buffer.split("\n", 1)
                    # Preserve bare newlines from upstream framing.
                    fixed = normalize_zai_sse_line(line.rstrip("\r"))
                    yield (fixed + "\n").encode("utf-8")
            if buffer:
                fixed = normalize_zai_sse_line(buffer.rstrip("\r"))
                yield (fixed + "\n").encode("utf-8")
        finally:
            await upstream.aclose()

    return StreamingResponse(
        body(),
        status_code=upstream.status_code,
        media_type="text/event-stream",
        headers=response_headers(upstream, tier),
    )


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global agent, questions, http_client, classify_lock
    if settings is None:
        raise RuntimeError("Settings were not loaded before startup")

    classify_lock = asyncio.Lock()
    log.info("Loading Laya checkpoint convaiinnovations/laya")
    agent = laya.load("convaiinnovations/laya")
    questions = laya.router_questions()
    http_client = httpx.AsyncClient(timeout=httpx.Timeout(600.0, connect=30.0))
    log.info("Laya ready")
    try:
        yield
    finally:
        await http_client.aclose()


app = FastAPI(lifespan=lifespan)


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@app.post("/v1/chat/completions")
async def chat_completions(request: Request) -> Response:
    try:
        body = await request.json()
    except Exception:
        return JSONResponse(
            error_body("Invalid JSON body", "invalid_request_error"),
            status_code=400,
        )

    if not isinstance(body, dict):
        return JSONResponse(
            error_body("JSON body must be an object", "invalid_request_error"),
            status_code=400,
        )

    messages = body.get("messages")
    if not isinstance(messages, list) or not messages:
        return JSONResponse(
            error_body("messages must be a non-empty array", "invalid_request_error"),
            status_code=400,
        )

    tier = await classify_tier(latest_user_text(messages))
    url, api_key, model = backend_for(tier)
    payload = dict(body)
    payload["model"] = model
    if tier == "complex":
        payload = prepare_zai_payload(payload)
    stream = bool(payload.get("stream"))
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "text/event-stream" if stream else "application/json",
    }
    if tier == "complex":
        headers["Accept-Language"] = "en-US,en"
    log.info("forwarding tier=%s model=%s url=%s stream=%s", tier, model, url, stream)

    if stream:
        return await forward_stream(url, headers, payload, tier)
    return await forward_complete(url, headers, payload, tier)


def main() -> None:
    global settings
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        stream=sys.stdout,
    )
    try:
        settings = Settings()
    except ValueError as exc:
        raise SystemExit(f"Invalid LISTEN_PORT: {exc}") from exc

    uvicorn.run(app, host=settings.listen_host, port=settings.listen_port, log_level="info")


if __name__ == "__main__":
    main()
