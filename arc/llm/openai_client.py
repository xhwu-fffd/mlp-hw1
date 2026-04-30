from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import time
from dataclasses import dataclass
from pathlib import Path
from urllib import error, request


DEFAULT_CONFIG_PATH = Path(__file__).resolve().with_name("config.json")


@dataclass
class OpenAIConfig:
    model: str
    api_key: str
    base_url: str
    temperature: float = 0.0
    seed: int = 42
    max_tokens: int = 4096

    @property
    def chat_completions_url(self) -> str:
        return f"{self.base_url.rstrip('/')}/chat/completions"


def load_config(config_path: str | Path | None = None) -> OpenAIConfig:
    path = Path(config_path) if config_path is not None else DEFAULT_CONFIG_PATH
    payload = json.loads(path.read_text(encoding="utf-8"))
    return OpenAIConfig(
        model=payload["model"],
        api_key=payload["api_key"],
        base_url=payload["base_url"],
        temperature=payload.get("temperature", 0.0),
        seed=payload.get("seed", 42),
        max_tokens=payload.get("max_tokens", 4096),
    )


def image_file_to_data_url(image_path: str | Path) -> str:
    image_path = Path(image_path)
    mime_type, _ = mimetypes.guess_type(image_path.name)
    if mime_type is None:
        mime_type = "image/jpeg"
    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def _extract_message_text(payload: dict) -> str:
    choices = payload.get("choices", [])
    if not choices:
        raise ValueError("API response does not contain any choices.")
    message = choices[0].get("message", {})
    content = message.get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
        return "\n".join(parts).strip()
    return str(content)


def _strip_code_fences(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()
    return cleaned


def _find_json_object(text: str) -> str:
    start = text.find("{")
    if start < 0:
        raise ValueError("No JSON object found in model response.")

    depth = 0
    in_string = False
    escape = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]

    raise ValueError("JSON object in model response is incomplete.")


def parse_json_response(text: str) -> dict:
    cleaned = _strip_code_fences(text)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        return json.loads(_find_json_object(cleaned))


def _post_json(
    url: str,
    headers: dict[str, str],
    payload: dict,
    timeout: int,
) -> dict:
    raw = json.dumps(payload).encode("utf-8")
    req = request.Request(url, data=raw, headers=headers, method="POST")
    with request.urlopen(req, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def create_chat_completion(
    messages: list[dict],
    config_path: str | Path | None = None,
    timeout: int = 180,
    expect_json: bool = False,
    retries: int = 3,
) -> dict:
    config = load_config(config_path)
    headers = {
        "Authorization": f"Bearer {config.api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": config.model,
        "messages": messages,
        "temperature": config.temperature,
        "seed": config.seed,
        "max_tokens": config.max_tokens,
    }

    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            response_payload = _post_json(
                url=config.chat_completions_url,
                headers=headers,
                payload={**payload, "response_format": {"type": "json_object"}} if expect_json else payload,
                timeout=timeout,
            )
            message_text = _extract_message_text(response_payload)
            return {
                "config": {
                    "model": config.model,
                    "base_url": config.base_url,
                },
                "raw_response": response_payload,
                "message_text": message_text,
                "parsed_json": parse_json_response(message_text) if expect_json else None,
            }
        except error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            # Some OpenAI-compatible services do not support response_format for multimodal requests.
            if expect_json and attempt == 1 and "response_format" in body:
                try:
                    response_payload = _post_json(
                        url=config.chat_completions_url,
                        headers=headers,
                        payload=payload,
                        timeout=timeout,
                    )
                    message_text = _extract_message_text(response_payload)
                    return {
                        "config": {
                            "model": config.model,
                            "base_url": config.base_url,
                        },
                        "raw_response": response_payload,
                        "message_text": message_text,
                        "parsed_json": parse_json_response(message_text),
                    }
                except Exception as retry_error:  # pragma: no cover - fallback branch
                    last_error = retry_error
            last_error = RuntimeError(f"HTTP {exc.code}: {body}")
        except Exception as exc:  # pragma: no cover - network/runtime failures
            last_error = exc

        if attempt < retries:
            time.sleep(2 * attempt)

    raise RuntimeError(f"Chat completion request failed after {retries} attempts: {last_error}") from last_error


def create_vision_json_completion(
    image_path: str | Path,
    prompt: str,
    system_prompt: str,
    config_path: str | Path | None = None,
    timeout: int = 180,
    retries: int = 3,
) -> dict:
    image_data_url = image_file_to_data_url(image_path)
    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": image_data_url,
                        "detail": "high",
                    },
                },
            ],
        },
    ]
    return create_chat_completion(
        messages=messages,
        config_path=config_path,
        timeout=timeout,
        expect_json=True,
        retries=retries,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Send one image-plus-prompt request to an OpenAI-compatible multimodal API.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument(
        "--system-prompt",
        type=str,
        default="You are a careful visual analyst. Return JSON only.",
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = create_vision_json_completion(
        image_path=args.image,
        prompt=args.prompt,
        system_prompt=args.system_prompt,
        config_path=args.config,
    )
    print(json.dumps(result["parsed_json"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
