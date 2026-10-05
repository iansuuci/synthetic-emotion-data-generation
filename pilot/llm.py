"""OpenRouter client. Retry, backoff, and checkpoint patterns follow notebook cells 6 and 12.

API keys come from the environment (OPENROUTER_API_KEY). Nothing here reads or writes a key.
"""

from __future__ import annotations

import json
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from openai import OpenAI

from pilot.config import IDEA_MODEL, WORKERS

ROOT = Path(__file__).resolve().parents[1]
_write_lock = threading.Lock()
_client: OpenAI | None = None


def load_env() -> None:
    path = ROOT / ".env"
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def redact(text: str) -> str:
    text = re.sub(r"sk-or-v1-[A-Za-z0-9]+", "sk-or-v1-REDACTED", text)
    text = re.sub(r"hf_[A-Za-z0-9]+", "hf_REDACTED", text)
    return text[:220]


def client() -> OpenAI:
    global _client
    if _client is None:
        load_env()
        key = os.environ.get("OPENROUTER_API_KEY", "").strip()
        if not key:
            raise RuntimeError("OPENROUTER_API_KEY is not set")
        _client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=key, timeout=120)
    return _client


def _retryable(exc: Exception) -> bool:
    message = str(exc)
    return any(
        token in message
        for token in ("429", "503", "402", "rate_limit", "overloaded", "credits", "in-flight", "timeout", "Unavailable")
    )


def chat(
    prompt: str,
    *,
    model: str,
    temperature: float,
    max_tokens: int,
    system: str | None = None,
    reasoning: str | None = None,
    seed: int | None = None,
) -> dict:
    """One completion. Returns content, finish_reason, token counts.

    reasoning is an OpenRouter effort: "none" for Luna stories, "minimal" for Gemini
    (that endpoint rejects "none", and its default reasoning budget truncates short JSON).
    """
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    extra = {"reasoning": {"effort": reasoning}} if reasoning else None
    last_error = "unknown"
    for attempt in range(6):
        try:
            kwargs = dict(
                model=model,
                temperature=temperature,
                max_completion_tokens=max_tokens,
                messages=messages,
            )
            if extra:
                kwargs["extra_body"] = extra
            if seed is not None:
                kwargs["seed"] = seed
            started = time.time()
            try:
                response = client().chat.completions.create(**kwargs)
            except TypeError:
                kwargs.pop("seed", None)
                response = client().chat.completions.create(**kwargs)
            choice = response.choices[0]
            reasoning_tokens = 0
            try:
                reasoning_tokens = response.usage.completion_tokens_details.reasoning_tokens or 0
            except Exception:
                pass
            return {
                "output": choice.message.content,
                "finish_reason": choice.finish_reason,
                "out_tok": getattr(response.usage, "completion_tokens", 0) or 0,
                "reasoning_tok": reasoning_tokens,
                "secs": round(time.time() - started, 1),
            }
        except Exception as exc:
            last_error = redact(str(exc))
            if _retryable(exc):
                time.sleep(min(5 * (2 ** attempt), 90))
                continue
            return {
                "output": None,
                "finish_reason": "ERROR",
                "out_tok": 0,
                "reasoning_tok": 0,
                "secs": 0,
                "error": last_error,
            }
    return {
        "output": None,
        "finish_reason": "RETRIES",
        "out_tok": 0,
        "reasoning_tok": 0,
        "secs": 0,
        "error": last_error,
    }


def gen_sync(prompts: list[str], temperature: float, *, model: str = IDEA_MODEL, max_tokens: int = 1500, workers: int = WORKERS) -> list[str | None]:
    """Batched idea generation. Same shape as notebook gen_sync: one string per prompt, None on failure."""

    def one(prompt: str) -> str | None:
        result = chat(prompt, model=model, temperature=temperature, max_tokens=max_tokens, reasoning="minimal")
        text = (result.get("output") or "").strip().strip("`")
        if text.lower().startswith("json"):
            text = text[4:].strip()
        return text or None

    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(one, prompts))


def parse_json(text: str):
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?", "", cleaned, flags=re.I).strip()
    cleaned = re.sub(r"```$", "", cleaned).strip()
    if cleaned.lower().startswith("json"):
        cleaned = cleaned[4:].strip()
    start_obj, start_arr = cleaned.find("{"), cleaned.find("[")
    if start_arr >= 0 and (start_obj < 0 or start_arr < start_obj):
        end = cleaned.rfind("]")
        cleaned = cleaned[start_arr : end + 1]
    elif start_obj >= 0:
        end = cleaned.rfind("}")
        cleaned = cleaned[start_obj : end + 1]
    return json.loads(cleaned)


def append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, ensure_ascii=False) + "\n"
    with _write_lock:
        with path.open("a") as handle:
            handle.write(line)
            handle.flush()


def load_jsonl(path: Path) -> dict[str, dict]:
    found: dict[str, dict] = {}
    if not path.exists():
        return found
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        key = record.get("key")
        if key:
            found[key] = record
    return found


def map_indexed(items: list, fn, workers: int = WORKERS) -> list:
    """Run fn over items, preserving order. fn returns a record that includes no ordering of its own."""
    if not items:
        return []
    results: list = [None] * len(items)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(fn, item): index for index, item in enumerate(items)}
        done = 0
        total = len(items)
        for future in as_completed(futures):
            index = futures[future]
            results[index] = future.result()
            done += 1
            if done % 25 == 0 or done == total:
                print(f"  {done}/{total}", flush=True)
    return results
