"""Small auditable Kimi JSON client. Secrets never enter persisted requests."""
from __future__ import annotations

import hashlib
import json
import random
import socket
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from skillrsi_m1.judge import _json_object, read_model_config


class KimiClient:
    def __init__(self, config_path: Path, cache_dir: Path, *, max_retries: int = 6):
        self.key, self.model, self.url = read_model_config(config_path)
        self.cache_dir = cache_dir
        self.max_retries = max_retries
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.calls: list[dict[str, Any]] = []

    def ask(self, stage: str, system: str, user: str, *, max_tokens: int = 2200) -> tuple[dict[str, Any], str]:
        digest = hashlib.sha256(json.dumps([self.model, system, user, max_tokens], ensure_ascii=False).encode()).hexdigest()
        call_id = f"{stage}-{digest[:16]}"
        cache = self.cache_dir / f"{digest}.json"
        if cache.is_file():
            saved = json.loads(cache.read_text())
            self.calls.append({"call_id": call_id, "stage": stage, "cache_hit": True,
                               "api_calls": 0, "retries": 0, "prompt_tokens": 0,
                               "completion_tokens": 0, "cache_tokens": 0, "elapsed_sec": 0,
                               "origin_usage": saved.get("usage", {})})
            return saved["response"], call_id
        payload = {"model": self.model, "messages": [{"role": "system", "content": system},
                    {"role": "user", "content": user}], "temperature": 0,
                   "max_tokens": max_tokens, "thinking": {"type": "disabled"}}
        request = urllib.request.Request(self.url, data=json.dumps(payload, ensure_ascii=False).encode(),
                                         method="POST", headers={"Authorization": f"Bearer {self.key}",
                                                                  "Content-Type": "application/json"})
        start = time.monotonic()
        retries = 0
        for attempt in range(self.max_retries + 1):
            try:
                with urllib.request.urlopen(request, timeout=300) as response:
                    raw = json.load(response)
                content = raw["choices"][0]["message"]["content"]
                parsed = _json_object(content)
                usage = raw.get("usage") or {}
                prompt = int(usage.get("prompt_tokens") or 0)
                completion = int(usage.get("completion_tokens") or 0)
                cached = int((usage.get("prompt_tokens_details") or {}).get("cached_tokens") or
                             usage.get("cached_tokens") or 0)
                record = {"call_id": call_id, "stage": stage, "cache_hit": False,
                          "api_calls": attempt + 1, "retries": retries, "prompt_tokens": prompt,
                          "completion_tokens": completion, "cache_tokens": cached,
                          "elapsed_sec": round(time.monotonic() - start, 3),
                          "usage_raw": usage}
                self.calls.append(record)
                cache.write_text(json.dumps({"model": self.model, "request": {"system": system, "user": user},
                                             "response": parsed, "usage": record}, ensure_ascii=False, indent=2))
                return parsed, call_id
            except urllib.error.HTTPError as exc:
                if exc.code not in (429, 500, 502, 503, 504) or attempt >= self.max_retries:
                    raise RuntimeError(f"Kimi HTTP {exc.code}") from exc
                retry_after = exc.headers.get("Retry-After")
                delay = float(retry_after) if retry_after and retry_after.isdigit() else min(2 ** attempt, 60)
            except (urllib.error.URLError, TimeoutError, socket.timeout, OSError) as exc:
                if attempt >= self.max_retries:
                    raise RuntimeError(f"Kimi transport error: {type(exc).__name__}") from exc
                delay = min(2 ** attempt, 30)
            retries += 1
            time.sleep(delay + random.random())
        raise RuntimeError("Kimi retry exhausted")

    def usage(self) -> dict[str, Any]:
        return {"model": self.model, "api_calls": sum(x["api_calls"] for x in self.calls),
                "retries": sum(x["retries"] for x in self.calls),
                "prompt_tokens": sum(x["prompt_tokens"] for x in self.calls),
                "completion_tokens": sum(x["completion_tokens"] for x in self.calls),
                "cache_tokens": sum(x["cache_tokens"] for x in self.calls),
                "reused_prompt_tokens": sum(x.get("origin_usage", {}).get("prompt_tokens", 0) for x in self.calls if x["cache_hit"]),
                "reused_completion_tokens": sum(x.get("origin_usage", {}).get("completion_tokens", 0) for x in self.calls if x["cache_hit"]),
                "calls": self.calls}
