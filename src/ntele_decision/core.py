"""Typed decisions, backend transport and deterministic wide-candidate policy.

Written independently from the public algorithm description, not vendor source.
SGLang owns tokenization, causal attention, prompt v1 and temperature application.
"""
import asyncio
import math
import time
from dataclasses import dataclass

import httpx


class InvalidRequest(ValueError):
    pass


class CapacityError(ValueError):
    pass


class BackendError(RuntimeError):
    pass


@dataclass(frozen=True)
class Settings:
    model: str = "NTele-Hybrid-27B-v1"
    backend_url: str = "http://127.0.0.1:7801"
    temperature: float = 1.26964
    max_options: int = 255
    backend_concurrency: int = 4
    timeout: float = 600

    def __post_init__(self):
        if not math.isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError("temperature must be finite and positive")
        if not 26 <= self.max_options <= 255 or self.backend_concurrency < 1:
            raise ValueError("invalid capacity or concurrency")


def distribution(value, keys):
    if not isinstance(value, dict) or set(value) != set(keys):
        raise BackendError("backend probability keys differ from requested options")
    if any(isinstance(p, bool) or not isinstance(p, (int, float)) or
           not math.isfinite(p) or not 0 <= p <= 1 for p in value.values()):
        raise BackendError("invalid backend probability")
    total = math.fsum(value.values())
    if abs(total - 1) > 0.01 or total <= 0:
        raise BackendError("backend probabilities do not sum to one")
    return {k: value[k] / total for k in keys}


def text(value):
    # Preserve structured text; represent empty inputs literally for RequiredDecisionText.
    return '""' if value == "" else "{}" if value == {} else "[]" if value == [] else value


def validate(body, cfg):
    if not isinstance(body, dict) or set(body) - {"model", "state", "questions"}:
        raise InvalidRequest("allowed fields: model, state, questions")
    if body.get("model", cfg.model) != cfg.model:
        raise InvalidRequest("model does not match the loaded backend")
    if "state" not in body or not isinstance(body["state"], (str, dict, list)):
        raise InvalidRequest("state must be text, object or list")
    qs = body.get("questions")
    if not isinstance(qs, dict) or not qs:
        raise InvalidRequest("questions must be a nonempty object")
    for key, q in qs.items():
        if not isinstance(key, str) or not key.strip() or not isinstance(q, dict):
            raise InvalidRequest("invalid question ID or specification")
        if set(q) - {"type", "instructions", "criteria"}:
            raise InvalidRequest("unknown question fields")
        if not isinstance(q.get("instructions", ""), (str, dict, list)):
            raise InvalidRequest("instructions must be text, object or list")
        typ, criteria = q.get("type"), q.get("criteria")
        if typ == "choice":
            if not isinstance(criteria, dict) or not criteria or any(not k.strip() for k in criteria):
                raise InvalidRequest("choice criteria must be a nonempty named object")
            if len(criteria) > cfg.max_options:
                raise CapacityError("options per choice exceed declared maximum")
        elif typ == "noul":
            if criteria is not None and (not isinstance(criteria, dict) or
                    set(criteria) - {"yes", "no", "true", "false"}):
                raise InvalidRequest("noul criteria use yes/no or true/false")
            if criteria and (("yes" in criteria and "true" in criteria) or
                             ("no" in criteria and "false" in criteria)):
                raise InvalidRequest("ambiguous binary criteria")
        elif typ == "score":
            if not isinstance(criteria, (dict, list)) or not 2 <= len(criteria) <= 10:
                raise CapacityError("score supports 2 to 10 ordered levels")
            if isinstance(criteria, dict):
                try:
                    levels = [float(k) for k in criteria]
                except ValueError as exc:
                    raise InvalidRequest("score keys must be numeric") from exc
                if any(not math.isfinite(v) for v in levels) or any(a >= b for a, b in zip(levels, levels[1:])):
                    raise InvalidRequest("score keys must be finite and strictly increasing")
        else:
            raise InvalidRequest("unsupported question type")
        if criteria is not None:
            values = criteria.values() if isinstance(criteria, dict) else criteria
            if any(v is not None and not isinstance(v, (str, dict, list)) for v in values):
                raise InvalidRequest("criteria descriptions must be structured text")
    return qs


class SGLangBackend:
    def __init__(self, cfg):
        self.cfg = cfg
        # Explicit local backend must not inherit corporate/external HTTP proxies.
        self.client = httpx.AsyncClient(base_url=cfg.backend_url, timeout=cfg.timeout, trust_env=False)

    async def score(self, state, question):
        r = await self.client.post("/v1/decisions", json={
            "model": self.cfg.model, "input": text(state), "questions": [question],
            "temperature": self.cfg.temperature, "prompt_format_version": 1,
            "chat_template_kwargs": {"enable_thinking": False},
        })
        if r.is_error:
            # Only explicit backend capacity errors become unsupported; other errors stay errors.
            markers = ("maximum context length", "context window", "max_model_len", "context length exceeded")
            if r.status_code in (400, 413, 422) and any(m in r.text.lower() for m in markers):
                raise CapacityError("maximum context length exceeded by backend")
            raise BackendError(f"backend HTTP {r.status_code}")
        data = r.json()
        if data.get("model") != self.cfg.model or data.get("prompt_format_version") != 1:
            raise BackendError("backend model or prompt version mismatch")
        if set(data.get("answers", {})) != {question["id"]}:
            raise BackendError("backend question IDs mismatch")
        usage = data.get("usage", {}).get("prompt_tokens")
        if isinstance(usage, bool) or not isinstance(usage, int) or usage < 0:
            raise BackendError("backend omitted valid prompt token usage")
        return data["answers"][question["id"]], usage

    async def close(self):
        await self.client.aclose()


class DecisionService:
    def __init__(self, cfg, backend):
        self.cfg, self.backend = cfg, backend
        self.slots = asyncio.Semaphore(cfg.backend_concurrency)

    async def read(self, state, native, keys):
        async with self.slots:
            result, tokens = await self.backend.score(state, native)
        return distribution(result.get("probabilities"), keys), tokens

    async def choice(self, state, key, q):
        options = q["criteria"]
        keys = list(options)
        if len(keys) == 1:
            return {keys[0]: 1.0}, 0, 0

        async def run(subset):
            native = {"id": key, "type": "choice", "question": text(q.get("instructions", "")),
                      "options": [{"name": k, "description": options[k]} for k in subset]}
            return await self.read(state, native, subset)

        if len(keys) <= 26:
            probs, tokens = await run(keys)
            return probs, tokens, 1
        count = math.ceil(len(keys) / 25)
        width, extra = divmod(len(keys), count)
        groups, offset = [], 0
        for i in range(count):
            end = offset + width + (i < extra)
            groups.append(keys[offset:end])
            offset = end
        rounds = await asyncio.gather(*(run(g) for g in groups))
        first, selected = {}, []
        for group, (probs, _) in zip(groups, rounds):
            first.update(probs)
            selected.extend(sorted(group, key=lambda k: -probs[k])[:3])
        if len(selected) > 26:
            selected = sorted(selected, key=lambda k: -first[k])[:26]
        final, tokens = await run(selected)
        eliminated = [k for k in keys if k not in final]
        total = math.fsum(first[k] for k in eliminated)
        probs = {k: final[k] * 0.999 if k in final else
                 0.001 * (first[k] / total if total else 1 / len(eliminated)) for k in keys}
        return distribution(probs, keys), tokens + sum(t for _, t in rounds), count + 1

    async def decide(self, body):
        qs = validate(body, self.cfg)
        start = time.perf_counter()
        answers, tokens, passes = {}, 0, 0
        for key, q in qs.items():
            typ = q["type"]
            if typ == "choice":
                p, n, calls = await self.choice(body["state"], key, q)
                answers[key] = {"type": typ, "choice": max(p, key=p.get), "probabilities": p}
            else:
                native = {"id": key, "question": text(q.get("instructions", ""))}
                c = q.get("criteria") or {}
                if typ == "noul":
                    native.update(type="yes_no", yes=c.get("yes", c.get("true")), no=c.get("no", c.get("false")))
                    p, n = await self.read(body["state"], native, ["no", "yes"])
                    answers[key] = {"type": typ, "noul": p["yes"], "probabilities": p}
                else:
                    levels = list(c.values()) if isinstance(c, dict) else c
                    native.update(type="score", levels=levels)
                    ids = [str(i) for i in range(len(levels))]
                    p, n = await self.read(body["state"], native, ids)
                    names = list(c) if isinstance(c, dict) else ids
                    mapped = dict(zip(names, p.values()))
                    answers[key] = {"type": typ, "score": math.fsum(float(k) * v for k, v in mapped.items()), "probabilities": mapped}
                calls = 1
            tokens += n
            passes += calls
        return {"model": self.cfg.model, "answers": answers,
                "usage": {"input_tokens": tokens, "prompt_tokens": tokens, "total_tokens": tokens},
                "inference": {"backend_calls": passes, "wall_ms": (time.perf_counter() - start) * 1000}}
