"""Batch decision rounds through vLLM's /v1/systemone endpoint."""
import asyncio
import json
import time

import httpx

from .core import (
    BackendError, CapacityError, answer, choice_distribution, choice_finalists,
    choice_groups, distribution, validate,
)


def structured_text(value):
    return value if value is None or isinstance(value, str) else json.dumps(value, ensure_ascii=False)


def question_spec(q, subset=None):
    criteria = q.get("criteria")
    if q["type"] == "choice":
        criteria = {k: structured_text(criteria[k]) for k in
                    (subset if subset is not None else criteria)}
    elif q["type"] == "noul":
        criteria = criteria or {}
        criteria = {"true": structured_text(criteria.get("true", criteria.get("yes"))),
                    "false": structured_text(criteria.get("false", criteria.get("no")))}
    else:
        levels = criteria.values() if isinstance(criteria, dict) else criteria
        criteria = [structured_text(level) for level in levels]
    return {"type": q["type"], "instructions": structured_text(q.get("instructions", "")),
            "criteria": criteria}


def probability_keys(q):
    if q["type"] == "choice":
        return list(q["criteria"])
    if q["type"] == "noul":
        return ["no", "yes"]
    return [str(i) for i in range(len(q["criteria"]))]


class VLLMBackend:
    max_questions = 64

    def __init__(self, cfg):
        self.cfg = cfg
        self.client = httpx.AsyncClient(base_url=cfg.backend_url, timeout=cfg.timeout, trust_env=False)

    async def score_batch(self, state, questions):
        response = await self.client.post("/v1/systemone", json={
            "model": self.cfg.model, "state": state, "questions": questions,
        })
        if response.is_error:
            markers = ("maximum context length", "context window", "max_model_len", "context length exceeded")
            if response.status_code in (400, 413, 422) and any(m in response.text.lower() for m in markers):
                raise CapacityError("maximum context length exceeded by backend")
            raise BackendError(f"backend HTTP {response.status_code}")
        try:
            data = response.json()
            if data["model"] != self.cfg.model or set(data["answers"]) != set(questions):
                raise BackendError("backend model or question IDs mismatch")
            usage = {k: data["usage"][k] for k in ("input_tokens", "output_tokens")}
            if any(isinstance(v, bool) or not isinstance(v, int) or v < 0 for v in usage.values()):
                raise BackendError("backend omitted valid token usage")
            probs = {key: distribution(data["answers"][key]["probabilities"], probability_keys(q))
                     for key, q in questions.items()}
        except (KeyError, TypeError, ValueError) as exc:
            raise BackendError("invalid vLLM decision response") from exc
        return probs, usage

    async def close(self):
        await self.client.aclose()


class VLLMDecisionService:
    def __init__(self, cfg, backend):
        self.cfg, self.backend = cfg, backend
        self.slots = asyncio.Semaphore(cfg.backend_concurrency)

    async def decide(self, body):
        questions = validate(body, self.cfg)
        start = time.perf_counter()
        probabilities, wide, first_round = {}, {}, {}
        usage = {"input_tokens": 0, "output_tokens": 0}
        calls = 0

        def add(target, q, subset=None):
            key = f"q{len(target)}"
            target[key] = question_spec(q, subset)
            return key

        async def run(specs):
            nonlocal calls
            result = {}
            items = list(specs.items())
            for offset in range(0, len(items), self.backend.max_questions):
                batch = dict(items[offset:offset + self.backend.max_questions])
                async with self.slots:
                    probs, tokens = await self.backend.score_batch(body["state"], batch)
                result.update(probs)
                for name in usage:
                    usage[name] += tokens[name]
                calls += 1
            return result

        ordinary = {}
        for key, q in questions.items():
            if q["type"] == "choice" and len(q["criteria"]) == 1:
                probabilities[key] = {next(iter(q["criteria"])): 1.0}
            elif q["type"] == "choice" and len(q["criteria"]) > 26:
                groups = choice_groups(list(q["criteria"]))
                wide[key] = (groups, [add(first_round, q, group) for group in groups])
            else:
                ordinary[key] = add(first_round, q)

        first_results = await run(first_round)
        for key, internal in ordinary.items():
            probabilities[key] = first_results[internal]

        finals, finalists = {}, {}
        for key, (groups, ids) in wide.items():
            first, selected = choice_finalists(groups, [first_results[i] for i in ids])
            finalists[key] = (first, add(finals, questions[key], selected))
        final_results = await run(finals)
        for key, (first, internal) in finalists.items():
            probabilities[key] = choice_distribution(list(questions[key]["criteria"]), first,
                                                     final_results[internal])

        return {"model": self.cfg.model,
                "answers": {key: answer(q, probabilities[key]) for key, q in questions.items()},
                "usage": {**usage, "prompt_tokens": usage["input_tokens"],
                          "total_tokens": sum(usage.values())},
                "inference": {"backend_calls": calls, "wall_ms": (time.perf_counter() - start) * 1000}}
