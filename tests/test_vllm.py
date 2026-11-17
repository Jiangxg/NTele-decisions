import json
import unittest

import httpx

from ntele_decision.core import DecisionService, Settings
from ntele_decision.server import create_app
from ntele_decision.vllm import VLLMBackend
from test_service import Fake, request


class VLLMTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.cfg = Settings(backend="vllm", model="decision-lora")
        self.backend = VLLMBackend(self.cfg)
        await self.backend.client.aclose()
        self.calls = []
        self.failure = None
        self.backend.client = httpx.AsyncClient(
            base_url="http://vllm", transport=httpx.MockTransport(self.respond))
        self.client = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=create_app(self.cfg, self.backend)), base_url="http://frontend")

    async def asyncTearDown(self):
        await self.client.aclose()
        await self.backend.close()

    def respond(self, req):
        self.assertEqual(req.url.path, "/v1/systemone")
        body = json.loads(req.content)
        self.assertEqual(set(body), {"model", "state", "questions"})
        self.assertEqual(body["model"], "decision-lora")
        self.assertLessEqual(len(body["questions"]), 64)
        self.calls.append(body)
        if self.failure:
            return self.failure
        answers = {}
        for key, q in body["questions"].items():
            if q["type"] == "choice":
                keys = list(q["criteria"])
                self.assertLessEqual(len(keys), 26)
            elif q["type"] == "noul":
                self.assertEqual(set(q["criteria"]), {"true", "false"})
                keys = ["yes", "no"]
            else:
                self.assertIsInstance(q["criteria"], list)
                keys = [str(i) for i in range(len(q["criteria"]))]
            total = len(keys) * (len(keys) + 1) / 2
            answers[key] = {"probabilities": {k: (i + 1) / total for i, k in enumerate(keys)}}
        n = len(answers)
        return httpx.Response(200, json={"model": self.cfg.model, "answers": answers,
                                        "usage": {"input_tokens": n * 17, "output_tokens": n}})

    async def test_ordinary_questions_share_one_backend_request(self):
        body = {"state": {"context": "x"}, "questions": {
            "q0": {"type": "choice", "criteria": {"a": "A", "b": "B"}},
            "binary": {"type": "noul", "criteria": {"yes": "positive"}},
            "score": {"type": "score", "criteria": {"-1": "low", "0": "middle", "2": "high"}},
        }}
        response = await self.client.post("/v1/systemone", json=body)
        self.assertEqual(response.status_code, 200, response.text)
        result = response.json()
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(self.calls[0]["state"], body["state"])
        self.assertEqual(list(result["answers"]), list(body["questions"]))
        self.assertAlmostEqual(result["answers"]["score"]["score"], 5 / 6)
        self.assertEqual(result["usage"]["total_tokens"], 54)
        self.assertEqual(result["inference"]["backend_calls"], 1)

    async def test_wide_choices_reuse_existing_policy_in_two_batched_rounds(self):
        questions = {f"width{n}": request(n)["questions"]["q"] for n in [1, 2, 26, 27, 77, 151, 255]}
        body = {"state": "context", "questions": questions}
        expected = await DecisionService(Settings(), Fake()).decide(body)
        response = await self.client.post("/v1/systemone", json=body)
        self.assertEqual(response.status_code, 200, response.text)
        result = response.json()
        self.assertEqual(result["answers"], expected["answers"])
        self.assertEqual(result["usage"]["input_tokens"], expected["usage"]["input_tokens"])
        self.assertEqual(len(self.calls), 2)
        self.assertEqual(len(self.calls[0]["questions"]), 26)
        self.assertEqual(len(self.calls[1]["questions"]), 4)
        self.assertEqual(result["inference"]["backend_calls"], 2)

    async def test_vllm_question_capacity_splits_batches(self):
        body = {"state": "x", "questions": {str(i): request(2)["questions"]["q"] for i in range(65)}}
        response = await self.client.post("/v1/systemone", json=body)
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual([len(b["questions"]) for b in self.calls], [64, 1])
        self.assertEqual(len(response.json()["answers"]), 65)

    async def test_single_option_skips_backend_and_limit_fails_before_inference(self):
        response = await self.client.post("/v1/systemone", json=request(1))
        self.assertEqual(response.json()["inference"]["backend_calls"], 0)
        response = await self.client.post("/v1/systemone", json=request(256))
        self.assertEqual(response.status_code, 413)
        self.assertEqual(self.calls, [])

    async def test_transport_error_is_redacted(self):
        def fail(req):
            raise httpx.ConnectError("cannot reach http://10.0.0.1:8000/private", request=req)
        await self.backend.client.aclose()
        self.backend.client = httpx.AsyncClient(base_url="http://vllm", transport=httpx.MockTransport(fail))
        response = await self.client.post("/v1/systemone", json=request(2))
        self.assertEqual(response.status_code, 502)
        self.assertEqual(response.json(), {"detail": "backend request failed"})

    async def test_backend_failure_stops_before_final_round(self):
        for failure, status in [
            (httpx.Response(400, text="prompt exceeds max_model_len"), 413),
            (httpx.Response(501, text="unsupported model"), 502),
            (httpx.Response(200, json={"model": "wrong", "answers": {}}), 502),
            (httpx.Response(200, text="not JSON"), 502),
            (httpx.Response(200, json={"model": self.cfg.model,
                "answers": {"q0": {"probabilities": {"0": .5, "1": .5}}},
                "usage": {"input_tokens": True, "output_tokens": 1}}), 502),
            (httpx.Response(500, text="upstream http://10.0.0.1:8000/internal"), 502),
        ]:
            with self.subTest(status=status, failure=failure):
                self.calls.clear()
                self.failure = failure
                response = await self.client.post("/v1/systemone", json=request(77))
                self.assertEqual(response.status_code, status, response.text)
                if status == 502:
                    self.assertNotIn("10.0.0.1", response.text)
                self.assertEqual(len(self.calls), 1)

    async def test_invalid_probability_and_usage_are_backend_errors(self):
        for probabilities, usage in [
            ({"0": .2, "1": .2}, {"input_tokens": 17, "output_tokens": 1}),
            ({"0": .5, "wrong": .5}, {"input_tokens": 17, "output_tokens": 1}),
            ({"0": .5, "1": .5}, {"input_tokens": True, "output_tokens": 1}),
        ]:
            self.failure = httpx.Response(200, json={"model": self.cfg.model,
                "answers": {"q0": {"probabilities": probabilities}}, "usage": usage})
            response = await self.client.post("/v1/systemone", json=request(2))
            self.assertEqual(response.status_code, 502, response.text)

    async def test_structured_descriptions_and_ids_survive_translation(self):
        body = {"state": [], "questions": {"q0": {"type": "choice",
            "instructions": {"ask": "选择"}, "criteria": {"a": {"label": "甲"}, "b": ["乙"]}}}}
        response = await self.client.post("/v1/systemone", json=body)
        self.assertEqual(response.status_code, 200, response.text)
        sent = self.calls[0]["questions"]["q0"]
        self.assertEqual(json.loads(sent["instructions"]), {"ask": "选择"})
        self.assertEqual(json.loads(sent["criteria"]["a"]), {"label": "甲"})
        self.assertEqual(self.calls[0]["state"], [])
        self.assertEqual(list(response.json()["answers"]), ["q0"])

    async def test_metadata_does_not_advertise_sglang_temperature(self):
        response = await self.client.get("/v1/models")
        config = response.json()["decision_config"]
        self.assertEqual(config["backend"], "vllm")
        self.assertNotIn("temperature", config)
        self.assertNotIn("prompt_format_version", config)
