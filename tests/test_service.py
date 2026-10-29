import asyncio
import math
import unittest

import httpx

from ntele_decision.core import BackendError, CapacityError, DecisionService, InvalidRequest, Settings, SGLangBackend
from ntele_decision.server import create_app


class Fake:
    def __init__(self):
        self.calls = []

    async def score(self, state, q):
        self.calls.append((state, q))
        keys = ([o['name'] for o in q['options']] if q['type'] == 'choice' else
                ['yes', 'no'] if q['type'] == 'yes_no' else [str(i) for i in range(len(q['levels']))])
        weights = list(range(1, len(keys) + 1))
        return {'probabilities': {k: w / sum(weights) for k, w in zip(keys, weights)}}, 17

    async def close(self):
        pass


def request(n):
    return {'state': 'context', 'questions': {'q': {'type': 'choice', 'instructions': 'pick',
            'criteria': {str(i): 'description' for i in range(n)}}}}


class Tests(unittest.IsolatedAsyncioTestCase):
    async def test_widths_and_official_validator(self):
        from decision_index.engines.base import validate
        for n in [1, 2, 26, 27, 77, 151, 255]:
            with self.subTest(n=n):
                fake = Fake()
                service = DecisionService(Settings(), fake)
                body = request(n)
                result = await service.decide(body)
                validate(body['questions'], result)
                p = result['answers']['q']['probabilities']
                self.assertEqual(list(p), list(body['questions']['q']['criteria']))
                self.assertAlmostEqual(sum(p.values()), 1)
                self.assertEqual(result['usage']['input_tokens'], 17 * len(fake.calls))
                expected = 0 if n == 1 else 1 if n <= 26 else math.ceil(n / 25) + 1
                self.assertEqual(len(fake.calls), expected)
                self.assertTrue(all(len(q['options']) <= 26 for _, q in fake.calls))
                if n > 26:
                    offered = [o['name'] for _, q in fake.calls[:-1] for o in q['options']]
                    self.assertEqual(offered, list(p))

    async def test_noul_score_and_multiple(self):
        fake = Fake()
        service = DecisionService(Settings(), fake)
        r = await service.decide({'state': {}, 'questions': {
            'n': {'type': 'noul', 'criteria': {'true': 'yes description', 'false': 'no description'}},
            's': {'type': 'score', 'criteria': {'-1': 'low', '0': 'middle', '2': 'high'}},
        }})
        self.assertAlmostEqual(r['answers']['n']['noul'], 1 / 3)
        self.assertAlmostEqual(r['answers']['s']['score'], 5 / 6)
        self.assertEqual(fake.calls[0][1]['yes'], 'yes description')

    async def test_rejects_invalid_and_capacity(self):
        s = DecisionService(Settings(), Fake())
        with self.assertRaises(CapacityError):
            await s.decide(request(256))
        with self.assertRaises(InvalidRequest):
            await s.decide(dict(request(2), temperature=2))
        with self.assertRaises(InvalidRequest):
            await s.decide(dict(request(2), model='wrong'))

    async def test_transport_pins_temperature_once(self):
        cfg = Settings()
        backend = SGLangBackend(cfg)
        await backend.client.aclose()
        def handler(req):
            import json
            body = json.loads(req.content)
            self.assertEqual(body['temperature'], 1.26964)
            self.assertEqual(body['prompt_format_version'], 1)
            self.assertEqual(body['input'], '{}')
            self.assertFalse(body['chat_template_kwargs']['enable_thinking'])
            return httpx.Response(200, json={'model': cfg.model, 'prompt_format_version': 1,
                'answers': {'q': {'probabilities': {'a': .7, 'b': .3}}}, 'usage': {'prompt_tokens': 12}})
        backend.client = httpx.AsyncClient(base_url=cfg.backend_url, transport=httpx.MockTransport(handler))
        s = DecisionService(cfg, backend)
        r = await s.decide({'state': {}, 'questions': {'q': {'type': 'choice', 'criteria': {'a': '', 'b': ''}}}})
        self.assertEqual(r['answers']['q']['probabilities'], {'a': .7, 'b': .3})
        await backend.close()

    async def test_http(self):
        app = create_app(Settings(), Fake())
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://test') as client:
            self.assertEqual((await client.post('/v1/systemone', json=request(77))).status_code, 200)
            self.assertEqual((await client.post('/v1/systemone', json=request(256))).status_code, 413)


if __name__ == '__main__':
    unittest.main()
