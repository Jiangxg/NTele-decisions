# NTele Decision

A typed text-decision service for [NTele-Hybrid-27B-v1](https://huggingface.co/ZTE-AIM/NTele-Hybrid-27B-v1), backed by SGLang or vLLM.
Exposes `/v1/systemone` for the official Decision Index HTTP engine, inspired by the serving design and publicly described wide-choice algorithm of
[StartLux-Decision](https://github.com/StartLuxLabs/StartLux-Decision).


## Installation

Python >=3.11. Use a separate environment:

For an existing backend, install the frontend only:

```bash
pip install -e .
```

For automatic SGLang launch:

```bash
pip install -e '.[sglang]'
```

The optional extra installs SGLang >=0.5.20.
For automatic vLLM launch, install a compatible vLLM build supporting
`/v1/systemone` in the same environment, then install this package with
`pip install -e .`. The package does not install vLLM automatically.

## Launch SGLang automatically

Download the model first, pinning its HF revision for evaluation:

```bash
ntele-decide serve \
  --model-path /path/to/NTele-Hybrid-27B-v1 \
  --model NTele-Hybrid-27B-v1 \
  --launch-sglang --gpus 0 \
  --temperature 1.26964 --port 8090
```

The launcher starts `python -m sglang.launch_server` on localhost:7801, in an
owned process group. 

Use `--gpus 0,1 --tp 2` for two GPUs; TP defaults to the visible GPU count.
Options include `--backend-port`, `--backend-log`, `--startup-timeout` and
`--backend-concurrency`. 

## Connect to an existing SGLang backend

```bash
ntele-decide serve --backend sglang --backend-url http://127.0.0.1:7801 \
  --model NTele-Hybrid-27B-v1 --temperature 1.26964 --port 8090
```

The backend must use the matching served-model-name. This mode also verifies
the backend at startup, but never starts or stops it.

## Launch vLLM automatically

To launch vLLM automatically, install a vLLM build supporting `/v1/systemone`
in the same Python environment as this package, then run:

```bash
ntele-decide serve --launch-vllm \
  --model-path /path/to/model --model qwen-chat \
  --gpus 0,1 --tp 2 --port 8090
```

For base-model chat and LoRA decisions together:

```bash
ntele-decide serve --launch-vllm \
  --model-path /path/to/model \
  --served-model-name qwen-chat --model decision-lora \
  --gpus 0,1 --tp 2 --port 8090 \
  --vllm-arg=--enable-lora \
  --vllm-arg=--lora-modules \
  --vllm-arg=decision-lora=/path/to/decision-lora \
  --vllm-arg=--max-lora-rank --vllm-arg=64
```

`--launch-vllm` implies `--backend vllm`. The child runs via the current Python
environment on localhost:7801 by default. Use `--backend-port`, `--backend-log`
(default `vllm.log`), and `--startup-timeout` to configure it. Extra vLLM arguments
are passed as individual `--vllm-arg=VALUE` tokens. Model/address/TP/authentication
arguments are owned by the launcher and cannot be overridden this way.
The launcher waits for `/health`, then probes an actual decision before serving.
Exit and startup failure clean up its child process group; external backends are
never stopped. The package does not install vLLM automatically.

Chat clients use `http://127.0.0.1:7801/v1/chat/completions` with `model=qwen-chat`;
decision clients use frontend port 8090. The model must support both the decision
strategy and the requested LoRA configuration.

For example, send a chat request directly to the launched base model:

```bash
curl http://127.0.0.1:7801/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen-chat","messages":[{"role":"user","content":"Hello"}]}'
```

## Connect to an existing vLLM backend

Start a vLLM server whose model supports `/v1/systemone`, then run:

```bash
ntele-decide serve --backend vllm \
  --backend-url http://127.0.0.1:8000 \
  --model decision-lora --port 8090
```

Use the backend's served model name, or its registered LoRA name (as above).
The adapter name is forwarded on every round. Chat clients can continue to call
vLLM's `/v1/chat/completions` directly with the base model name.
External-backend mode does not start or stop vLLM. The frontend does not proxy chat requests.

## Decision execution and limits

The vLLM mode uses the existing wide-choice policy and response format:

- Choices with 2–26 options, binary questions, and scores are sent together in
  one `/v1/systemone` call. vLLM schedules the individual model reads.
- Choices with 27–255 options are split into balanced groups of at most 25.
  These groups share the first call with ordinary questions. The top three
  options of each group advance; if necessary, their first-round probabilities
  select at most 26 finalists. All final questions share the second round.
- Each round is split into calls of at most 64 questions, matching the current
  vLLM limit. Capacity chunks are sent sequentially; there is no per-question
  HTTP fan-out. `--backend-concurrency` limits concurrent batch HTTP calls
  across frontend requests.
- A one-option choice returns probability 1 without a backend call.
- Finalists share probability mass 0.999; eliminated options share 0.001 using
  their first-round scores. This is a synthesized distribution, not a single
  model read over all options. Group-local probabilities are not globally
  calibrated.
- Numeric-key score criteria retain their original score values. Structured
  question text/descriptions are serialized to JSON text; state is forwarded
  unchanged. Original question IDs and order are restored in the response.
- `inference.backend_calls` counts actual batch HTTP calls. Token usage sums
  backend `input_tokens` and `output_tokens` once per call, including all rounds.
  `total_tokens` includes both; `prompt_tokens` aliases `input_tokens`.

The vLLM endpoint does not accept this package's SGLang temperature or prompt
version fields. `--temperature` is therefore rejected in vLLM mode; probabilities
follow the served vLLM model's decision strategy. Responses retain this package's
existing `answers` format and do not expose vLLM confidence/diagnostic fields.
The default backend remains SGLang.

SGLang mode retains its original execution: questions within one frontend
request are processed sequentially; groups within a wide-choice first round
are concurrent. Its `inference.backend_calls` counts individual question/group
calls. vLLM mode batches questions by round as described above.

## API

| Endpoint | Purpose |
| --- | --- |
| `POST /v1/systemone` | Typed decisions: `choice`, `noul`, and `score` |
| `GET /v1/models` | Configured model and decision limits |
| `GET /health` | Perform a backend decision probe |

Requests accept `model` (optional; must match `--model`), `state`, and
`questions`. Choices support 1–255 options, scores 2–10 levels. Media fields
and vLLM-specific request extensions are not exposed by this frontend.

### Request example

```bash
curl http://127.0.0.1:8090/v1/systemone -H 'Content-Type: application/json' \
  -d '{"state":"The marker is blue.","questions":{"color":{"type":"choice","instructions":"What color is the marker?","criteria":{"blue":"Blue","red":"Red"}}}}'
```

## Tests

Local tests (HTTP calls use mock backends; no GPU required):

```bash
python -m unittest discover -s tests -v
```

The optional official-validator test is skipped if `decision_index` is not
installed. Install your Decision Index evaluation distribution to run that test
and the evaluation command below. Local tests do not establish model accuracy
or GPU performance.

## Official evaluation

Set the evaluation model to the frontend's `--model`, including the adapter
name when using LoRA decisions.

```bash
decision-index run --edition 0.3 --suite-dir /path/to/verified-suite-0.3 \
  --engine http --option base_url=http://127.0.0.1:8090 \
  --option model=NTele-Hybrid-27B-v1 --out runs/NTele-Hybrid-27B-v1 --compact
```

## Deployment privacy

The frontend defaults to localhost and does not provide authentication. When
exposing it to other hosts, use an authenticated gateway and HTTPS. Use HTTPS
for external backends when requests cross an untrusted network.

Backend failures return a generic HTTP 502 message rather than exception details.
Automatically launched backend logs are created with owner-only permissions
(0600); existing log permissions are tightened, and symbolic links are rejected.
Logs can still contain sensitive details produced by the backend. Keep vLLM
request/output logging disabled (do not pass `--enable-log-requests` or
`--enable-log-outputs`), and review the logging defaults of your installed
backend version. External backend logs must be managed separately. Do not
include runtime logs or local environments when publishing the project.
