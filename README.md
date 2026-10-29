# NTele Decision

A typed text-decision service for NTele-Hybrid-27B-v1, backed by SGLang.
Exposes `/v1/systemone` for the official Decision Index HTTP engine.

Inspired by the serving design and publicly described wide-choice algorithm of
[StartLux-Decision](https://github.com/StartLuxLabs/StartLux-Decision).
Independently implemented: no StartLux code is copied, vendored or imported.
NTele retains its SGLang prompt and causal attention, not StartLux's attention changes.

## Installation

Python >=3.11. Use a separate environment:

```bash
pip install -e '.[sglang]'
```

The optional extra installs SGLang >=0.5.20. Match PyTorch/CUDA to your GPU.
Version eligibility alone does not guarantee model or protocol compatibility.
The integration targets the local 0.5.21 protocol and checks it at startup.
Pin the exact tested runtime and required patches before public submission.
This package does not patch SGLang or automatically download weights.

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
owned process group. It waits for health and verifies a real decision response
before serving traffic: model identity, prompt v1, probabilities and token usage.
This is a protocol probe, not an accuracy test. Backend logs append to `sglang.log`.
Startup failure, timeout, Ctrl-C and SIGTERM/shutdown clean up the owned process
group. SIGKILL/host failure cannot run cleanup; use an OS supervisor if needed.
An occupied port causes failure: unrelated processes are never terminated.

Use `--gpus 0,1 --tp 2` for two GPUs; TP defaults to the visible GPU count.
Options include `--backend-port`, `--backend-log`, `--startup-timeout` and
`--backend-concurrency`. Additional SGLang arguments are separate tokens:

```bash
ntele-decide serve --launch-sglang --model-path /path/to/model --gpus 0,1 \
  --sglang-arg=--context-length --sglang-arg=131072
```

Extra arguments cannot override owned model/address/TP/authentication flags.
No shell command construction is used.

## Connect to an existing backend

```bash
ntele-decide serve --backend-url http://127.0.0.1:7801 \
  --model NTele-Hybrid-27B-v1 --temperature 1.26964 --port 8090
```

The backend must use the matching served-model-name. This mode also verifies
the backend at startup, but never starts or stops it.

## API

```bash
curl http://127.0.0.1:8090/v1/systemone -H 'Content-Type: application/json' \
  -d '{"state":"The marker is blue.","questions":{"color":{"type":"choice","instructions":"What color is the marker?","criteria":{"blue":"Blue","red":"Red"}}}}'
```

- Choice: 1–255 named options; returns all probabilities and the selected option.
- NOUL: yes/no probability, with optional yes/no or true/false descriptions.
- Score: 2–10 ordered levels. Numeric object keys must strictly increase; lists
  use zero-based indices. Returns probabilities and the expected numeric value.
- `/v1/models` exposes settings. `/health` performs a real decision probe and
  returns 503 if unavailable/incompatible; it consumes a small inference request.
- Localhost by default; no frontend authentication. Do not expose directly online.

## Inference policy

SGLang prompt version 1, thinking disabled. Temperature is applied once per
backend readout; the combined distribution is not calibrated a second time.
Up to 26 options use one pass; a singleton returns one without a backend call.
Wider lists preserve order and use near-equal groups of at most 25. Each group's
top three advance, capped at 26 using first-round probability and stable ties.
Finalists share 0.999 mass; the rest share 0.001 proportionally to their first-round
probabilities (uniformly if their sum is zero). Every option is read and returned.
This composite distribution is not an all-option simultaneous softmax.

All calls count toward tokens and wall time. Questions are processed independently;
group concurrency is globally bounded. No silent truncation, benchmark-specific
prompts or automatic retries. Capacity failures are 413, invalid requests 422,
backend failures 502. Any failed question fails the entire request.

## Official evaluation

```bash
decision-index run --edition 0.3 --suite-dir /path/to/verified-suite-0.3 \
  --engine http --option base_url=http://127.0.0.1:8090 \
  --option model=NTele-Hybrid-27B-v1 --out runs/NTele-Hybrid-27B-v1 --compact
```

Publish weights revision, serving commit, dependency lock, settings and complete
run artifacts. Count wide-choice rounds in latency. Multi-GPU throughput is not
the official single-GPU latency measurement.

## Validation and release status

With the official kit installed for response validation:

```bash
python -m unittest discover -s tests -v
```

CPU tests use mock backends/processes. Real-GPU launch, probability parity,
single-GPU latency/capacity and full official-suite evaluation remain to validate.
No leaderboard approval is implied. The project owner must choose a license before
public release. No StartLux source is included; earlier research viewed its source,
so this is not a claim of legally isolated clean-room development.
