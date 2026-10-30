# NTele Decision

A typed text-decision service for NTele-Hybrid-27B-v1, backed by SGLang.
Exposes `/v1/systemone` for the official Decision Index HTTP engine, inspired by the serving design and publicly described wide-choice algorithm of
[StartLux-Decision](https://github.com/StartLuxLabs/StartLux-Decision).


## Installation

Python >=3.11. Use a separate environment:

```bash
pip install -e '.[sglang]'
```

The optional extra installs SGLang >=0.5.20. 

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

## Official evaluation

```bash
decision-index run --edition 0.3 --suite-dir /path/to/verified-suite-0.3 \
  --engine http --option base_url=http://127.0.0.1:8090 \
  --option model=NTele-Hybrid-27B-v1 --out runs/NTele-Hybrid-27B-v1 --compact
```
