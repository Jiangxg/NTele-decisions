import argparse
import signal
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, HTTPException

from . import __version__
from .core import BackendError, CapacityError, InvalidRequest, Settings, create_service


def create_app(cfg, backend=None, verify_backend=False):
    service = create_service(cfg, backend)
    backend = service.backend

    @asynccontextmanager
    async def lifespan(app):
        try:
            if verify_backend:
                from .launcher import probe
                await probe(cfg)
            yield
        finally:
            await backend.close()

    app = FastAPI(title="NTele Decision", version=__version__, lifespan=lifespan)

    @app.get("/health")
    async def health():
        from .launcher import probe
        try:
            await probe(cfg)
        except Exception as exc:
            raise HTTPException(503, "backend decision probe failed") from exc
        return {"status": "ready", "backend_verified": True}

    @app.get("/v1/models")
    async def models():
        backend_config = ({"temperature": cfg.temperature, "prompt_format_version": 1}
                          if cfg.backend == "sglang" else {})
        return {"object": "list", "data": [{"id": cfg.model, "object": "model"}],
                "decision_config": {"backend": cfg.backend, **backend_config,
                                    "thinking": False, "max_options": cfg.max_options,
                                    "wide": {"group_max": 25, "advance": 3, "final_max": 26, "residual": 0.001}}}

    @app.post("/v1/systemone")
    async def decide(body: dict):
        try:
            return await service.decide(body)
        except CapacityError as exc:
            raise HTTPException(413, str(exc)) from exc
        except InvalidRequest as exc:
            raise HTTPException(422, str(exc)) from exc
        except (BackendError, httpx.HTTPError) as exc:
            raise HTTPException(502, "backend request failed") from exc

    return app


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["serve"])
    parser.add_argument("--backend", choices=["sglang", "vllm"])
    parser.add_argument("--backend-url", help="Connect to an existing backend; incompatible with automatic launch")
    launches = parser.add_mutually_exclusive_group()
    launches.add_argument("--launch-sglang", action="store_true")
    launches.add_argument("--launch-vllm", action="store_true")
    parser.add_argument("--model-path", help="Local weights directory or Hugging Face model ID")
    parser.add_argument("--gpus", default="0")
    parser.add_argument("--tp", type=int, help="Tensor parallelism; defaults to visible GPU count")
    parser.add_argument("--backend-port", type=int, default=7801)
    parser.add_argument("--startup-timeout", type=float, default=900)
    parser.add_argument("--backend-log", help="Defaults to sglang.log or vllm.log")
    parser.add_argument("--sglang-arg", action="append", default=[],
                        help="Extra backend argument token, repeat using --sglang-arg=VALUE")
    parser.add_argument("--vllm-arg", action="append", default=[],
                        help="Extra vLLM argument token, repeat using --vllm-arg=VALUE")
    parser.add_argument("--served-model-name", help="Launched vLLM base model alias; --model may name a LoRA")
    parser.add_argument("--model", default="NTele-Hybrid-27B-v1")
    parser.add_argument("--temperature", type=float, help="SGLang only; default 1.26964")
    parser.add_argument("--backend-concurrency", type=int, default=4)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8090)
    args = parser.parse_args()
    launched_backend = "vllm" if args.launch_vllm else "sglang" if args.launch_sglang else None
    if launched_backend and args.backend and args.backend != launched_backend:
        parser.error("--backend conflicts with the selected launcher")
    args.backend = args.backend or launched_backend or "sglang"
    if args.backend == "vllm" and args.temperature is not None:
        parser.error("vllm does not accept --temperature")
    if launched_backend and (not args.model_path or args.backend_url):
        parser.error("automatic launch requires --model-path and cannot use --backend-url")
    if not launched_backend and (args.model_path or args.sglang_arg or args.vllm_arg or args.tp is not None or args.served_model_name):
        parser.error("model-path, tp and backend launch arguments require automatic launch")
    if args.backend == "vllm" and not args.launch_vllm and not args.backend_url:
        parser.error("vllm requires --backend-url or --launch-vllm")
    if args.launch_vllm and args.sglang_arg:
        parser.error("--sglang-arg requires --launch-sglang")
    if args.launch_sglang and (args.vllm_arg or args.served_model_name):
        parser.error("--vllm-arg and --served-model-name require --launch-vllm")
    if not 1 <= args.port <= 65535 or not 1 <= args.backend_port <= 65535 or args.port == args.backend_port:
        parser.error("frontend and backend ports must be valid and distinct")
    import uvicorn
    cfg = Settings(model=args.model, backend_url=args.backend_url or f"http://127.0.0.1:{args.backend_port}",
                   temperature=args.temperature if args.temperature is not None else 1.26964,
                   backend_concurrency=args.backend_concurrency, backend=args.backend)
    owner = None
    old_handlers = {}
    try:
        if launched_backend:
            from .launcher import ManagedSGLang, ManagedVLLM
            launcher = ManagedVLLM if args.launch_vllm else ManagedSGLang
            options = {"served_model_name": args.served_model_name} if args.launch_vllm else {}
            owner = launcher(args.model_path, cfg, args.backend_port, args.gpus, args.tp,
                             args.startup_timeout, args.backend_log or f"{args.backend}.log",
                             args.vllm_arg if args.launch_vllm else args.sglang_arg, **options)
            def interrupted(signum, frame):
                raise KeyboardInterrupt
            for sig in (signal.SIGINT, signal.SIGTERM):
                old_handlers[sig] = signal.signal(sig, interrupted)
            owner.start()
            owner.wait_ready()
        # Startup fails before serving requests if model/version/probability protocol differs.
        uvicorn.run(create_app(cfg, verify_backend=True), host=args.host, port=args.port)
    finally:
        if owner is not None:
            owner.stop()
        for sig, handler in old_handlers.items():
            signal.signal(sig, handler)


if __name__ == "__main__":
    main()
