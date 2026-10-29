import argparse
import signal
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, HTTPException

from . import __version__
from .core import BackendError, CapacityError, DecisionService, InvalidRequest, Settings, SGLangBackend


def create_app(cfg, backend=None, verify_backend=False):
    backend = backend or SGLangBackend(cfg)
    service = DecisionService(cfg, backend)

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
        return {"object": "list", "data": [{"id": cfg.model, "object": "model"}],
                "decision_config": {"temperature": cfg.temperature, "prompt_format_version": 1,
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
            raise HTTPException(502, str(exc)) from exc

    return app


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["serve"])
    parser.add_argument("--backend-url", help="Connect to an existing backend; incompatible with --launch-sglang")
    parser.add_argument("--launch-sglang", action="store_true")
    parser.add_argument("--model-path", help="Local weights directory or Hugging Face model ID")
    parser.add_argument("--gpus", default="0")
    parser.add_argument("--tp", type=int, help="Tensor parallelism; defaults to visible GPU count")
    parser.add_argument("--backend-port", type=int, default=7801)
    parser.add_argument("--startup-timeout", type=float, default=900)
    parser.add_argument("--backend-log", default="sglang.log")
    parser.add_argument("--sglang-arg", action="append", default=[],
                        help="Extra backend argument token, repeat using --sglang-arg=VALUE")
    parser.add_argument("--model", default="NTele-Hybrid-27B-v1")
    parser.add_argument("--temperature", type=float, default=1.26964)
    parser.add_argument("--backend-concurrency", type=int, default=4)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8090)
    args = parser.parse_args()
    if args.launch_sglang and (not args.model_path or args.backend_url):
        parser.error("--launch-sglang requires --model-path and cannot use --backend-url")
    if not args.launch_sglang and (args.model_path or args.sglang_arg or args.tp is not None):
        parser.error("model-path, tp and sglang-arg require --launch-sglang")
    if not 1 <= args.port <= 65535 or not 1 <= args.backend_port <= 65535 or args.port == args.backend_port:
        parser.error("frontend and backend ports must be valid and distinct")
    import uvicorn
    cfg = Settings(model=args.model, backend_url=args.backend_url or f"http://127.0.0.1:{args.backend_port}", temperature=args.temperature,
                   backend_concurrency=args.backend_concurrency)
    owner = None
    old_handlers = {}
    try:
        if args.launch_sglang:
            from .launcher import ManagedSGLang
            owner = ManagedSGLang(args.model_path, cfg, args.backend_port, args.gpus, args.tp,
                                 args.startup_timeout, args.backend_log, args.sglang_arg)
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
