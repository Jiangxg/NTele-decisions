"""Own a local SGLang process group; never stop an external backend."""
import importlib.metadata
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time

import httpx
from packaging.version import Version

from .core import DecisionService, Settings, SGLangBackend


async def probe(cfg):
    backend = SGLangBackend(cfg)
    try:
        result = await DecisionService(cfg, backend).decide({
            "model": cfg.model, "state": "The marker is blue.",
            "questions": {"readiness": {"type": "choice", "instructions": "What color is the marker?",
                "criteria": {"blue": "Blue", "red": "Red"}}}})
        return result
    finally:
        await backend.close()


class ManagedSGLang:
    def __init__(self, model_path, cfg, port=7801, gpus="0", tp=None,
                 startup_timeout=900, log_path="sglang.log", extra=()):
        self.cfg, self.port, self.gpus = cfg, port, gpus
        ids = gpus.split(",")
        if not ids or any(not x.strip() for x in ids) or len(set(ids)) != len(ids):
            raise ValueError("--gpus must be a nonempty list of distinct device IDs")
        self.tp = tp or len(ids)
        if self.tp < 1 or self.tp > len(ids):
            raise ValueError("--tp must be between 1 and the number of visible GPUs")
        if startup_timeout <= 0:
            raise ValueError("startup timeout must be positive")
        forbidden = {"--model-path", "--model", "--served-model-name", "--host", "--port",
                     "--tp", "--tp-size", "--tensor-parallel-size", "--api-key"}
        if any(x.split("=", 1)[0] in forbidden for x in extra):
            raise ValueError("backend arguments cannot override owned model, address, TP or authentication settings")
        self.command = [sys.executable, "-m", "sglang.launch_server", "--model-path", model_path,
                        "--served-model-name", cfg.model, "--host", "127.0.0.1", "--port", str(port),
                        "--tp", str(self.tp), *extra]
        self.timeout, self.log_path = startup_timeout, Path(log_path)
        self.process = self.log = None

    def start(self):
        version = importlib.metadata.version("sglang")
        if Version(version) < Version("0.5.20"):
            raise RuntimeError(f"SGLang >=0.5.20 required; installed {version}")
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", self.port))  # refuse occupied ports; never attach silently
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.log = self.log_path.open("a", encoding="utf-8")
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=self.gpus)
        self.process = subprocess.Popen(self.command, env=env, stdout=self.log,
                                        stderr=subprocess.STDOUT, start_new_session=True)

    def wait_ready(self):
        deadline = time.monotonic() + self.timeout
        with httpx.Client(timeout=3, trust_env=False) as client:
            while time.monotonic() < deadline:
                if self.process.poll() is not None:
                    raise RuntimeError(f"SGLang exited with {self.process.returncode}; see {self.log_path}")
                try:
                    if client.get(self.cfg.backend_url + "/health").is_success:
                        return
                except httpx.HTTPError:
                    pass
                time.sleep(1)
        raise TimeoutError(f"SGLang startup timed out; see {self.log_path}")

    def stop(self):
        if self.process is not None:
            try:
                os.killpg(self.process.pid, signal.SIGTERM)
                self.process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(self.process.pid, signal.SIGKILL)
                self.process.wait(timeout=5)
            except ProcessLookupError:
                pass
        if self.log is not None:
            self.log.close()
