"""Own a local inference process group; never stop an external backend."""
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

from .core import create_service


def open_backend_log(path):
    """Create private logs and tighten permissions on existing regular files."""
    flags = os.O_WRONLY | os.O_CREAT | os.O_APPEND | os.O_NOFOLLOW | os.O_NONBLOCK
    fd = os.open(path, flags, 0o600)
    try:
        import stat
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise ValueError("backend log must be a regular file")
        os.fchmod(fd, 0o600)
        return os.fdopen(fd, "a", encoding="utf-8")
    except BaseException:
        os.close(fd)
        raise


async def probe(cfg):
    service = create_service(cfg)
    try:
        result = await service.decide({
            "model": cfg.model, "state": "The marker is blue.",
            "questions": {"readiness": {"type": "choice", "instructions": "What color is the marker?",
                "criteria": {"blue": "Blue", "red": "Red"}}}})
        return result
    finally:
        await service.backend.close()


class ManagedSGLang:
    name = "SGLang"
    def __init__(self, model_path, cfg, port=7801, gpus="0", tp=None,
                 startup_timeout=900, log_path="sglang.log", extra=()):
        self.cfg, self.port, self.gpus = cfg, port, gpus
        ids = gpus.split(",")
        if not ids or any(not x.strip() for x in ids) or len(set(ids)) != len(ids):
            raise ValueError("--gpus must be a nonempty list of distinct device IDs")
        self.tp = len(ids) if tp is None else tp
        if self.tp < 1 or self.tp > len(ids):
            raise ValueError("--tp must be between 1 and the number of visible GPUs")
        if startup_timeout <= 0:
            raise ValueError("startup timeout must be positive")
        forbidden = {"--model-path", "--model", "--served-model-name", "--host", "--port",
                     "--tp", "--tp-size", "--tensor-parallel-size", "-tp", "--api-key"}
        if any(x.split("=", 1)[0] in forbidden for x in extra):
            raise ValueError("backend arguments cannot override owned model, address, TP or authentication settings")
        self.command = [sys.executable, "-m", "sglang.launch_server", "--model-path", model_path,
                        "--served-model-name", cfg.model, "--host", "127.0.0.1", "--port", str(port),
                        "--tp", str(self.tp), *extra]
        self.timeout, self.log_path = startup_timeout, Path(log_path)
        self.process = self.log = None

    def check_installation(self):
        version = importlib.metadata.version("sglang")
        if Version(version) < Version("0.5.20"):
            raise RuntimeError(f"SGLang >=0.5.20 required; installed {version}")

    def start(self):
        self.check_installation()
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", self.port))  # refuse occupied ports; never attach silently
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.log = open_backend_log(self.log_path)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=self.gpus)
        self.process = subprocess.Popen(self.command, env=env, stdout=self.log,
                                        stderr=subprocess.STDOUT, start_new_session=True)

    def wait_ready(self):
        deadline = time.monotonic() + self.timeout
        with httpx.Client(timeout=3, trust_env=False) as client:
            while time.monotonic() < deadline:
                if self.process.poll() is not None:
                    raise RuntimeError(f"{self.name} exited with {self.process.returncode}; see {self.log_path}")
                try:
                    if client.get(self.cfg.backend_url + "/health").is_success:
                        return
                except httpx.HTTPError:
                    pass
                time.sleep(1)
        raise TimeoutError(f"{self.name} startup timed out; see {self.log_path}")

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


class ManagedVLLM(ManagedSGLang):
    name = "vLLM"

    def __init__(self, model_path, cfg, port=7801, gpus="0", tp=None,
                 startup_timeout=900, log_path="vllm.log", extra=(), served_model_name=None):
        super().__init__(model_path, cfg, port, gpus, tp, startup_timeout, log_path, extra)
        self.command = [sys.executable, "-m", "vllm.entrypoints.cli.main", "serve", model_path,
                        "--served-model-name", served_model_name or cfg.model,
                        "--host", "127.0.0.1", "--port", str(port),
                        "--tensor-parallel-size", str(self.tp), *extra]

    def check_installation(self):
        # Protocol support is verified by the decision probe after /health is ready.
        importlib.metadata.version("vllm")
