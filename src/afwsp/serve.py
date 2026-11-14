"""Serve a checkpoint with vLLM's OpenAI-compatible server, for the agentic evals.

with vllm_server(ckpt, vllm_python, log_path, reasoning_parser="qwen3") as base_url: ...

Starts `vllm serve` from the vLLM virtualenv with Qwen3.5's tool-call parser, waits until /health answers, and on
exit kills the whole process group. vLLM's engine core is a child process that outlives a plain kill of its
parent and keeps holding the GPU (docs/REPRODUCE.md).
"""
import os
import signal
import socket
import subprocess
import time
import urllib.request
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def model_endpoint(ckpt, log_path, reasoning_parser=None, max_model_len=131072):
    """The server base URL for ckpt: a fresh vLLM server from $AFWSP_VLLM_PYTHON, or, when $AFWSP_MODEL_URL is set,
    a server that is already running (on a cluster, or a stub in tests), used as is."""
    url = os.environ.get("AFWSP_MODEL_URL")
    if url:
        yield url.rstrip("/").removesuffix("/v1")
        return
    with vllm_server(ckpt, os.environ["AFWSP_VLLM_PYTHON"], log_path, reasoning_parser=reasoning_parser,
                     max_model_len=max_model_len) as base:
        yield base


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@contextmanager
def vllm_server(ckpt, vllm_python, log_path, served_name="afwsp", reasoning_parser=None,
                tool_parser="qwen3_coder", max_model_len=131072, gpu_mem=0.85, timeout=1200):
    port = free_port()
    vllm_bin = Path(vllm_python).parent / "vllm"
    head = [str(vllm_bin), "serve"] if vllm_bin.exists() else [str(vllm_python), "-m", "vllm.entrypoints.cli.main", "serve"]
    cmd = head + [str(ckpt), "--served-model-name", served_name, "--host", "127.0.0.1", "--port", str(port),
                  "--dtype", "bfloat16", "--max-model-len", str(max_model_len), "--gpu-memory-utilization",
                  str(gpu_mem), "--seed", "0", "--enable-auto-tool-choice", "--tool-call-parser", tool_parser]
    if reasoning_parser:
        cmd += ["--reasoning-parser", reasoning_parser]
    env = dict(os.environ)
    env["PATH"] = str(Path(vllm_python).parent) + os.pathsep + env.get("PATH", "")
    env.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    log = open(log_path, "a")
    log.write("$ " + " ".join(cmd) + "\n")
    log.flush()
    p = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True)
    base = f"http://127.0.0.1:{port}"
    try:
        t0 = time.time()
        while True:
            if p.poll() is not None:
                raise RuntimeError(f"vLLM server exited with code {p.returncode}; see {log_path}")
            try:
                with urllib.request.urlopen(base + "/health", timeout=5) as r:
                    if r.status == 200:
                        break
            except OSError:
                pass
            if time.time() - t0 > timeout:
                raise TimeoutError(f"vLLM server not healthy after {timeout}s; see {log_path}")
            time.sleep(5)
        yield base
    finally:
        try:
            os.killpg(p.pid, signal.SIGTERM)
            p.wait(timeout=120)
        except ProcessLookupError:
            pass
        except subprocess.TimeoutExpired:
            os.killpg(p.pid, signal.SIGKILL)
            p.wait()
        log.close()
