"""Run the agent-free pipeline end to end for one model, and log what a write-up needs.

  uv run python scripts/run_experiment.py configs/rerun_0.8b.json --dry-run   # print the plan, run nothing
  uv run python scripts/run_experiment.py configs/rerun_0.8b.json             # run whatever is not done yet
  uv run python scripts/run_experiment.py configs/rerun_0.8b.json --only layers,edits --force

Stages, in order. A finished step is skipped next time; --force reruns the selected stages.

  base_ckpt   text-only checkpoint of the unedited model                      afwsp.ckpt
              (first derives the model's text-only config if it is missing    scripts/make_text_config.py)
  traces      base answers on fit_gen, the traces the pairs are cut from     afwsp.eval.maths (vLLM)
              (reused when the folder's .model stamp names this model; refused when it names another)
  pairs       matched continuations                                          scripts/gen_pairs.py (vLLM)
  directions  the direction at every layer, and every pair's difference      scripts/pair_directions.py
  acts        activations at every layer, in two passes                      scripts/collect_layers.py
  layers      the LP at every layer, ranked                                  scripts/layer_search.py
  edits       one edit per arm: "top2" takes the ranking's top pair, a list fixes the layers
  ckpts       a checkpoint per arm                                           afwsp.ckpt
  evals       every benchmark for every arm, then base unless a base run is reused (vLLM)
  observe     observation summaries                                          scripts/observe.py
  compare     paired comparisons and report.md                               scripts/compare.py

Outputs: runs/<name>/ (manifest.json, events.jsonl, gpu.csv, logs/, observations/, report.md), cache/<name>/
(pairs, activations, LP solutions), edits/<name>/, ckpts/<name>/, results/gpu/<name>/<arm>/<benchmark>/.
While a run is going, `tail -f runs/<name>/events.jsonl` shows each step starting and ending, and gpu.csv samples
utilisation, memory and power every 30 s; each step's mean utilisation, peak memory and power go in the manifest.
"""
import argparse
import hashlib
import json
import os
import platform
import re
import shlex
import shutil
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STAGES = ["base_ckpt", "traces", "pairs", "directions", "acts", "layers", "edits", "ckpts", "evals", "observe",
          "compare"]
HF, VLLM = "hf", "vllm"
_MATHS = ["-m", "afwsp.eval.maths", "{ckpt}", "{out}"]
BENCHMARKS = {   # name -> (interpreter, arguments; {ckpt} and {out} are filled in)
    "dev_a": (VLLM, _MATHS + ["--sets", "gsm8k_train,math_train", "--n", "1", "--ids", "split:dev_a"]),
    "dev_b": (VLLM, _MATHS + ["--sets", "gsm8k_train,math_train", "--n", "1", "--ids", "split:dev_b"]),
    "final_practice": (VLLM, _MATHS + ["--sets", "gsm8k_train,math_train", "--n", "1", "--ids", "split:final_practice"]),
    "math500": (VLLM, _MATHS + ["--sets", "math500", "--n", "1"]),
    "gsm8k_test": (VLLM, _MATHS + ["--sets", "gsm8k", "--n", "1", "--limit", "500"]),   # as in RESULTS.md
    "ifeval": (VLLM, ["-m", "afwsp.eval.ifeval", "{ckpt}", "{out}"]),
    "medqa": (VLLM, ["-m", "afwsp.eval.medqa", "{ckpt}", "{out}"]),
    "medqa_open": (VLLM, ["-m", "afwsp.eval.medqa_open", "{ckpt}", "{out}"]),
    # agentic: these serve the checkpoint with vLLM themselves and drive BFCL / tau2 in their own environments
    "bfcl_v4": (HF, ["-m", "afwsp.eval.bfcl", "{ckpt}", "{out}"]),
    "tau2": (HF, ["-m", "afwsp.eval.tau2", "{ckpt}", "{out}"]),
}
AGENTIC = ("bfcl_v4", "tau2")


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Run:
    def __init__(self, cfg_path: Path, dry: bool):
        self.cfg_path, self.dry = cfg_path, dry
        c = self.cfg = json.load(open(cfg_path))
        self.name = c["name"]
        self.dir = ROOT / "runs" / self.name
        self.cache, self.edits = ROOT / "cache" / self.name, ROOT / "edits" / self.name
        self.ckpts, self.results = ROOT / "ckpts" / self.name, ROOT / "results" / "gpu" / self.name
        self.model_dir, self.text_config = ROOT / c["model_dir"], ROOT / c["text_config"]
        self.traces = ROOT / c["traces"]["dir"]
        self.hf = list(c.get("hf_python", ["uv", "run", "python"]))
        self.vllm = c.get("vllm_python", "/workspace/vllmenv/bin/python")
        self.lp = {"EPS": 1.0, "K": 8, "MULT": 4, "SEED": 0, **c.get("lp", {})}
        self.k = int(self.lp["K"])
        self.tag = f"{self.name}_eps{self.lp['EPS']}_m{self.lp['MULT']}_s{self.lp['SEED']}"
        self.arms = list(c["arms"])
        # an arm given as {"edit": "<file>"} evaluates an existing edit; the pipeline runs only if some arm needs it
        self.file_arms = {a for a, s in c["arms"].items() if isinstance(s, dict) and "edit" in s}
        self.build = any(a not in self.file_arms for a in self.arms)
        mp = self.dir / "manifest.json"
        self.manifest = json.load(open(mp)) if mp.exists() else {"name": self.name, "created": now(), "steps": {}}

    # ------------------------------------------------------------------------------------------------ helpers
    def n_layers(self) -> int:
        f = self.text_config / "config.json"
        if f.exists():
            return int(json.load(open(f))["num_hidden_layers"])
        if self.dry:
            return int(self.cfg.get("n_layers", 24))
        raise SystemExit(f"{f} is missing: run scripts/fetch_exp8.sh (0.8B) or scripts/make_text_config.py")

    def py(self, kind, args):
        return (self.hf if kind == HF else [self.vllm]) + [str(a) for a in args]

    def env(self, extra):
        e = dict(os.environ)
        e.update({"AFWSP_MODEL": str(self.model_dir), "AFWSP_TEXT_CONFIG": str(self.text_config),
                  "AFWSP_CACHE": str(self.cache), "AFWSP_EDITS": str(self.edits), "BASE_TRACES": str(self.traces),
                  "VLLM_USE_FLASHINFER_SAMPLER": "0",          # see docs/REPRODUCE.md
                  "VLLM_WORKER_MULTIPROC_METHOD": "spawn", "PYTHONUNBUFFERED": "1"})
        e["PATH"] = str(Path(self.vllm).parent) + os.pathsep + e.get("PATH", "")   # vLLM needs its venv's ninja
        e.update({k: str(v) for k, v in extra.items()})
        return e

    def edit_path(self, arm) -> Path:
        if arm in self.file_arms:
            return ROOT / self.cfg["arms"][arm]["edit"]
        return self.edits / f"{arm}_k{self.k}.safetensors"

    def arm_layers(self, arm):
        spec = self.cfg["arms"][arm]
        if arm in self.file_arms:
            return edit_layers(self.edit_path(arm))
        if isinstance(spec, list):
            return sorted(int(x) for x in spec)
        if spec != "top2":
            raise SystemExit(f"arm {arm}: use \"top2\" or a list of layers, not {spec!r}")
        f = self.cache / f"layer_search_{self.tag}_k{self.k}.json"
        return sorted(r["layer"] for r in json.load(open(f))[:2]) if f.exists() else None

    def save(self):
        self.manifest["updated"] = now()
        self.dir.mkdir(parents=True, exist_ok=True)
        json.dump(self.manifest, open(self.dir / "manifest.json", "w"), indent=1)

    def event(self, **kw):
        self.dir.mkdir(parents=True, exist_ok=True)
        with open(self.dir / "events.jsonl", "a") as f:
            f.write(json.dumps({"time": now(), **kw}) + "\n")

    # ------------------------------------------------------------------------------------------------ stages
    def s_base_ckpt(self):
        steps = []
        if not (self.text_config / "config.json").exists():     # any size but the 0.8B: derive it
            steps.append(("text_config", self.py(HF, ["scripts/make_text_config.py", self.model_dir, self.text_config]),
                          {}, None))
        return steps + [("base_ckpt", self.py(HF, ["-m", "afwsp.ckpt", self.ckpts / "base", "-"]), {}, None)]

    def s_traces(self):
        if not self.build:
            print("[skip] traces, pairs, directions, acts, layers: every arm is an existing edit file")
            return []
        stamp = self.traces / ".model"
        if any(self.traces.glob("*_samples.jsonl.gz")):
            got = stamp.read_text().strip() if stamp.exists() else None
            if got != self.cfg["model"]:
                raise SystemExit(f"{self.traces} holds traces stamped {got!r}, not {self.cfg['model']!r}. Point "
                                 "traces.dir at an empty folder to generate this model's own (docs/REPRODUCE.md).")
            self.manifest["traces"] = {"reused": rel(self.traces)}
            print(f"[reuse] traces: {rel(self.traces)} (stamped {got})")
            return []
        t = self.cfg["traces"]
        args = ["-m", "afwsp.eval.maths", self.ckpts / "base", self.traces, "--sets", "gsm8k_train,math_train",
                "--n", "1", "--ids", f"split:{t.get('split', 'fit_gen')}"] + (["--limit", t["limit"]] if t.get("limit") else [])
        return [("traces", self.py(VLLM, args), {}, lambda: stamp.write_text(self.cfg["model"] + "\n"))]

    def s_pairs(self):
        if not self.build:
            return []
        env = {"PAIR_MAX_PREFIX": 300, "PAIR_N_LOOP": 8, "PAIR_N_PROD": 0, **self.cfg.get("pairs", {})}
        return [("pairs", self.py(VLLM, ["scripts/gen_pairs.py", self.model_dir]), env, None)]

    def s_directions(self):
        if not self.build:
            return []
        env = {"PAIR_LAYERS": self.n_layers(), **self.cfg.get("directions", {})}
        return [("directions", self.py(HF, ["scripts/pair_directions.py"]), env, None)]

    def s_acts(self):
        if not self.build:
            return []
        n = self.n_layers()
        a = {"TRACES": 300, "OUT": "acts_all", **self.cfg.get("acts", {})}
        return [(f"acts/{lo}-{hi}", self.py(HF, ["scripts/collect_layers.py"]), {**a, "LAYERS": f"{lo}-{hi}"}, None)
                for lo, hi in ((0, n // 2 - 1), (n // 2, n - 1))]   # two passes keep peak RAM near half

    def s_layers(self):
        if not self.build:
            return []
        env = {"ACTS": "acts_all", "N_LAYERS": self.n_layers(), "LS_TAG": self.tag, **self.lp}
        return [("layers", self.py(HF, ["scripts/layer_search.py"]), env, None)]

    def s_edits(self):
        steps, chosen = [], {}
        for arm in self.arms:
            layers = self.arm_layers(arm)
            if arm in self.file_arms:              # nothing to build: record which layers the file touches
                chosen[arm] = layers
                continue
            if layers is None and not self.dry:
                raise SystemExit(f"arm {arm} needs the layer search: run the 'layers' stage first")
            chosen[arm] = layers
            lay = ",".join(map(str, layers)) if layers else "<top-2-of-layer-search>"
            steps.append((f"edits/{arm}", self.py(HF, ["scripts/make_edit_layers.py", f"{arm}_k{self.k}", self.k, lay]),
                          {"LS_TAG": self.tag}, None))
        self.manifest["arms"] = chosen
        return steps

    def s_ckpts(self):
        for arm in self.file_arms:
            if not self.edit_path(arm).exists():
                msg = f"edit file {rel(self.edit_path(arm))} for arm '{arm}' is missing"
                if not self.dry:
                    raise SystemExit(msg + ": put the edit there, or change the arm in the config")
                print(f"[warn] {msg}")
        return [(f"ckpts/{arm}", self.py(HF, ["-m", "afwsp.ckpt", self.ckpts / arm, self.edit_path(arm)]), {}, None)
                for arm in self.arms]

    def s_evals(self):
        reuse = self.cfg.get("base_runs", {})
        self.manifest["base_runs"] = reuse
        steps = []
        for bench in self.cfg["benchmarks"]:
            kind, argv = BENCHMARKS[bench]
            for arm in self.arms + ["base"]:             # the edits first: base runs are the slow ones
                if arm == "base" and bench in reuse:
                    continue
                out = self.results / arm / bench
                args = [a.format(ckpt=self.ckpts / arm, out=out) for a in argv] + self.cfg.get("bench_args", {}).get(bench, [])
                extra = {"MED_N": self.cfg.get("med_n", 512)} if bench.startswith("medqa") else {}
                if bench in AGENTIC:     # API keys (OPENAI_API_KEY, SERPAPI_API_KEY) come from the environment only
                    extra = {"AFWSP_VLLM_PYTHON": self.vllm,
                             **({"AFWSP_BFCL_PYTHON": self.cfg["bfcl_python"]} if bench == "bfcl_v4" else {}),
                             **({"AFWSP_TAU2": self.cfg["tau2_bin"]} if bench == "tau2" else {}),
                             **({"AFWSP_SAMPLING": json.dumps(self.cfg["agent_sampling"])}
                                if self.cfg.get("agent_sampling") else {})}
                steps.append((f"evals/{arm}/{bench}", self.py(kind, args), extra, None))
        return steps

    def s_observe(self):
        return [("observe", self.py(HF, ["scripts/observe.py", self.dir]), {}, None)]

    def s_compare(self):
        return [("compare", self.py(HF, ["scripts/compare.py", self.dir]), {}, None)]

    # ------------------------------------------------------------------------------------------------ running
    def describe_environment(self):
        info = {"python": sys.version.split()[0], "platform": platform.platform(),
                "config_file": str(self.cfg_path), "model_revision": None, "code_sha256": code_hash()}
        rev = self.model_dir / ".revision"
        info["model_revision"] = rev.read_text().strip() if rev.exists() else self.cfg.get("model_revision")
        info["gpu"] = probe(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"])
        info["git"] = probe(["git", "-C", str(ROOT), "describe", "--always", "--dirty"])
        for kind, names in ((HF, ["torch", "transformers", "numpy", "safetensors"]),
                            (VLLM, ["vllm", "torch", "transformers", "math-verify", "lm-eval", "datasets"])):
            code = ("import json, importlib.metadata as m\n"
                    "def v(p):\n    try:\n        return m.version(p)\n    except Exception:\n        return None\n"
                    f"print(json.dumps({{p: v(p) for p in {names!r}}}))")
            info[f"packages_{kind}"] = probe(self.py(kind, ["-c", code]))
        return info

    def run(self, stages, force):
        self.manifest.update({"config": self.cfg, "paths": {
            "run": rel(self.dir), "cache": rel(self.cache), "edits": rel(self.edits), "ckpts": rel(self.ckpts),
            "results": rel(self.results), "traces": rel(self.traces), "model_dir": rel(self.model_dir),
            "text_config": rel(self.text_config), "layer_search_tag": f"{self.tag}_k{self.k}"}})
        if not self.dry:
            self.manifest["environment"] = self.describe_environment()
            self.save()
        for i, stage in enumerate(stages, 1):
            print(f"== stage {i}/{len(stages)}: {stage}", flush=True)
            for key, cmd, extra, after in getattr(self, f"s_{stage}")():
                marker = self.dir / "state" / (key.replace("/", "__") + ".done")
                shown = " ".join(f"{k}={v}" for k, v in extra.items())
                if self.dry:
                    print(f"[plan] {key}\n    {shown + ' ' if shown else ''}{' '.join(shlex.quote(c) for c in cmd)}")
                    continue
                if marker.exists() and not force:
                    print(f"[done] {key}")
                    continue
                log = self.dir / "logs" / (key.replace("/", "__") + ".log")
                log.parent.mkdir(parents=True, exist_ok=True)
                rec = {"cmd": cmd, "env": {k: str(v) for k, v in extra.items()}, "log": rel(log), "start": now()}
                print(f"[run ] {key}  (log: {rel(log)})", flush=True)
                self.event(step=key, event="start")
                t0 = time.time()
                with GpuMonitor(self.dir / "gpu.csv", key) as gpu:
                    code = stream(cmd, self.env(extra), log)
                rec.update({"end": now(), "seconds": round(time.time() - t0), "exit_code": code,
                            "status": "ok" if code == 0 else "failed", **gpu.summary()})
                self.manifest["steps"][key] = rec
                self.save()
                self.event(step=key, event="end", status=rec["status"], seconds=rec["seconds"], exit_code=code,
                           **gpu.summary())
                if code != 0:
                    raise SystemExit(f"{key} failed with exit code {code}; see {rel(log)}")
                if after:
                    after()
                marker.parent.mkdir(parents=True, exist_ok=True)
                marker.write_text(now() + "\n")
        if not self.dry:
            self.save()


class GpuMonitor:
    """Samples nvidia-smi every `every` seconds while a step runs and appends to gpu.csv; summary() gives the step's
    mean utilisation, peak memory and mean power. A no-op where nvidia-smi is missing."""
    QUERY = ["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used,memory.total,power.draw",
             "--format=csv,noheader,nounits"]

    def __init__(self, path: Path, step: str, every: float = 30.0):
        self.path, self.step, self.every = path, step, every
        self.util, self.mem, self.power, self.mem_total = [], [], [], None
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._loop, daemon=True)

    def __enter__(self):
        if shutil.which("nvidia-smi"):
            self.thread.start()
        return self

    def __exit__(self, *exc):
        self.stop.set()
        if self.thread.is_alive():
            self.thread.join(timeout=30)

    def _loop(self):
        while True:
            self._sample()
            if self.stop.wait(self.every):
                return

    def _sample(self):
        try:
            out = subprocess.run(self.QUERY, capture_output=True, text=True, timeout=20).stdout
        except Exception:  # noqa: BLE001
            return
        new = not self.path.exists()
        with open(self.path, "a") as f:
            if new:
                f.write("time,step,gpu,util_pct,mem_used_mib,mem_total_mib,power_w\n")
            for line in out.strip().splitlines():
                parts = [x.strip() for x in line.split(",")]
                if len(parts) != 5:
                    continue
                f.write(",".join([now(), self.step, *parts]) + "\n")
                try:
                    self.util.append(float(parts[1]))
                    self.mem.append(float(parts[2]))
                    self.mem_total = float(parts[3])
                    self.power.append(float(parts[4]))
                except ValueError:       # e.g. power.draw reported as [N/A]
                    pass

    def summary(self) -> dict:
        if not self.util:
            return {}
        return {"gpu_util_mean_pct": round(sum(self.util) / len(self.util), 1), "gpu_mem_peak_mib": max(self.mem),
                "gpu_mem_total_mib": self.mem_total,
                "gpu_power_mean_w": round(sum(self.power) / len(self.power), 1) if self.power else None,
                "gpu_samples": len(self.util)}


def stream(cmd, env, log_path) -> int:
    with open(log_path, "a") as log:
        log.write(f"\n# {now()}\n$ {' '.join(shlex.quote(c) for c in cmd)}\n")
        log.flush()
        p = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                             bufsize=1, errors="replace")
        for line in p.stdout:
            sys.stdout.write(line)
            log.write(line)
        return p.wait()


def probe(cmd):
    try:
        out = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=300).stdout.strip()
    except Exception:  # noqa: BLE001
        return None
    try:
        return json.loads(out)
    except ValueError:
        return out or None


def edit_layers(path: Path) -> list[int]:
    """The layers an edit file touches, read from its safetensors header without loading any tensor."""
    if not Path(path).exists():
        raise SystemExit(f"edit file {path} is missing: put the edit there, or change the arm in the config")
    with open(path, "rb") as f:
        n = int.from_bytes(f.read(8), "little")
        keys = json.loads(f.read(n)).keys()
    return sorted({int(m.group(1)) for k in keys if (m := re.search(r"layers\.(\d+)\.mlp\.down_proj", k))})


def code_hash():
    h = hashlib.sha256()
    for f in sorted([*ROOT.glob("src/afwsp/**/*.py"), *ROOT.glob("scripts/*.py")]):
        h.update(str(f.relative_to(ROOT)).encode())
        h.update(f.read_bytes())
    return h.hexdigest()[:16]


def rel(p: Path) -> str:
    p = Path(p)
    return str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("config", type=Path)
    ap.add_argument("--dry-run", action="store_true", help="print every step and run nothing")
    ap.add_argument("--only", help="comma-separated stages to run")
    ap.add_argument("--from", dest="start", help="run from this stage on")
    ap.add_argument("--force", action="store_true", help="rerun steps of the selected stages even if done")
    a = ap.parse_args()
    stages = STAGES
    if a.only:
        stages = [s for s in STAGES if s in a.only.split(",")]
    elif a.start:
        stages = STAGES[STAGES.index(a.start):]
    unknown = set((a.only or "").split(",")) - set(STAGES) - {""}
    if unknown or (a.start and a.start not in STAGES):
        raise SystemExit(f"unknown stage(s): {unknown or a.start}; stages are {', '.join(STAGES)}")
    Run(a.config, a.dry_run).run(stages, a.force)


if __name__ == "__main__":
    main()
