"""Run the method end to end from one config, and log everything.

  python scripts/run.py configs/rerun_0.8b.json [--dry-run] [--only STAGE,...] [--from STAGE] [--force]

Stages, in order. A step that finished is skipped next time, unless --force.
  checkpoint   a vLLM checkpoint of the unedited model
  traces       the base model's answers to the 512 fit_gen problems
  pairs        matched continuations of the answers that loop
  direction    u at every layer
  activations  the MLP inputs h at every layer
  layers       the LP at every layer, ranked
  edits        one edit per arm: "top2" takes the ranking's top pair, a list fixes the layers
  arms         a checkpoint per arm
  evals        every benchmark, for every arm and for base
  report       observations, paired comparisons, report.md

Everything a run produces goes in runs/<name>/. The logs are logs/<step>.log (each step's output),
events.jsonl (each step's start and end), gpu.csv (nvidia-smi every 30 s) and manifest.json (config, commands,
timings, exit codes, GPU, versions, model revision).
"""
import argparse
import importlib.metadata
import json
import shlex
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STAGES = ["checkpoint", "traces", "pairs", "direction", "activations", "layers", "edits", "arms", "evals", "report"]
BENCHMARKS = {"math500": ["--sets", "math500", "--n", "1"],
              "gsm8k_test": ["--sets", "gsm8k", "--n", "1", "--limit", "500"]}


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def python(module, *args):
    return [sys.executable, "-m", module, *map(str, args)]


def arm_layers(spec, run):
    """The layers an arm edits, or None while the layer search has not run."""
    if isinstance(spec, list):
        return sorted(spec)
    ranking = run / "layers/layers.json"
    return sorted(r["layer"] for r in json.load(open(ranking))[:2]) if ranking.exists() else None


def plan(stage, config, run):
    """The steps of one stage, as (name, command) pairs."""
    model = ROOT / config["model"]
    arms = config["arms"]
    lp = {"eps": 1.0, "k": 8, "mult": 4, "seed": 0, **config.get("lp", {})}
    pairs = {"n_loop": 8, "n_prod": 0, "max_prefix": 300, **config.get("pairs", {})}
    if stage == "checkpoint":
        return [("checkpoint", python("afwsp.model", "checkpoint", model, run / "checkpoints/base"))]
    if stage == "traces":
        return [("traces", python("afwsp.evals", run / "checkpoints/base", run / "traces",
                                  "--sets", "gsm8k_train,math_train", "--n", 1, "--split", "fit_gen"))]
    if stage == "pairs":
        return [("pairs", python("afwsp.direction", "pairs", model, run / "traces", run / "pairs",
                                 "--n-loop", pairs["n_loop"], "--n-prod", pairs["n_prod"],
                                 "--max-prefix", pairs["max_prefix"]))]
    if stage == "direction":
        return [("direction", python("afwsp.direction", "build", model, run / "pairs"))]
    if stage == "activations":
        return [("activations", python("afwsp.edit", "collect", model, run / "traces", run / "activations",
                                       "--traces", config.get("activation_traces", 300)))]
    if stage == "layers":
        return [("layers", python("afwsp.edit", "search", model, run / "pairs", run / "activations", run / "layers",
                                  "--eps", lp["eps"], "--k", lp["k"], "--mult", lp["mult"], "--seed", lp["seed"]))]
    if stage == "edits":
        steps = []
        for arm, spec in arms.items():
            layers = arm_layers(spec, run)
            layers = ",".join(map(str, layers)) if layers else "<top pair of the layer search>"
            steps.append((f"edit-{arm}", python("afwsp.edit", "write", model, run / "layers", layers,
                                                 run / f"edits/{arm}.safetensors")))
        return steps
    if stage == "arms":
        return [(f"checkpoint-{arm}", python("afwsp.model", "checkpoint", model, run / f"checkpoints/{arm}",
                                             "--edit", run / f"edits/{arm}.safetensors"))
                for arm in arms]
    if stage == "evals":
        steps = []
        for benchmark in config["benchmarks"]:
            if benchmark not in BENCHMARKS:
                raise SystemExit(f"unknown benchmark {benchmark}; known: {', '.join(BENCHMARKS)}")
            for arm in list(arms) + ["base"]:          # the edits first: base runs are the slow ones
                steps.append((f"eval-{arm}-{benchmark}",
                              python("afwsp.evals", run / f"checkpoints/{arm}", run / f"results/{arm}/{benchmark}",
                                     *BENCHMARKS[benchmark])))
        return steps
    if stage == "report":
        return [("report", python("afwsp.report", run))]
    raise SystemExit(f"unknown stage {stage}")


def describe_machine(config):
    model = ROOT / config["model"]
    gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                         capture_output=True, text=True).stdout.strip() if shutil.which("nvidia-smi") else None
    git = subprocess.run(["git", "-C", ROOT, "describe", "--always", "--dirty"], capture_output=True, text=True)
    versions = {}
    for package in ("vllm", "torch", "transformers", "math-verify"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    revision = model / ".revision"
    return {"gpu": gpu or None, "git": git.stdout.strip() or None, "python": sys.version.split()[0],
            "versions": versions, "model_revision": revision.read_text().strip() if revision.exists() else None}


def run_step(name, command, run, manifest):
    log_file = run / "logs" / f"{name}.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    print(f"[run]  {name}   (log: {log_file.relative_to(ROOT)})", flush=True)
    record_event(run, step=name, event="start")
    began = time.time()
    with open(log_file, "a") as log:
        log.write(f"\n# {now()}\n$ {shlex.join(map(str, command))}\n")
        log.flush()
        process = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                   bufsize=1, errors="replace")
        for line in process.stdout:
            sys.stdout.write(line)
            log.write(line)
        code = process.wait()
    seconds = round(time.time() - began)
    status = "ok" if code == 0 else "failed"
    manifest["steps"][name] = {"command": shlex.join(map(str, command)), "started": now(), "seconds": seconds,
                               "status": status, "exit_code": code}
    save_manifest(run, manifest)
    record_event(run, step=name, event="end", status=status, seconds=seconds)
    if code != 0:
        raise SystemExit(f"{name} failed (exit code {code}); see {log_file.relative_to(ROOT)}")


def record_event(run, **fields):
    with open(run / "events.jsonl", "a") as f:
        f.write(json.dumps({"time": now(), **fields}) + "\n")


def save_manifest(run, manifest):
    manifest["updated"] = now()
    json.dump(manifest, open(run / "manifest.json", "w"), indent=1)


def main():
    parser = argparse.ArgumentParser(description="Run the method end to end from one config.")
    parser.add_argument("config")
    parser.add_argument("--dry-run", action="store_true", help="print every step and run nothing")
    parser.add_argument("--only", help="comma-separated stages")
    parser.add_argument("--from", dest="start", help="run from this stage on")
    parser.add_argument("--force", action="store_true", help="rerun steps that already finished")
    args = parser.parse_args()
    stages = STAGES
    if args.only:
        stages = [s for s in STAGES if s in args.only.split(",")]
    elif args.start:
        stages = STAGES[STAGES.index(args.start):]

    config = json.load(open(args.config))
    run = ROOT / "runs" / config["name"]
    if args.dry_run:
        for stage in stages:
            for name, command in plan(stage, config, run):
                print(f"[plan] {name}\n       {shlex.join(map(str, command))}")
        return

    run.mkdir(parents=True, exist_ok=True)
    manifest_file = run / "manifest.json"
    manifest = json.load(open(manifest_file)) if manifest_file.exists() else {"created": now(), "steps": {}}
    manifest.update({"config": config, **describe_machine(config)})
    save_manifest(run, manifest)
    gpu_log = None
    if shutil.which("nvidia-smi"):
        gpu_log = subprocess.Popen(["nvidia-smi", "--query-gpu=timestamp,utilization.gpu,memory.used,power.draw",
                                    "--format=csv", "-l", "30"], stdout=open(run / "gpu.csv", "a"))
    try:
        for stage in stages:
            print(f"== {stage}", flush=True)
            for name, command in plan(stage, config, run):
                if manifest["steps"].get(name, {}).get("status") == "ok" and not args.force:
                    print(f"[done] {name}", flush=True)
                    continue
                run_step(name, command, run, manifest)
            if stage == "layers" or stage == "edits":
                manifest["arms"] = {arm: arm_layers(spec, run) for arm, spec in config["arms"].items()}
                save_manifest(run, manifest)
    finally:
        if gpu_log:
            gpu_log.terminate()


if __name__ == "__main__":
    main()
