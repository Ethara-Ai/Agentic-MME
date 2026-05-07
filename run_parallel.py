#!/usr/bin/env python3
"""
Parallel model runner — launches atomic LiteLLM runner for multiple models simultaneously.

Usage:
    python run_parallel.py --models kimi nova --task_dir ./akatsuki_dataset/json --dataset_root ./akatsuki_dataset

Models are resolved from configs/api.json under the "models" key:
    {
      "models": {
        "kimi": { "api_key": "ABSK...", "model": "bedrock/converse/arn:...", "base_model": "moonshotai.kimi-k2.5" },
        "nova": { "api_key": "ABSK...", "model": "bedrock/converse/arn:...", "base_model": "amazon.nova-pro-v1:0" }
      }
    }

The "base_model" field maps the ARN to a known model ID for LiteLLM cost tracking.
Each model runs in a separate subprocess. Results go to <output_dir>/<model_name>/
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed


def _load_model_config(config_path: Path, name: str) -> dict:
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    models = cfg.get("models", {})
    if name not in models:
        available = ", ".join(models.keys()) or "(none defined)"
        raise ValueError(
            f"Model '{name}' not found in {config_path}. "
            f"Available: {available}. Add it under the 'models' key."
        )
    entry = models[name]
    if not entry.get("model"):
        raise ValueError(f"Model '{name}' in {config_path} is missing 'model' field")
    if not entry.get("api_key"):
        top_key = cfg.get("api_key", "")
        if top_key:
            entry["api_key"] = top_key
        else:
            raise ValueError(f"Model '{name}' in {config_path} has no 'api_key' (and no top-level fallback)")
    return entry


def run_model(
    model_config: dict,
    task_dir: str,
    dataset_root: str,
    images_dir: str,
    output_dir: str,
    label: str,
    temperature: float,
    max_rounds: int,
    max_tool_calls: int,
    max_tasks: int,
    tasks: list,
    enable_search: bool,
    search_config: str,
    task_delay: float,
    max_retries: int,
    tasks_per_minute: int,
) -> dict:
    model_output = str(Path(output_dir) / label)
    Path(model_output).mkdir(parents=True, exist_ok=True)

    # Write a per-model temp config so the runner reads model, api_key, base_model from it
    tmp_cfg = tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", prefix=f"cfg_{label}_", delete=False
    )
    json.dump(model_config, tmp_cfg, indent=2)
    tmp_cfg.close()

    cmd = [
        sys.executable,
        str(Path(__file__).parent / "atomic" / "run_atomic_tools_litellm.py"),
        "--api_config", tmp_cfg.name,
        "--task_dir", task_dir,
        "--out_dir", model_output,
        "--temperature", str(temperature),
        "--max_rounds", str(max_rounds),
        "--max_tool_calls", str(max_tool_calls),
    ]

    if max_tasks > 0:
        cmd.extend(["--max_tasks", str(max_tasks)])
    if tasks:
        cmd.extend(["--tasks"] + tasks)
    if dataset_root:
        cmd.extend(["--dataset_root", dataset_root])
    if images_dir:
        cmd.extend(["--images_dir", images_dir])
    if enable_search and search_config:
        cmd.extend(["--enable_search", "--search_config", search_config])
    cmd.extend(["--task_delay", str(task_delay)])
    cmd.extend(["--max_retries", str(max_retries)])
    if tasks_per_minute > 0:
        cmd.extend(["--tasks_per_minute", str(tasks_per_minute)])

    print(f"\n{'='*60}")
    print(f"[LAUNCH] {label} ({model_config['model'][:80]}...)")
    print(f"[OUTPUT] {model_output}")
    print(f"{'='*60}\n")

    t0 = time.time()
    log_path = Path(model_output) / "runner.log"

    with open(log_path, "w") as log_f:
        proc = subprocess.run(
            cmd,
            stdout=log_f,
            stderr=subprocess.STDOUT,
            text=True,
        )

    elapsed = time.time() - t0
    Path(tmp_cfg.name).unlink(missing_ok=True)
    status = "SUCCESS" if proc.returncode == 0 else f"FAILED (exit={proc.returncode})"

    return {
        "model": model_config["model"],
        "label": label,
        "status": status,
        "returncode": proc.returncode,
        "elapsed_s": round(elapsed, 1),
        "output_dir": model_output,
        "log": str(log_path),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Run atomic LiteLLM runner for multiple models in parallel",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--models", nargs="+", required=True,
                        help="Model names as defined in api.json 'models' key (e.g. kimi nova)")
    parser.add_argument("--api_config", default="configs/api.json",
                        help="Path to api.json config (default: configs/api.json)")
    parser.add_argument("--task_dir", required=True,
                        help="Directory with task JSON files")
    parser.add_argument("--dataset_root", default="",
                        help="Dataset root directory")
    parser.add_argument("--images_dir", default="",
                        help="Images directory override")
    parser.add_argument("--output_dir", default="./results",
                        help="Root output directory (each model gets a subdirectory)")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max_rounds", type=int, default=50)
    parser.add_argument("--max_tool_calls", type=int, default=50)
    parser.add_argument("--enable_search", action="store_true")
    parser.add_argument("--search_config", default="")
    parser.add_argument("--max_tasks", type=int, default=0,
                        help="Max tasks per model (0 = unlimited)")
    parser.add_argument("--tasks", nargs="+", default=[],
                        help="Specific task IDs to run (stems without .json, e.g. 0001 0005 0012)")
    parser.add_argument("--max_workers", type=int, default=None,
                        help="Max parallel processes (default: number of models)")
    parser.add_argument("--task_delay", type=float, default=2.0,
                        help="Delay between tasks in seconds (forwarded to runner)")
    parser.add_argument("--max_retries", type=int, default=3,
                        help="Max retries for rate limit errors (forwarded to runner)")
    parser.add_argument("--tasks_per_minute", type=int, default=0,
                        help="Max tasks per 60s window per model (0=unlimited, forwarded to runner)")

    args = parser.parse_args()

    config_path = Path(args.api_config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}. Copy from configs/api.json.example")

    model_configs = []
    for name in args.models:
        entry = _load_model_config(config_path, name)
        model_configs.append((name, entry))

    max_workers = args.max_workers or len(model_configs)

    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = Path(args.output_dir) / run_timestamp

    print(f"\n{'#'*60}")
    print(f"# Parallel Atomic Runner — {len(model_configs)} models")
    print(f"# Config: {config_path}")
    print(f"# Task dir: {args.task_dir}")
    print(f"# Output: {output_root}/<model_name>/")
    print(f"# Workers: {max_workers}")
    for name, entry in model_configs:
        print(f"#   {name}: {entry['model'][:70]}...")
    print(f"{'#'*60}\n")

    results = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        for name, entry in model_configs:
            fut = executor.submit(
                run_model,
                model_config=entry,
                task_dir=args.task_dir,
                dataset_root=args.dataset_root,
                images_dir=args.images_dir,
                output_dir=str(output_root),
                label=name,
                temperature=args.temperature,
                max_rounds=args.max_rounds,
                max_tool_calls=args.max_tool_calls,
                max_tasks=args.max_tasks,
                tasks=args.tasks,
                enable_search=args.enable_search,
                search_config=args.search_config,
                task_delay=args.task_delay,
                max_retries=args.max_retries,
                tasks_per_minute=args.tasks_per_minute,
            )
            futures[fut] = name

        for fut in as_completed(futures):
            label = futures[fut]
            try:
                result = fut.result()
                results.append(result)
                print(f"\n[DONE] {label}: {result['status']} in {result['elapsed_s']}s")
            except Exception as e:
                print(f"\n[ERROR] {label}: {e}")
                results.append({"label": label, "status": f"EXCEPTION: {e}", "returncode": -1})

    print(f"\n\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for r in sorted(results, key=lambda x: x.get("label", "")):
        emoji = "✓" if r.get("returncode") == 0 else "✗"
        print(f"  {emoji} {r['label']:20s} {r['status']:20s} {r.get('elapsed_s', '?')}s")
        if r.get("log"):
            print(f"    Log: {r['log']}")
    print()


if __name__ == "__main__":
    main()
