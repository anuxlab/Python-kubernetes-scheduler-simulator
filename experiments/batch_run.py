#!/usr/bin/env python3
"""
experiments/batch_run.py – Batch experiment automation for the simulator.

This script enables large‑scale parameter sweeps and parallel execution.

Features:
    - Define a base configuration (YAML) with placeholders for parameters.
    - Sweep over policies, seeds, workload parameters (num_pods, gang_prob, etc.).
    - Run simulations in parallel using multiprocessing.
    - Collect results into a single CSV/JSON summary file.
    - Optionally generate shell scripts for cluster execution.

Usage:
    # Simple sweep over policies
    python batch_run.py --base configs/basic.yaml --sweep policy:bestfit,spread,fgd --runs 5

    # Sweep over multiple parameters
    python batch_run.py --base configs/medium_experiment.yaml \\
        --sweep policy:bestfit,spread,fgd gang_prob:0.1,0.3,0.5 num_pods:200,500 \\
        --runs 3 --parallel 4

    # Generate shell scripts instead of running directly
    python batch_run.py --base configs/medium_experiment.yaml \\
        --sweep policy:bestfit,spread --runs 3 --generate-scripts

    # Use a custom output directory
    python batch_run.py --base configs/basic.yaml --sweep policy:bestfit,fgd \\
        --output batch_results/
"""

import argparse
import csv
import itertools
import json
import multiprocessing
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

# -------------------------------------------------------------------------
# Utilities
# -------------------------------------------------------------------------


def load_yaml(filepath: str) -> dict[str, Any]:
    """Load YAML file."""
    with open(filepath, "r") as f:
        return yaml.safe_load(f)


def save_yaml(data: dict[str, Any], filepath: str) -> None:
    """Save YAML file."""
    with open(filepath, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)


def merge_dicts(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Deep merge two dictionaries."""
    result = base.copy()
    for key, value in override.items():
        if isinstance(value, dict) and key in result and isinstance(result[key], dict):
            result[key] = merge_dicts(result[key], value)
        else:
            result[key] = value
    return result


def parse_sweep_spec(spec: str) -> tuple[str, list[str]]:
    """
    Parse a sweep specification like "policy:bestfit,spread,fgd".
    Returns (key, list_of_values).
    """
    if ":" not in spec:
        raise ValueError(
            f"Invalid sweep spec: {spec}. Expected 'key:value1,value2,...'"
        )
    key, values_str = spec.split(":", 1)
    values = [v.strip() for v in values_str.split(",") if v.strip()]
    # Try to convert to appropriate types
    converted = []
    for v in values:
        # Try int, float, else string
        try:
            if "." in v:
                converted.append(float(v))
            else:
                converted.append(int(v))
        except ValueError:
            converted.append(v)
    return key, converted


def flatten_parameters(
    base_params: dict[str, Any], sweep_params: dict[str, list[Any]]
) -> list[dict[str, Any]]:
    """
    Generate all combinations of sweep parameters.
    Returns a list of dicts, each containing the parameter values for one run.
    """
    # Ensure sweep_params are lists
    keys = list(sweep_params.keys())
    values = list(sweep_params.values())
    combinations = list(itertools.product(*values))
    result = []
    for combo in combinations:
        param_dict = dict(zip(keys, combo))
        result.append(param_dict)
    return result


def update_config_with_params(
    config: dict[str, Any], params: dict[str, Any]
) -> dict[str, Any]:
    """
    Update a configuration dict with parameter values.
    Supports dotted paths like "workload.num_pods" or "simulation.policy".
    """
    result = config.copy()
    for key, value in params.items():
        if "." in key:
            # Dotted path: e.g., workload.num_pods
            parts = key.split(".")
            current = result
            for part in parts[:-1]:
                if part not in current:
                    current[part] = {}
                current = current[part]
            current[parts[-1]] = value
        else:
            # Top-level key
            result[key] = value
    return result


# -------------------------------------------------------------------------
# Single run execution
# -------------------------------------------------------------------------


def run_single_experiment(
    config_file: str, output_dir: str, seed: int | None = None
) -> dict[str, Any]:
    """
    Run a single simulation using the `main.py apply` or `run` command.
    For simplicity, we assume we have a unified config file.
    We'll call `main.py run --config <config> --output <output_dir> [--seed]`.
    Returns a dictionary with results (or raises on error).
    """
    cmd = [
        sys.executable,
        "main.py",
        "run",
        "--config",
        config_file,
        "--output",
        output_dir,
    ]
    if seed is not None:
        cmd.extend(["--seed", str(seed)])

    # Run subprocess
    try:
        subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
            timeout=3600,  # 1 hour timeout
        )
        # Parse the JSON stats file
        Path(output_dir) / "bestfit_stats.json"  # TODO: policy may differ
        # Actually, the policy is in the config, we need to read it.
        # Better: we can read the config to know the policy.
        # But we can also just scan for a .json file in output_dir
        json_files = list(Path(output_dir).glob("*.json"))
        if not json_files:
            raise RuntimeError(f"No JSON stats file found in {output_dir}")
        with open(json_files[0], "r") as f:
            stats = json.load(f)
        return stats
    except subprocess.CalledProcessError as e:
        print(f"Error running experiment: {e.stderr}")
        raise
    except Exception as e:
        print(f"Unexpected error: {e}")
        raise


# -------------------------------------------------------------------------
# Batch orchestrator
# -------------------------------------------------------------------------


def run_batch(
    base_config_file: str,
    sweep_params: dict[str, list[Any]],
    runs_per_combination: int,
    output_dir: str,
    parallel: int = 1,
    generate_scripts: bool = False,
    seeds: list[int] | None = None,
) -> None:
    """
    Run a batch of experiments.
    """
    base_config = load_yaml(base_config_file)
    # Generate all parameter combinations
    param_combinations = flatten_parameters(base_config, sweep_params)

    # For each combination, we will create runs (with different seeds)
    all_work_items = []
    for combo in param_combinations:
        for run_idx in range(runs_per_combination):
            # Determine seed: either use provided list or generate from run index
            if seeds is not None and run_idx < len(seeds):
                seed = seeds[run_idx]
            else:
                seed = 42 + run_idx  # simple sequential
            # Create a config file for this specific run
            config = update_config_with_params(base_config, combo)
            # Add the seed to the simulation section (override)
            if "simulation" not in config:
                config["simulation"] = {}
            config["simulation"]["seed"] = seed

            # Generate a unique output subdirectory
            # Build a descriptive name
            combo_str = "_".join([f"{k}_{v}" for k, v in combo.items()])
            run_dir_name = f"run_{combo_str}_seed{seed}"
            if runs_per_combination > 1:
                run_dir_name = f"{run_dir_name}_{run_idx}"

            # Create temporary config file
            # We'll use a temp directory or a dedicated configs/ subdir
            temp_config_dir = Path(output_dir) / "configs"
            temp_config_dir.mkdir(parents=True, exist_ok=True)
            config_file = temp_config_dir / f"{run_dir_name}.yaml"
            save_yaml(config, config_file)

            output_run_dir = Path(output_dir) / "results" / run_dir_name
            output_run_dir.mkdir(parents=True, exist_ok=True)

            work_item = {
                "config_file": str(config_file),
                "output_dir": str(output_run_dir),
                "seed": seed,
                "params": combo,
                "run_id": run_idx,
            }
            all_work_items.append(work_item)

    print(f"Total work items: {len(all_work_items)}")

    if generate_scripts:
        # Generate shell scripts instead of running directly
        generate_shell_scripts(all_work_items, output_dir)
        return

    # Run in parallel
    if parallel > 1:
        with multiprocessing.Pool(processes=parallel) as pool:
            results = pool.map(run_work_item, all_work_items)
    else:
        results = [run_work_item(item) for item in all_work_items]

    # Collect results into a summary
    summary_file = Path(output_dir) / "summary.json"
    with open(summary_file, "w") as f:
        json.dump(results, f, indent=2)

    # Also produce a CSV summary for easy analysis
    csv_file = Path(output_dir) / "summary.csv"
    if results:
        # Flatten stats and params
        flat_rows = []
        for res in results:
            row = {}
            # Include parameters
            for k, v in res["params"].items():
                row[f"param_{k}"] = v
            # Include run info
            row["seed"] = res["seed"]
            row["run_id"] = res["run_id"]
            # Include aggregate stats
            stats = res.get("stats", {})
            for stat_key, stat_val in stats.items():
                row[stat_key] = stat_val
            flat_rows.append(row)

        with open(csv_file, "w", newline="") as f:
            if flat_rows:
                fieldnames = flat_rows[0].keys()
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(flat_rows)

    print(f"Batch completed. Results saved to {output_dir}")


def run_work_item(item: dict[str, Any]) -> dict[str, Any]:
    """
    Execute a single work item (for multiprocessing).
    """
    try:
        stats = run_single_experiment(
            config_file=item["config_file"],
            output_dir=item["output_dir"],
            seed=item["seed"],
        )
        return {
            "success": True,
            "params": item["params"],
            "seed": item["seed"],
            "run_id": item["run_id"],
            "stats": stats,
        }
    except Exception as e:  # noqa: BLE001
        # Catch-all for any unexpected error during the experiment
        return {
            "success": False,
            "params": item["params"],
            "seed": item["seed"],
            "run_id": item["run_id"],
            "error": str(e),
        }


def generate_shell_scripts(work_items: list[dict[str, Any]], output_dir: str) -> None:
    """
    Generate a shell script with all run commands.
    """
    scripts_dir = Path(output_dir) / "scripts"
    scripts_dir.mkdir(parents=True, exist_ok=True)

    # Single script with all commands
    script_file = scripts_dir / "run_all.sh"
    with open(script_file, "w") as f:
        f.write("#!/bin/bash\n")
        f.write(f"# Generated batch script for {len(work_items)} experiments\n\n")
        for item in work_items:
            cmd = (
                f"python main.py run --config {item['config_file']} "
                f"--output {item['output_dir']} --seed {item['seed']}"
            )
            f.write(f"echo 'Running {item['run_id']}...'\n")
            f.write(f"{cmd}\n\n")
    os.chmod(script_file, 0o755)
    print(f"Generated shell script: {script_file}")

    # Also generate individual scripts for each run (optional)
    for item in work_items:
        run_script = scripts_dir / f"run_{item['run_id']}.sh"
        with open(run_script, "w") as f:
            f.write("#!/bin/bash\n")
            cmd = (
                f"python main.py run --config {item['config_file']} "
                f"--output {item['output_dir']} --seed {item['seed']}"
            )
            f.write(cmd + "\n")
        os.chmod(run_script, 0o755)


# -------------------------------------------------------------------------
# CLI
# -------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Batch experiment automation for scheduler simulator.",
        epilog="Example: python batch_run.py --base configs/basic.yaml --sweep policy:bestfit,spread,fgd --runs 5",
    )
    parser.add_argument(
        "--base", "-b", required=True, help="Base YAML configuration file."
    )
    parser.add_argument(
        "--sweep",
        "-s",
        action="append",
        help="Sweep parameter specification, e.g., policy:bestfit,spread,fgd. Can be repeated.",
    )
    parser.add_argument(
        "--runs",
        "-r",
        type=int,
        default=1,
        help="Number of runs per parameter combination (each with different seed).",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="batch_results",
        help="Output directory for batch results (default: batch_results).",
    )
    parser.add_argument(
        "--parallel",
        "-p",
        type=int,
        default=1,
        help="Number of parallel processes (default: 1).",
    )
    parser.add_argument(
        "--generate-scripts",
        "-g",
        action="store_true",
        help="Generate shell scripts instead of running directly.",
    )
    parser.add_argument(
        "--seeds",
        "-S",
        nargs="+",
        type=int,
        help="List of seeds to use for the runs (will be used cyclically).",
    )

    args = parser.parse_args()

    # Parse sweep parameters
    sweep_params = {}
    if args.sweep:
        for spec in args.sweep:
            key, values = parse_sweep_spec(spec)
            sweep_params[key] = values
    else:
        # Default: sweep over policies if not specified
        sweep_params = {"simulation.policy": ["bestfit", "spread", "fgd", "gpupack"]}

    # Determine seeds
    seeds = args.seeds
    if seeds is None:
        # If not provided, we'll generate seeds based on run index
        seeds = None  # handled in orchestrator

    run_batch(
        base_config_file=args.base,
        sweep_params=sweep_params,
        runs_per_combination=args.runs,
        output_dir=args.output,
        parallel=args.parallel,
        generate_scripts=args.generate_scripts,
        seeds=seeds,
    )


if __name__ == "__main__":
    main()
