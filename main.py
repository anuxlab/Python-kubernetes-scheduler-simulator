#!/usr/bin/env python3
"""
main.py - CLI entry point for the Kubernetes scheduler simulator.

This script provides command-line access to the simulation framework.
It supports two modes:

  1. run   : Run a complete experiment from a single YAML configuration file.
  2. apply : Run a simulation using separate cluster, scheduler, and workload
             configuration files (mimicking the 'simon apply' command).

Usage examples:
    # Single config file
    python main.py run --config configs/basic.yaml --output results/run1/

    # Separate config files (like simon apply)
    python main.py apply --cluster cluster.yaml --scheduler scheduler.yaml \\
                         --workload workload.yaml --output results/apply1/

    # Override seed and enable plots
    python main.py run --config configs/medium_experiment.yaml --seed 42 --plot
"""

import sys
import os
import argparse
import yaml
import json
import csv
import random
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Union

# Import the simulation package
from sim import (
    Cluster, Node, Pod, ResourceRequest,
    Scheduler, Simulator, SimulationStats,
    default_runtime_func, resource_based_runtime,
    get_policy
)


# -------------------------------------------------------------------------
# Configuration loaders
# -------------------------------------------------------------------------

def load_cluster_from_yaml(cluster_file: str) -> Cluster:
    """
    Load cluster configuration from a YAML file.

    Expected format:
        nodes:
          - name: node-1
            capacity: { cpu: 4.0, memory: 8.0, gpu: 1 }
          ...

    Returns:
        Cluster object.
    """
    with open(cluster_file, 'r') as f:
        data = yaml.safe_load(f)
    nodes = []
    for node_cfg in data.get('nodes', []):
        cap = node_cfg['capacity']
        res = ResourceRequest(
            cpu=float(cap.get('cpu', 0.0)),
            memory=float(cap.get('memory', 0.0)),
            gpu=int(cap.get('gpu', 0))
        )
        node = Node(name=node_cfg['name'], capacity=res)
        nodes.append(node)
    return Cluster(nodes)


def load_scheduler_from_yaml(scheduler_file: str) -> Dict[str, Any]:
    """
    Load scheduler configuration from a YAML file.

    Expected format:
        policy: bestfit
        until: 5000.0
        log_interval: 100.0
        seed: 42
        runtime_function: default   # or resource_based

    Returns:
        Dictionary with scheduler parameters.
    """
    with open(scheduler_file, 'r') as f:
        data = yaml.safe_load(f)
    # Provide defaults if missing
    data.setdefault('policy', 'bestfit')
    data.setdefault('until', 10000.0)
    data.setdefault('log_interval', None)
    data.setdefault('seed', None)
    data.setdefault('runtime_function', 'default')
    return data


def load_workload_from_yaml(workload_file: str) -> Dict[str, Any]:
    """
    Load workload configuration from a YAML file.

    Expected format:
        type: synthetic          # or trace
        num_pods: 100
        inter_arrival: { type: exponential, mean: 1.5 }
        cpu: { type: uniform, min: 0.2, max: 1.0 }
        memory: { type: uniform, min: 0.5, max: 2.0 }
        gpu: { type: choice, values: [0, 0, 1] }
        gang_prob: 0.1
        gang_size: { type: uniform, min: 2, max: 3 }

    Returns:
        Dictionary with workload parameters.
    """
    with open(workload_file, 'r') as f:
        data = yaml.safe_load(f)
    return data


# -------------------------------------------------------------------------
# Workload builders (reused from experiments.py)
# -------------------------------------------------------------------------

def build_workload_from_config(workload_config: Dict[str, Any],
                               rng: random.Random) -> List[tuple]:
    """
    Build workload events from configuration. This is a copy of the function
    from experiments.py, but kept here for self‑contained CLI.

    Returns:
        List of (submit_time, pod_or_gang) events.
    """
    from collections import defaultdict

    wtype = workload_config.get('type', 'synthetic')

    if wtype == 'trace':
        trace_file = workload_config.get('file')
        if not trace_file:
            raise ValueError("Trace file not specified in workload config.")
        return load_trace_from_csv(trace_file)

    elif wtype == 'synthetic':
        return generate_synthetic_workload(workload_config, rng)

    else:
        raise ValueError(f"Unknown workload type: {wtype}")


def load_trace_from_csv(csv_file: str) -> List[tuple]:
    """Load workload trace from a CSV file (same as in experiments.py)."""
    from collections import defaultdict

    events_by_time = defaultdict(list)
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            t = float(row['time'])
            uid = row.get('uid', f"pod-{len(events_by_time)}")
            cpu = float(row.get('cpu', 0.0))
            memory = float(row.get('memory', 0.0))
            gpu = int(row.get('gpu', 0))
            gang_id = row.get('gang_id', None)
            pod = Pod(
                uid=uid,
                name=uid,
                resources=ResourceRequest(cpu=cpu, memory=memory, gpu=gpu),
                gang_id=gang_id,
            )
            events_by_time[t].append((gang_id, pod))

    events = []
    for t, items in events_by_time.items():
        gangs = defaultdict(list)
        non_gang = []
        for gang_id, pod in items:
            if gang_id:
                gangs[gang_id].append(pod)
            else:
                non_gang.append(pod)
        for gang_id, gang_pods in gangs.items():
            events.append((t, gang_pods))
        for pod in non_gang:
            events.append((t, pod))

    return sorted(events, key=lambda x: x[0])


def generate_synthetic_workload(config: Dict[str, Any],
                                rng: random.Random) -> List[tuple]:
    """Generate synthetic workload (same as in experiments.py)."""
    num_pods = config.get('num_pods', 100)
    inter_arrival_cfg = config.get('inter_arrival', {'type': 'exponential', 'mean': 5.0})
    cpu_cfg = config.get('cpu', {'type': 'uniform', 'min': 0.5, 'max': 2.0})
    memory_cfg = config.get('memory', {'type': 'uniform', 'min': 1.0, 'max': 4.0})
    gpu_cfg = config.get('gpu', {'type': 'choice', 'values': [0, 0, 1]})
    gang_prob = config.get('gang_prob', 0.2)
    gang_size_cfg = config.get('gang_size', {'type': 'uniform', 'min': 2, 'max': 5})

    events = []
    current_time = 0.0
    pod_count = 0
    gang_counter = 0

    for _ in range(num_pods):
        delta = sample_distribution(inter_arrival_cfg, rng)
        current_time += delta

        cpu = sample_distribution(cpu_cfg, rng)
        memory = sample_distribution(memory_cfg, rng)
        gpu = int(sample_distribution(gpu_cfg, rng))

        if rng.random() < gang_prob:
            size = max(1, int(sample_distribution(gang_size_cfg, rng)))
            gang_id = f"gang-{gang_counter}"
            gang_counter += 1
            gang_pods = []
            for _ in range(size):
                uid = f"pod-{pod_count}"
                pod_count += 1
                cpu_pod = sample_distribution(cpu_cfg, rng)
                memory_pod = sample_distribution(memory_cfg, rng)
                gpu_pod = int(sample_distribution(gpu_cfg, rng))
                pod = Pod(
                    uid=uid,
                    name=uid,
                    resources=ResourceRequest(cpu=cpu_pod, memory=memory_pod, gpu=gpu_pod),
                    gang_id=gang_id,
                )
                gang_pods.append(pod)
            events.append((current_time, gang_pods))
        else:
            uid = f"pod-{pod_count}"
            pod_count += 1
            pod = Pod(
                uid=uid,
                name=uid,
                resources=ResourceRequest(cpu=cpu, memory=memory, gpu=gpu),
                gang_id=None,
            )
            events.append((current_time, pod))

    return events


def sample_distribution(dist_cfg: Dict[str, Any], rng: random.Random) -> Union[float, int]:
    """Sample from a distribution (same as in experiments.py)."""
    dtype = dist_cfg.get('type', 'constant')
    if dtype == 'constant':
        return dist_cfg.get('value', 0.0)
    elif dtype == 'uniform':
        return rng.uniform(dist_cfg.get('min', 0.0), dist_cfg.get('max', 1.0))
    elif dtype == 'exponential':
        mean = dist_cfg.get('mean', 1.0)
        return rng.expovariate(1.0 / mean) if mean > 0 else 0.0
    elif dtype == 'normal':
        return rng.gauss(dist_cfg.get('mean', 0.0), dist_cfg.get('std', 1.0))
    elif dtype == 'choice':
        values = dist_cfg.get('values', [0])
        weights = dist_cfg.get('weights', None)
        if weights is None:
            return rng.choice(values)
        else:
            return rng.choices(values, weights=weights)[0]
    else:
        raise ValueError(f"Unknown distribution type: {dtype}")


# -------------------------------------------------------------------------
# Core simulation runner
# -------------------------------------------------------------------------

def run_simulation(cluster: Cluster,
                   workload_events: List[tuple],
                   scheduler_params: Dict[str, Any],
                   output_dir: str,
                   seed: Optional[int] = None,
                   plot: bool = False) -> SimulationStats:
    """
    Run a simulation with the given components and save results.

    Args:
        cluster: Cluster object.
        workload_events: List of (time, pod_or_gang) events.
        scheduler_params: Dictionary with policy, until, log_interval, runtime_function.
        output_dir: Directory to save outputs.
        seed: Random seed (overrides scheduler_params['seed']).
        plot: Whether to generate plots.

    Returns:
        SimulationStats object.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Determine seed
    if seed is not None:
        final_seed = seed
    else:
        final_seed = scheduler_params.get('seed', 42)

    # Policy
    policy = scheduler_params.get('policy', 'bestfit')

    # Scheduler
    scheduler = Scheduler(cluster, policy_name=policy, seed=final_seed)

    # Runtime function
    runtime_type = scheduler_params.get('runtime_function', 'default')
    if runtime_type == 'resource_based':
        runtime_func = resource_based_runtime
    else:
        runtime_func = default_runtime_func

    # Simulation parameters
    until = scheduler_params.get('until', 10000.0)
    log_interval = scheduler_params.get('log_interval', None)

    # Simulator
    sim = Simulator(
        cluster=cluster,
        scheduler=scheduler,
        workload_events=workload_events,
        runtime_func=runtime_func,
        seed=final_seed,
        log_interval=log_interval,
    )

    # Run
    sim.run(until=until)

    # Collect stats
    stats = sim.get_stats()

    # Save results
    prefix = f"{policy}"
    csv_file = out_path / f"{prefix}_stats.csv"
    json_file = out_path / f"{prefix}_stats.json"

    # CSV summary
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['metric', 'value'])
        writer.writerow(['total_pods_submitted', stats.total_pods_submitted])
        writer.writerow(['total_pods_scheduled', stats.total_pods_scheduled])
        writer.writerow(['total_pods_completed', stats.total_pods_completed])
        writer.writerow(['total_pods_failed', stats.total_pods_failed])
        avg_wait = sum(stats.wait_times)/len(stats.wait_times) if stats.wait_times else 0
        avg_run = sum(stats.run_times)/len(stats.run_times) if stats.run_times else 0
        avg_turn = sum(stats.turnaround_times)/len(stats.turnaround_times) if stats.turnaround_times else 0
        writer.writerow(['avg_wait_time', avg_wait])
        writer.writerow(['avg_run_time', avg_run])
        writer.writerow(['avg_turnaround_time', avg_turn])
        if stats.wait_times:
            sorted_wait = sorted(stats.wait_times)
            writer.writerow(['p50_wait_time', sorted_wait[int(len(sorted_wait)*0.5)]])
            writer.writerow(['p90_wait_time', sorted_wait[int(len(sorted_wait)*0.9)]])
            writer.writerow(['p99_wait_time', sorted_wait[int(len(sorted_wait)*0.99)]])

    # JSON full stats
    stats_dict = {
        'total_pods_submitted': stats.total_pods_submitted,
        'total_pods_scheduled': stats.total_pods_scheduled,
        'total_pods_completed': stats.total_pods_completed,
        'total_pods_failed': stats.total_pods_failed,
        'wait_times': stats.wait_times,
        'run_times': stats.run_times,
        'turnaround_times': stats.turnaround_times,
        'avg_wait_time': avg_wait,
        'avg_run_time': avg_run,
        'avg_turnaround_time': avg_turn,
        'final_utilization': cluster.get_cluster_utilization(),
    }
    with open(json_file, 'w') as f:
        json.dump(stats_dict, f, indent=2)

    # Optional plots
    if plot:
        try:
            import matplotlib.pyplot as plt
            if stats.wait_times:
                plt.figure()
                sorted_wait = sorted(stats.wait_times)
                cdf = [i/len(sorted_wait) for i in range(len(sorted_wait))]
                plt.plot(sorted_wait, cdf, label='Wait Time CDF')
                plt.xlabel('Wait Time')
                plt.ylabel('CDF')
                plt.title(f'Wait Time CDF - Policy: {policy}')
                plt.grid(True)
                plt.savefig(out_path / f"{prefix}_wait_cdf.png")
                plt.close()
        except ImportError:
            print("matplotlib not installed; skipping plots.")

    return stats


# -------------------------------------------------------------------------
# CLI subcommand implementations
# -------------------------------------------------------------------------

def cmd_run(args: argparse.Namespace) -> None:
    """
    'run' subcommand: run a single experiment from a unified YAML config.
    """
    config_file = args.config
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)

    # Extract components from the unified config
    cluster_cfg = config.get('cluster')
    if not cluster_cfg:
        raise ValueError("No 'cluster' section found in config.")
    # Build cluster using the same builder as experiments.py (we'll replicate)
    cluster = build_cluster_from_dict(cluster_cfg)

    workload_cfg = config.get('workload', {})
    rng = random.Random(args.seed if args.seed is not None else 42)
    workload_events = build_workload_from_config(workload_cfg, rng)

    scheduler_cfg = config.get('simulation', {})
    # Override seed from args
    if args.seed is not None:
        scheduler_cfg['seed'] = args.seed

    output_dir = args.output or 'results'
    run_simulation(
        cluster=cluster,
        workload_events=workload_events,
        scheduler_params=scheduler_cfg,
        output_dir=output_dir,
        seed=args.seed,
        plot=args.plot,
    )


def cmd_apply(args: argparse.Namespace) -> None:
    """
    'apply' subcommand: run simulation with separate cluster, scheduler, workload YAMLs.
    """
    cluster = load_cluster_from_yaml(args.cluster)
    scheduler_params = load_scheduler_from_yaml(args.scheduler)
    workload_cfg = load_workload_from_yaml(args.workload)

    # Override seed from args
    if args.seed is not None:
        scheduler_params['seed'] = args.seed

    rng = random.Random(args.seed if args.seed is not None else scheduler_params.get('seed', 42))
    workload_events = build_workload_from_config(workload_cfg, rng)

    output_dir = args.output or 'results'
    run_simulation(
        cluster=cluster,
        workload_events=workload_events,
        scheduler_params=scheduler_params,
        output_dir=output_dir,
        seed=args.seed,
        plot=args.plot,
    )


def build_cluster_from_dict(cluster_cfg: Dict[str, Any]) -> Cluster:
    """Helper to build Cluster from a dict (used by cmd_run)."""
    nodes = []
    for node_cfg in cluster_cfg.get('nodes', []):
        cap = node_cfg['capacity']
        res = ResourceRequest(
            cpu=float(cap.get('cpu', 0.0)),
            memory=float(cap.get('memory', 0.0)),
            gpu=int(cap.get('gpu', 0))
        )
        node = Node(name=node_cfg['name'], capacity=res)
        nodes.append(node)
    return Cluster(nodes)


# -------------------------------------------------------------------------
# Main parser
# -------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Kubernetes Scheduler Simulator CLI",
        usage="main.py <command> [options]"
    )
    subparsers = parser.add_subparsers(dest='command', required=True, help='Subcommands')

    # ---------- run subcommand ----------
    parser_run = subparsers.add_parser('run', help='Run experiment from a single YAML config')
    parser_run.add_argument('--config', '-c', required=True,
                           help='Path to the experiment YAML file')
    parser_run.add_argument('--output', '-o', default='results',
                           help='Output directory (default: results/)')
    parser_run.add_argument('--seed', type=int, default=None,
                           help='Random seed (overrides config)')
    parser_run.add_argument('--plot', '-p', action='store_true',
                           help='Generate plots (requires matplotlib)')
    parser_run.set_defaults(func=cmd_run)

    # ---------- apply subcommand ----------
    parser_apply = subparsers.add_parser('apply', help='Apply simulation with separate configs')
    parser_apply.add_argument('--cluster', '-f', required=True,
                             help='Path to cluster YAML configuration')
    parser_apply.add_argument('--scheduler', '-s', required=True,
                             help='Path to scheduler YAML configuration')
    parser_apply.add_argument('--workload', '-w', required=True,
                             help='Path to workload YAML configuration')
    parser_apply.add_argument('--output', '-o', default='results',
                             help='Output directory (default: results/)')
    parser_apply.add_argument('--seed', type=int, default=None,
                             help='Random seed (overrides scheduler config)')
    parser_apply.add_argument('--plot', '-p', action='store_true',
                             help='Generate plots (requires matplotlib)')
    parser_apply.set_defaults(func=cmd_apply)
    parser_apply.add_argument('--scheduler-config', '-s', required=True,
                         help='Path to scheduler YAML (KubeSchedulerConfiguration)')
    
    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()