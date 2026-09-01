#!/usr/bin/env python3
"""
experiments.py

Orchestration script for running Kubernetes scheduler simulations.

This script loads configuration from a YAML file, builds the cluster and
workload, runs the simulation, and outputs statistics (CSV, JSON) and
optionally generates plots.

Usage:
    python experiments.py --config configs/experiment.yaml [--output results/] [--seed 42] [--plot]

Configuration file format (YAML):
    cluster:
      nodes:
        - name: node-1
          capacity:
            cpu: 4.0
            memory: 8.0
            gpu: 1
        - name: node-2
          capacity:
            cpu: 8.0
            memory: 16.0
            gpu: 0

    workload:
      type: trace          # or 'synthetic'
      file: workloads/trace.csv   # for 'trace'
      synthetic:           # for 'synthetic'
        num_pods: 100
        inter_arrival: exponential(mean=5.0)
        runtime: exponential(mean=60.0)
        cpu: uniform(0.5, 2.0)
        memory: uniform(1.0, 4.0)
        gpu: choice([0, 0, 1])   # 1/3 chance of GPU
        gang_prob: 0.2
        gang_size: uniform(2, 5)

    simulation:
      policy: bestfit      # bestfit, spread, gpupack, fgd, random
      until: 10000.0       # simulation horizon
      log_interval: 100.0  # progress logging interval

    output:
      csv: results_$policy.csv
      json: results_$policy.json
      plots:
        - wait_time_cdf
        - utilization_over_time

Command-line arguments override config options where appropriate.
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
from typing import List, Dict, Any, Optional, Tuple, Union
from collections import defaultdict
import math

# Import the simulation package
from sim import (
    Cluster, Node, Pod, ResourceRequest,
    Scheduler, Simulator, SimulationStats,
    default_runtime_func, resource_based_runtime,
)


# -------------------------------------------------------------------------
# Configuration Loading
# -------------------------------------------------------------------------

def load_config(config_file: str) -> Dict[str, Any]:
    """
    Load and parse a YAML configuration file.

    Args:
        config_file: Path to the YAML file.

    Returns:
        Dictionary with configuration.
    """
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    return config


def build_cluster_from_config(cluster_config: Dict[str, Any]) -> Cluster:
    """
    Build a Cluster object from the configuration.

    Expected cluster_config format:
        nodes:
          - name: node-1
            capacity:
              cpu: 4.0
              memory: 8.0
              gpu: 1
          ...

    Args:
        cluster_config: Dictionary with 'nodes' list.

    Returns:
        Cluster instance.
    """
    nodes = []
    for node_cfg in cluster_config['nodes']:
        cap = node_cfg['capacity']
        res = ResourceRequest(
            cpu=float(cap.get('cpu', 0.0)),
            memory=float(cap.get('memory', 0.0)),
            gpu=int(cap.get('gpu', 0))
        )
        node = Node(name=node_cfg['name'], capacity=res)
        nodes.append(node)
    return Cluster(nodes)


def build_workload_from_config(workload_config: Dict[str, Any],
                               rng: random.Random) -> List[Tuple[float, Union[Pod, List[Pod]]]]:
    """
    Build workload events from configuration.

    Supports:
        - 'trace': load from a CSV file with columns: time, uid, cpu, memory, gpu, gang_id(optional)
        - 'synthetic': generate workload according to specified distributions.

    Args:
        workload_config: Dictionary with workload settings.
        rng: Random number generator.

    Returns:
        List of (submit_time, pod_or_gang) events.
    """
    wtype = workload_config.get('type', 'synthetic')

    if wtype == 'trace':
        trace_file = workload_config.get('file')
        if not trace_file:
            raise ValueError("Trace file not specified in workload config.")
        events = load_trace_from_csv(trace_file)
        # Convert to events list (already in (time, Pod) or (time, list of Pods))
        # The trace loader can return the same format.
        return events

    elif wtype == 'synthetic':
        return generate_synthetic_workload(workload_config, rng)

    else:
        raise ValueError(f"Unknown workload type: {wtype}")


def load_trace_from_csv(csv_file: str) -> List[Tuple[float, Union[Pod, List[Pod]]]]:
    """
    Load workload trace from a CSV file.

    Expected columns:
        time, uid, cpu, memory, gpu, gang_id(optional)

    If gang_id is present, pods with the same gang_id at the same time
    will be grouped into a gang.

    Args:
        csv_file: Path to CSV file.

    Returns:
        List of (time, pod_or_gang) events.
    """
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

    # Group by (time, gang_id) to form gangs
    events = []
    for t, items in events_by_time.items():
        # Separate gang and non-gang
        gangs: Dict[str, List[Pod]] = defaultdict(list)
        non_gang: List[Pod] = []
        for gang_id, pod in items:
            if gang_id:
                gangs[gang_id].append(pod)
            else:
                non_gang.append(pod)
        # Add gangs
        for gang_id, gang_pods in gangs.items():
            events.append((t, gang_pods))
        # Add non-gang pods
        for pod in non_gang:
            events.append((t, pod))

    return sorted(events, key=lambda x: x[0])


def generate_synthetic_workload(config: Dict[str, Any],
                                rng: random.Random) -> List[Tuple[float, Union[Pod, List[Pod]]]]:
    """
    Generate a synthetic workload based on configuration.

    Expected config keys:
        num_pods: int
        inter_arrival: dict with distribution type and parameters
        runtime: dict with distribution (only used for completion, not for workload)
        cpu: distribution for CPU request
        memory: distribution for memory request
        gpu: distribution for GPU count
        gang_prob: float (0-1) probability a pod belongs to a gang
        gang_size: distribution for gang size (number of pods in a gang)

    Distribution formats:
        - "constant": {"type": "constant", "value": 5.0}
        - "uniform":  {"type": "uniform", "min": 1.0, "max": 4.0}
        - "exponential": {"type": "exponential", "mean": 60.0}
        - "choice":    {"type": "choice", "values": [1, 2, 3], "weights": [0.5, 0.3, 0.2]} (optional)
        - "normal":    {"type": "normal", "mean": 2.0, "std": 0.5}

    We only use inter_arrival, cpu, memory, gpu, gang_prob, gang_size for generation.
    runtime is handled by the simulator's runtime function.

    Returns:
        List of (submit_time, pod_or_gang) events.
    """
    num_pods = config.get('num_pods', 100)
    inter_arrival_cfg = config.get('inter_arrival', {'type': 'exponential', 'mean': 5.0})
    cpu_cfg = config.get('cpu', {'type': 'uniform', 'min': 0.5, 'max': 2.0})
    memory_cfg = config.get('memory', {'type': 'uniform', 'min': 1.0, 'max': 4.0})
    gpu_cfg = config.get('gpu', {'type': 'choice', 'values': [0, 0, 1]})  # default 1/3 GPU
    gang_prob = config.get('gang_prob', 0.2)
    gang_size_cfg = config.get('gang_size', {'type': 'uniform', 'min': 2, 'max': 5})

    events = []
    current_time = 0.0
    pod_count = 0
    gang_counter = 0

    for i in range(num_pods):
        # Inter-arrival
        delta = sample_distribution(inter_arrival_cfg, rng)
        current_time += delta

        # Resource requests
        cpu = sample_distribution(cpu_cfg, rng)
        memory = sample_distribution(memory_cfg, rng)
        gpu = int(sample_distribution(gpu_cfg, rng))  # GPU is integer

        # Gang or not?
        if rng.random() < gang_prob:
            # Create a gang
            size = int(sample_distribution(gang_size_cfg, rng))
            # Ensure at least 1
            size = max(1, size)
            gang_id = f"gang-{gang_counter}"
            gang_counter += 1
            gang_pods = []
            for j in range(size):
                uid = f"pod-{pod_count}"
                pod_count += 1
                # Each pod in gang can have different resource requests? The original FGD sums them.
                # We'll use same distribution for each.
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
            # Single pod
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
    """
    Sample a value from a distribution described by a configuration dictionary.

    Supported distributions:
        - constant: value
        - uniform: min, max
        - exponential: mean (rate = 1/mean)
        - normal: mean, std
        - choice: values, weights (optional, if weights not provided, uniform)

    Args:
        dist_cfg: Dictionary with 'type' and parameters.
        rng: Random generator.

    Returns:
        Sampled value (float or int).
    """
    dtype = dist_cfg.get('type', 'constant')

    if dtype == 'constant':
        return dist_cfg.get('value', 0.0)

    elif dtype == 'uniform':
        min_val = dist_cfg.get('min', 0.0)
        max_val = dist_cfg.get('max', 1.0)
        return rng.uniform(min_val, max_val)

    elif dtype == 'exponential':
        mean = dist_cfg.get('mean', 1.0)
        # rate = 1/mean
        return rng.expovariate(1.0 / mean) if mean > 0 else 0.0

    elif dtype == 'normal':
        mean = dist_cfg.get('mean', 0.0)
        std = dist_cfg.get('std', 1.0)
        return rng.gauss(mean, std)

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
# Experiment Runner
# -------------------------------------------------------------------------

def run_experiment(config: Dict[str, Any], output_dir: str,
                   seed: Optional[int] = None, plot: bool = False) -> SimulationStats:
    """
    Run a single simulation experiment based on the configuration.

    Args:
        config: Full configuration dictionary.
        output_dir: Directory to save outputs.
        seed: Random seed (overrides config['simulation']['seed'] if present).
        plot: Whether to generate plots.

    Returns:
        SimulationStats object.
    """
    # Set up output directory
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Determine seed
    if seed is None:
        seed = config.get('simulation', {}).get('seed', 42)
    rng = random.Random(seed)

    # Build cluster
    cluster_cfg = config['cluster']
    cluster = build_cluster_from_config(cluster_cfg)

    # Build workload
    workload_cfg = config.get('workload', {})
    workload_events = build_workload_from_config(workload_cfg, rng)

    # Scheduler policy
    policy = config.get('simulation', {}).get('policy', 'bestfit')
    scheduler = Scheduler(cluster, policy_name=policy, seed=seed)

    # Runtime function: from config or default
    runtime_type = config.get('simulation', {}).get('runtime_function', 'default')
    if runtime_type == 'default':
        runtime_func = default_runtime_func
    elif runtime_type == 'resource_based':
        runtime_func = resource_based_runtime
    else:
        runtime_func = default_runtime_func  # fallback

    # Simulation parameters
    until = config.get('simulation', {}).get('until', 10000.0)
    log_interval = config.get('simulation', {}).get('log_interval', None)

    # Create simulator
    sim = Simulator(
        cluster=cluster,
        scheduler=scheduler,
        workload_events=workload_events,
        runtime_func=runtime_func,
        seed=seed,
        log_interval=log_interval,
    )

    # Run
    sim.run(until=until)

    # Collect stats
    stats = sim.get_stats()

    # Save results
    output_prefix = f"{policy}"
    csv_file = out_path / f"{output_prefix}_stats.csv"
    json_file = out_path / f"{output_prefix}_stats.json"

    # Save to CSV
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['metric', 'value'])
        writer.writerow(['total_pods_submitted', stats.total_pods_submitted])
        writer.writerow(['total_pods_scheduled', stats.total_pods_scheduled])
        writer.writerow(['total_pods_completed', stats.total_pods_completed])
        writer.writerow(['total_pods_failed', stats.total_pods_failed])
        writer.writerow(['avg_wait_time', sum(stats.wait_times)/len(stats.wait_times) if stats.wait_times else 0])
        writer.writerow(['avg_run_time', sum(stats.run_times)/len(stats.run_times) if stats.run_times else 0])
        writer.writerow(['avg_turnaround_time', sum(stats.turnaround_times)/len(stats.turnaround_times) if stats.turnaround_times else 0])
        writer.writerow(['min_wait_time', min(stats.wait_times) if stats.wait_times else 0])
        writer.writerow(['max_wait_time', max(stats.wait_times) if stats.wait_times else 0])
        # Percentiles
        if stats.wait_times:
            sorted_wait = sorted(stats.wait_times)
            writer.writerow(['p50_wait_time', sorted_wait[int(len(sorted_wait)*0.5)]])
            writer.writerow(['p90_wait_time', sorted_wait[int(len(sorted_wait)*0.9)]])
            writer.writerow(['p99_wait_time', sorted_wait[int(len(sorted_wait)*0.99)]])

    # Save to JSON
    stats_dict = {
        'total_pods_submitted': stats.total_pods_submitted,
        'total_pods_scheduled': stats.total_pods_scheduled,
        'total_pods_completed': stats.total_pods_completed,
        'total_pods_failed': stats.total_pods_failed,
        'wait_times': stats.wait_times,
        'run_times': stats.run_times,
        'turnaround_times': stats.turnaround_times,
        'avg_wait_time': sum(stats.wait_times)/len(stats.wait_times) if stats.wait_times else 0,
        'avg_run_time': sum(stats.run_times)/len(stats.run_times) if stats.run_times else 0,
        'avg_turnaround_time': sum(stats.turnaround_times)/len(stats.turnaround_times) if stats.turnaround_times else 0,
        'final_utilization': cluster.get_cluster_utilization(),
    }
    with open(json_file, 'w') as f:
        json.dump(stats_dict, f, indent=2)

    # Optionally generate plots
    if plot:
        try:
            import matplotlib.pyplot as plt
            # Plot CDF of wait times
            if stats.wait_times:
                plt.figure()
                sorted_wait = sorted(stats.wait_times)
                cdf = [i/len(sorted_wait) for i in range(len(sorted_wait))]
                plt.plot(sorted_wait, cdf, label='Wait Time CDF')
                plt.xlabel('Wait Time')
                plt.ylabel('CDF')
                plt.title(f'Wait Time CDF - Policy: {policy}')
                plt.grid(True)
                plt.savefig(out_path / f"{output_prefix}_wait_cdf.png")
                plt.close()

            # Plot utilization over time (if recorded)
            # For simplicity, we can sample utilization periodically during simulation.
            # We'll add a periodic sampling mechanism if needed.
            # For now, we just print a note.
        except ImportError:
            print("matplotlib not installed; skipping plots.")

    return stats


# -------------------------------------------------------------------------
# Main Entry Point
# -------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Run Kubernetes scheduler simulation experiments."
    )
    parser.add_argument(
        '--config', '-c', required=True,
        help='Path to YAML configuration file.'
    )
    parser.add_argument(
        '--output', '-o', default='results',
        help='Output directory for results (default: results/).'
    )
    parser.add_argument(
        '--seed', type=int, default=None,
        help='Random seed (overrides config).'
    )
    parser.add_argument(
        '--plot', action='store_true',
        help='Generate plots (requires matplotlib).'
    )
    parser.add_argument(
        '--verbose', '-v', action='store_true',
        help='Enable verbose logging.'
    )
    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Load config
    config = load_config(args.config)

    # Run experiment
    stats = run_experiment(
        config=config,
        output_dir=args.output,
        seed=args.seed,
        plot=args.plot,
    )

    print(f"\nExperiment completed. Results saved to {args.output}")
    print(f"Total pods completed: {stats.total_pods_completed}")
    print(f"Average wait time: {sum(stats.wait_times)/len(stats.wait_times) if stats.wait_times else 0:.2f}")
    print(f"Average turnaround time: {sum(stats.turnaround_times)/len(stats.turnaround_times) if stats.turnaround_times else 0:.2f}")


if __name__ == '__main__':
    main()