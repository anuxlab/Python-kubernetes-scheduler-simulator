#!/usr/bin/env python3
"""
data/prepare_trace.py - Data preparation pipeline for raw CSV traces.

This script converts raw cluster and workload CSV files into YAML configuration
files that can be used by the simulator. It supports:

  1. Node list CSV: columns -> node_name, cpu_capacity, memory_capacity, gpu_capacity
  2. Pod list CSV: columns -> submit_time, pod_uid, cpu_request, memory_request,
                                gpu_request, gang_id (optional)

The output can be a unified experiment YAML (for `main.py run`) or separate
cluster, scheduler, and workload YAMLs (for `main.py apply`).

Usage:
    # Generate a unified YAML for `main.py run`
    python prepare_trace.py --nodes nodes.csv --pods pods.csv --output experiment.yaml

    # Generate separate configs for `main.py apply`
    python prepare_trace.py --nodes nodes.csv --pods pods.csv --output-dir configs/
                             --separate

    # Also generate a scheduler configuration with a given policy
    python prepare_trace.py --nodes nodes.csv --pods pods.csv --output configs/ \\
                             --policy fgd --until 10000

Example CSV formats:

nodes.csv:
    node_name,cpu_capacity,memory_capacity,gpu_capacity
    node-1,8.0,16.0,1
    node-2,8.0,16.0,0

pods.csv:
    submit_time,pod_uid,cpu_request,memory_request,gpu_request,gang_id
    0.0,pod-1,1.0,2.0,0,
    2.5,pod-2,0.5,1.0,1,gang-A
    5.0,pod-3,2.0,4.0,0,gang-A
"""

import sys
import os
import csv
import yaml
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional
from collections import defaultdict


# -------------------------------------------------------------------------
# CSV Parsers
# -------------------------------------------------------------------------

def parse_node_csv(csv_file: str) -> List[Dict[str, Any]]:
    """
    Parse a CSV file containing node specifications.

    Expected columns:
        node_name, cpu_capacity, memory_capacity, gpu_capacity

    Returns:
        List of node dictionaries.
    """
    nodes = []
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            node = {
                'name': row.get('node_name', row.get('name')),
                'capacity': {
                    'cpu': float(row.get('cpu_capacity', row.get('cpu', 0.0))),
                    'memory': float(row.get('memory_capacity', row.get('memory', 0.0))),
                    'gpu': int(row.get('gpu_capacity', row.get('gpu', 0))),
                }
            }
            nodes.append(node)
    return nodes


def parse_pod_csv(csv_file: str) -> List[Dict[str, Any]]:
    """
    Parse a CSV file containing pod specifications.

    Expected columns:
        submit_time, pod_uid, cpu_request, memory_request, gpu_request, gang_id (optional)

    Returns:
        List of pod dictionaries.
    """
    pods = []
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            pod = {
                'time': float(row.get('submit_time', row.get('time', 0.0))),
                'pod': {
                    'uid': row.get('pod_uid', row.get('uid', 'pod-unknown')),
                    'name': row.get('pod_uid', row.get('uid', 'pod-unknown')),
                    'resources': {
                        'cpu': float(row.get('cpu_request', row.get('cpu', 0.0))),
                        'memory': float(row.get('memory_request', row.get('memory', 0.0))),
                        'gpu': int(row.get('gpu_request', row.get('gpu', 0))),
                    }
                }
            }
            # Optional gang_id
            gang_id = row.get('gang_id', None)
            if gang_id and gang_id.strip():
                pod['pod']['gang_id'] = gang_id.strip()
            pods.append(pod)
    return pods


# -------------------------------------------------------------------------
# YAML Generators
# -------------------------------------------------------------------------

def generate_unified_experiment_yaml(nodes: List[Dict[str, Any]],
                                     pods: List[Dict[str, Any]],
                                     policy: str = 'bestfit',
                                     until: float = 10000.0,
                                     seed: int = 42,
                                     runtime: str = 'default',
                                     log_interval: Optional[float] = None) -> Dict[str, Any]:
    """
    Build a unified experiment configuration dictionary.

    Returns:
        Dictionary suitable for YAML output.
    """
    # Cluster section
    cluster_cfg = {'nodes': nodes}

    # Workload section (trace type)
    workload_cfg = {
        'type': 'trace',
        'file': 'pods.csv'  # This is just a placeholder; the actual trace is embedded in the YAML
        # But for a full self-contained YAML, we could embed the pods directly.
        # However, we will generate separate workload YAML for trace type that references the CSV.
        # For simplicity in this pipeline, we generate a trace-based workload that points to the CSV.
        # But since we have the data, we can either:
        #   a) embed the events as a list in YAML (large)
        #   b) generate separate workload YAML with file reference.
        # We'll choose option b: we'll output separate files if needed.
        # For unified, we can just output a workload section with type trace and file path.
        # The user must ensure the file path is correct.
    }

    # However, the unified YAML expects a workload section. We'll generate one that references
    # the CSV file. The path will be relative to the YAML location? We'll let the user specify.
    # We'll add an option to embed events directly, but for large traces it's not recommended.
    # So for now, we'll generate separate YAMLs for cluster, scheduler, and workload.

    # So we'll create a function to generate separate YAMLs, which is more flexible.
    # The unified mode is less useful for large traces; we'll skip it in this script
    # and focus on generating the three separate files.

    # Return a dict with cluster, scheduler, workload sections (for unified) but we will
    # not implement embedding of the actual pod events.

    # Actually, the main.py 'run' command expects a single YAML with cluster, workload, simulation.
    # For trace workload, we need a workload section that points to a CSV file.
    # We can generate that YAML referencing the CSV file path.

    # So we'll support both: if --output is a file, we generate a unified YAML with
    # workload.file pointing to the CSV. We'll also generate a separate scheduler section.
    return {
        'cluster': cluster_cfg,
        'workload': {
            'type': 'trace',
            'file': 'pods.csv'  # This will be overridden by the actual path
        },
        'simulation': {
            'policy': policy,
            'until': until,
            'seed': seed,
            'log_interval': log_interval,
            'runtime_function': runtime,
        }
    }


def generate_cluster_yaml(nodes: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Generate a cluster YAML dictionary."""
    return {'nodes': nodes}


def generate_scheduler_yaml(policy: str = 'bestfit',
                            until: float = 10000.0,
                            seed: int = 42,
                            log_interval: Optional[float] = None,
                            runtime: str = 'default') -> Dict[str, Any]:
    """Generate a scheduler YAML dictionary."""
    cfg = {
        'policy': policy,
        'until': until,
        'seed': seed,
        'runtime_function': runtime,
    }
    if log_interval is not None:
        cfg['log_interval'] = log_interval
    return cfg


def generate_workload_yaml_from_csv(csv_file: str) -> Dict[str, Any]:
    """Generate a workload YAML that references a CSV trace file."""
    return {
        'type': 'trace',
        'file': csv_file,
    }


# -------------------------------------------------------------------------
# Main Pipeline
# -------------------------------------------------------------------------

def run_pipeline(args):
    """
    Execute the data preparation pipeline.
    """
    # Parse input files
    nodes = parse_node_csv(args.nodes)
    pods = parse_pod_csv(args.pods)

    print(f"Loaded {len(nodes)} nodes and {len(pods)} pods from CSV.")

    if args.output_dir:
        # Generate separate YAMLs for `main.py apply`
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        # Cluster YAML
        cluster_cfg = generate_cluster_yaml(nodes)
        cluster_file = out_dir / 'cluster.yaml'
        with open(cluster_file, 'w') as f:
            yaml.dump(cluster_cfg, f, default_flow_style=False, sort_keys=False)
        print(f"Cluster config written to: {cluster_file}")

        # Scheduler YAML
        scheduler_cfg = generate_scheduler_yaml(
            policy=args.policy,
            until=args.until,
            seed=args.seed,
            log_interval=args.log_interval,
            runtime=args.runtime
        )
        scheduler_file = out_dir / 'scheduler.yaml'
        with open(scheduler_file, 'w') as f:
            yaml.dump(scheduler_cfg, f, default_flow_style=False, sort_keys=False)
        print(f"Scheduler config written to: {scheduler_file}")

        # Workload YAML (references the CSV file)
        # We copy or link the pods CSV to the output dir for portability
        # We'll just reference the original path
        workload_cfg = generate_workload_yaml_from_csv(args.pods)
        workload_file = out_dir / 'workload.yaml'
        with open(workload_file, 'w') as f:
            yaml.dump(workload_cfg, f, default_flow_style=False, sort_keys=False)
        print(f"Workload config written to: {workload_file}")

        print(f"\nNow you can run: python main.py apply --cluster {cluster_file} "
              f"--scheduler {scheduler_file} --workload {workload_file} --output results/")

    elif args.output:
        # Generate a unified experiment YAML for `main.py run`
        # We need to output a single YAML that contains cluster, workload (referencing CSV), simulation.
        # The workload section should have 'file' pointing to the CSV.
        # We'll use the original CSV path.
        unified_cfg = {
            'cluster': generate_cluster_yaml(nodes)['nodes'],
            'workload': generate_workload_yaml_from_csv(args.pods),
            'simulation': generate_scheduler_yaml(
                policy=args.policy,
                until=args.until,
                seed=args.seed,
                log_interval=args.log_interval,
                runtime=args.runtime
            )
        }
        # But we need to ensure the structure is correct: cluster.nodes, workload, simulation.
        # We'll reconstruct properly.
        unified_cfg = {
            'cluster': {'nodes': nodes},
            'workload': {'type': 'trace', 'file': args.pods},
            'simulation': generate_scheduler_yaml(
                policy=args.policy,
                until=args.until,
                seed=args.seed,
                log_interval=args.log_interval,
                runtime=args.runtime
            )
        }
        output_file = Path(args.output)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w') as f:
            yaml.dump(unified_cfg, f, default_flow_style=False, sort_keys=False)
        print(f"Unified experiment config written to: {output_file}")
        print(f"\nNow you can run: python main.py run --config {output_file} --output results/")

    else:
        print("Error: either --output or --output-dir must be specified.")
        sys.exit(1)


# -------------------------------------------------------------------------
# CLI
# -------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Prepare YAML configs from raw CSV traces.",
        epilog="Example: python prepare_trace.py --nodes nodes.csv --pods pods.csv --output-dir configs/"
    )
    parser.add_argument('--nodes', required=True,
                        help='CSV file with node specifications.')
    parser.add_argument('--pods', required=True,
                        help='CSV file with pod specifications (trace).')
    parser.add_argument('--output-dir', '-d',
                        help='Directory to write separate cluster, scheduler, workload YAMLs.')
    parser.add_argument('--output', '-o',
                        help='Single unified YAML file for `main.py run`.')
    parser.add_argument('--policy', '-p', default='bestfit',
                        help='Scheduling policy (default: bestfit).')
    parser.add_argument('--until', type=float, default=10000.0,
                        help='Simulation horizon (default: 10000).')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42).')
    parser.add_argument('--log-interval', type=float, default=None,
                        help='Progress logging interval (optional).')
    parser.add_argument('--runtime', '-r', default='default',
                        choices=['default', 'resource_based'],
                        help='Runtime function (default: default).')

    args = parser.parse_args()
    run_pipeline(args)


if __name__ == '__main__':
    main()