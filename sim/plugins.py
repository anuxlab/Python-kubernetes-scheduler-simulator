"""
sim/plugins.py – Implementation of scoring plugins.

Each plugin provides a `Score` function that returns a float for a given
(pod, node) combination. The final node is chosen by summing weighted scores
across all enabled plugins.

Plugins available:
    - FGDScore: (gang only) returns 1.0 if the node can fit the entire gang,
      else 0.0. This is used as a filter.
    - BestFitScore: favours nodes with the smallest remaining capacity.
    - SpreadScore: favours nodes with the fewest running pods.
    - GpuPackingScore: favours nodes already using GPUs when the pod requests GPUs.
    - RandomScore: returns a random score (for random selection).
    - DotProductScore: (HKUST style) scores based on resource request and node free resources
      using dot product, with optional normalisation.
"""

import math
import random
from typing import List, Optional, Dict, Any, Callable
from .cluster import Cluster
from .pod import Pod, ResourceRequest
from .node import Node


# -------------------------------------------------------------------------
# Base plugin interface
# -------------------------------------------------------------------------

class ScoringPlugin:
    """Base class for all scoring plugins."""
    name: str = "base"

    def __init__(self, args: Optional[Dict[str, Any]] = None):
        self.args = args or {}

    def Score(self, pod: Pod, node: Node, cluster: Cluster) -> float:
        """
        Compute a score for placing the pod on the given node.

        Higher scores are better. The score can be any float, but it is
        recommended to keep it in a reasonable range (e.g., 0-100).
        """
        raise NotImplementedError


# -------------------------------------------------------------------------
# Concrete plugins
# -------------------------------------------------------------------------

class BestFitScore(ScoringPlugin):
    """
    Best‑fit: score is inversely proportional to the remaining capacity
    after placing the pod. The node with the smallest remaining capacity
    (tightest fit) gets the highest score.
    """
    name = "BestFitScore"

    def Score(self, pod: Pod, node: Node, cluster: Cluster) -> float:
        if not node.can_fit(pod.resources):
            return 0.0

        remaining_cpu = node.capacity.cpu - node.allocated.cpu - pod.resources.cpu
        remaining_mem = node.capacity.memory - node.allocated.memory - pod.resources.memory
        remaining_gpu = node.capacity.gpu - node.allocated.gpu - pod.resources.gpu

        # Normalise by capacity to get fraction, then invert (smaller remaining -> higher score)
        cpu_frac = remaining_cpu / node.capacity.cpu if node.capacity.cpu > 0 else 1.0
        mem_frac = remaining_mem / node.capacity.memory if node.capacity.memory > 0 else 1.0
        gpu_frac = remaining_gpu / node.capacity.gpu if node.capacity.gpu > 0 else 1.0

        # Score = 1 / (1 + sum of fractions) – smaller sum => higher score
        total_frac = cpu_frac + mem_frac + gpu_frac
        return 1.0 / (1.0 + total_frac)


class SpreadScore(ScoringPlugin):
    """
    Spread: score is inversely proportional to the number of running pods
    on the node. Nodes with fewer pods get higher scores.
    """
    name = "SpreadScore"

    def Score(self, pod: Pod, node: Node, cluster: Cluster) -> float:
        if not node.can_fit(pod.resources):
            return 0.0
        # More pods => lower score
        return 1.0 / (1.0 + len(node.pods))


class GpuPackingScore(ScoringPlugin):
    """
    GPU packing: for pods requesting GPUs, nodes that already have GPU
    allocations get a bonus. The score is:
        - If pod requests no GPU: 0.5 (neutral)
        - If pod requests GPU:
            score = 1.0 if node has allocated GPUs else 0.0
    """
    name = "GpuPackingScore"

    def Score(self, pod: Pod, node: Node, cluster: Cluster) -> float:
        if not node.can_fit(pod.resources):
            return 0.0
        if pod.resources.gpu > 0:
            # Prefer nodes that already have GPUs allocated
            return 1.0 if node.allocated.gpu > 0 else 0.0
        else:
            return 0.5  # neutral for non‑GPU pods


class FGDScore(ScoringPlugin):
    """
    FGD (First Gang Deployed) – this is a special plugin that operates
    on a gang of pods. It returns 1.0 if the node can fit the entire gang,
    else 0.0. It should be used as a filter (weight high or as mandatory).
    """
    name = "FGDScore"

    def Score(self, pod: Pod, node: Node, cluster: Cluster) -> float:
        # For single‑pod scheduling, fall back to normal feasibility
        return 1.0 if node.can_fit(pod.resources) else 0.0

    def ScoreGang(self, gang_pods: List[Pod], node: Node, cluster: Cluster) -> float:
        """
        Score for a whole gang. Checks if the node can fit all pods.
        """
        total_req = ResourceRequest(0.0, 0.0, 0)
        for p in gang_pods:
            total_req.cpu += p.resources.cpu
            total_req.memory += p.resources.memory
            total_req.gpu += p.resources.gpu
        # Create a temporary pod with total resources for feasibility
        temp_pod = Pod(uid="tmp", name="tmp", resources=total_req)
        return 1.0 if node.can_fit(temp_pod.resources) else 0.0


class DotProductScore(ScoringPlugin):
    """
    DotProductScore as defined in HKUST: computes the dot product of
    the pod's resource request vector and the node's free resource vector,
    optionally normalised.
    """
    name = "DotProductScore"

    def __init__(self, args: Optional[Dict[str, Any]] = None):
        super().__init__(args)
        self.dim_ext_method = self.args.get('dimExtMethod', 'share')  # 'share' or 'full'
        self.norm_method = self.args.get('normMethod', 'max')         # 'max' or 'none'

    def Score(self, pod: Pod, node: Node, cluster: Cluster) -> float:
        if not node.can_fit(pod.resources):
            return 0.0

        # Free resources on node
        free_cpu = node.capacity.cpu - node.allocated.cpu
        free_mem = node.capacity.memory - node.allocated.memory
        free_gpu = node.capacity.gpu - node.allocated.gpu

        # Pod request
        req_cpu = pod.resources.cpu
        req_mem = pod.resources.memory
        req_gpu = pod.resources.gpu

        # Dot product
        dot = req_cpu * free_cpu + req_mem * free_mem + req_gpu * free_gpu

        # Normalisation (optional)
        if self.norm_method == 'max':
            # Max possible dot product (pod request * total capacity)
            max_dot = (req_cpu * node.capacity.cpu +
                       req_mem * node.capacity.memory +
                       req_gpu * node.capacity.gpu)
            if max_dot > 0:
                dot = dot / max_dot
            else:
                dot = 0.0
        # else: no normalisation

        return dot


class RandomScore(ScoringPlugin):
    """Random score – used for random scheduling."""
    name = "RandomScore"

    def __init__(self, args: Optional[Dict[str, Any]] = None, seed: Optional[int] = None):
        super().__init__(args)
        self.rng = random.Random(seed) if seed is not None else random.Random()

    def Score(self, pod: Pod, node: Node, cluster: Cluster) -> float:
        if not node.can_fit(pod.resources):
            return 0.0
        return self.rng.random()


# -------------------------------------------------------------------------
# Plugin Registry
# -------------------------------------------------------------------------

PLUGIN_REGISTRY = {
    "BestFitScore": BestFitScore,
    "SpreadScore": SpreadScore,
    "GpuPackingScore": GpuPackingScore,
    "FGDScore": FGDScore,
    "DotProductScore": DotProductScore,
    "RandomScore": RandomScore,
}


def get_plugin_class(name: str):
    """Retrieve a plugin class by name."""
    cls = PLUGIN_REGISTRY.get(name)
    if cls is None:
        raise ValueError(f"Unknown plugin: {name}")
    return cls


def instantiate_plugin(name: str, args: Optional[Dict[str, Any]] = None, seed: Optional[int] = None):
    """Create an instance of a plugin with given arguments."""
    cls = get_plugin_class(name)
    if name == "RandomScore" and seed is not None:
        return cls(args=args, seed=seed)
    return cls(args=args)