"""
sim/config.py – KubeSchedulerConfiguration parser for plugin‑based scheduling.

This module defines the YAML schema for scheduler configuration,
following the Kubernetes style (apiVersion, kind, plugins, pluginConfig).

Example YAML:
    apiVersion: simon/v1alpha1
    kind: KubeSchedulerConfiguration
    plugins:
      score:
        enabled:
          - name: FGDScore
            weight: 2
          - name: DotProductScore
            weight: 1
          - name: GpuPackingScore
            weight: 1
    pluginConfig:
      - name: DotProductScore
        args:
          dimExtMethod: share
          normMethod: max
      - name: GpuPackingScore
        args:
          gpuResourceName: nvidia.com/gpu
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import yaml


# -------------------------------------------------------------------------
# Data models for the configuration
# -------------------------------------------------------------------------

@dataclass
class PluginArgs:
    """Arguments for a specific plugin."""
    # Generic dict for arbitrary key-value pairs
    data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Plugin:
    """A single plugin entry (name and weight)."""
    name: str
    weight: int = 1


@dataclass
class PluginConfig:
    """Plugin configuration with optional arguments."""
    name: str
    args: Optional[Dict[str, Any]] = None


@dataclass
class Plugins:
    """Container for score plugins."""
    score: Optional[List[Plugin]] = None


@dataclass
class KubeSchedulerConfiguration:
    """
    The main scheduler configuration object.

    Matches the schema:
        apiVersion: simon/v1alpha1
        kind: KubeSchedulerConfiguration
        plugins:
          score:
            enabled: [...]
        pluginConfig: [...]
    """
    apiVersion: str = "simon/v1alpha1"
    kind: str = "KubeSchedulerConfiguration"
    plugins: Optional[Plugins] = None
    pluginConfig: Optional[List[PluginConfig]] = None

    @classmethod
    def from_yaml(cls, filepath: str) -> "KubeSchedulerConfiguration":
        """Load and parse a YAML file."""
        with open(filepath, 'r') as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "KubeSchedulerConfiguration":
        """Parse a dictionary into a configuration object."""
        api_version = data.get('apiVersion', 'simon/v1alpha1')
        kind = data.get('kind', 'KubeSchedulerConfiguration')
        plugins_data = data.get('plugins', {})
        plugin_config_data = data.get('pluginConfig', [])

        plugins = None
        if plugins_data:
            score_plugins = []
            score_data = plugins_data.get('score', {})
            enabled = score_data.get('enabled', [])
            for p in enabled:
                if isinstance(p, dict):
                    name = p.get('name')
                    weight = p.get('weight', 1)
                    score_plugins.append(Plugin(name=name, weight=weight))
                else:
                    score_plugins.append(Plugin(name=p, weight=1))
            plugins = Plugins(score=score_plugins)

        plugin_config = []
        for pc in plugin_config_data:
            name = pc.get('name')
            args = pc.get('args')
            plugin_config.append(PluginConfig(name=name, args=args))

        return cls(
            apiVersion=api_version,
            kind=kind,
            plugins=plugins,
            pluginConfig=plugin_config,
        )

    def get_plugin_args(self, plugin_name: str) -> Dict[str, Any]:
        """Retrieve the args for a given plugin name, or empty dict."""
        if self.pluginConfig:
            for pc in self.pluginConfig:
                if pc.name == plugin_name:
                    return pc.args or {}
        return {}

    def get_plugins_with_weights(self) -> List[Plugin]:
        """Return the list of score plugins with weights."""
        if self.plugins and self.plugins.score:
            return self.plugins.score
        return []