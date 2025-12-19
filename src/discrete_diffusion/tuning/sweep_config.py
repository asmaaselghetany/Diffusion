"""Sweep configuration parsing and validation.

Supports grid search, random search, and Latin hypercube sampling
with constraint validation and conditional parameter handling.
"""

from __future__ import annotations
import itertools
import json
import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import yaml
from scipy.stats import qmc


@dataclass
class ParameterSpec:
    """Specification for a single hyperparameter."""
    name: str
    values: List[Any]
    baseline: Any = None
    param_type: str = "categorical"  # categorical, continuous, log_continuous
    
    @classmethod
    def from_dict(cls, name: str, spec: Dict) -> 'ParameterSpec':
        values = spec.get('values', [])
        baseline = spec.get('baseline')
        
        # Infer type from values
        if all(isinstance(v, (int, float)) for v in values):
            if len(values) >= 2:
                ratios = [values[i+1] / values[i] for i in range(len(values)-1) if values[i] != 0]
                if ratios and all(abs(r - ratios[0]) < 0.1 for r in ratios):
                    param_type = "log_continuous"
                else:
                    param_type = "continuous"
            else:
                param_type = "continuous"
        else:
            param_type = "categorical"
        
        return cls(name=name, values=values, baseline=baseline, param_type=param_type)


@dataclass
class Constraint:
    """Constraint between parameters."""
    constraint_type: str  # divisible, less_than, greater_than, equal
    params: List[str]
    
    def validate(self, config: Dict[str, Any]) -> bool:
        """Check if config satisfies constraint."""
        vals = [config.get(p) for p in self.params]
        if any(v is None for v in vals):
            return True  # Skip if params not in config
        
        if self.constraint_type == "divisible":
            return vals[0] % vals[1] == 0
        elif self.constraint_type == "less_than":
            return vals[0] < vals[1]
        elif self.constraint_type == "greater_than":
            return vals[0] > vals[1]
        elif self.constraint_type == "equal":
            return vals[0] == vals[1]
        return True


@dataclass
class Conditional:
    """Conditional parameter overrides."""
    condition: str  # Python expression to evaluate
    overrides: Dict[str, Any]
    
    def applies(self, config: Dict[str, Any]) -> bool:
        """Check if condition applies to config."""
        try:
            # Safe evaluation with limited context
            return eval(self.condition, {"__builtins__": {}}, config)
        except Exception:
            return False
    
    def apply(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Apply overrides if condition is met."""
        if self.applies(config):
            config = config.copy()
            config.update(self.overrides)
        return config


@dataclass  
class SweepConfig:
    """Complete sweep configuration."""
    name: str
    method: str  # grid, random, latin_hypercube
    parameters: Dict[str, ParameterSpec]
    fixed: Dict[str, Any]
    constraints: List[Constraint] = field(default_factory=list)
    conditionals: List[Conditional] = field(default_factory=list)
    metric: Dict[str, str] = field(default_factory=lambda: {"name": "loss", "goal": "minimize"})
    early_terminate: Optional[Dict[str, Any]] = None
    requires_checkpoint: Optional[Dict[str, Any]] = None
    
    def get_baseline_config(self) -> Dict[str, Any]:
        """Get configuration with all baseline values."""
        config = self.fixed.copy()
        for name, spec in self.parameters.items():
            if spec.baseline is not None:
                config[name] = spec.baseline
            else:
                config[name] = spec.values[len(spec.values) // 2]  # Middle value
        return config
    
    def generate_configs(self, num_runs: Optional[int] = None, seed: int = 42) -> List[Dict[str, Any]]:
        """Generate all configurations based on sweep method."""
        if self.method == "grid":
            return self._generate_grid_configs()
        elif self.method == "random":
            return self._generate_random_configs(num_runs or 50, seed)
        elif self.method == "latin_hypercube":
            return self._generate_lhs_configs(num_runs or 50, seed)
        else:
            raise ValueError(f"Unknown sweep method: {self.method}")
    
    def _generate_grid_configs(self) -> List[Dict[str, Any]]:
        """Generate full grid of configurations."""
        param_names = list(self.parameters.keys())
        param_values = [self.parameters[n].values for n in param_names]
        
        configs = []
        for combo in itertools.product(*param_values):
            config = self.fixed.copy()
            config.update(dict(zip(param_names, combo)))
            
            # Apply conditionals
            for cond in self.conditionals:
                config = cond.apply(config)
            
            # Validate constraints
            if all(c.validate(config) for c in self.constraints):
                configs.append(config)
        
        return configs
    
    def _generate_random_configs(self, num_runs: int, seed: int) -> List[Dict[str, Any]]:
        """Generate random configurations."""
        rng = np.random.default_rng(seed)
        param_names = list(self.parameters.keys())
        
        configs = []
        attempts = 0
        max_attempts = num_runs * 10
        
        while len(configs) < num_runs and attempts < max_attempts:
            config = self.fixed.copy()
            for name in param_names:
                spec = self.parameters[name]
                config[name] = rng.choice(spec.values)
            
            # Apply conditionals
            for cond in self.conditionals:
                config = cond.apply(config)
            
            # Validate constraints and uniqueness
            if all(c.validate(config) for c in self.constraints):
                config_hash = self._config_hash(config)
                if not any(self._config_hash(c) == config_hash for c in configs):
                    configs.append(config)
            
            attempts += 1
        
        return configs
    
    def _generate_lhs_configs(self, num_runs: int, seed: int) -> List[Dict[str, Any]]:
        """Generate Latin hypercube sampled configurations."""
        param_names = list(self.parameters.keys())
        n_params = len(param_names)
        
        if n_params == 0:
            return [self.fixed.copy()]
        
        # Generate LHS samples in [0, 1]^n
        sampler = qmc.LatinHypercube(d=n_params, seed=seed)
        samples = sampler.random(n=num_runs * 2)  # Oversample for constraint filtering
        
        configs = []
        for sample in samples:
            if len(configs) >= num_runs:
                break
            
            config = self.fixed.copy()
            for i, name in enumerate(param_names):
                spec = self.parameters[name]
                # Map [0, 1] to parameter index
                idx = int(sample[i] * len(spec.values))
                idx = min(idx, len(spec.values) - 1)
                config[name] = spec.values[idx]
            
            # Apply conditionals
            for cond in self.conditionals:
                config = cond.apply(config)
            
            # Validate constraints
            if all(c.validate(config) for c in self.constraints):
                configs.append(config)
        
        return configs
    
    def _config_hash(self, config: Dict[str, Any]) -> str:
        """Generate hash for config deduplication."""
        # Sort and serialize for consistent hashing
        sorted_items = sorted(config.items())
        return hashlib.md5(json.dumps(sorted_items, default=str).encode()).hexdigest()
    
    # def config_to_hydra_overrides(self, config: Dict[str, Any]) -> List[str]:
    #     """Convert config dict to Hydra CLI overrides."""
    #     overrides = []
    #     for key, value in config.items():
    #         if isinstance(value, bool):
    #             value = str(value).lower()
    #         elif isinstance(value, str):
    #             # Quote strings if they contain special chars
    #             if any(c in value for c in ' =[]{}'):
    #                 value = f'"{value}"'
    #         overrides.append(f"{key}={value}")
    #     return overrides

    def config_to_hydra_overrides(self, config: Dict[str, Any]) -> List[str]:
        """Convert config dict to Hydra CLI overrides."""
        overrides = []
        for key, value in config.items():
            if isinstance(value, bool):
                value = str(value).lower()
            elif isinstance(value, str):
                # Quote strings if they contain special chars (including =, spaces, etc.)
                if any(c in value for c in ' =[]{}\'\"'):
                    # Use single quotes for Hydra overrides with special chars
                    value = f"'{value}'"
            overrides.append(f"{key}={value}")
        return overrides
    
    def get_run_name(self, config: Dict[str, Any], run_idx: int) -> str:
        """Generate descriptive run name."""
        # Include key varying parameters in name
        varying = []
        for name, spec in self.parameters.items():
            if name in config:
                short_name = name.split('.')[-1]
                value = config[name]
                if isinstance(value, float):
                    value = f"{value:.2g}"
                varying.append(f"{short_name}_{value}")
        
        varying_str = "_".join(varying[:4])  # Limit length
        return f"{self.name}_{run_idx:04d}_{varying_str}"


def load_sweep_config(path: Union[str, Path]) -> SweepConfig:
    """Load sweep configuration from YAML file."""
    path = Path(path)
    with open(path, 'r') as f:
        data = yaml.safe_load(f)
    
    sweep_data = data.get('sweep', {})
    
    # Parse parameters
    parameters = {}
    for name, spec in sweep_data.get('parameters', {}).items():
        parameters[name] = ParameterSpec.from_dict(name, spec)
    
    # Parse constraints
    constraints = []
    for c in data.get('constraints', []):
        constraints.append(Constraint(
            constraint_type=c.get('type', 'divisible'),
            params=c.get('params', [])
        ))
    
    # Parse conditionals
    conditionals = []
    for c in data.get('conditionals', []):
        conditionals.append(Conditional(
            condition=c.get('condition', 'False'),
            overrides=c.get('overrides', {})
        ))
    
    return SweepConfig(
        name=sweep_data.get('name', 'sweep'),
        method=sweep_data.get('method', 'grid'),
        parameters=parameters,
        fixed=data.get('fixed', {}),
        constraints=constraints,
        conditionals=conditionals,
        metric=sweep_data.get('metric', {"name": "loss", "goal": "minimize"}),
        early_terminate=sweep_data.get('early_terminate'),
        requires_checkpoint=data.get('requires_checkpoint')
    )


def estimate_grid_size(sweep_config: SweepConfig) -> int:
    """Estimate total number of configurations in grid."""
    total = 1
    for spec in sweep_config.parameters.values():
        total *= len(spec.values)
    return total

