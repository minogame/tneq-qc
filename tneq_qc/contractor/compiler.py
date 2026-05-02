"""
Strategy compiler for selecting and compiling optimal contraction strategies.

This module provides the StrategyCompiler class that manages strategy selection.
"""

from __future__ import annotations
from typing import Dict, List, Any, Tuple, Callable, Union

from .base import ContractionStrategy


class StrategyCompiler:
    """Strategy compiler, responsible for selecting and compiling the optimal strategy"""
    
    # Strategy list for three modes
    MODES = {
        'fast': ['einsum_default'],
        'balanced': ['row_priority'],
        'full': ['row_priority']
    }
    
    # Global strategy registry
    _strategies: Dict[str, ContractionStrategy] = {}
    
    def __init__(
        self,
        strategy: Union[str, List[str], None] = None,
        mode: str = None,
    ):
        """
        Initialize compiler
        
        Args:
            strategy: Strategy name or ordered list of candidate strategy names.
            mode: Deprecated compatibility option: 'fast', 'balanced', or 'full'.
        """
        if strategy is None:
            if mode is None:
                strategy = 'row_priority'
            else:
                if mode not in self.MODES:
                    raise ValueError(
                        f"Invalid mode '{mode}'. Must be one of {list(self.MODES.keys())}"
                    )
                strategy = self.MODES[mode]

        if isinstance(strategy, str):
            strategy_names = [strategy]
        else:
            strategy_names = list(strategy)

        if not strategy_names:
            raise ValueError("strategy must contain at least one strategy name")

        self.mode = mode
        self.strategy_names = strategy_names
        self.strategy_key = "+".join(strategy_names)
    
    @classmethod
    def register_strategy(cls, strategy: ContractionStrategy, modes: List[str] = None):
        """
        Register a strategy (static method for registration in __init__.py)
        
        Args:
            strategy: Strategy instance
            modes: Which modes to register to, e.g. ['fast', 'balanced', 'full']
                   If None, only register to the strategy registry without adding to any mode
        """
        cls._strategies[strategy.name] = strategy
        
        if modes is not None:
            for mode in modes:
                if mode not in cls.MODES:
                    raise ValueError(
                        f"Invalid mode '{mode}'. Must be one of {list(cls.MODES.keys())}"
                    )
                if strategy.name not in cls.MODES[mode]:
                    cls.MODES[mode].append(strategy.name)
    
    @classmethod
    def get_registered_strategies(cls) -> Dict[str, ContractionStrategy]:
        """Get all registered strategies"""
        return cls._strategies.copy()
    
    @property
    def strategies(self) -> Dict[str, ContractionStrategy]:
        """Get strategies (uses class-level registry)"""
        return self._strategies
    
    def compile(self, qctn, shapes_info: Dict[str, Any], backend, **kwargs) -> Tuple[Callable, str, float]:
        """
        Compile: Select optimal strategy and return computation function
        
        Compilation process:
        1. Check structure compatibility
        2. Estimate cost
        3. Generate computation function
        4. Select strategy with lowest cost
        
        Args:
            qctn: QCTN object
            shapes_info: Shape information dict
            backend: Computation backend
        
        Returns:
            tuple: (compute_fn, strategy_name, estimated_cost)
        """
        candidates = []
        
        print(
            f"[Compiler] Strategy candidates: {self.strategy_names}, "
            f"Testing {len(self.strategy_names)} strategies..."
        )
        
        # Iterate over all candidate strategies
        for name in self.strategy_names:
            if name not in self._strategies:
                print(f"  [{name}] Strategy not registered, skipping...")
                continue
                
            strategy = self._strategies[name]
            
            is_compatible = strategy.check_compatibility(qctn, shapes_info)
            print(f"  [{name}] Compatibility: {is_compatible}")
            
            if not is_compatible:
                continue
            
            # Estimate cost
            cost = strategy.estimate_cost(qctn, shapes_info)
            print(f"  [{name}] Estimated cost: {cost:.2e} FLOPs")
            
            # Generate computation function
            compute_fn = strategy.get_compute_function(qctn, shapes_info, backend, **kwargs)
            
            candidates.append({
                'name': name,
                'strategy': strategy,
                'compute_fn': compute_fn,
                'cost': cost
            })
        
        # Select strategy with lowest cost
        if not candidates:
            raise RuntimeError("No compatible strategy found!")
        
        best = min(candidates, key=lambda x: x['cost'])
        print(f"[Compiler] Selected strategy: {best['name']} (cost: {best['cost']:.2e})")
        
        return best['compute_fn'], best['name'], best['cost']
    
    def register_custom_strategy(self, strategy: ContractionStrategy, modes: List[str]):
        """
        Register custom strategy (instance method for runtime registration)
        
        Args:
            strategy: Strategy instance
            modes: Which modes to register to, e.g. ['balanced', 'full']
        """
        self.register_strategy(strategy, modes)

    @classmethod
    def get_mode_strategies(cls) -> Dict[str, List[str]]:
        """Return the configured strategy names for each mode."""
        return {mode: names.copy() for mode, names in cls.MODES.items()}
