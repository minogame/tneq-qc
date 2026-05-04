"""
Contractor module for generating tensor contraction expressions and managing strategies.

This module provides:
- ContractionStrategy: Abstract base class for contraction strategies
- Concrete strategies: EinsumStrategy, MPSChainStrategy, RowPriorityStrategy
- StrategyCompiler: Compiles and selects strategies by name
- Public registration helpers for user-defined strategies

Strategy registration is performed here to initialize all built-in strategies.
"""

from .base import ContractionStrategy
from .einsum_strategy import EinsumStrategy
from .mps_strategy import MPSChainStrategy
from .row_priority_strategy import RowPriorityStrategy
from .compiler import StrategyCompiler


# =============================================================================
# Strategy Registration
# =============================================================================

def _register_builtin_strategies():
    """Register all built-in strategies"""

    StrategyCompiler.register_strategy(EinsumStrategy())

    StrategyCompiler.register_strategy(RowPriorityStrategy())


def register_contraction_strategy(strategy: ContractionStrategy):
    """Register a user-defined contraction strategy."""
    StrategyCompiler.register_strategy(strategy)


def get_registered_contraction_strategies():
    """Return all registered contraction strategies."""
    return StrategyCompiler.get_registered_strategies()


# Perform registration when module is imported
_register_builtin_strategies()


# =============================================================================
# Public API
# =============================================================================

__all__ = [
    'ContractionStrategy',
    'EinsumStrategy',
    'MPSChainStrategy',
    'RowPriorityStrategy',
    'StrategyCompiler',
    'register_contraction_strategy',
    'get_registered_contraction_strategies',
]
