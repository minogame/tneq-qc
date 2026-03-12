"""
Contractor module for generating tensor contraction expressions and managing strategies.

This module provides:
- ContractionStrategy: Abstract base class for contraction strategies
- Concrete strategies: EinsumStrategy, MPSChainStrategy, GreedyStrategy
- StrategyCompiler: Compiles and selects optimal strategy based on mode

Strategy registration is performed here to initialize all built-in strategies.
"""

from .base import ContractionStrategy
from .einsum_strategy import EinsumStrategy
from .mps_strategy import MPSChainStrategy
from .greedy_strategy import GreedyStrategy
from .row_priority_strategy import RowPriorityStrategy
from .compiler import StrategyCompiler


# =============================================================================
# Strategy Registration
# =============================================================================

def _register_builtin_strategies():
    """Register all built-in strategies"""
    
    # Register EinsumStrategy for fast mode
    StrategyCompiler.register_strategy(
        EinsumStrategy(),
        modes=['fast']
    )
    
    # Register MPSChainStrategy for balanced and full modes
    # TODO: Temporarily disable MPSChainStrategy registration
    # StrategyCompiler.register_strategy(
    #     MPSChainStrategy(),
    #     modes=['balanced', 'full']
    # )
    
    # Register GreedyStrategy for balanced and full modes
    StrategyCompiler.register_strategy(
        GreedyStrategy(),
        modes=['balanced']
    )

    # Register RowPriorityStrategy for balanced and full modes
    StrategyCompiler.register_strategy(
        RowPriorityStrategy(),
        modes=['full']
    )


# Perform registration when module is imported
_register_builtin_strategies()


# =============================================================================
# Public API
# =============================================================================

__all__ = [
    'ContractionStrategy',
    'EinsumStrategy',
    'MPSChainStrategy',
    'GreedyStrategy',
    'RowPriorityStrategy',
    'StrategyCompiler',
]
