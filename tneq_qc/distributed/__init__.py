"""
TNEQ Distributed Computing Module

Provides distributed training capabilities for TNEQ-QC tensor network models.

Key Components:
- comm: Communication backends (MPI, torch.distributed)
- parallel: Data parallel training strategy
- engine: Distributed engine with graph partitioning
- trainer: High-level distributed trainer

Existing (Genetic Algorithm):
- mpi_overlord: Overlord for genetic algorithm distributed search
- mpi_agent: Agent workers for genetic algorithm
- mpi_core: Core MPI utilities

Example:
    >>> from tneq_qc.distributed import DistributedTrainer, DistributedConfig
    >>> config = DistributedConfig(backend_type='pytorch', num_qubits=4)
    >>> trainer = DistributedTrainer(config)
    >>> stats = trainer.train(data_list, circuit_states_list)
"""

# Communication backends
from .comm import (
    # Interface
    CommBase,
    ReduceOp,
    DistributedContext,
    AsyncHandle,
    # MPI backend
    CommMPI,
    MockCommMPI,
    get_comm_mpi,
    # Torch backend
    CommTorch,
    MockCommTorch,
    get_comm_torch,
    # Factory
    get_comm_backend,
    get_auto_backend,
    # Backward compatibility
    MPIBackend,
    MockMPIBackend,
)

# Data parallel training
from .parallel import (
    DataParallelTrainer, 
    TrainingConfig, 
    TrainingStats,
    ModelParallelManager, 
    ModelParallelTrainer, 
    ModelParallelConfig,
    CorePartition, 
    create_model_parallel_trainer
)

# Distributed engine
from .engine import EngineDistributed, DistributedEngineSiamese
from .engine.distributed_engine import (
    PartitionConfig,
    ContractStage,
    DistributedContractPlan,
)

# High-level trainer
from .trainer import DistributedTrainer
from .trainer.distributed_trainer import DistributedConfig

# Distributed optimization
from .optim import (
    AllReduceGrad,
    allreduce_with_grad,
    DistributedSGDG,
)
from .optim.distributed_sgdg import LRScheduler

# Legacy genetic algorithm modules (existing)
from .mpi_core import TAGS, SURVIVAL, REASONS, AGENT_STATUS, INDIVIDUAL_STATUS

__all__ = [
    # Communication interface
    'CommBase',
    'ReduceOp',
    'DistributedContext',
    'AsyncHandle',
    
    # MPI backend
    'CommMPI',
    'MockCommMPI',
    'get_comm_mpi',
    
    # Torch backend
    'CommTorch',
    'MockCommTorch',
    'get_comm_torch',
    
    # Factory
    'get_comm_backend',
    'get_auto_backend',
    
    # Backward compatibility (comm)
    'MPIBackend',
    'MockMPIBackend',
    
    # Data Parallel
    'DataParallelTrainer',
    'TrainingConfig',
    'TrainingStats',
    
    # Model Parallel
    'ModelParallelManager',
    'ModelParallelTrainer',
    'ModelParallelConfig',
    'CorePartition',
    'create_model_parallel_trainer',
    
    # Engine
    'EngineDistributed',
    'DistributedEngineSiamese',
    'PartitionConfig',
    'ContractStage',
    'DistributedContractPlan',
    
    # High-level trainer
    'DistributedTrainer',
    'DistributedConfig',
    
    # Distributed optimization
    'AllReduceGrad',
    'allreduce_with_grad',
    'DistributedSGDG',
    'LRScheduler',
    
    # Legacy (genetic algorithm)
    'TAGS',
    'SURVIVAL',
    'REASONS',
    'AGENT_STATUS',
    'INDIVIDUAL_STATUS',
]
