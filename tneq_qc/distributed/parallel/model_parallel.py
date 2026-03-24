"""
Model Parallel for Tensor Network

Implements model parallelism by partitioning core tensors across workers.
Each worker owns a subset of cores and computes gradients only for its local cores.

Key concepts:
1. Core Partition: Cores are divided evenly by index across workers
2. Weight Management: Each worker stores and updates only its local cores
3. Forward Pass: Sequential contraction with communication at partition boundaries
4. Backward Pass: Each worker computes gradients for its local cores

Architecture:
    Worker 0: cores [0, 1, 2]     (cores A, B, C)
    Worker 1: cores [3, 4, 5]     (cores D, E, F)  
    Worker 2: cores [6, 7, 8]     (cores G, H, I)
    ...

Communication Pattern:
    For a sequential contraction order (qubit-by-qubit):
    - When contracting a core on the partition boundary,
      the intermediate result is sent to the next worker
    - Each worker receives the partial result, continues contraction
"""

from __future__ import annotations
from typing import List, Dict, Tuple, Optional, Any, TYPE_CHECKING
from dataclasses import dataclass
from copy import deepcopy

from ..comm import CommMPI, MockCommMPI, ReduceOp, get_comm_mpi, DistributedContext

# Backward compatibility aliases
MPIBackend = CommMPI
MockMPIBackend = MockCommMPI
get_backend = get_comm_mpi

if TYPE_CHECKING:
    from ...core.qctn import QCTN
    from ...backends.backend_interface import ComputeBackend


@dataclass
class ModelParallelConfig:
    """Configuration for model parallel training."""
    
    # Partition strategy: 'even' or 'balanced'
    partition_strategy: str = 'even'
    
    # Whether to overlap communication with computation
    overlap_comm: bool = False
    
    # Communication buffer size for async operations
    comm_buffer_size: int = 10


@dataclass
class CorePartition:
    """Information about core tensor partition for a worker."""
    
    # Worker rank
    rank: int
    
    # Total number of workers
    world_size: int
    
    # List of core indices owned by this worker
    local_core_indices: List[int]
    
    # List of core names owned by this worker
    local_core_names: List[str]
    
    # Mapping from core name to worker rank
    core_to_worker: Dict[str, int]
    
    # Total number of cores
    total_cores: int
    
    def __repr__(self):
        return (f"CorePartition(rank={self.rank}/{self.world_size}, "
                f"local_cores={self.local_core_names})")
    
    def is_local_core(self, core_name: str) -> bool:
        """Check if a core is owned by this worker."""
        return core_name in self.local_core_names
    
    def get_core_owner(self, core_name: str) -> int:
        """Get the worker rank that owns a core."""
        return self.core_to_worker.get(core_name, -1)


class ModelParallelManager:
    """
    Manages model parallel distribution of core tensors.
    
    Handles:
    - Core tensor partitioning across workers
    - Local weight management (loading/saving only local cores)
    - Distributed contraction with communication
    - Gradient synchronization (only for local cores)
    
    Example:
        >>> manager = ModelParallelManager(qctn, mpi_backend)
        >>> partition = manager.partition
        >>> 
        >>> # Each worker only has local cores
        >>> local_weights = manager.get_local_weights()
        >>> 
        >>> # Update only local cores
        >>> manager.set_local_weights(new_weights)
    """
    
    def __init__(self,
                 qctn: 'QCTN',
                 mpi: Optional[MPIBackend] = None,
                 config: Optional[ModelParallelConfig] = None):
        """
        Initialize model parallel manager.
        
        Args:
            qctn: QCTN model to partition
            mpi: MPI backend for communication
            config: Model parallel configuration
        """
        self.qctn = qctn
        self.mpi = mpi or get_backend(use_mpi=True)
        self.ctx = self.mpi.get_context()
        self.config = config or ModelParallelConfig()
        
        # Create partition
        self.partition = self._create_partition()
        
        self._log(f"ModelParallelManager initialized: {self.partition}")
    
    def _log(self, msg: str):
        """Log on main process only."""
        if self.mpi.is_main_process():
            print(f"[ModelParallel] {msg}")
    
    def _create_partition(self) -> CorePartition:
        """
        Create core partition for current worker.
        
        Distributes cores evenly across workers by index.
        """
        cores = self.qctn.cores  # Sorted list of core names
        n_cores = len(cores)
        world_size = self.ctx.world_size
        rank = self.ctx.rank
        
        # Calculate partition
        cores_per_worker = n_cores // world_size
        remainder = n_cores % world_size
        
        # Assign cores to this worker
        if rank < remainder:
            start_idx = rank * (cores_per_worker + 1)
            end_idx = start_idx + cores_per_worker + 1
        else:
            start_idx = remainder * (cores_per_worker + 1) + (rank - remainder) * cores_per_worker
            end_idx = start_idx + cores_per_worker
        
        local_core_indices = list(range(start_idx, end_idx))
        local_core_names = [cores[i] for i in local_core_indices]
        
        # Build mapping from core to worker
        core_to_worker = {}
        for i, core_name in enumerate(cores):
            if i < remainder * (cores_per_worker + 1):
                owner = i // (cores_per_worker + 1)
            else:
                owner = remainder + (i - remainder * (cores_per_worker + 1)) // cores_per_worker
            core_to_worker[core_name] = owner
        
        return CorePartition(
            rank=rank,
            world_size=world_size,
            local_core_indices=local_core_indices,
            local_core_names=local_core_names,
            core_to_worker=core_to_worker,
            total_cores=n_cores
        )
    
    # ==================== Weight Management ====================
    
    def get_local_weights(self) -> Dict[str, Any]:
        """
        Get weights for local cores only.
        
        Returns:
            Dict mapping core name to weight tensor
        """
        weights = {}
        for core_name in self.partition.local_core_names:
            weights[core_name] = self.qctn.cores_weights[core_name]
        return weights
    
    def set_local_weights(self, weights: Dict[str, Any]):
        """
        Set weights for local cores.
        
        Args:
            weights: Dict mapping core name to weight tensor
        """
        for core_name, weight in weights.items():
            if core_name in self.partition.local_core_names:
                self.qctn.cores_weights[core_name] = weight
    
    def broadcast_all_weights(self):
        """
        Broadcast all weights from each owner to all workers.
        
        This synchronizes the full model across all workers.
        Used at initialization or checkpoint loading.
        """
        from ...core.tn_tensor import TNTensor
        
        for core_name in self.qctn.cores:
            owner = self.partition.get_core_owner(core_name)
            weight = self.qctn.cores_weights[core_name]
            
            if isinstance(weight, TNTensor):
                # Broadcast tensor and scale separately
                synced_tensor = self.mpi.broadcast(weight.tensor, src=owner)
                if hasattr(weight.scale, 'detach'):
                    synced_scale = self.mpi.broadcast(weight.scale, src=owner)
                else:
                    import numpy as np
                    scale_arr = np.array([weight.scale], dtype=np.float32)
                    scale_tensor = self.qctn.backend.convert_to_tensor(scale_arr)
                    synced_scale = self.mpi.broadcast(scale_tensor, src=owner)
                    synced_scale = float(self.qctn.backend.tensor_to_numpy(synced_scale)[0])
                
                # IMPORTANT: Only overwrite if we are NOT the owner.
                # If we are owner, we must keep the original tensor to preserve Autograd graph.
                if self.ctx.rank != owner:
                    self.qctn.cores_weights[core_name] = TNTensor(synced_tensor, synced_scale)
            else:
                synced_weight = self.mpi.broadcast(weight, src=owner)
                if self.ctx.rank != owner:
                    self.qctn.cores_weights[core_name] = synced_weight
        
        self._log("All weights synchronized via broadcast")
    
    def gather_local_gradients(self, local_grads: Dict[str, Any]) -> Dict[str, Any]:
        """
        Gather gradients from all workers.
        
        Each worker contributes gradients for its local cores.
        Result contains gradients for all cores.
        
        Args:
            local_grads: Dict mapping local core names to gradients
            
        Returns:
            Dict mapping all core names to gradients
        """
        # Prepare gradient list for each core (None for non-local)
        all_grads = {}
        for core_name in self.qctn.cores:
            if core_name in local_grads:
                all_grads[core_name] = local_grads[core_name]
            else:
                all_grads[core_name] = None
        
        # Gather from each worker
        for core_name in self.qctn.cores:
            owner = self.partition.get_core_owner(core_name)
            
            # Broadcast gradient from owner
            if self.ctx.rank == owner:
                grad = local_grads.get(core_name)
                if grad is not None:
                    synced_grad = self.mpi.broadcast(grad, src=owner)
                    all_grads[core_name] = synced_grad
            else:
                # Receive gradient from owner
                if all_grads[core_name] is None:
                    # Need shape info - get from weight
                    weight = self.qctn.cores_weights[core_name]
                    if hasattr(weight, 'tensor'):
                        shape = weight.tensor.shape
                    else:
                        shape = weight.shape
                    
                    # Create placeholder and receive
                    import numpy as np
                    placeholder = self.qctn.backend.zeros(shape)
                    synced_grad = self.mpi.broadcast(placeholder, src=owner)
                    all_grads[core_name] = synced_grad
        
        return all_grads
    
    # ==================== Distributed Contraction ====================
    
    def get_contraction_order(self) -> List[Dict]:
        """
        Get ordered list of contraction steps with communication points.
        
        Returns:
            List of step dicts with:
            - 'type': 'local_contract' or 'communicate'
            - 'cores': list of core names involved
            - 'from_worker': source worker (for communicate)
            - 'to_worker': destination worker (for communicate)
        """
        # For now, use simple sequential order by qubit
        # This will be optimized in future versions
        
        steps = []
        prev_owner = None
        
        # Group cores by their contraction order
        for qubit_idx in range(self.qctn.nqubits):
            # Find cores connected to this qubit
            qubit_cores = []
            for core_info in self.qctn.adjacency_table:
                for edge in core_info['in_edge_list'] + core_info['out_edge_list']:
                    if edge.get('qubit_idx') == qubit_idx:
                        qubit_cores.append(core_info['core_name'])
                        break
            
            qubit_cores = list(set(qubit_cores))
            
            if not qubit_cores:
                continue
            
            # Check if we need communication (cores span multiple workers)
            owners = set(self.partition.get_core_owner(c) for c in qubit_cores)
            
            if len(owners) == 1:
                # All cores on same worker - local contraction
                steps.append({
                    'type': 'local_contract',
                    'qubit_idx': qubit_idx,
                    'cores': qubit_cores,
                    'owner': list(owners)[0]
                })
            else:
                # Cores span multiple workers - need communication
                steps.append({
                    'type': 'distributed_contract',
                    'qubit_idx': qubit_idx,
                    'cores': qubit_cores,
                    'owners': list(owners)
                })
        
        return steps


class ModelParallelTrainer:
    """
    Trainer for model parallel tensor network training.
    
    Combines ModelParallelManager with optimization to train
    large tensor networks that don't fit on a single worker.
    
    Features:
    - Each worker updates only its local cores
    - Forward pass with pipeline-style communication
    - Backward pass with local gradient computation
    - No gradient AllReduce needed (each worker has unique cores)
    
    Example:
        >>> trainer = ModelParallelTrainer(qctn, engine, mpi)
        >>> 
        >>> for epoch in range(epochs):
        ...     loss, grads = trainer.train_step(data)
        ...     # grads only contain local core gradients
        ...     trainer.update_local_weights(grads, optimizer)
    """
    
    def __init__(self,
                 qctn: 'QCTN',
                 engine,  # DistributedEngineSiamese or EngineSiamese
                 mpi: Optional[MPIBackend] = None,
                 config: Optional[ModelParallelConfig] = None):
        """
        Initialize model parallel trainer.
        
        Args:
            qctn: QCTN model
            engine: Engine for contraction
            mpi: MPI backend
            config: Model parallel config
        """
        self.qctn = qctn
        self.engine = engine
        self.mpi = mpi or get_backend(use_mpi=True)
        self.ctx = self.mpi.get_context()
        self.config = config or ModelParallelConfig()
        
        # Create model parallel manager
        self.manager = ModelParallelManager(qctn, self.mpi, self.config)
        self.partition = self.manager.partition
        
        # Initialize Distributed Hierarchical Contractor
        from .distributed_contractor import DistributedHierarchicalContractor
        self.contractor = DistributedHierarchicalContractor(engine, self.mpi, self.partition)

        # Synchronize initial weights
        self.manager.broadcast_all_weights()
        
        # Ensure local weights require gradients
        for core_name in self.partition.local_core_names:
             w = qctn.cores_weights[core_name]
             from ...core.tn_tensor import TNTensor
             if isinstance(w, TNTensor):
                 if not w.tensor.requires_grad:
                     w.tensor.requires_grad_(True)
             else:
                 if not w.requires_grad:
                     w.requires_grad_(True)
        
        self._log(f"ModelParallelTrainer initialized")
        self._log(f"  Local cores: {self.partition.local_core_names}")
    
    def _log(self, msg: str):
        """Log with rank info."""
        print(f"[Rank {self.ctx.rank}] {msg}")
    
    def forward(self, circuit_states_list: List, measure_input_list: List):
        """
        Forward pass with model parallelism.
        
        Each worker computes the full forward pass but only tracks
        gradients for local cores.
        
        Args:
            circuit_states_list: Circuit states
            measure_input_list: Measurement matrices
            
        Returns:
            Loss value
        """
        # All workers need full model for forward pass
        # (weights already synced via broadcast_all_weights)
        
        # Use base engine for contraction
        backend = self.engine.backend if hasattr(self.engine, 'backend') else self.engine._base_engine.backend
        
        if hasattr(self.engine, '_base_engine'):
            base_engine = self.engine._base_engine
        else:
            base_engine = self.engine
        
        result = base_engine.contract_with_compiled_strategy(
            self.qctn,
            circuit_states_list=circuit_states_list,
            measure_input_list=measure_input_list
        )
        
        return result
    
    def forward_with_gradient(self, circuit_states_list: List, measure_input_list: List):
        """
        Forward pass with gradient computation recursively.
        
        Returns:
            (loss, local_gradients) where local_gradients only contains
            gradients for cores owned by this worker
        """
        # Ensure weights are synced (currently full broadcast, can be optimized)
        self.manager.broadcast_all_weights()
        
        # Delegate to distributed contractor
        return self.contractor.forward_with_gradient(
            self.qctn,
            circuit_states_list,
            measure_input_list
        )

    
    def train_step(self, 
                   circuit_states_list: List, 
                   measure_input_list: List,
                   optimizer = None) -> Tuple[float, Dict[str, Any]]:
        """
        Execute one training step.
        
        Args:
            circuit_states_list: Circuit states
            measure_input_list: Measurement matrices
            optimizer: Optional optimizer to apply updates
            
        Returns:
            (loss, local_gradients)
        """
        # Forward + backward
        loss, local_grads = self.forward_with_gradient(
            circuit_states_list, measure_input_list
        )
        
        # Apply optimizer if provided
        if optimizer is not None:
            self._apply_optimizer(local_grads, optimizer)
        
        return loss, local_grads
    
    def _apply_optimizer(self, local_grads: Dict[str, Any], optimizer):
        """
        Apply optimizer updates to local cores.
        
        Args:
            local_grads: Gradients for local cores
            optimizer: Optimizer instance
        """
        # Get current local weights
        local_weights = self.manager.get_local_weights()
        
        # Apply update
        from ...core.tn_tensor import TNTensor
        
        for core_name, grad in local_grads.items():
            if core_name not in local_weights:
                continue
            
            weight = local_weights[core_name]
            
            # Handle TNTensor
            if isinstance(weight, TNTensor):
                # Update the underlying tensor
                if hasattr(optimizer, 'step_single'):
                    # Custom optimizer with single tensor update
                    new_tensor = optimizer.step_single(weight.tensor, grad)
                else:
                    # Simple SGD fallback
                    lr = getattr(optimizer, 'lr', 0.01)
                    new_tensor = weight.tensor - lr * grad
                
                # Create new TNTensor with updated tensor
                new_weight = TNTensor(new_tensor, weight.scale)
                # new_weight.auto_scale()
                local_weights[core_name] = new_weight
            else:
                # Regular tensor
                if hasattr(optimizer, 'step_single'):
                    new_weight = optimizer.step_single(weight, grad)
                else:
                    lr = getattr(optimizer, 'lr', 0.01)
                    new_weight = weight - lr * grad
                local_weights[core_name] = new_weight
        
        # Update QCTN weights
        self.manager.set_local_weights(local_weights)
    
    def sync_weights_after_update(self):
        """
        Synchronize weights after optimizer update.
        
        Each worker broadcasts its updated local cores to all others.
        Call this after optimizer.step() to ensure all workers have
        consistent model weights.
        """
        self.manager.broadcast_all_weights()
    
    def save_checkpoint(self, path: str, metadata: Optional[Dict] = None):
        """
        Save model checkpoint.
        
        Only main process saves, but gathers weights from all workers.
        
        Args:
            path: Path to save checkpoint
            metadata: Optional metadata dict
        """
        # First sync all weights to main process
        self.manager.broadcast_all_weights()
        
        # Main process saves
        if self.mpi.is_main_process():
            self.qctn.save_cores(path, metadata=metadata)
            self._log(f"Checkpoint saved: {path}")
    
    def load_checkpoint(self, path: str):
        """
        Load model checkpoint.
        
        Main process loads, then broadcasts to all workers.
        
        Args:
            path: Path to checkpoint file
        """
        # Main process loads
        if self.mpi.is_main_process():
            self.qctn.load_cores(path)
            self._log(f"Checkpoint loaded: {path}")
        
        # Broadcast to all workers
        self.manager.broadcast_all_weights()


def create_model_parallel_trainer(qctn: 'QCTN',
                                   backend = None,
                                   strategy_mode: str = 'balanced',
                                   config: Optional[ModelParallelConfig] = None,
                                   mpi: Optional[MPIBackend] = None):
    """
    Factory function to create a model parallel trainer.
    
    Args:
        qctn: QCTN model
        backend: Compute backend
        strategy_mode: Contraction strategy mode
        config: Model parallel config
        mpi: MPI backend
        
    Returns:
        ModelParallelTrainer instance
    """
    from ..engine.distributed_engine import DistributedEngineSiamese
    
    engine = DistributedEngineSiamese(
        backend=backend,
        strategy_mode=strategy_mode,
        mpi_backend=mpi
    )
    
    return ModelParallelTrainer(
        qctn=qctn,
        engine=engine,
        mpi=mpi or engine.mpi,
        config=config
    )
