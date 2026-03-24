"""
Data Parallel Training Module

Implements data parallel training strategy where training data is split
across multiple workers, each worker computes local gradients, and
gradients are synchronized via AllReduce.
"""

from __future__ import annotations
import time
from typing import List, Dict, Tuple, Optional, Union, TYPE_CHECKING
from dataclasses import dataclass, field

from ..comm import CommMPI, MockCommMPI, ReduceOp, get_comm_mpi

# Backward compatibility aliases
MPIBackend = CommMPI
MockMPIBackend = MockCommMPI
get_backend = get_comm_mpi

if TYPE_CHECKING:
    import torch
    from ...core.engine_siamese import EngineSiamese
    from ...core.qctn import QCTN
    from ...optim.optimizer import Optimizer


@dataclass
class TrainingConfig:
    """Configuration for distributed training."""
    
    # Training parameters
    max_steps: int = 1000
    log_interval: int = 10
    checkpoint_interval: int = 100
    gradient_accumulation_steps: int = 1
    
    # Learning rate configuration
    learning_rate: float = 1e-2
    lr_schedule: Optional[List[Tuple[int, float]]] = None
    
    # Optimizer parameters
    optimizer_method: str = 'sgdg'
    momentum: float = 0.9
    stiefel: bool = True
    
    # Data parallel settings
    sync_every_step: bool = True  # Sync gradients every step vs accumulate
    async_gradient_sync: bool = False  # Use async AllReduce
    
    # Convergence
    tol: Optional[float] = None  # Tolerance for convergence check


@dataclass
class TrainingStats:
    """Statistics from training."""
    final_loss: float = float('inf')
    total_steps: int = 0
    total_time: float = 0.0
    losses: List[float] = field(default_factory=list)
    converged: bool = False
    
    def to_dict(self) -> Dict:
        return {
            'final_loss': self.final_loss,
            'total_steps': self.total_steps,
            'total_time': self.total_time,
            'converged': self.converged,
        }


class DataParallelTrainer:
    """
    Data Parallel Trainer.
    
    Splits training data across multiple workers, computes local gradients,
    and synchronizes via AllReduce.
    
    Key features:
    1. Data partitioning and distribution
    2. Local forward/backward computation
    3. Gradient AllReduce synchronization
    4. Synchronized parameter updates
    
    Example:
        >>> mpi = MPIBackend()
        >>> trainer = DataParallelTrainer(engine, qctn, config, mpi)
        >>> stats = trainer.train(data_list, circuit_states_list)
    """
    
    def __init__(self, 
                 engine: 'EngineSiamese',
                 qctn: 'QCTN',
                 config: TrainingConfig,
                 mpi_backend: Optional[Union[MPIBackend, MockMPIBackend]] = None):
        """
        Initialize the data parallel trainer.
        
        Args:
            engine: EngineSiamese instance
            qctn: QCTN model
            config: Training configuration
            mpi_backend: MPI backend (auto-created if None)
        """
        self.engine = engine
        self.qctn = qctn
        self.config = config
        
        # Initialize MPI
        self.mpi = mpi_backend or get_backend(use_mpi=True)
        self.ctx = self.mpi.get_context()
        
        # Lazy import to avoid circular dependencies
        from ...optim.optimizer import Optimizer
        
        # Initialize optimizer
        self.optimizer = Optimizer(
            method=config.optimizer_method,
            learning_rate=config.learning_rate,
            max_iter=config.max_steps,
            lr_schedule=config.lr_schedule,
            momentum=config.momentum,
            stiefel=config.stiefel,
            engine=engine
        )
        
        # Training state
        self.global_step = 0
        self.accumulated_grads: Optional[List['torch.Tensor']] = None
        self.accumulation_count = 0
        
        self._log(f"DataParallelTrainer initialized: {self.ctx}")
    
    def _log(self, msg: str, level: str = "info"):
        """Log message only on main process."""
        if self.mpi.is_main_process():
            print(f"[{level.upper()}] {msg}")
    
    # ==================== Data Partitioning ====================
    
    def partition_data(self, data_list: List[Dict]) -> List[Dict]:
        """
        Partition data to this worker.
        
        Distributes data evenly across workers. If data count is not
        evenly divisible, earlier workers get one extra sample.
        
        Args:
            data_list: Complete data list
            
        Returns:
            This worker's data partition
        """
        n = len(data_list)
        per_worker = n // self.ctx.world_size
        remainder = n % self.ctx.world_size
        
        # Earlier workers get one extra if there's remainder
        if self.ctx.rank < remainder:
            start = self.ctx.rank * (per_worker + 1)
            end = start + per_worker + 1
        else:
            start = remainder * (per_worker + 1) + (self.ctx.rank - remainder) * per_worker
            end = start + per_worker
        
        local_data = data_list[start:end]
        self._log(f"Worker {self.ctx.rank}: data partition [{start}:{end}] ({len(local_data)} samples)")
        
        return local_data
    
    # ==================== Gradient Computation & Sync ====================
    
    def compute_local_gradients(self, data: Dict, 
                                 circuit_states_list: List) -> Tuple[float, List['torch.Tensor']]:
        """
        Compute gradients on local data.
        
        Args:
            data: Dictionary containing measure_input_list
            circuit_states_list: Circuit states list
            
        Returns:
            (loss, gradients)
        """
        loss, grads = self.engine.contract_with_compiled_strategy_for_gradient(
            self.qctn,
            circuit_states_list=circuit_states_list,
            **data
        )
        
        return float(loss), grads
    
    def sync_gradients(self, local_grads: List['torch.Tensor']) -> List['torch.Tensor']:
        """
        Synchronize gradients across all workers using AllReduce.
        
        Args:
            local_grads: Local gradients
            
        Returns:
            Globally averaged gradients
        """
        return self.mpi.allreduce_tensors(local_grads, op=ReduceOp.AVG)
    
    def sync_gradients_async(self, local_grads: List['torch.Tensor']):
        """
        Start asynchronous gradient synchronization.
        
        Args:
            local_grads: Local gradients
            
        Returns:
            AsyncHandle
        """
        return self.mpi.allreduce_tensors_async(local_grads, op=ReduceOp.AVG)
    
    def sync_loss(self, local_loss: float) -> float:
        """
        Synchronize loss across all workers.
        
        Args:
            local_loss: Local loss value
            
        Returns:
            Globally averaged loss
        """
        return self.mpi.allreduce_scalar(local_loss, op=ReduceOp.AVG)
    
    # ==================== Gradient Accumulation ====================
    
    def accumulate_gradients(self, grads: List['torch.Tensor']):
        """
        Accumulate gradients for gradient accumulation strategy.
        
        Args:
            grads: Gradients to accumulate
        """
        if self.accumulated_grads is None:
            self.accumulated_grads = [g.clone() for g in grads]
        else:
            for i, g in enumerate(grads):
                self.accumulated_grads[i] += g
        self.accumulation_count += 1
    
    def get_accumulated_gradients(self) -> List['torch.Tensor']:
        """
        Get averaged accumulated gradients and reset.
        
        Returns:
            Averaged accumulated gradients
        """
        if self.accumulated_grads is None:
            raise ValueError("No gradients accumulated")
        
        avg_grads = [g / self.accumulation_count for g in self.accumulated_grads]
        
        # Reset
        self.accumulated_grads = None
        self.accumulation_count = 0
        
        return avg_grads
    
    # ==================== Training Step ====================
    
    def train_step(self, data: Dict, circuit_states_list: List) -> float:
        """
        Execute a single training step.
        
        Args:
            data: Training data for this step
            circuit_states_list: Circuit states
            
        Returns:
            Loss value
        """
        # Compute local gradients
        local_loss, local_grads = self.compute_local_gradients(data, circuit_states_list)
        
        if self.config.gradient_accumulation_steps > 1:
            # Accumulate gradients
            self.accumulate_gradients(local_grads)
            
            if (self.global_step + 1) % self.config.gradient_accumulation_steps == 0:
                # Get accumulated gradients and sync
                accumulated = self.get_accumulated_gradients()
                global_grads = self.sync_gradients(accumulated)
                
                # Update parameters
                self.optimizer.step(self.qctn, global_grads)
        else:
            # Sync gradients every step
            if self.config.async_gradient_sync:
                # Async sync - overlap communication with computation
                handle = self.sync_gradients_async(local_grads)
                # ... could do other computation here ...
                global_grads = handle.wait()
            else:
                global_grads = self.sync_gradients(local_grads)
            
            # Update parameters
            self.optimizer.step(self.qctn, global_grads)
        
        # Sync loss for logging
        global_loss = self.sync_loss(local_loss)
        
        return global_loss
    
    # ==================== Training Loop ====================
    
    def train(self, data_list: List[Dict], circuit_states_list: List) -> TrainingStats:
        """
        Execute the distributed training loop.
        
        Args:
            data_list: Complete training data list
            circuit_states_list: Circuit states
            
        Returns:
            Training statistics
        """
        import torch
        import numpy as np
        
        stats = TrainingStats()
        start_time = time.time()
        
        # Partition data
        local_data = self.partition_data(data_list)
        n_local = len(local_data)
        
        if n_local == 0:
            self._log(f"Warning: Worker {self.ctx.rank} has no data!")
            return stats
        
        # Synchronize random seed for data shuffling
        seed = 42
        if self.mpi.is_main_process():
            seed = np.random.randint(0, 2**31)
        seed = self.mpi.broadcast_object(seed, src=0)
        rng = np.random.default_rng(seed)
        
        self._log(f"Starting training: {self.config.max_steps} steps, "
                  f"{n_local} local samples, lr={self.config.learning_rate}")
        
        # Training loop
        for step in range(self.config.max_steps):
            self.global_step = step
            self.optimizer.iter = step
            
            # Get data for this step
            data_idx = step % n_local
            data = local_data[data_idx]
            
            # Execute training step
            loss = self.train_step(data, circuit_states_list)
            stats.losses.append(loss)
            
            # Update learning rate
            self.optimizer._apply_lr_schedule()
            
            # Logging
            if step % self.config.log_interval == 0:
                self._log(f"Step {step}: loss = {loss:.6f}, lr = {self.optimizer.learning_rate:.2e}")
            
            # Convergence check
            if self.config.tol is not None and loss < self.config.tol:
                self._log(f"Converged at step {step} with loss {loss:.6f}")
                stats.converged = True
                break
            
            # Checkpoint
            if self.config.checkpoint_interval > 0 and (step + 1) % self.config.checkpoint_interval == 0:
                if self.mpi.is_main_process():
                    self._save_checkpoint(step)
        
        # Final sync
        self.mpi.barrier()
        
        stats.final_loss = stats.losses[-1] if stats.losses else float('inf')
        stats.total_steps = self.global_step + 1
        stats.total_time = time.time() - start_time
        
        self._log(f"Training completed: {stats.total_steps} steps, "
                  f"final loss = {stats.final_loss:.6f}, time = {stats.total_time:.1f}s")
        
        return stats
    
    def _save_checkpoint(self, step: int):
        """Save training checkpoint (main process only)."""
        # Placeholder for checkpoint saving
        self._log(f"Checkpoint at step {step}")
    
    # ==================== Evaluation ====================
    
    def evaluate(self, data_list: List[Dict], circuit_states_list: List) -> float:
        """
        Evaluate the model on given data (distributed).
        
        Args:
            data_list: Evaluation data
            circuit_states_list: Circuit states
            
        Returns:
            Average loss
        """
        import torch
        
        local_data = self.partition_data(data_list)
        
        total_loss = 0.0
        for data in local_data:
            with torch.no_grad():
                loss, _ = self.engine.contract_with_compiled_strategy_for_gradient(
                    self.qctn,
                    circuit_states_list=circuit_states_list,
                    **data
                )
            total_loss += float(loss)
        
        avg_local_loss = total_loss / len(local_data) if local_data else 0.0
        avg_global_loss = self.sync_loss(avg_local_loss)
        
        return avg_global_loss
