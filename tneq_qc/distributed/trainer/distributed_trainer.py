"""
Distributed Trainer

Main entry point for distributed TNEQ training. Combines all distributed
components into a simple, high-level API.

Provides distributed-specific configuration options beyond the standard trainer:
- Communication backend selection (MPI, torch.distributed)
- Graph partitioning strategy
- Tensor parallel settings
- Gradient synchronization options
"""

from __future__ import annotations
import os
import time
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any, Union, TYPE_CHECKING

from ..comm import CommBase, get_comm_backend, ReduceOp
from ..parallel.data_parallel import DataParallelTrainer, TrainingConfig, TrainingStats
from ..engine.distributed_engine import (
    EngineDistributed,
    PartitionConfig,
    DistributedContractPlan
)

if TYPE_CHECKING:
    import torch
    from ...core.qctn import QCTN


@dataclass
class DistributedConfig:
    """
    Configuration for distributed training.
    
    Extends standard training config with distributed-specific options.
    """
    
    # ==================== Backend Configuration ====================
    
    # Compute backend: 'pytorch' or 'jax'
    backend_type: str = 'pytorch'
    
    # Device: 'cpu', 'cuda', 'cuda:0', etc.
    device: str = 'cpu'
    
    # Contraction strategy mode: 'fast', 'balanced', 'full'
    strategy_mode: str = 'balanced'
    
    # Maximum Hermite polynomial order
    mx_K: int = 100
    
    # ==================== QCTN Configuration ====================

    # Pre-built QCTN object (preferred — preserves weights, requires_grad, etc.)
    qctn: Optional['QCTN'] = None

    # QCTN graph string (fallback if qctn is None)
    qctn_graph: Optional[str] = None

    # Number of qubits (used if both qctn and qctn_graph are None)
    num_qubits: int = 4
    
    # ==================== Communication Configuration ====================
    
    # Communication backend type: 'mpi', 'torch', 'auto'
    comm_backend: str = 'auto'
    
    # Whether to use real distributed communication or mock
    use_distributed: bool = True
    
    # Global rank of this process (None = auto-detect from environment)
    rank: Optional[int] = None
    
    # Total number of processes (None = auto-detect from environment)
    world_size: Optional[int] = None
    
    # Node rank / node index (for multi-node training, None = auto-detect)
    node_rank: Optional[int] = None
    
    # Number of nodes (for multi-node training, None = auto-detect)
    num_nodes: Optional[int] = None
    
    # ==================== Partitioning Configuration ====================
    
    # Partitioning strategy: 'layer', 'core', 'auto'
    partition_strategy: str = 'layer'
    partitions: Optional[List[List[str]]] = None
    
    # Minimum cores per partition
    min_cores_per_partition: int = 1
    
    # Whether to balance partition sizes
    balance_partitions: bool = True
    
    # ==================== Training Configuration ====================
    
    # Maximum training steps
    max_steps: int = 1000
    
    # Logging interval (steps)
    log_interval: int = 10
    
    # Checkpoint interval (steps)
    checkpoint_interval: int = 100
    
    # Learning rate
    learning_rate: float = 1e-2
    
    # Learning rate schedule: list of (step, lr) tuples
    lr_schedule: Optional[List[Tuple[int, float]]] = None
    
    # Optimizer method: 'sgdg', 'adam', etc.
    optimizer: str = 'sgdg'
    
    # Momentum (for SGD-based optimizers)
    momentum: float = 0.9
    
    # Whether to use Stiefel manifold optimization
    stiefel: bool = True
    
    # Convergence tolerance
    tol: Optional[float] = None
    
    # Gradient accumulation steps
    gradient_accumulation_steps: int = 1
    
    # ==================== Gradient Synchronization ====================
    
    # How often to sync gradients (in micro-batches)
    gradient_sync_interval: int = 1
    
    # Whether to overlap communication with computation
    overlap_comm_compute: bool = False
    
    # ==================== Checkpointing ====================
    
    # Checkpoint directory
    checkpoint_dir: str = './checkpoints'
    
    # Whether to save final model
    save_final_model: bool = True
    
    def to_training_config(self) -> TrainingConfig:
        """Convert to TrainingConfig for DataParallelTrainer."""
        return TrainingConfig(
            max_steps=self.max_steps,
            log_interval=self.log_interval,
            checkpoint_interval=self.checkpoint_interval,
            learning_rate=self.learning_rate,
            lr_schedule=self.lr_schedule,
            optimizer_method=self.optimizer,
            momentum=self.momentum,
            stiefel=self.stiefel,
            tol=self.tol,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
        )
    
    def to_partition_config(self, world_size: int) -> PartitionConfig:
        """Create PartitionConfig for engine."""
        return PartitionConfig(
            strategy=self.partition_strategy,
            num_partitions=world_size,
            min_cores_per_partition=self.min_cores_per_partition,
            balance_partitions=self.balance_partitions,
        )
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'DistributedConfig':
        """Create from dictionary."""
        return cls(**{k: v for k, v in config_dict.items() if hasattr(cls, k)})


class DistributedTrainer:
    """
    High-level Distributed Trainer.
    
    Provides a simple API for distributed TNEQ training, handling:
    - Communication backend initialization
    - QCTN model creation and partitioning
    - Data generation and distribution
    - Distributed training loop with gradient synchronization
    - Checkpoint management
    
    Key differences from standard Trainer:
    1. Graph partitioning: QCTN is split across workers
    2. Hierarchical contraction: log(n)+1 stage reduction
    3. Tensor parallel: Large matrix multiplications are sharded
    4. Gradient sync: Configurable synchronization strategies
    
    Example:
        >>> config = DistributedConfig(
        ...     backend_type='pytorch',
        ...     qctn_graph='-3-A-3-B-3-',
        ...     max_steps=1000,
        ...     partition_strategy='layer',
        ... )
        >>> trainer = DistributedTrainer(config)
        >>> data_list, circuit_states = trainer.prepare_data(N=100, B=128, K=3)
        >>> stats = trainer.train(data_list, circuit_states)
    
    Usage with mpiexec:
        $ mpiexec -n 4 python -m tneq_qc.distributed.trainer --config config.yaml
    """
    
    def __init__(self, config: Union[DistributedConfig, Dict[str, Any]]):
        """
        Initialize distributed trainer.
        
        Args:
            config: DistributedConfig or dictionary with configuration
        """
        # Parse config
        if isinstance(config, dict):
            self.config = DistributedConfig.from_dict(config)
            self._raw_config = config
        else:
            self.config = config
            self._raw_config = None
        
        # Initialize communication backend
        self._init_comm()
        
        # Initialize distributed engine with partitioning config
        partition_config = self.config.to_partition_config(self.comm.world_size)
        
        self.engine = EngineDistributed(
            backend=self.config.backend_type,
            strategy_mode=self.config.strategy_mode,
            comm=self.comm,
            partition_config=partition_config,
        )
        
        # Initialize QCTN
        self.qctn: Optional['QCTN'] = None
        self._init_qctn()
        
        # Initialize distributed contraction (partition the graph)
        # if self.qctn is not None and self.comm.world_size > 1:
        #     self._contract_plan = self.engine.init_distributed(self.qctn)
        # else:
        #     self._contract_plan = None
        self._contract_plan = self.engine.init_distributed(self.qctn, partitions=self.config.partitions)
        
        # Setup checkpoint directory
        self.checkpoint_dir = Path(self.config.checkpoint_dir)
        if self.comm.rank == 0:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self._log(f"DistributedTrainer initialized: "
                  f"rank={self.comm.rank}/{self.comm.world_size}")
    
    def _init_comm(self):
        """Initialize communication backend with config settings."""
        comm_type = self.config.comm_backend
        use_distributed = self.config.use_distributed
        
        # Build kwargs for comm backend
        comm_kwargs = {}
        if self.config.rank is not None:
            comm_kwargs['rank'] = self.config.rank
        if self.config.world_size is not None:
            comm_kwargs['world_size'] = self.config.world_size
        if self.config.node_rank is not None:
            comm_kwargs['node_rank'] = self.config.node_rank
        if self.config.num_nodes is not None:
            comm_kwargs['num_nodes'] = self.config.num_nodes
        
        if not use_distributed:
            # Use mock backend
            self.comm = get_comm_backend(backend='mock', **comm_kwargs)
        elif comm_type == 'auto':
            # Auto-detect: try MPI first, fall back to mock
            self.comm = get_comm_backend(backend=self.config.backend_type, **comm_kwargs)
        elif comm_type == 'mpi':
            self.comm = get_comm_backend(backend='mpi', **comm_kwargs)
        elif comm_type == 'torch':
            self.comm = get_comm_backend(backend='torch', **comm_kwargs)
        else:
            # Default to MPI
            self.comm = get_comm_backend(backend='auto', **comm_kwargs)
    
    def _log(self, msg: str, level: str = "info"):
        """Log message only on main process."""
        if self.comm.rank == 0:
            print(f"[DistributedTrainer] {msg}")
    
    def _init_qctn(self):
        """Initialize QCTN model on each process.

        If config.qctn is provided (a pre-built QCTN object), use it directly.
        This preserves user-set weights, requires_grad, hermit refs, etc.
        Otherwise, fall back to creating a new QCTN from config.qctn_graph.
        """
        from ...core.qctn import QCTN
        from ...utils.graph_generators import QCTNHelper

        if self.config.qctn is not None:
            # Use the pre-built QCTN directly (preserves weights + requires_grad)
            self.qctn = self.config.qctn
            self._log(f"Using pre-built QCTN: {self.qctn.nqubits} qubits, {len(self.qctn.cores)} cores")
            return

        qctn_graph = self.config.qctn_graph

        if qctn_graph is None:
            # Use default example graph based on num_qubits
            qctn_graph = QCTNHelper.generate_example_graph(n=self.config.num_qubits)
            self._log(f"Using default QCTN graph with {self.config.num_qubits} qubits")

        # Each process creates its own QCTN instance independently
        self.qctn = QCTN(qctn_graph, backend=self.engine.backend)
        self._log(f"QCTN initialized: {self.qctn.nqubits} qubits, {len(self.qctn.cores)} cores")
    
    def _sync_model_weights(self):
        """
        Synchronize model weights from main process to all workers.

        Note: This method is currently not used in the default initialization flow.
        Each process initializes QCTN independently and engine.init_distributed()
        handles partitioning. This method is kept for cases where explicit
        weight synchronization is needed (e.g., after modifying weights on rank 0).
        """
        from ...core.tn_tensor import TNTensor

        if self.comm.world_size == 1:
            return

        for core_name in self.qctn.cores:
            if core_name not in self.qctn.cores_weights:
                continue
            weight = self.qctn.cores_weights[core_name]

            # Handle TNTensor objects (use transparent proxy .is_leaf check)
            if isinstance(weight, TNTensor):
                synced_tensor = self.comm.broadcast_object(weight.tensor, src=0)
                synced_scale = self.comm.broadcast_object(weight.scale, src=0)
                self.qctn.cores_weights[core_name] = TNTensor(synced_tensor, synced_scale)
            else:
                synced_weight = self.comm.broadcast_object(weight, src=0)
                self.qctn.cores_weights[core_name] = synced_weight

        self._log("Model weights synchronized across workers")
    
    # ==================== Data Preparation ====================
    
    def prepare_data(self, N: int, B: int, K: int) -> Tuple[List[Dict], List]:
        """
        Prepare training data.
        
        Generates N batches of measurement matrices using Hermite polynomials.
        Data is generated on main process and broadcast to all workers.
        
        Args:
            N: Number of data batches
            B: Batch size (samples per batch)
            K: Hermite polynomial order
            
        Returns:
            (data_list, circuit_states_list)
        """
        import numpy as np
        
        backend = self.engine.backend
        D = self.qctn.nqubits
        
        data_list = []
        
        # Only main process generates data, then broadcast
        if self.comm.rank == 0:
            self._log(f"Generating data: N={N}, B={B}, D={D}, K={K}")
            
            for i in range(N):
                # Generate random data using numpy, then convert to backend tensor
                x_np = np.random.randn(B, D).astype(np.float32)
                x = backend.convert_to_tensor(x_np)
                Mx_list, _ = self.engine.generate_data(x, K=K)
                data_list.append({"measure_input_list": Mx_list})
        
        # Broadcast data list size
        n_batches = len(data_list) if self.comm.rank == 0 else 0
        n_batches = self.comm.broadcast_object(n_batches, src=0)
        
        # Broadcast each batch
        if self.comm.rank != 0:
            data_list = [None] * n_batches
        
        for i in range(n_batches):
            data_list[i] = self.comm.broadcast_object(data_list[i], src=0)
        
        # Generate circuit states
        circuit_states_list = [backend.zeros(K) for _ in range(D)]
        for s in circuit_states_list:
            s[-1] = 1.0
        
        self._log(f"Data prepared: {len(data_list)} batches")
        
        return data_list, circuit_states_list
    
    # ==================== Training ====================
    
    def train(self, 
              data_list: List[Dict], 
              circuit_states_list: List['torch.Tensor'],
              training_config: Optional[TrainingConfig] = None) -> TrainingStats:
        """
        Execute distributed training.
        
        Uses hierarchical contraction if world_size > 1:
        1. Each worker contracts its local subgraph
        2. log(n) reduction stages combine results using tensor parallel
        
        Args:
            data_list: Training data list
            circuit_states_list: Circuit states
            training_config: Training configuration (uses DistributedConfig if None)
            
        Returns:
            Training statistics
        """
        # Build training config from DistributedConfig if not provided
        if training_config is None:
            training_config = self.config.to_training_config()
        
        # Create data parallel trainer
        trainer = DataParallelTrainer(
            engine=self.engine._base_engine,  # Use base engine for now
            qctn=self.qctn,
            config=training_config,
            mpi_backend=self.comm  # Use our comm backend
        )
        
        # Execute training
        self._log("Starting distributed training...")
        stats = trainer.train(data_list, circuit_states_list)
        
        # Save final model if configured
        if self.config.save_final_model:
            self._save_final_model()
        
        return stats
    
    def train_distributed(self,
                          data_list: List[Dict],
                          circuit_states_list: List['torch.Tensor'],
                          num_epochs: int = None,
                          log_interval: int = None) -> Dict[str, Any]:
        """
        Execute distributed training using autograd through hierarchical contraction.
        
        This is the new training method that uses gradient-aware distributed
        contraction with SGDG optimizer.
        
        Args:
            data_list: Training data list (each item has 'measure_input_list')
            circuit_states_list: Circuit states for all qubits
            num_epochs: Number of epochs (default: config.max_steps)
            log_interval: Logging interval (default: config.log_interval)
            
        Returns:
            Training statistics dictionary
        """
        import time
        
        from ..optim.distributed_sgdg import DistributedSGDG, LRScheduler

        if num_epochs is None:
            num_epochs = self.config.max_steps
        if log_interval is None:
            log_interval = self.config.log_interval

        # Collect trainable params from local QCTN partition
        local_qctn = self.engine._local_qctn
        local_params = list(local_qctn.cores_weights.values())

        # Create distributed SGDG optimizer with params list (Phase 4.0 API)
        optimizer = DistributedSGDG(
            params=local_params,
            lr=self.config.learning_rate,
            momentum=self.config.momentum,
            stiefel=self.config.stiefel,
        )
        
        # Create LR scheduler if configured
        lr_scheduler = None
        if self.config.lr_schedule:
            lr_scheduler = LRScheduler(optimizer, self.config.lr_schedule)
        
        # Training loop
        total_loss = 0.0
        num_batches = 0
        start_time = time.time()
        loss_history = []
        
        self._log(f"Starting distributed training: {num_epochs} epochs, "
                  f"lr={self.config.learning_rate}, stiefel={self.config.stiefel}")
        
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            epoch_batches = 0
            
            for batch_idx, data in enumerate(data_list):
                measure_input_list = data.get('measure_input_list', data)
                
                # Execute training step
                loss = self.engine.train_step(
                    circuit_states_list=circuit_states_list,
                    measure_input_list=measure_input_list,
                    optimizer=optimizer,
                    measure_is_matrix=True,
                )
                
                epoch_loss += loss
                epoch_batches += 1
                total_loss += loss
                num_batches += 1
                
                # Update LR scheduler
                if lr_scheduler is not None:
                    lr_scheduler.step()
            
            # Log epoch summary
            avg_epoch_loss = epoch_loss / max(epoch_batches, 1)
            loss_history.append(avg_epoch_loss)
            
            if epoch % log_interval == 0 or epoch == num_epochs - 1:
                elapsed = time.time() - start_time
                if self.comm.rank == 0:
                    print(f"[Epoch {epoch}/{num_epochs}] "
                          f"loss={avg_epoch_loss:.6f}, lr={optimizer.lr:.6f}, "
                          f"elapsed={elapsed:.1f}s")
        
        # Final statistics
        elapsed = time.time() - start_time
        avg_loss = total_loss / max(num_batches, 1)
        
        stats = {
            'final_loss': avg_loss,
            'num_epochs': num_epochs,
            'num_batches': num_batches,
            'elapsed_time': elapsed,
            'loss_history': loss_history,
        }
        
        self._log(f"Training completed: final_loss={avg_loss:.6f}, "
                  f"elapsed={elapsed:.1f}s")
        
        # Save final model if configured
        if self.config.save_final_model:
            self._save_final_model()
        
        return stats

    # ==================== Trainer-compatible API (Phase 4.0) ====================

    def fit(
        self,
        target=None,
        loss=None,
        data_fn=None,
    ) -> List[float]:
        """Execute distributed training with single-process Trainer-compatible API.

        Data is managed through QCTN cores (embedded model + data) and optional
        ``data_fn``, exactly like single-process :class:`Trainer.fit`.

        - **Static mode**: QCTN cores are fixed, train to match ``target``.
        - **Dynamic data mode**: ``data_fn(step)`` updates data cores in-place
          before each forward pass.

        Uses ``EngineCommon`` for contraction and gradient computation (same as
        single-process Trainer), with distributed gradient sync for world_size > 1.

        Args:
            target: Learning target (float, tensor, QCTN, or None).
            loss: Loss specification — string name (``'mse'``, ``'nll'``, etc.),
                callable ``fn(result, target, backend)``, or ``BaseLoss`` instance.
            data_fn: Optional ``(step) -> None`` that updates data cores in the
                QCTN before each step.

        Returns:
            List of loss values, one per completed step.
        """
        import time

        from ..optim.distributed_sgdg import DistributedSGDG, LRScheduler
        from ...core.engine_common import EngineCommon

        max_steps = self.config.max_steps
        log_interval = self.config.log_interval

        # --- EngineCommon for contraction (data embedded in QCTN) ----------------
        engine_common = EngineCommon(
            backend=self.engine.backend,
            strategy_mode=self.config.strategy_mode,
        )

        # --- Collect trainable params from QCTN ---------------------------------
        params = self.qctn.parameters()

        # --- Create optimizer ----------------------------------------------------
        optimizer = DistributedSGDG(
            params=params,
            lr=self.config.learning_rate,
            momentum=self.config.momentum,
            stiefel=self.config.stiefel,
        )

        lr_scheduler = None
        if self.config.lr_schedule:
            lr_scheduler = LRScheduler(optimizer, self.config.lr_schedule)

        # --- Training loop -------------------------------------------------------
        loss_history: List[float] = []
        start_time = time.time()

        self._log(f"fit(): {max_steps} steps, lr={self.config.learning_rate}, "
                  f"stiefel={self.config.stiefel}")

        # Clear any stale strategy cache on the QCTN.
        # init_distributed() may have compiled a strategy with non-batch shapes;
        # data_fn will change the shapes, so we need a fresh compile.
        for attr in list(vars(self.qctn)):
            if attr.startswith('_compiled_strategy_'):
                delattr(self.qctn, attr)

        for step in range(1, max_steps + 1):
            # 1. Optional per-step data update (updates QCTN cores in-place)
            if data_fn is not None:
                data_fn(step)

            # 2. Forward + backward via EngineCommon (same as single-process)
            loss_tensor, grads = engine_common.contract_for_gradient(
                self.qctn, target=target, loss=loss,
            )

            # 3. Optimizer step
            optimizer.step(list(grads))

            # 4. LR schedule
            if lr_scheduler is not None:
                lr_scheduler.step()

            lv = float(loss_tensor)
            loss_history.append(lv)

            # 5. Logging
            if log_interval and (step % log_interval == 0 or step == 1):
                if self.comm.rank == 0:
                    lr_str = f"  lr={optimizer.lr:.1e}" if lr_scheduler else ""
                    print(f"  Step {step:4d}/{max_steps}  loss={lv:.6f}{lr_str}")

            # 6. Early stop
            if self.config.tol is not None and lv < self.config.tol:
                self._log(f"Converged at step {step} (loss={lv:.6f})")
                break

        elapsed = time.time() - start_time
        self._log(f"fit() completed: {len(loss_history)} steps, "
                  f"final_loss={loss_history[-1]:.6f}, elapsed={elapsed:.1f}s")

        if self.config.save_final_model:
            self._save_final_model()

        return loss_history

    # --- Internal helpers for fit() -------------------------------------------

    def _resolve_target(self, target):
        """Resolve target for distributed training."""
        if target is None:
            return None
        return target

    # ==================== Checkpointing ====================
    
    def _save_final_model(self):
        """Save final model (main process only)."""
        if self.comm.rank != 0:
            return
        
        try:
            model_path = self.checkpoint_dir / "final_model.safetensors"
            config_dict = self._raw_config if self._raw_config else {}
            self.qctn.save_cores(str(model_path), metadata={
                'config': json.dumps(config_dict)
            })
            self._log(f"Final model saved: {model_path}")
        except Exception as e:
            self._log(f"Warning: Could not save model: {e}", level="warn")
    
    def save_checkpoint(self, step: int, stats: Optional[TrainingStats] = None):
        """
        Save training checkpoint.
        
        Args:
            step: Current training step
            stats: Optional training stats
        """
        if self.comm.rank != 0:
            return
        
        checkpoint_path = self.checkpoint_dir / f"checkpoint_step_{step}.safetensors"
        
        config_dict = self._raw_config if self._raw_config else {}
        metadata = {
            'step': str(step),
            'config': json.dumps(config_dict),
        }
        
        if stats:
            metadata['final_loss'] = str(stats.final_loss)
        
        try:
            self.qctn.save_cores(str(checkpoint_path), metadata=metadata)
            self._log(f"Checkpoint saved: {checkpoint_path}")
        except Exception as e:
            self._log(f"Warning: Could not save checkpoint: {e}", level="warn")
    
    def load_checkpoint(self, checkpoint_path: str):
        """
        Load model from checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
        """
        from ...core.qctn import QCTN
        
        # Each process loads the checkpoint independently
        self.qctn = QCTN.from_pretrained(
            self.qctn.graph, 
            checkpoint_path, 
            backend=self.engine.backend
        )
        
        # Re-initialize distributed contraction plan
        # This will partition the graph and each process keeps only its local cores
        # if self.comm.world_size > 1:
        #     self._contract_plan = self.engine.init_distributed(self.qctn)
        
        self._contract_plan = self.engine.init_distributed(self.qctn)
        
        self._log(f"Loaded checkpoint: {checkpoint_path}")
    
    # ==================== Evaluation ====================
    
    def evaluate(self, data_list: List[Dict], 
                 circuit_states_list: List['torch.Tensor']) -> float:
        """
        Evaluate model on given data.
        
        Args:
            data_list: Evaluation data
            circuit_states_list: Circuit states
            
        Returns:
            Average loss
        """
        # Create temporary trainer for evaluation
        config = TrainingConfig(max_steps=0)
        trainer = DataParallelTrainer(
            engine=self.engine._base_engine,
            qctn=self.qctn,
            config=config,
            mpi_backend=self.comm
        )
        
        return trainer.evaluate(data_list, circuit_states_list)
    
    # ==================== Properties for Backward Compatibility ====================
    
    @property
    def mpi(self):
        """Alias for comm (backward compatibility)."""
        return self.comm
    
    @property
    def ctx(self):
        """Get distributed context."""
        return self.comm.get_context()


def main():
    """Command-line entry point for distributed training."""
    import argparse
    
    try:
        import yaml
        has_yaml = True
    except ImportError:
        has_yaml = False
    
    parser = argparse.ArgumentParser(description='TNEQ Distributed Training')
    parser.add_argument('--config', type=str, help='Config file path (YAML or JSON)')
    parser.add_argument('--backend', type=str, default='pytorch', help='Backend type')
    parser.add_argument('--max-steps', type=int, default=1000, help='Max training steps')
    parser.add_argument('--learning-rate', type=float, default=1e-2, help='Learning rate')
    parser.add_argument('--batch-size', type=int, default=128, help='Batch size')
    parser.add_argument('--data-batches', type=int, default=100, help='Number of data batches')
    parser.add_argument('--hermite-order', type=int, default=3, help='Hermite polynomial order')
    parser.add_argument('--num-qubits', type=int, default=4, help='Number of qubits')
    parser.add_argument('--partition-strategy', type=str, default='layer', 
                        choices=['layer', 'core', 'auto'], help='Partition strategy')
    
    args = parser.parse_args()
    
    # Load config
    if args.config:
        config_path = Path(args.config)
        if config_path.suffix in ['.yaml', '.yml'] and has_yaml:
            with open(config_path, 'r') as f:
                config_dict = yaml.safe_load(f)
        else:
            with open(config_path, 'r') as f:
                config_dict = json.load(f)
        config = DistributedConfig.from_dict(config_dict)
    else:
        config = DistributedConfig(
            backend_type=args.backend,
            max_steps=args.max_steps,
            learning_rate=args.learning_rate,
            num_qubits=args.num_qubits,
            partition_strategy=args.partition_strategy,
        )
    
    # Create trainer
    trainer = DistributedTrainer(config)
    
    # Prepare data
    data_list, circuit_states_list = trainer.prepare_data(
        N=args.data_batches,
        B=args.batch_size,
        K=args.hermite_order
    )
    
    # Train
    stats = trainer.train(data_list, circuit_states_list)
    
    if trainer.comm.rank == 0:
        print(f"\nTraining completed: {stats.to_dict()}")


if __name__ == "__main__":
    main()
