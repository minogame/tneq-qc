"""
MPI Communication Backend

Provides MPI-based communication primitives for distributed training:
- Point-to-point communication
- Collective operations (AllReduce, Broadcast, AllGather)
- Tensor serialization/deserialization
- Asynchronous communication support
"""

from __future__ import annotations
import numpy as np
from typing import List, Optional, Union, Any, TYPE_CHECKING
from dataclasses import dataclass
from enum import Enum

# Lazy import mpi4py to allow module to be imported without MPI
_MPI = None
_MPI_COMM_WORLD = None


def _get_mpi():
    """Lazy load MPI to avoid import errors when MPI is not installed."""
    global _MPI, _MPI_COMM_WORLD
    if _MPI is None:
        try:
            from mpi4py import MPI
            _MPI = MPI
            _MPI_COMM_WORLD = MPI.COMM_WORLD
        except ImportError:
            raise ImportError(
                "mpi4py is required for distributed training. "
                "Install with: pip install mpi4py"
            )
    return _MPI, _MPI_COMM_WORLD


class ReduceOp(Enum):
    """Reduction operation types for collective communications."""
    SUM = "SUM"
    AVG = "AVG"  # Custom: SUM then divide by world_size
    MAX = "MAX"
    MIN = "MIN"
    
    def to_mpi_op(self):
        """Convert to MPI operation."""
        MPI, _ = _get_mpi()
        mapping = {
            ReduceOp.SUM: MPI.SUM,
            ReduceOp.AVG: MPI.SUM,  # AVG uses SUM then divide
            ReduceOp.MAX: MPI.MAX,
            ReduceOp.MIN: MPI.MIN,
        }
        return mapping[self]


@dataclass
class DistributedContext:
    """Distributed computing context information."""
    world_size: int
    rank: int
    local_rank: int
    is_main_process: bool
    backend: str = "mpi"
    
    def __repr__(self):
        return f"DistributedContext(rank={self.rank}/{self.world_size}, main={self.is_main_process})"


class AsyncHandle:
    """Handle for asynchronous communication operations."""
    
    def __init__(self, requests: List, results: List[np.ndarray], 
                 op: ReduceOp, world_size: int):
        """
        Initialize async handle.
        
        Args:
            requests: List of MPI Request objects
            results: List of result buffers
            op: Reduction operation type
            world_size: Number of workers
        """
        self.requests = requests
        self.results = results
        self.op = op
        self.world_size = world_size
        self._completed = False
        self._tensors = None
    
    def wait(self) -> List['torch.Tensor']:
        """
        Wait for all communications to complete.
        
        Returns:
            List of result tensors
        """
        if self._completed:
            return self._tensors
        
        import torch
        MPI, _ = _get_mpi()
        
        MPI.Request.Waitall(self.requests)
        
        tensors = []
        for result in self.results:
            if self.op == ReduceOp.AVG:
                result = result / self.world_size
            tensors.append(torch.from_numpy(result.copy()))
        
        self._tensors = tensors
        self._completed = True
        return tensors
    
    def is_completed(self) -> bool:
        """Check if all communications are completed."""
        if self._completed:
            return True
        return all(req.Test() for req in self.requests)


class MPIBackend:
    """
    MPI Communication Backend.
    
    Encapsulates MPI communication primitives and provides tensor-level
    communication interfaces for distributed training.
    
    Example:
        >>> mpi = MPIBackend()
        >>> if mpi.is_main_process():
        ...     print("Hello from main process")
        >>> 
        >>> # AllReduce gradients
        >>> local_grad = torch.randn(10, 10)
        >>> global_grad = mpi.allreduce(local_grad, op=ReduceOp.AVG)
    """
    
    def __init__(self, comm=None):
        """
        Initialize MPI backend.
        
        Args:
            comm: MPI communicator. If None, uses MPI.COMM_WORLD.
        """
        MPI, COMM_WORLD = _get_mpi()
        
        self.comm = comm if comm is not None else COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.world_size = self.comm.Get_size()
        
        # Local rank (within the same node)
        # For simple case, assume one process per node
        self.local_rank = self.rank
        
        self.context = DistributedContext(
            world_size=self.world_size,
            rank=self.rank,
            local_rank=self.local_rank,
            is_main_process=(self.rank == 0)
        )
        
        self._initialized = True
    
    def get_context(self) -> DistributedContext:
        """Get the distributed context."""
        return self.context
    
    def is_main_process(self) -> bool:
        """Check if this is the main (rank 0) process."""
        return self.rank == 0
    
    # ==================== Basic Communications ====================
    
    def barrier(self):
        """Global synchronization barrier."""
        self.comm.Barrier()
    
    def broadcast(self, tensor: 'torch.Tensor', src: int = 0) -> 'torch.Tensor':
        """
        Broadcast a tensor from source to all processes.
        
        Args:
            tensor: Tensor to broadcast (valid on src process)
            src: Source process rank
            
        Returns:
            Broadcasted tensor (same on all processes)
        """
        import torch
        
        # Broadcast shape and dtype first
        if self.rank == src:
            data = tensor.detach().cpu().contiguous().numpy()
            shape = data.shape
            dtype_str = str(data.dtype)
        else:
            shape = None
            dtype_str = None
        
        shape = self.comm.bcast(shape, root=src)
        dtype_str = self.comm.bcast(dtype_str, root=src)
        dtype = np.dtype(dtype_str)
        
        if self.rank != src:
            data = np.empty(shape, dtype=dtype)
        
        self.comm.Bcast(data, root=src)
        
        return torch.from_numpy(data.copy())
    
    def broadcast_object(self, obj: Any, src: int = 0) -> Any:
        """
        Broadcast a Python object from source to all processes.
        
        Args:
            obj: Object to broadcast (valid on src process)
            src: Source process rank
            
        Returns:
            Broadcasted object (same on all processes)
        """
        return self.comm.bcast(obj, root=src)
    
    def allreduce(self, tensor: 'torch.Tensor', op: ReduceOp = ReduceOp.SUM) -> 'torch.Tensor':
        """
        AllReduce operation on a tensor.
        
        All processes contribute their tensor and receive the reduced result.
        
        Args:
            tensor: Input tensor
            op: Reduction operation type
            
        Returns:
            Reduced tensor (same on all processes)
        """
        import torch
        
        data = tensor.detach().cpu().contiguous().numpy()
        result = np.zeros_like(data)
        
        self.comm.Allreduce(data, result, op=op.to_mpi_op())
        
        if op == ReduceOp.AVG:
            result = result / self.world_size
        
        # Preserve original device
        result_tensor = torch.from_numpy(result)
        if tensor.device.type != 'cpu':
            result_tensor = result_tensor.to(tensor.device)
        
        return result_tensor
    
    def allreduce_inplace(self, tensor: 'torch.Tensor', op: ReduceOp = ReduceOp.SUM):
        """
        In-place AllReduce operation to reduce memory allocation.
        
        Args:
            tensor: Input/output tensor (will be modified)
            op: Reduction operation type
        """
        import torch
        MPI, _ = _get_mpi()
        
        data = tensor.detach().cpu().contiguous().numpy()
        
        self.comm.Allreduce(MPI.IN_PLACE, data, op=op.to_mpi_op())
        
        if op == ReduceOp.AVG:
            data = data / self.world_size
        
        tensor.copy_(torch.from_numpy(data))
    
    def allgather(self, tensor: 'torch.Tensor') -> List['torch.Tensor']:
        """
        AllGather operation: gather tensors from all processes.
        
        Args:
            tensor: Local tensor
            
        Returns:
            List of tensors from all processes
        """
        import torch
        
        data = tensor.detach().cpu().contiguous().numpy()
        all_data = self.comm.allgather(data)
        return [torch.from_numpy(d.copy()) for d in all_data]
    
    def reduce_scatter(self, tensor: 'torch.Tensor', op: ReduceOp = ReduceOp.SUM) -> 'torch.Tensor':
        """
        ReduceScatter operation (useful for tensor parallelism).
        
        Each process receives a portion of the reduced result.
        
        Args:
            tensor: Input tensor (size should be divisible by world_size)
            op: Reduction operation type
            
        Returns:
            This process's portion of the reduced result
        """
        import torch
        
        data = tensor.detach().cpu().contiguous().numpy()
        chunk_size = data.size // self.world_size
        
        result = np.zeros(chunk_size, dtype=data.dtype)
        
        self.comm.Reduce_scatter(data, result, op=op.to_mpi_op())
        
        if op == ReduceOp.AVG:
            result = result / self.world_size
        
        return torch.from_numpy(result)
    
    # ==================== Tensor List Communications ====================
    
    def allreduce_tensors(self, tensors: List['torch.Tensor'], 
                          op: ReduceOp = ReduceOp.AVG) -> List['torch.Tensor']:
        """
        AllReduce a list of tensors (e.g., gradients).
        
        Args:
            tensors: List of tensors
            op: Reduction operation type
            
        Returns:
            List of reduced tensors
        """
        return [self.allreduce(t, op) for t in tensors]
    
    def allreduce_tensors_async(self, tensors: List['torch.Tensor'], 
                                 op: ReduceOp = ReduceOp.AVG) -> AsyncHandle:
        """
        Asynchronous AllReduce for communication-computation overlap.
        
        Args:
            tensors: List of tensors
            op: Reduction operation type
            
        Returns:
            AsyncHandle that can be waited on
        """
        MPI, _ = _get_mpi()
        
        requests = []
        results = []
        
        for tensor in tensors:
            data = tensor.detach().cpu().contiguous().numpy()
            result = np.zeros_like(data)
            results.append(result)
            
            req = self.comm.Iallreduce(data, result, op=op.to_mpi_op())
            requests.append(req)
        
        return AsyncHandle(requests, results, op, self.world_size)
    
    # ==================== Scalar Communications ====================
    
    def allreduce_scalar(self, value: float, op: ReduceOp = ReduceOp.SUM) -> float:
        """
        AllReduce a scalar value.
        
        Args:
            value: Local scalar value
            op: Reduction operation type
            
        Returns:
            Reduced scalar value
        """
        local = np.array([value])
        result = np.zeros(1)
        
        self.comm.Allreduce(local, result, op=op.to_mpi_op())
        
        if op == ReduceOp.AVG:
            result[0] /= self.world_size
        
        return float(result[0])
    
    # ==================== Point-to-Point Communications ====================
    
    def send(self, tensor: 'torch.Tensor', dest: int, tag: int = 0):
        """
        Send a tensor to a destination process.
        
        Args:
            tensor: Tensor to send
            dest: Destination rank
            tag: Message tag
        """
        data = tensor.detach().cpu().contiguous().numpy()
        self.comm.Send(data, dest=dest, tag=tag)
    
    def recv(self, shape: tuple, dtype: np.dtype, src: int, tag: int = 0) -> 'torch.Tensor':
        """
        Receive a tensor from a source process.
        
        Args:
            shape: Expected tensor shape
            dtype: Expected dtype
            src: Source rank
            tag: Message tag
            
        Returns:
            Received tensor
        """
        import torch
        
        data = np.empty(shape, dtype=dtype)
        self.comm.Recv(data, source=src, tag=tag)
        return torch.from_numpy(data)
    
    def isend(self, tensor: 'torch.Tensor', dest: int, tag: int = 0):
        """
        Non-blocking send.
        
        Args:
            tensor: Tensor to send
            dest: Destination rank
            tag: Message tag
            
        Returns:
            MPI Request object
        """
        data = tensor.detach().cpu().contiguous().numpy()
        return self.comm.Isend(data, dest=dest, tag=tag)
    
    def irecv(self, shape: tuple, dtype: np.dtype, src: int, tag: int = 0):
        """
        Non-blocking receive.
        
        Args:
            shape: Expected tensor shape
            dtype: Expected dtype
            src: Source rank
            tag: Message tag
            
        Returns:
            Tuple of (buffer, MPI Request)
        """
        data = np.empty(shape, dtype=dtype)
        req = self.comm.Irecv(data, source=src, tag=tag)
        return data, req


class MockMPIBackend:
    """
    Mock MPI backend for testing without MPI.
    
    Simulates single-process MPI behavior.
    """
    
    def __init__(self):
        self.rank = 0
        self.world_size = 1
        self.local_rank = 0
        self.context = DistributedContext(
            world_size=1,
            rank=0,
            local_rank=0,
            is_main_process=True
        )
    
    def get_context(self) -> DistributedContext:
        return self.context
    
    def is_main_process(self) -> bool:
        return True
    
    def barrier(self):
        pass
    
    def broadcast(self, tensor: 'torch.Tensor', src: int = 0) -> 'torch.Tensor':
        return tensor.clone()
    
    def broadcast_object(self, obj: Any, src: int = 0) -> Any:
        return obj
    
    def allreduce(self, tensor: 'torch.Tensor', op: ReduceOp = ReduceOp.SUM) -> 'torch.Tensor':
        return tensor.clone()
    
    def allreduce_inplace(self, tensor: 'torch.Tensor', op: ReduceOp = ReduceOp.SUM):
        pass
    
    def allgather(self, tensor: 'torch.Tensor') -> List['torch.Tensor']:
        return [tensor.clone()]
    
    def allreduce_tensors(self, tensors: List['torch.Tensor'], 
                          op: ReduceOp = ReduceOp.AVG) -> List['torch.Tensor']:
        return [t.clone() for t in tensors]
    
    def allreduce_scalar(self, value: float, op: ReduceOp = ReduceOp.SUM) -> float:
        return value


def get_backend(use_mpi: bool = True) -> Union[MPIBackend, MockMPIBackend]:
    """
    Get the appropriate backend based on environment.
    
    Args:
        use_mpi: Whether to use real MPI backend
        
    Returns:
        MPIBackend or MockMPIBackend
    """
    if use_mpi:
        try:
            return MPIBackend()
        except ImportError:
            print("Warning: mpi4py not available, falling back to MockMPIBackend")
            return MockMPIBackend()
    else:
        return MockMPIBackend()
