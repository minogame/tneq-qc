"""Distributed Engine — inherits EngineCommon."""

from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any, Union, TYPE_CHECKING

import numpy as np

from ..comm import CommBase, get_comm_backend, ReduceOp
from ...core.engine_common import EngineCommon

if TYPE_CHECKING:
    import torch
    from ...core.qctn import QCTN
    from ...backends.backend_interface import ComputeBackend

from ...core.tn_tensor import TNTensor

local_debug = False

def local_print(*args, **kwargs):
    if local_debug:
        print(*args, **kwargs)


@dataclass
class PartitionConfig:
    """Configuration for QCTN graph partitioning."""
    
    # Partitioning strategy: 'layer' (by layers/stages) or 'core' (by cores)
    strategy: str = 'layer'
    
    # Number of partitions (usually equals world_size)
    num_partitions: int = 1
    
    # Minimum cores per partition
    min_cores_per_partition: int = 1
    
    # Whether to balance partition sizes
    balance_partitions: bool = True


@dataclass
class ContractStage:
    """Information about a contraction stage."""
    
    stage_idx: int
    stage_type: str  # 'local' or 'reduce'
    
    # For 'local' stage: which cores to contract
    local_cores: List[str] = field(default_factory=list)
    
    # For 'reduce' stage: group info
    group_ranks: List[int] = field(default_factory=list)  # ranks in this group
    group_size: int = 1
    is_group_leader: bool = False  # True if this rank should send result
    partner_rank: int = -1  # rank to receive from or send to
    
    def __str__(self) -> str:
        lines = [f"  Stage {self.stage_idx} ({self.stage_type}):"]
        if self.stage_type == 'local':
            cores_str = ', '.join(self.local_cores[:5])
            if len(self.local_cores) > 5:
                cores_str += f", ... (+{len(self.local_cores) - 5} more)"
            lines.append(f"    Local cores: [{cores_str}]")
        else:
            lines.append(f"    Group size: {self.group_size}")
            if self.group_ranks:
                lines.append(f"    Group ranks: {self.group_ranks}")
            if self.partner_rank >= 0:
                lines.append(f"    Partner rank: {self.partner_rank}")
            lines.append(f"    Is group leader: {self.is_group_leader}")
        return '\n'.join(lines)


@dataclass
class DistributedContractPlan:
    """Execution plan for distributed contraction."""
    
    # Total number of stages
    num_stages: int
    
    # List of stages to execute
    stages: List[ContractStage] = field(default_factory=list)
    
    # Partition info for this rank
    local_partition_idx: int = 0
    local_cores: List[str] = field(default_factory=list)
    
    # Inter-node contraction graph (computed by master)
    inter_node_graph: Optional[Dict[str, Any]] = None
    
    def __str__(self) -> str:
        lines = [
            "=" * 60,
            "DistributedContractPlan",
            "=" * 60,
            f"Partition Index: {self.local_partition_idx}",
            f"Total Stages: {self.num_stages}",
        ]
        
        # Local cores summary
        cores_str = ', '.join(self.local_cores[:5])
        if len(self.local_cores) > 5:
            cores_str += f", ... (+{len(self.local_cores) - 5} more)"
        lines.append(f"Local Cores ({len(self.local_cores)}): [{cores_str}]")
        
        # Stages
        lines.append("-" * 40)
        lines.append("Stages:")
        for stage in self.stages:
            lines.append(str(stage))
        
        # Inter-node graph summary
        if self.inter_node_graph:
            lines.append("-" * 40)
            lines.append("Inter-Node Graph:")
            lines.append(f"  Num partitions: {self.inter_node_graph.get('num_partitions', 'N/A')}")
            lines.append(f"  Partition sizes: {self.inter_node_graph.get('partition_sizes', [])}")
            
            cross_edges = self.inter_node_graph.get('cross_edges', [])
            lines.append(f"  Cross edges: {len(cross_edges)}")
            if cross_edges and len(cross_edges) <= 5:
                for edge in cross_edges:
                    lines.append(f"    {edge['from_core']} -> {edge['to_core']} (rank={edge['edge_rank']} qubit={edge['qubit_idx']})")
            elif cross_edges:
                for edge in cross_edges[:3]:
                    lines.append(f"    {edge['from_core']} -> {edge['to_core']} (rank={edge['edge_rank']} qubit={edge['qubit_idx']})")
                lines.append(f"    ... (+{len(cross_edges) - 3} more edges)")
            
            # Partition adjacency tables summary
            partition_tables = self.inter_node_graph.get('partition_adjacency_tables', [])
            if partition_tables:
                lines.append(f"  Partition adjacency tables: {len(partition_tables)} partitions")
                for p_idx, table in enumerate(partition_tables):
                    # Count out_edge_list
                    out_internal_edges = sum(
                        1 for entry in table 
                        for e in entry.get('out_edge_list', []) 
                        if not e.get('is_cross_partition', False) and e.get('neighbor_name')
                    )
                    out_cross_edges = sum(
                        1 for entry in table 
                        for e in entry.get('out_edge_list', []) 
                        if e.get('is_cross_partition', False)
                    )
                    # Count in_edge_list
                    in_internal_edges = sum(
                        1 for entry in table 
                        for e in entry.get('in_edge_list', []) 
                        if not e.get('is_cross_partition', False) and e.get('neighbor_name')
                    )
                    in_cross_edges = sum(
                        1 for entry in table 
                        for e in entry.get('in_edge_list', []) 
                        if e.get('is_cross_partition', False)
                    )
                    in_external = sum(
                        1 for entry in table 
                        for e in entry.get('in_edge_list', []) 
                        if not e.get('neighbor_name')
                    )
                    out_external = sum(
                        1 for entry in table 
                        for e in entry.get('out_edge_list', []) 
                        if not e.get('neighbor_name')
                    )
                    lines.append(f"    P{p_idx}: {len(table)} cores | "
                                 f"in: {in_internal_edges} internal, {in_cross_edges} cross, {in_external} ext | "
                                 f"out: {out_internal_edges} internal, {out_cross_edges} cross, {out_external} ext")
        
        lines.append("=" * 60)
        return '\n'.join(lines)
    
    def __repr__(self) -> str:
        return self.__str__()


class EngineDistributed(EngineCommon):
    """Distributed engine — inherits EngineCommon, adds partitioning and reduce.

    Usage::

        engine = EngineDistributed(backend='pytorch', comm=comm)
        engine.init_distributed(combined)
        result = engine.contract(combined)
    """

    def __init__(self,
                 backend=None,
                 strategy: Optional[str] = None,
                 strategy_mode: Optional[str] = None,
                 comm: Optional[CommBase] = None,
                 partition_config: Optional[PartitionConfig] = None,
                 comm_timeout: float = 300.0):
        super().__init__(backend=backend, strategy=strategy, strategy_mode=strategy_mode)

        self.comm = comm or get_comm_backend(self.backend.backend_info.backend_type)

        self.rank = self.comm.rank
        self.world_size = self.comm.world_size
        self.comm_timeout = comm_timeout

        self.partition_config = partition_config or PartitionConfig(
            num_partitions=self.world_size
        )

        self._is_initialized = False
        self._qctn: Optional['QCTN'] = None
        self._local_qctn: Optional['QCTN'] = None
        self._contract_plan: Optional[DistributedContractPlan] = None
        self._current_partitions = set()
        self._current_boundary_edges: List[Dict[str, Any]] = []

        self.num_stages = math.ceil(math.log2(self.world_size)) + 1 if self.world_size > 1 else 1

        import torch.distributed as dist
        self._pg_cache = {}
        for stage_idx in range(self.num_stages):
            for j in range(0, self.world_size, 2**stage_idx):
                group = list(range(j, min(j + 2**stage_idx, self.world_size)))
                dist_group = dist.new_group(group)
                self._pg_cache[tuple(group)] = dist_group        
        
    
    def _log(self, msg: str, level: str = "info"):
        """Log message only on main process."""
        if self.rank == 0:
            local_print(f"[DistributedEngine] {msg}")
    
    def check_comm_health(self, timeout: float = 5.0) -> bool:
        """Check communication health via allreduce."""
        import torch
        import time
        try:
            test_tensor = torch.tensor([self.rank], dtype=torch.float32,
                                      device=self.backend.backend_info.device)
            start_time = time.time()
            result_list = self.comm.allgather(test_tensor)
            elapsed = time.time() - start_time
            expected_sum = sum(range(self.world_size))
            actual_sum = sum([t.item() for t in result_list])
            if abs(actual_sum - expected_sum) < 1e-6:
                local_print(f"[Rank {self.rank}] Comm health check passed ({elapsed:.3f}s)")
                return True
            return False
        except Exception as e:
            print(f"[Rank {self.rank}] ERROR: Comm health check failed: {e}")
            return False

    def init_distributed(self, qctn: 'QCTN', partitions=None) -> DistributedContractPlan:
        """Partition QCTN graph and build contraction plan."""
        self._qctn = qctn

        partitions = self._partition_qctn(qctn, partitions=partitions)
        contract_plan = self._compute_contract_plan(qctn, partitions)
        
        # local_print(f"[Rank {self.rank}] Distributed contraction plan computed.")
        # local_print(f"[Rank {self.rank}] Partitions: {partitions}")
        # local_print(f"[Rank {self.rank}] Contract Plan: \n{contract_plan}")

        # Step 2: Extract this worker's local partition based on rank
        local_cores = partitions[self.rank] if self.rank < len(partitions) else []
        
        # Update plan with local info
        contract_plan.local_partition_idx = self.rank
        contract_plan.local_cores = local_cores
        self._contract_plan = contract_plan
        
        # Step 3: Create local QCTN (subgraph) and load corresponding weights
        if local_cores:
            self._local_qctn = self._create_local_qctn(qctn, local_cores)
        else:
            self._local_qctn = None

        # self._print_local_qctn(self._local_qctn)
        
        self._is_initialized = True
        
        self._log(f"Distributed init complete: {len(local_cores)} local cores, "
                  f"{contract_plan.num_stages} stages")
        
        return contract_plan
    
    def _partition_qctn(self, qctn: 'QCTN', partitions=None) -> List[List[str]]:
        """
        Partition QCTN cores across workers.
        
        Uses a simple layer-based partitioning strategy:
        - Cores are ordered by their topological order
        - Cores are distributed evenly across partitions
        
        Args:
            qctn: QCTN to partition
            
        Returns:
            List of core name lists, one per partition
        """
        num_partitions = self.partition_config.num_partitions
        cores = qctn.cores  # Already sorted
        ncores = len(cores)

        if partitions is not None:
            self._log(f"Using provided partition dict for partitioning.")
            return partitions
        
        if ncores < num_partitions:
            # More partitions than cores: some partitions will be empty
            partitions = [[cores[i]] if i < ncores else [] for i in range(num_partitions)]
        else:
            # Distribute cores evenly
            base_size = ncores // num_partitions
            remainder = ncores % num_partitions
            
            partitions = []
            idx = 0
            for i in range(num_partitions):
                size = base_size + (1 if i < remainder else 0)
                partitions.append(cores[idx:idx + size])
                idx += size
        
        self._log(f"Partitioned {ncores} cores into {num_partitions} partitions: "
                  f"{[len(p) for p in partitions]}")
        
        return partitions
    
    def _compute_contract_plan(self, qctn: 'QCTN', 
                                partitions: List[List[str]]) -> DistributedContractPlan:
        """
        Compute the hierarchical contraction plan.
        
        For n workers, we have log2(n) + 1 stages:
        - Stage 0: Local contraction within each partition
        - Stages 1 to log2(n): Hierarchical reduction
        
        Args:
            qctn: Full QCTN
            partitions: Core assignments per partition
            
        Returns:
            DistributedContractPlan
        """
        n = self.world_size
        num_reduce_stages = int(math.ceil(math.log2(n))) if n > 1 else 0
        num_stages = 1 + num_reduce_stages  # 1 for local + reduction stages
        
        stages = []
        
        # Stage 0: Local contraction
        stage0 = ContractStage(
            stage_idx=0,
            stage_type='local',
            local_cores=partitions[0] if partitions else [],  # Will be updated per-rank
        )
        stages.append(stage0)
        
        # Reduction stages
        for s in range(1, num_stages):
            # Group size doubles each stage: 2, 4, 8, ...
            group_size = 2 ** s
            
            # Determine which group this rank belongs to
            # Not needed here as this is master-side; each rank computes its own
            
            stage = ContractStage(
                stage_idx=s,
                stage_type='reduce',
                group_size=group_size,
            )
            stages.append(stage)
        
        # Compute inter-node graph edges (which partitions need to contract together)
        inter_node_graph = self._compute_inter_node_graph(qctn, partitions)
        
        plan = DistributedContractPlan(
            num_stages=num_stages,
            stages=stages,
            inter_node_graph=inter_node_graph,
        )
        
        return plan
    
    def _compute_inter_node_graph(self, qctn: 'QCTN', 
                                   partitions: List[List[str]]) -> Dict[str, Any]:
        """
        Compute the inter-node contraction graph.
        
        This determines which edges connect different partitions and how
        they should be contracted in the reduction stages.
        
        Args:
            qctn: Full QCTN
            partitions: Core assignments per partition
            
        Returns:
            Dict containing:
            - raw_cross_edges: original cross-partition edges with core names
            - cross_edges: cross-partition edges with partition+index naming
            - partition_adjacency_tables: adjacency table for each partition
            - num_partitions: number of partitions
            - partition_sizes: size of each partition
        """
        # Map each core to its partition and index within partition
        core_to_partition = {}
        core_to_partition_idx = {}
        for p_idx, cores in enumerate(partitions):
            for local_idx, core in enumerate(cores):
                core_to_partition[core] = p_idx
                core_to_partition_idx[core] = local_idx
        
        # Build partition adjacency tables
        partition_adjacency_tables = self._build_partition_adjacency_tables(
            qctn, partitions, core_to_partition, core_to_partition_idx
        )
        
        # Find edges that cross partition boundaries
        raw_cross_edges = []
        cross_edges = []
        
        for core_info in qctn.adjacency_table:
            core_name = core_info['core_name']
            core_partition = core_to_partition.get(core_name, -1)
            core_local_idx = core_to_partition_idx.get(core_name, -1)
            
            for edge in core_info['out_edge_list']:
                neighbor_name = edge['neighbor_name']
                if neighbor_name and neighbor_name in core_to_partition:
                    neighbor_partition = core_to_partition[neighbor_name]
                    neighbor_local_idx = core_to_partition_idx[neighbor_name]
                    
                    if core_partition != neighbor_partition:
                        # Raw cross edge with original core names
                        raw_cross_edges.append({
                            'from_core': core_name,
                            'to_core': neighbor_name,
                            'from_partition': core_partition,
                            'to_partition': neighbor_partition,
                            'edge_rank': edge['edge_rank'],
                            'qubit_idx': edge['qubit_idx'],
                        })
                        
                        # Cross edge with partition naming (P0, P1, etc.)
                        cross_edges.append({
                            'from_core': f"P{core_partition}",
                            'to_core': f"P{neighbor_partition}",
                            'from_partition': core_partition,
                            'to_partition': neighbor_partition,
                            'from_core_idx': core_local_idx,
                            'to_core_idx': neighbor_local_idx,
                            'edge_rank': edge['edge_rank'],
                            'qubit_idx': edge['qubit_idx'],
                            # Keep original names for reference
                            'from_core_raw': core_name,
                            'to_core_raw': neighbor_name,
                        })
        
        return {
            'raw_cross_edges': raw_cross_edges,
            'cross_edges': cross_edges,
            'partition_adjacency_tables': partition_adjacency_tables,
            'num_partitions': len(partitions),
            'partition_sizes': [len(p) for p in partitions],
        }
    
    def _build_partition_adjacency_tables(
        self, 
        qctn: 'QCTN', 
        partitions: List[List[str]],
        core_to_partition: Dict[str, int],
        core_to_partition_idx: Dict[str, int]
    ) -> List[List[Dict[str, Any]]]:
        """
        Build adjacency table for each partition.
        
        Each partition's adjacency table contains entries for its local cores,
        with edges pointing to either:
        - Other local cores (internal edges)
        - External cores from other partitions (cross edges, marked with partition info)
        
        Args:
            qctn: Full QCTN
            partitions: Core assignments per partition
            core_to_partition: Mapping from core name to partition index
            core_to_partition_idx: Mapping from core name to index within partition
            
        Returns:
            List of adjacency tables, one per partition
        """
        partition_adjacency_tables = []
        
        for p_idx, partition_cores in enumerate(partitions):
            partition_table = []
            partition_core_set = set(partition_cores)
            
            for local_idx, core_name in enumerate(partition_cores):
                # Find original core info
                original_info = None
                for info in qctn.adjacency_table:
                    if info['core_name'] == core_name:
                        original_info = info
                        break
                
                if original_info is None:
                    continue
                
                # Build local adjacency entry using original core names
                local_entry = {
                    'core_idx': local_idx,
                    'core_name': core_name,  # Use original core name
                    'in_edge_list': [],
                    'out_edge_list': [],
                    'input_shape': original_info['input_shape'].copy(),
                    'output_shape': original_info['output_shape'].copy(),
                    'input_dim': original_info['input_dim'],
                    'output_dim': original_info['output_dim'],
                }
                
                # Process in_edge_list
                for edge in original_info['in_edge_list']:
                    neighbor_name = edge['neighbor_name']
                    if not neighbor_name:
                        # Input from circuit (external input)
                        local_entry['in_edge_list'].append({
                            'neighbor_idx': -1,
                            'neighbor_name': "",
                            'edge_rank': edge['edge_rank'],
                            'qubit_idx': edge['qubit_idx'],
                            'is_cross_partition': False,
                        })
                    elif neighbor_name in partition_core_set:
                        # Internal edge within partition
                        neighbor_local_idx = core_to_partition_idx[neighbor_name]
                        local_entry['in_edge_list'].append({
                            'neighbor_idx': neighbor_local_idx,
                            'neighbor_name': neighbor_name,  # Use original name
                            'edge_rank': edge['edge_rank'],
                            'qubit_idx': edge['qubit_idx'],
                            'is_cross_partition': False,
                        })
                    else:
                        # Cross-partition edge
                        neighbor_partition = core_to_partition.get(neighbor_name, -1)
                        neighbor_local_idx = core_to_partition_idx.get(neighbor_name, -1)
                        local_entry['in_edge_list'].append({
                            # 'neighbor_idx': neighbor_local_idx,
                            'neighbor_idx': -1,
                            'neighbor_name': neighbor_name,  # Use original name
                            'neighbor_partition': neighbor_partition,
                            'edge_rank': edge['edge_rank'],
                            'qubit_idx': edge['qubit_idx'],
                            'is_cross_partition': True,
                        })
                
                # Process out_edge_list
                for edge in original_info['out_edge_list']:
                    neighbor_name = edge['neighbor_name']
                    if not neighbor_name:
                        # Output to circuit (external output)
                        local_entry['out_edge_list'].append({
                            'neighbor_idx': -1,
                            'neighbor_name': "",
                            'edge_rank': edge['edge_rank'],
                            'qubit_idx': edge['qubit_idx'],
                            'is_cross_partition': False,
                        })
                    elif neighbor_name in partition_core_set:
                        # Internal edge within partition
                        neighbor_local_idx = core_to_partition_idx[neighbor_name]
                        local_entry['out_edge_list'].append({
                            'neighbor_idx': neighbor_local_idx,
                            'neighbor_name': neighbor_name,  # Use original name
                            'edge_rank': edge['edge_rank'],
                            'qubit_idx': edge['qubit_idx'],
                            'is_cross_partition': False,
                        })
                    else:
                        # Cross-partition edge
                        neighbor_partition = core_to_partition.get(neighbor_name, -1)
                        neighbor_local_idx = core_to_partition_idx.get(neighbor_name, -1)
                        local_entry['out_edge_list'].append({
                            # 'neighbor_idx': neighbor_local_idx,
                            'neighbor_idx': -1,
                            'neighbor_name': neighbor_name,  # Use original name
                            'neighbor_partition': neighbor_partition,
                            'edge_rank': edge['edge_rank'],
                            'qubit_idx': edge['qubit_idx'],
                            'is_cross_partition': True,
                        })
                
                partition_table.append(local_entry)
            
            partition_adjacency_tables.append(partition_table)
        
        return partition_adjacency_tables
    
    def _create_local_qctn(self, qctn: 'QCTN', local_cores: List[str]) -> 'QCTN':
        """
        Create a local QCTN subgraph containing only the specified cores.
        
        Each process retrieves its partition's adjacency table from the contract plan
        and extracts only the local cores' weights from the full QCTN.
        Creates a new QCTN instance with the local partition's data.
        
        Args:
            qctn: Full QCTN (with all cores initialized)
            local_cores: List of core names for this partition
            
        Returns:
            QCTN instance with:
            - adjacency_table: Local partition's adjacency table (using original core names)
            - cores_weights: Dict mapping core names to their weights
            - cores: List of core names (original naming)
        """
        from ...core.qctn import QCTN
        
        # Get partition adjacency table from contract plan
        partition_tables = self._contract_plan.inter_node_graph.get('partition_adjacency_tables', [])
        if self.rank < len(partition_tables):
            local_adjacency_table = partition_tables[self.rank]
        else:
            local_adjacency_table = []
        
        # Extract core names and weights from adjacency table
        # Now core_name is the original name
        local_weights = {}
        local_cores_list = []  # Original naming
        qubit_indices_set = set()  # Collect all qubit indices involved in this partition
        
        for entry in local_adjacency_table:
            core_name = entry.get('core_name', '')
            
            if core_name and core_name in qctn.cores_weights:
                local_weights[core_name] = qctn.cores_weights[core_name]
                local_cores_list.append(core_name)
            
            # Collect qubit indices from in_edge_list and out_edge_list
            for e in entry.get('in_edge_list', []):
                qubit_idx = e.get('qubit_idx', -1)
                if qubit_idx >= 0:
                    qubit_indices_set.add(qubit_idx)
            for e in entry.get('out_edge_list', []):
                qubit_idx = e.get('qubit_idx', -1)
                if qubit_idx >= 0:
                    qubit_indices_set.add(qubit_idx)
        
        # Sort qubit indices to maintain order
        qubit_indices = sorted(qubit_indices_set)

        if self.rank != 0:
            # reverse qubit_indices
            qubit_indices = qubit_indices[::-1]
        
        # Create a new QCTN-like object with local partition data
        # We use object.__new__ to avoid calling __init__ which parses a graph string
        local_qctn = object.__new__(QCTN)
        
        # Set essential attributes
        local_qctn.graph = None  # No graph string for partitioned QCTN
        local_qctn.nqubits = qctn.nqubits  # Keep original nqubits for reference
        local_qctn.cores = local_cores_list  # Original core names
        local_qctn.ncores = len(local_cores_list)
        local_qctn.adjacency_table = local_adjacency_table
        local_qctn.cores_weights = local_weights
        local_qctn.backend = qctn.backend
        local_qctn.einsum_expr = None
        local_qctn._loaded_metadata = None
        
        # Store partition-specific info as extra attribute
        local_qctn.partition_idx = self.rank
        local_qctn.qubit_indices = qubit_indices  # All qubit indices involved in this partition
        
        self._log(f"Rank {self.rank}: Local QCTN created with {len(local_cores_list)} cores: "
                  f"{local_cores_list}, qubits: {qubit_indices}")
        
        return local_qctn
    
    def _print_local_qctn(self, local_qctn: 'QCTN'):
        """
        Print formatted local QCTN information.
        
        Args:
            local_qctn: Local QCTN instance
        """
        lines = [
            "=" * 60,
            f"Local QCTN - Partition {local_qctn.partition_idx} (Rank {self.rank})",
            "=" * 60,
        ]
        
        # Core names (using original names)
        lines.append(f"Cores ({local_qctn.ncores}):")
        for i, core_name in enumerate(local_qctn.cores):
            weight = local_qctn.cores_weights.get(core_name)
            if weight is not None:
                shape_str = f"shape={tuple(weight.tensor.shape)}"
            else:
                shape_str = "no weight"
            lines.append(f"  [{i}] {core_name} - {shape_str}")
        
        # Adjacency table
        lines.append("-" * 40)
        lines.append("Adjacency Table:")
        for entry in local_qctn.adjacency_table:
            core_name = entry.get('core_name', '')
            lines.append(f"  {core_name}:")
            
            # In edges
            in_edges = entry.get('in_edge_list', [])
            if in_edges:
                lines.append(f"    In edges ({len(in_edges)}):")
                for e in in_edges:
                    neighbor = e.get('neighbor_name', '')
                    is_cross = e.get('is_cross_partition', False)
                    edge_rank = e.get('edge_rank', 0)
                    qubit_idx = e.get('qubit_idx', -1)
                    cross_mark = " [CROSS]" if is_cross else ""
                    if neighbor:
                        lines.append(f"      <- {neighbor} (rank={edge_rank}, qubit={qubit_idx}){cross_mark}")
                    else:
                        lines.append(f"      <- [INPUT] (rank={edge_rank}, qubit={qubit_idx})")
            
            # Out edges
            out_edges = entry.get('out_edge_list', [])
            if out_edges:
                lines.append(f"    Out edges ({len(out_edges)}):")
                for e in out_edges:
                    neighbor = e.get('neighbor_name', '')
                    is_cross = e.get('is_cross_partition', False)
                    edge_rank = e.get('edge_rank', 0)
                    qubit_idx = e.get('qubit_idx', -1)
                    cross_mark = " [CROSS]" if is_cross else ""
                    if neighbor:
                        lines.append(f"      -> {neighbor} (rank={edge_rank}, qubit={qubit_idx}){cross_mark}")
                    else:
                        lines.append(f"      -> [OUTPUT] (rank={edge_rank}, qubit={qubit_idx})")
        
        lines.append("=" * 60)
        
        # Print all lines (each rank prints its own)
        print('\n'.join(lines))
    
    # ==================== Distributed Contraction ====================
    
    def contract_distributed(self) -> 'torch.Tensor':
        """Execute distributed tensor contraction.

        Data must be embedded in QCTN cores before calling.

        Returns:
            Final contracted result tensor.
        """
        if not self._is_initialized:
            raise RuntimeError("Must call init_distributed() before contract_distributed()")

        import torch

        plan = self._contract_plan

        import time
        tic = time.time()

        local_result = self._contract_local()
        self._current_partitions = {self.rank}
        self._current_boundary_edges = self._boundary_edges_for_partitions(
            self._current_partitions
        )

        toc = time.time()

        if self.world_size == 1:
            local_print(f"[Rank {self.rank}] Single rank execution complete. Result shape: {local_result.shape}")
            return local_result

        print(f"[Rank {self.rank}] Local contraction complete. Result shape: {local_result.shape} cost time {toc - tic}")

        current_result = local_result

        for stage_idx in range(1, plan.num_stages):
            current_result = self._contract_reduce_stage(
                stage_idx, current_result
            )

        return current_result
    
    def _contract_local(self) -> 'torch.Tensor':
        """Execute local contraction (Stage 0) via EngineCommon."""
        return super().contract(self._local_qctn)
    
    def _contract_reduce_stage(self, stage_idx: int,
                                local_result: 'torch.Tensor') -> 'torch.Tensor':
        """Execute a reduction stage by exchanging results and contracting via einsum.

        Each partition's local contraction result has open boundary dims
        corresponding to cross-partition edges. The dims are ordered by
        ``_contract_remaining``: in-boundaries sorted by qubit_idx, then
        out-boundaries sorted by qubit_idx.

        Matching rule:
        - My out-boundary edges = partner's in-boundary edges (same cross edges)
        - My in-boundary edges = partner's out-boundary edges

        So the einsum contracts all spatial dims: ``a I O, a O I -> a``.
        """
        import torch
        import opt_einsum

        self.comm.barrier()

        # --- Determine partner rank (pairwise in stage 1) ---
        group_size = 2 ** stage_idx
        half = group_size // 2
        my_group = self.rank // group_size
        my_pos = self.rank % group_size
        group_start = my_group * group_size

        if my_pos < half:
            partner_rank = group_start + my_pos + half
        else:
            partner_rank = group_start + my_pos - half

        # --- Unwrap TNTensor ---
        if isinstance(local_result, TNTensor):
            my_log_scale = local_result.log_scale
            my_scale = local_result.scale
            local_tensor = local_result.tensor
        else:
            my_log_scale = 0.0
            my_scale = 1.0
            local_tensor = local_result

        # --- Exchange tensors ---
        partner_tensor = self._exchange_tensor_with_partner(local_tensor, partner_rank)

        # --- Exchange TNTensor scales ---
        # ``TNTensor`` stores both scale and log_scale.  The effective value is
        # tensor * scale, so the reduced TNTensor must preserve scale as well
        # as log_scale.
        my_scale_meta = torch.tensor(
            [my_log_scale, my_scale],
            device=local_tensor.device,
            dtype=torch.float64,
        )
        partner_scale_meta = torch.zeros(
            2,
            device=local_tensor.device,
            dtype=torch.float64,
        )
        tag = 200 + stage_idx
        if self.rank < partner_rank:
            self.comm.send(my_scale_meta, partner_rank, tag=tag)
            self.comm.recv(partner_rank, tag=tag, tensor=partner_scale_meta)
        else:
            self.comm.recv(partner_rank, tag=tag, tensor=partner_scale_meta)
            self.comm.send(my_scale_meta, partner_rank, tag=tag)

        prev_group_size = 2 ** (stage_idx - 1)
        partner_group_start = (partner_rank // prev_group_size) * prev_group_size
        partner_partitions = set(
            range(
                partner_group_start,
                min(partner_group_start + prev_group_size, self.world_size),
            )
        )
        my_partitions = set(self._current_partitions)
        union_partitions = my_partitions | partner_partitions

        my_edges = self._current_boundary_edges
        partner_edges = self._boundary_edges_for_partitions(partner_partitions)
        output_edges = self._boundary_edges_for_partitions(union_partitions)

        if local_tensor.ndim == len(my_edges) + 1:
            my_has_batch = True
        elif local_tensor.ndim == len(my_edges):
            my_has_batch = False
        else:
            raise RuntimeError(
                f"Rank {self.rank} boundary metadata mismatch: tensor ndim "
                f"{local_tensor.ndim}, metadata edges {len(my_edges)}."
            )

        if partner_tensor.ndim == len(partner_edges) + 1:
            partner_has_batch = True
        elif partner_tensor.ndim == len(partner_edges):
            partner_has_batch = False
        else:
            raise RuntimeError(
                f"Rank {self.rank} partner boundary metadata mismatch: tensor ndim "
                f"{partner_tensor.ndim}, metadata edges {len(partner_edges)}."
            )

        contract_keys = {
            edge['key']
            for edge in my_edges
            if self._edge_other_partition(edge) in partner_partitions
        }
        contract_keys.update(
            edge['key']
            for edge in partner_edges
            if self._edge_other_partition(edge) in my_partitions
        )

        batch_sym = 'a'
        key_to_symbol: Dict[Tuple, str] = {}

        def symbol_for(key: Tuple) -> str:
            if key not in key_to_symbol:
                # Start after 'a' to keep batch symbol stable.
                key_to_symbol[key] = opt_einsum.get_symbol(len(key_to_symbol) + 1)
            return key_to_symbol[key]

        my_part = (batch_sym if my_has_batch else '') + ''.join(
            symbol_for(edge['key']) for edge in my_edges
        )
        partner_part = (batch_sym if partner_has_batch else '') + ''.join(
            symbol_for(edge['key']) for edge in partner_edges
        )
        rhs = (batch_sym if my_has_batch or partner_has_batch else '') + ''.join(
            symbol_for(edge['key']) for edge in output_edges
        )
        einsum_eq = f"{my_part},{partner_part}->{rhs}"

        local_print(
            f"[Rank {self.rank}] Reduce stage {stage_idx}: "
            f"my_parts={sorted(my_partitions)}, partner_parts={sorted(partner_partitions)}, "
            f"contract_edges={len(contract_keys)}, output_edges={len(output_edges)}, "
            f"eq={einsum_eq}, shapes={local_tensor.shape} x {partner_tensor.shape}"
        )

        result = torch.einsum(einsum_eq, local_tensor, partner_tensor)
        self._current_partitions = union_partitions
        self._current_boundary_edges = output_edges

        partner_log_scale = float(partner_scale_meta[0].item())
        partner_scale = float(partner_scale_meta[1].item())
        combined_log_scale = my_log_scale + partner_log_scale
        combined_scale = my_scale * partner_scale

        self.comm.barrier()

        return TNTensor(result, scale=combined_scale, log_scale=combined_log_scale)

    @staticmethod
    def _cross_edge_key(edge: Dict[str, Any]) -> Tuple:
        """Stable key for the two partition views of the same cross edge."""
        return (
            edge.get('from_partition'),
            edge.get('to_partition'),
            edge.get('from_core_raw'),
            edge.get('to_core_raw'),
            edge.get('qubit_idx'),
            edge.get('edge_rank'),
        )

    @staticmethod
    def _edge_other_partition(edge: Dict[str, Any]) -> int:
        if edge['direction'] == 'out':
            return edge['to_partition']
        return edge['from_partition']

    def _boundary_edges_for_partitions(self, partitions) -> List[Dict[str, Any]]:
        """Return open cross-partition boundaries for a partition group.

        The order intentionally matches RowPriorityStrategy._contract_remaining:
        all in-boundaries sorted by qubit, then all out-boundaries sorted by
        qubit.  Tensor dimensions after the batch dimension follow this order.
        """
        partition_set = set(partitions)
        boundary_edges = []
        for edge in self._contract_plan.inter_node_graph.get('cross_edges', []):
            from_p = edge['from_partition']
            to_p = edge['to_partition']
            if from_p in partition_set and to_p not in partition_set:
                item = edge.copy()
                item['direction'] = 'out'
                item['key'] = self._cross_edge_key(edge)
                boundary_edges.append(item)
            elif to_p in partition_set and from_p not in partition_set:
                item = edge.copy()
                item['direction'] = 'in'
                item['key'] = self._cross_edge_key(edge)
                boundary_edges.append(item)

        direction_order = {'in': 0, 'out': 1}
        boundary_edges.sort(
            key=lambda e: (
                direction_order[e['direction']],
                e.get('qubit_idx', -1),
                e.get('from_partition', -1),
                e.get('to_partition', -1),
                e.get('from_core_idx', -1),
                e.get('to_core_idx', -1),
            )
        )
        return boundary_edges
    
    def _get_cross_edges_for_stage(self, stage_idx: int, my_group: int) -> List[Dict]:
        """
        Get cross-partition edges relevant for this reduction stage.
        
        In stage 1: edges between partition 0-1, 2-3, etc.
        In stage 2: edges between partitions 0,1 and 2,3, etc.
        
        Args:
            stage_idx: Which reduction stage
            my_group: Which group this rank belongs to
            
        Returns:
            List of cross edges for this stage
        """
        all_cross_edges = self._contract_plan.inter_node_graph.get('cross_edges', [])
        
        # In stage k, group size is 2^k
        # Left half partitions: [group_start, group_start + 2^(k-1) - 1]
        # Right half partitions: [group_start + 2^(k-1), group_start + 2^k - 1]
        group_size = 2 ** stage_idx
        half_size = group_size // 2
        group_start = my_group * group_size
        left_partitions = set(range(group_start, group_start + half_size))
        right_partitions = set(range(group_start + half_size, group_start + group_size))
        
        # Find edges going from left partitions to right partitions (or vice versa)
        relevant_edges = []
        for edge in all_cross_edges:
            from_p = edge['from_partition']
            to_p = edge['to_partition']
            # Edge crosses between left and right halves of this group
            if (from_p in left_partitions and to_p in right_partitions) or \
               (from_p in right_partitions and to_p in left_partitions):
                relevant_edges.append(edge)
        
        return relevant_edges
    
    def _tensor_parallel_matmul(self, 
                                 local_tensor: 'torch.Tensor',
                                 stage_idx: int,
                                 group_ranks: List[int],
                                 my_position: int,
                                 is_left_half: bool,
                                 contract_qubit_indices: List[int]) -> 'torch.Tensor':
        """
        Perform tensor parallel matrix multiplication for distributed contraction.
        
        The tensor has structure: [batch, in_dims..., out_dims...]
        where in_dims and out_dims correspond to qubit indices.
        
        For contraction:
        1. Identify which dims correspond to contract_qubit_indices
        2. Transpose to put non-contract dims first, contract dims last
        3. Reshape to [batch, N, K] where K is product of contract dims
        4. Partner matrix is [batch, M, K]
        5. Shard N across group, each computes partial [batch, N/group_size, K] @ [batch, M, K]^T
        6. AllGather to get full [batch, N, M] result
        
        Args:
            local_tensor: Local result tensor
            stage_idx: Current reduction stage
            group_ranks: Ranks in current group
            my_position: Position within group
            is_left_half: Whether this rank is in left half of group
            contract_qubit_indices: Qubit indices being contracted in this stage
            
        Returns:
            Result tensor after TP matmul
        """
        import torch
        
        group_size = len(group_ranks)
        half_group_size = group_size // 2
        
        # Print cross-edge information for this partition from _contract_plan
        my_partition = self.rank  # Assuming 1:1 mapping between rank and partition
        all_cross_edges = self._contract_plan.inter_node_graph.get('cross_edges', [])
        
        # print(f'all_cross_edges : {all_cross_edges}')


        batch_size = local_tensor.shape[0]
        remaining_dims = list(local_tensor.shape[1:])
        n_dims = len(remaining_dims)
        
        # =====================================================================
        # Define left_partitions, right_partitions based on group structure
        # =====================================================================
        
        my_partition = self.rank
        my_group_idx = my_partition // group_size
        group_start = my_group_idx * group_size
        
        # Define left and right partitions for this group
        left_partitions = set(range(group_start, group_start + half_group_size))
        right_partitions = set(range(group_start + half_group_size, group_start + group_size))
        
        local_print(f"[Rank {self.rank}] Stage {stage_idx}: left_partitions={sorted(left_partitions)}, "
              f"right_partitions={sorted(right_partitions)}, is_left_half={is_left_half}")
        
        # Handle TNTensor input
        is_tntensor = isinstance(local_tensor, TNTensor)
        if is_tntensor:
            my_log_scale = local_tensor.log_scale
            local_tensor = local_tensor.tensor
            local_print(f"[Rank {self.rank}] TNTensor input detected, log_scale={my_log_scale}")
        else:
            my_log_scale = 0.0
            
        # Get all cross edges from contract plan
        all_cross_edges = self._contract_plan.inter_node_graph.get('cross_edges', [])
        

        def compute_partition_info(partitions, partner_partitions, n_dims):
            """
            Compute in_edges, out_edges, permute, contract_dim info for a set of partitions.
            
            Returns:
                dict with keys:
                - in_edges: sorted by qubit_idx
                - out_edges: sorted by qubit_idx
                - contract_dim_indices: list of dim indices (in original tensor)
                - non_contract_dim_indices: list of dim indices (in original tensor)
                - perm: permutation to apply [batch, non_contract..., contract...]
                - dim_info: list of dicts, one per dim after permute (excluding batch)
                    Each dict: {'is_contract': bool, 'edge_type': 'in'|'out', 'edge_idx': int}
            """
            # Find edges relevant to these partitions
            # OUT edges: from these partitions to any other
            # IN edges: from any other to these partitions
            out_edges = [e for e in all_cross_edges if e['from_partition'] in partitions and e['to_partition'] not in partitions]
            in_edges = [e for e in all_cross_edges if e['to_partition'] in partitions and e['from_partition'] not in partitions]
            
            # Sort by qubit_idx to match tensor dim order
            in_edges.sort(key=lambda x: x['qubit_idx'])
            out_edges.sort(key=lambda x: x['qubit_idx'])

            local_print(f"[Rank {self.rank}] in_edges {in_edges}, out_edges {out_edges}")
            
            contract_dim_indices = []
            non_contract_dim_indices = []
            
            # For in_edges: dim_idx is position in sorted in_edges list
            for dim_idx, in_edge in enumerate(in_edges):
                neighbor_part = in_edge.get('from_partition', -1)
                if neighbor_part in partner_partitions:
                    # Contract dim: add both original and its mirror
                    contract_dim_indices.append(dim_idx)
                    contract_dim_indices.append(n_dims - dim_idx - 1)
                else:
                    non_contract_dim_indices.append(dim_idx)
                    non_contract_dim_indices.append(n_dims - dim_idx - 1)
            
            # For out_edges: offset by number of in_edges
            offset = len(in_edges)
            for dim_idx, out_edge in enumerate(out_edges):
                neighbor_part = out_edge.get('to_partition', -1)
                if neighbor_part in partner_partitions:
                    contract_dim_indices.append(offset + dim_idx)
                    contract_dim_indices.append(n_dims - (offset + dim_idx) - 1)
                else:
                    non_contract_dim_indices.append(offset + dim_idx)
                    non_contract_dim_indices.append(n_dims - (offset + dim_idx) - 1)
            
            # print(f'[Rank {self.rank}] contract_dim_indices: {contract_dim_indices} non_contract_dim_indices: {non_contract_dim_indices}')

            # Build permutation: [batch, non_contract_dims..., contract_dims...]
            perm = [0] + [d + 1 for d in non_contract_dim_indices] + [d + 1 for d in contract_dim_indices]
            
            # Build dim_info for each dim after permute (excluding batch dim)
            # Order: non_contract_dims first, then contract_dims
            dim_info = []
            
            # Non-contract dims first
            for orig_dim_idx in non_contract_dim_indices:
                # Determine if this is an in_edge or out_edge dim
                if orig_dim_idx < len(in_edges):
                    edge_type = 'in'
                    edge_idx = orig_dim_idx
                elif orig_dim_idx >= n_dims - len(in_edges):
                    # Mirror of in_edge
                    edge_type = 'in'
                    edge_idx = n_dims - orig_dim_idx - 1
                elif orig_dim_idx < offset + len(out_edges):
                    edge_type = 'out'
                    edge_idx = orig_dim_idx - offset
                else:
                    # Mirror of out_edge
                    edge_type = 'out'
                    edge_idx = n_dims - orig_dim_idx - 1 - offset
                
                dim_info.append({
                    'is_contract': False,
                    'edge_type': edge_type,
                    'edge_idx': edge_idx,
                    'orig_dim_idx': orig_dim_idx
                })
            
            # Contract dims next
            for orig_dim_idx in contract_dim_indices:
                if orig_dim_idx < len(in_edges):
                    edge_type = 'in'
                    edge_idx = orig_dim_idx
                elif orig_dim_idx >= n_dims - len(in_edges):
                    edge_type = 'in'
                    edge_idx = n_dims - orig_dim_idx - 1
                elif orig_dim_idx < offset + len(out_edges):
                    edge_type = 'out'
                    edge_idx = orig_dim_idx - offset
                else:
                    edge_type = 'out'
                    edge_idx = n_dims - orig_dim_idx - 1 - offset
                
                dim_info.append({
                    'is_contract': True,
                    'edge_type': edge_type,
                    'edge_idx': edge_idx,
                    'orig_dim_idx': orig_dim_idx
                })
            
            return {
                'in_edges': in_edges,
                'out_edges': out_edges,
                'contract_dim_indices': contract_dim_indices,
                'non_contract_dim_indices': non_contract_dim_indices,
                'perm': perm,
                'dim_info': dim_info
            }
        
        # Compute info for both left and right partitions
        left_info = compute_partition_info(left_partitions, right_partitions, n_dims)
        right_info = compute_partition_info(right_partitions, left_partitions, n_dims)
        
        # Select my partition's info based on is_left_half
        if is_left_half:
            my_info = left_info
            partner_info = right_info
        else:
            my_info = right_info
            partner_info = left_info
        
        contract_dim_indices = my_info['contract_dim_indices']
        non_contract_dim_indices = my_info['non_contract_dim_indices']
        perm = my_info['perm']
        dim_info = my_info['dim_info']
        
        local_print(f"[Rank {self.rank}] Using {'left' if is_left_half else 'right'} partition info")
        local_print(f"[Rank {self.rank}] Contract dim indices: {contract_dim_indices}")
        local_print(f"[Rank {self.rank}] Non-contract dim indices: {non_contract_dim_indices}")
        local_print(f"[Rank {self.rank}] Permutation: {perm}")
        local_print(f"[Rank {self.rank}] Dim info after permute: {dim_info}")
        
        # If no contract dims identified, fallback
        if not contract_dim_indices:
            local_print(f"[Rank {self.rank}] Warning: no contract dims found, using identity")
            return local_tensor

        # Transpose: [batch, non_contract_dims..., contract_dims...]
        # Use the perm already computed from partition info
        transposed = local_tensor.permute(perm)
        
        # Compute shapes
        non_contract_shape = [remaining_dims[d] for d in non_contract_dim_indices]
        contract_shape = [remaining_dims[d] for d in contract_dim_indices]
        
        # If no non-contract dims, N = 1; if no contract dims, K = 1
        N = 1
        for s in non_contract_shape:
            N *= s
        K = 1
        for s in contract_shape:
            K *= s
        
        # Reshape to [batch, N, K] - N=1 if no non-contract dims, K=1 if no contract dims
        reshaped = transposed.reshape(batch_size, N, K)
        
        # Synchronize reshaped tensor shapes across all ranks
        # Each rank broadcasts its [batch_size, N, K] shape
        # Result: all_shapes is a [world_size, 3] tensor
        my_shape_tensor = torch.tensor([batch_size, N, K], dtype=torch.long, device=local_tensor.device)
        all_shapes_list = self.comm.allgather(my_shape_tensor)
        all_shapes = torch.stack(all_shapes_list, dim=0)  # [world_size, 3]
        
        local_print(f"[Rank {self.rank}] Synchronized reshaped shapes across all ranks: {all_shapes}")
        
        local_print(f"[Rank {self.rank}] TP matmul: tensor shape {local_tensor.shape} -> "
              f"reshaped {reshaped.shape}, N={N}, K={K}, is_left={is_left_half}")
        
        # =====================================================================
        # TP Matrix Multiplication with K-dimension sharding
        # =====================================================================
        
        # Identify left and right partitions for this stage
        left_partition_ranks = group_ranks[:half_group_size]
        right_partition_ranks = group_ranks[half_group_size:]
        
        # Get shapes from first node of each partition (they hold the authoritative shape)
        left_first_rank = left_partition_ranks[0]
        right_first_rank = right_partition_ranks[0]
        
        left_shape = all_shapes[left_first_rank]  # [batch, N_left, K_left]
        right_shape = all_shapes[right_first_rank]  # [batch, N_right, K_right]
        
        # For matmul: left [B, N, K] @ right [B, M, K]^T = [B, N, M]
        # K dimension must match and will be sharded across all nodes
        K_total = left_shape[2].item()  # K dimension to shard
        N_left = left_shape[1].item()
        M_right = right_shape[1].item()  # This is M (right's N dimension)
        
        local_print(f"[Rank {self.rank}] TP setup: left_partition={left_partition_ranks}, "
              f"right_partition={right_partition_ranks}, K_total={K_total}, "
              f"N_left={N_left}, M_right={M_right}")
        
        # Compute K sharding across all nodes in the group
        # Total nodes = group_size, shard K evenly with remainder to first nodes
        total_nodes = group_size
        base_k = K_total // total_nodes
        remainder_k = K_total % total_nodes
        
        # Compute start/end indices for each rank's K shard
        k_shards = []
        k_start = 0
        for i in range(total_nodes):
            k_size = base_k + (1 if i < remainder_k else 0)
            k_shards.append((k_start, k_start + k_size))
            k_start += k_size
        
        my_global_position = my_position  # Position within group
        my_k_start, my_k_end = k_shards[my_global_position]
        my_k_size = my_k_end - my_k_start
        
        local_print(f"[Rank {self.rank}] K sharding: position={my_global_position}, "
              f"k_range=[{my_k_start}:{my_k_end}], k_size={my_k_size}")
        local_print(f"[Rank {self.rank}] All K shards: {k_shards}")
        
        # Determine partner rank for data exchange
        # Left node i pairs with right node i
        # e.g., rank 0 <-> rank 2, rank 1 <-> rank 3
        if is_left_half:
            partner_rank = right_partition_ranks[my_position]  # my_position is position in left half
            partner_position = my_position + half_group_size  # partner's position in group
        else:
            partner_rank = left_partition_ranks[my_position - half_group_size]
            partner_position = my_position - half_group_size
        
        # Get partner's K shard range
        partner_k_start, partner_k_end = k_shards[partner_position]
        
        local_print(f"[Rank {self.rank}] Partner for data exchange: rank {partner_rank}, "
              f"partner_position={partner_position}, partner_k_range=[{partner_k_start}:{partner_k_end}]")
        
        # Extract my K shard from local tensor (for my own computation)
        my_k_shard = reshaped[:, :, my_k_start:my_k_end]  # [B, N, K_shard] or [B, M, K_shard]
        
        # Extract the K shard to send to partner (using partner's K range)
        k_shard_to_partner = reshaped[:, :, partner_k_start:partner_k_end]  # Partner needs this shard
        
        local_print(f"[Rank {self.rank}] My K shard shape: {my_k_shard.shape}, "
              f"K shard to send to partner: {k_shard_to_partner.shape}")
        
        # Exchange K shards with partner
        # I send my tensor sliced at partner's K range, partner sends their tensor sliced at my K range
        partner_k_shard = self._exchange_tensor_with_partner(k_shard_to_partner, partner_rank)
        
        local_print(f"[Rank {self.rank}] Partner K shard received shape: {partner_k_shard.shape}")

        # Exchange log scales with partner to maintain TNTensor precision
        partner_log_scale = 0.0
        if partner_rank >= 0:
            my_log_scale_tensor = torch.tensor([my_log_scale], device=local_tensor.device, dtype=torch.float64)
            partner_log_scale_tensor = torch.zeros(1, device=local_tensor.device, dtype=torch.float64)
            
            tag_log_scale = 103
            try:
                if self.rank < partner_rank:
                    self.comm.send(my_log_scale_tensor, partner_rank, tag=tag_log_scale)
                    self.comm.recv(partner_rank, tag=tag_log_scale, tensor=partner_log_scale_tensor)
                else:
                    self.comm.recv(partner_rank, tag=tag_log_scale, tensor=partner_log_scale_tensor)
                    self.comm.send(my_log_scale_tensor, partner_rank, tag=tag_log_scale)
                partner_log_scale = partner_log_scale_tensor.item()
                pair_log_scale = my_log_scale + partner_log_scale
                local_print(f"[Rank {self.rank}] Log scale exchange successful: "
                           f"my={my_log_scale:.2e}, partner={partner_log_scale:.2e}")
            except Exception as e:
                print(f"[Rank {self.rank}] ERROR: Log scale exchange with rank {partner_rank} failed: {e}")
                raise
        else:
            # If no partner, this rank contributes zeros, so log_scale can be effectively -inf
            pair_log_scale = -1e10 
            
        # Synchronize pair_log_scale across the group to find the max log scale for allreduce safety
        pair_log_scale_tensor = torch.tensor([pair_log_scale], device=local_tensor.device, dtype=torch.float64)
        all_pair_log_scales_list = self.comm.allgather(pair_log_scale_tensor)
        all_pair_log_scales = torch.stack(all_pair_log_scales_list, dim=0) # [world_size, 1]
        
        # Determine target log scale for this group's reduction (use max log scale across group)
        group_pair_log_scales = all_pair_log_scales[group_ranks]
        target_log_scale = group_pair_log_scales.max().item()
        
        # Scaling factor to align this rank's partial result with the target log scale
        scaling_factor = math.exp(pair_log_scale - target_log_scale)
        
        # Perform partial matrix multiplication
        # Left: [B, N, K_shard] @ [B, K_shard, M] -> [B, N, M]
        # Right: [B, M, K_shard] @ [B, K_shard, N] -> [B, M, N]
        if is_left_half:
            # I'm in left partition, I have left[B, N, K_shard], partner has right[B, M, K_shard]
            # Compute: left @ right^T = [B, N, K_shard] @ [B, K_shard, M] = [B, N, M]
            partner_T = partner_k_shard.transpose(1, 2)  # [B, K_shard, M]
            partial_result = torch.bmm(my_k_shard, partner_T)  # [B, N, M]
        else:
            # I'm in right partition, I have right[B, M, K_shard], partner has left[B, N, K_shard]
            # Compute: left @ right^T = [B, N, K_shard] @ [B, K_shard, M] = [B, N, M]
            # partner has [B, N, K_shard], I have [B, M, K_shard]
            my_T = my_k_shard.transpose(1, 2)  # [B, K_shard, M]
            partial_result = torch.bmm(partner_k_shard, my_T)  # [B, N, M]
        
        # Scale partial_result to match the target log scale
        if scaling_factor != 1.0:
            partial_result = partial_result * scaling_factor

        local_print(f"[Rank {self.rank}] Partial result shape: {partial_result.shape}, scaled by {scaling_factor}")
        
        # AllReduce within group to sum up partial results
        # Use gradient-aware allreduce with the specific group
        full_result = self._allreduce_with_grad(partial_result, group_ranks=group_ranks)
        
        local_print(f"[Rank {self.rank}] Full result after allreduce: {full_result.shape}, log_scale={target_log_scale}")
        
        # =====================================================================
        # Reshape back to multi-dimensional tensor where each dim = one edge
        # Result shape: [batch, left_non_contract_dims..., right_non_contract_dims...]
        # Each dim corresponds to one edge from left_info or right_info's non_contract dims
        # =====================================================================
        
        # Get non_contract_dim shapes from left_info and right_info
        # left_info['non_contract_dim_indices'] tells us which original dims are non-contract for left
        # right_info['non_contract_dim_indices'] tells us which original dims are non-contract for right
        
        left_non_contract_indices = left_info['non_contract_dim_indices']
        right_non_contract_indices = right_info['non_contract_dim_indices']
        
        # Get the shapes for each non-contract dim
        # remaining_dims contains the original tensor dim sizes (excluding batch)
        left_non_contract_shapes = [remaining_dims[d] for d in left_non_contract_indices]
        right_non_contract_shapes = [remaining_dims[d] for d in right_non_contract_indices]
        
        # Build the final shape: [batch, left_non_contract_shapes..., right_non_contract_shapes...]
        # This gives us one dim per edge
        result_shape = [batch_size] + left_non_contract_shapes + right_non_contract_shapes
        result = full_result.reshape(result_shape)
        
        local_print(f"[Rank {self.rank}] Reshaped result to per-edge dims:")
        local_print(f"  left_non_contract_indices: {left_non_contract_indices}")
        local_print(f"  left_non_contract_shapes: {left_non_contract_shapes}")
        local_print(f"  right_non_contract_indices: {right_non_contract_indices}")
        local_print(f"  right_non_contract_shapes: {right_non_contract_shapes}")
        local_print(f"  Intermediate result shape: {result.shape}")
        
        # =====================================================================
        # Reorder dims to match symmetric core tensor structure:
        # [B, in_edges sorted by qubit, out_edges sorted by qubit, 
        #     out_edges reversed, in_edges reversed]
        # 
        # Note: result already has mirrored structure within each partition:
        # - left_non_contract has pairs (original, mirror) for each edge
        # - right_non_contract has pairs (original, mirror) for each edge
        # 
        # Example: left has out_q1, right has in_q2
        # Current result: [B, left_out_q1, left_out_q1, right_in_q2, right_in_q2]
        # After reorder:  [B, right_in_q2, left_out_q1, left_out_q1, right_in_q2]
        # =====================================================================
        
        # Collect unique edges (only the first of each pair) from both left and right
        # Each partition's non_contract_indices is already paired: [orig, mirror, orig, mirror, ...]
        # We take only the first half as unique edges, and record both positions
        
        def extract_unique_edges_with_positions(non_contract_indices, partition_info, source, result_dim_offset):
            """
            Extract unique edges from non_contract_indices (which has mirrored pairs).
            Returns list of edge info with both primary and mirror positions in result.
            """
            unique_edges = []
            n_in = len(partition_info['in_edges'])
            offset = n_in
            
            # non_contract_indices has pairs: [idx0, mirror0, idx1, mirror1, ...]
            # Each pair corresponds to one unique edge
            n_pairs = len(non_contract_indices) // 2
            
            for pair_idx in range(n_pairs):
                primary_orig_dim = non_contract_indices[pair_idx * 2]
                mirror_orig_dim = non_contract_indices[pair_idx * 2 + 1]
                
                # Determine edge type and qubit_idx from primary dim
                if primary_orig_dim < n_in:
                    edge_type = 'in'
                    edge_idx = primary_orig_dim
                    qubit_idx = partition_info['in_edges'][edge_idx]['qubit_idx']
                elif primary_orig_dim >= n_dims - n_in:
                    edge_type = 'in'
                    edge_idx = n_dims - primary_orig_dim - 1
                    qubit_idx = partition_info['in_edges'][edge_idx]['qubit_idx']
                elif primary_orig_dim < offset + len(partition_info['out_edges']):
                    edge_type = 'out'
                    edge_idx = primary_orig_dim - offset
                    qubit_idx = partition_info['out_edges'][edge_idx]['qubit_idx']
                else:
                    edge_type = 'out'
                    edge_idx = n_dims - primary_orig_dim - 1 - offset
                    qubit_idx = partition_info['out_edges'][edge_idx]['qubit_idx']
                
                unique_edges.append({
                    'edge_type': edge_type,
                    'qubit_idx': qubit_idx,
                    'source': source,
                    'primary_dim_in_result': result_dim_offset + pair_idx * 2 + 1,  # +1 for batch
                    'mirror_dim_in_result': result_dim_offset + pair_idx * 2 + 2,   # +1 for batch
                    'shape': remaining_dims[primary_orig_dim]
                })
            
            return unique_edges
        
        # Extract unique edges from left and right partitions
        left_unique_edges = extract_unique_edges_with_positions(
            left_non_contract_indices, left_info, 'left', 0)
        right_unique_edges = extract_unique_edges_with_positions(
            right_non_contract_indices, right_info, 'right', len(left_non_contract_indices))
        
        all_unique_edges = left_unique_edges + right_unique_edges
        
        local_print(f"[Rank {self.rank}] Left unique edges: {left_unique_edges}")
        local_print(f"[Rank {self.rank}] Right unique edges: {right_unique_edges}")
        
        # Sort unique edges: first in_edges by qubit, then out_edges by qubit
        in_edges_sorted = sorted([e for e in all_unique_edges if e['edge_type'] == 'in'], 
                                  key=lambda x: x['qubit_idx'])
        out_edges_sorted = sorted([e for e in all_unique_edges if e['edge_type'] == 'out'], 
                                   key=lambda x: x['qubit_idx'])
        
        # Build first half: [in_edges sorted, out_edges sorted] using primary positions
        # Build second half: [out_edges reversed, in_edges reversed] using mirror positions
        first_half_edges = in_edges_sorted + out_edges_sorted
        second_half_edges = list(reversed(out_edges_sorted)) + list(reversed(in_edges_sorted))
        
        # Construct permutation
        perm_after_reshape = [0]  # batch dim stays at 0
        
        # First half uses primary positions
        for edge in first_half_edges:
            perm_after_reshape.append(edge['primary_dim_in_result'])
        
        # Second half uses mirror positions
        for edge in second_half_edges:
            perm_after_reshape.append(edge['mirror_dim_in_result'])
        
        local_print(f"[Rank {self.rank}] Reorder permutation: {perm_after_reshape}")
        local_print(f"[Rank {self.rank}] First half (in sorted, out sorted): "
              f"{[(e['edge_type'], e['qubit_idx'], e['source']) for e in first_half_edges]}")
        local_print(f"[Rank {self.rank}] Second half (out rev, in rev): "
              f"{[(e['edge_type'], e['qubit_idx'], e['source']) for e in second_half_edges]}")
        
        # Apply permutation to reorder dims
        result = result.permute(perm_after_reshape)
        
        # Build final dim_info in the new order
        # First half uses primary dims, second half uses mirror dims
        result_dim_info = []
        for edge in first_half_edges:
            result_dim_info.append({
                'source': edge['source'],
                'edge_type': edge['edge_type'],
                'qubit_idx': edge['qubit_idx'],
                'shape': edge['shape'],
                'is_mirror': False
            })
        for edge in second_half_edges:
            result_dim_info.append({
                'source': edge['source'],
                'edge_type': edge['edge_type'],
                'qubit_idx': edge['qubit_idx'],
                'shape': edge['shape'],
                'is_mirror': True
            })
        
        local_print(f"[Rank {self.rank}] Final result shape after reorder: {result.shape}")
        local_print(f"[Rank {self.rank}] Result dim info (symmetric): {result_dim_info}")
        
        # If input was TNTensor, return the result as a TNTensor to maintain precision
        if is_tntensor:
            return TNTensor(result, log_scale=target_log_scale)
        else:
            return result
    
    def _gather_shards_in_subgroup(self, my_shard: 'torch.Tensor', sub_group_ranks: List[int]) -> List['torch.Tensor']:
        """
        Gather shards from all ranks in a sub-group using P2P communication.
        
        Each rank sends its shard to all other ranks in the sub-group and receives
        shards from all other ranks.
        
        Args:
            my_shard: This rank's shard tensor
            sub_group_ranks: List of ranks in the sub-group
            
        Returns:
            List of shards from all ranks in sub-group (in rank order)
        """
        import torch
        
        shards = [None] * len(sub_group_ranks)
        my_position = sub_group_ranks.index(self.rank)
        shards[my_position] = my_shard
        
        # Exchange with all other ranks in sub-group
        for i, other_rank in enumerate(sub_group_ranks):
            if other_rank == self.rank:
                continue
            
            # Exchange shard with other_rank using P2P
            other_shard = self._exchange_tensor_with_partner(my_shard, other_rank)
            shards[i] = other_shard
        
        return shards
    
    def _exchange_tensor_with_partner(self, tensor: 'torch.Tensor', partner_rank: int) -> 'torch.Tensor':
        """
        Exchange tensor with partner rank using gradient-aware point-to-point send/recv.
        
        Uses SendRecvGrad autograd function to maintain gradient flow through
        the exchange operation during backpropagation.
        
        Optimized version: combines shape metadata into single message to reduce
        communication rounds and deadlock risk.
        
        Args:
            tensor: Tensor to send
            partner_rank: Rank of partner
            
        Returns:
            Tensor received from partner (with gradient support)
        """
        import torch
        from ..optim.allreduce_grad import SendRecvGrad
        
        # Step 1: Exchange shape metadata.  Local contraction can expose one
        # dimension per cross-partition boundary, so the tensor rank is not
        # bounded by a small constant.
        my_shape = list(tensor.shape)
        n_dims = len(my_shape)

        # Use sendrecv pattern: lower rank sends first to avoid deadlock
        tag_metadata = 100
        tag_shape = 101
        
        try:
            my_n_dims = torch.tensor([n_dims], dtype=torch.long, device=tensor.device)
            partner_n_dims_tensor = torch.zeros(1, dtype=torch.long, device=tensor.device)

            if self.rank < partner_rank:
                self.comm.send(my_n_dims, partner_rank, tag=tag_metadata)
                self.comm.recv(partner_rank, tag=tag_metadata, tensor=partner_n_dims_tensor)
            else:
                self.comm.recv(partner_rank, tag=tag_metadata, tensor=partner_n_dims_tensor)
                self.comm.send(my_n_dims, partner_rank, tag=tag_metadata)

            partner_n_dims = int(partner_n_dims_tensor.item())
            my_shape_tensor = torch.tensor(my_shape, dtype=torch.long, device=tensor.device)
            partner_shape_tensor = torch.zeros(
                partner_n_dims, dtype=torch.long, device=tensor.device
            )

            if self.rank < partner_rank:
                self.comm.send(my_shape_tensor, partner_rank, tag=tag_shape)
                self.comm.recv(partner_rank, tag=tag_shape, tensor=partner_shape_tensor)
            else:
                self.comm.recv(partner_rank, tag=tag_shape, tensor=partner_shape_tensor)
                self.comm.send(my_shape_tensor, partner_rank, tag=tag_shape)
            
            # Unpack partner's shape
            partner_shape = tuple(partner_shape_tensor.tolist())
            
            local_print(f"[Rank {self.rank}] Metadata exchange with rank {partner_rank} successful: "
                       f"my_shape={my_shape}, partner_shape={partner_shape}")
            
        except Exception as e:
            print(f"[Rank {self.rank}] ERROR: Metadata exchange with rank {partner_rank} failed: {e}")
            raise
        
        # Step 2: Allocate receive buffer and use gradient-aware send/recv
        recv_buffer = torch.zeros(partner_shape, dtype=tensor.dtype, device=tensor.device)
        
        # Use SendRecvGrad for gradient-aware exchange
        # This allows gradients to flow back through the exchange during backpropagation
        try:
            result = SendRecvGrad.apply(tensor.contiguous(), recv_buffer, partner_rank, self.rank)
            local_print(f"[Rank {self.rank}] Tensor exchange with rank {partner_rank} successful")
            return result
        except Exception as e:
            print(f"[Rank {self.rank}] ERROR: Tensor exchange with rank {partner_rank} failed: {e}")
            raise
    
    # ==================== Unified Engine Interface ====================

    def contract(self, qctn) -> Any:
        """Override: distributed contraction if initialized, else local."""
        if self._is_initialized and self.world_size > 1:
            return self.contract_distributed()
        return super().contract(qctn)

    def contract_for_gradient(self, qctn, target=None, loss=None):
        """Override: uses super() for gradient computation."""
        return super().contract_for_gradient(qctn, target=target, loss=loss)

    # ==================== Gradient-Aware Operations ====================
    
    def _get_process_group(self, rank_list: List[int]):
        """
        Get or create a process group for the given ranks.
        
        Args:
            rank_list: List of ranks in the group
            
        Returns:
            Process group object for torch.distributed
        """
        import torch.distributed as dist
        
        if not hasattr(self, '_pg_cache'):
            self._pg_cache = {}
            
        # Group list must be sorted for consistent key
        ranks = tuple(sorted(rank_list))
        
        local_print(f"[Rank {self.rank}] Retrieving process group for ranks: {ranks} cache {self._pg_cache}")

        if ranks not in self._pg_cache:
            if len(ranks) <= 1:
                self._pg_cache[ranks] = None
            elif len(ranks) == self.world_size:
                self._pg_cache[ranks] = None # Use WORLD group
            else:
                print(f"[Rank {self.rank}] Creating new process group for ranks: {ranks}")
                self._pg_cache[ranks] = dist.new_group(ranks)

        local_print(f"[Rank {self.rank}] Using process group for ranks: {ranks} -> {self._pg_cache[ranks]}")

        return self._pg_cache[ranks]

    def _allreduce_with_grad(self, tensor: 'torch.Tensor', group_ranks: Optional[List[int]] = None) -> 'torch.Tensor':
        """
        Perform allreduce with gradient support.
        
        Uses custom autograd function to maintain gradient flow through
        the allreduce operation during backpropagation.
        
        Args:
            tensor: Input tensor to allreduce
            group_ranks: Optional list of ranks for local allreduce
            
        Returns:
            Allreduced tensor (gradients flow through)
        """
        import torch
        import torch.distributed as dist
        from torch.distributed import ReduceOp as TorchReduceOp
        
        # If not using distributed or single process, return as-is
        if self.world_size == 1 or not dist.is_initialized():
            return tensor
        
        local_print(f"[Rank {self.rank}] Performing gradient-aware allreduce on tensor shape {tensor.shape} {group_ranks}")
        # Determine process group
        pg = None
        if group_ranks is not None:
            pg = self._get_process_group(group_ranks)
        
        # Use gradient-aware allreduce
        from ..optim.allreduce_grad import allreduce_with_grad

        return allreduce_with_grad(tensor, TorchReduceOp.SUM, group=pg)
    
    def contract_distributed_with_gradient(self, target=None, loss=None):
        """Execute distributed contraction and compute gradients.

        Args:
            target: Target tensor for loss computation.
            loss: Loss specification (str, callable, or BaseLoss).

        Returns:
            tuple: (loss_value, grads)
        """
        import torch
        from ...losses import LossRegistry
        from ...losses.target import TargetResolver

        # Ensure local weights require gradients
        for name in self._local_qctn.cores:
            weight = self._local_qctn.cores_weights[name]
            raw = weight.tensor if isinstance(weight, TNTensor) else weight
            if not isinstance(raw, torch.Tensor):
                continue
            if hasattr(raw, 'is_leaf') and not raw.is_leaf:
                continue
            if raw.grad is not None:
                raw.grad.zero_()

        result = self.contract_distributed()

        local_print(f"[Rank {self.rank}] Contraction result shape: {result.shape}")

        # Resolve loss
        loss_obj = LossRegistry.resolve(loss)
        resolved_target = TargetResolver.resolve(
            target, result.shape, self.backend, engine=self
        )
        loss_val = loss_obj(result, resolved_target, self.backend)

        local_print(f"[Rank {self.rank}] loss.requires_grad: {loss_val.requires_grad}")

        # Collect raw tensors for autograd
        raw_core_tensors = []
        core_names = []
        for name in self._local_qctn.cores:
            weight = self._local_qctn.cores_weights[name]
            raw = weight.tensor if isinstance(weight, TNTensor) else weight
            if not raw.requires_grad:
                continue
            if hasattr(raw, 'is_leaf') and not raw.is_leaf:
                continue
            raw_core_tensors.append(raw)
            core_names.append(name)
        # print(f"train core_names: {core_names}")
        # print(f"raw_core_tensors: {raw_core_tensors}")
        # exit()
        gradients = torch.autograd.grad(
            outputs=loss_val,
            inputs=raw_core_tensors,
            create_graph=False,
            retain_graph=False,
            allow_unused=False
        )

        grads = []
        for i, grad in enumerate(gradients):
            if grad is None:
                grads.append(torch.zeros_like(raw_core_tensors[i]))
            else:
                grads.append(grad.contiguous())

        return loss_val, grads
    
    def train_step(self, optimizer, target=None, loss=None) -> float:
        """Execute a single training step.

        Args:
            optimizer: Optimizer with step(grads) interface.
            target: Target for loss computation.
            loss: Loss specification.

        Returns:
            Loss value as float.
        """
        loss_val, grads = self.contract_distributed_with_gradient(
            target=target, loss=loss
        )
        optimizer.step(grads)
        return loss_val.item() if hasattr(loss_val, 'item') else float(loss_val)
    
    # Backward compatibility aliases
    @property
    def mpi(self):
        """Alias for comm (backward compatibility)."""
        return self.comm
    
    @property
    def ctx(self):
        """Get distributed context."""
        return self.comm.get_context()
    
    def is_main_process(self) -> bool:
        """Check if this is the main process."""
        return self.rank == 0

    # ==================== Distributed Model Saving ====================
    
    def save_cores_distributed(self, file_path: str, metadata: Optional[Dict[str, str]] = None):
        """
        Save all core weights from all processes to a single file.
        
        This method gathers core weights from all processes to rank 0,
        then saves them to a safetensors file.
        
        Args:
            file_path: Path to save the safetensors file
            metadata: Optional metadata dictionary
        """
        import torch
        import torch.distributed as dist

        if not self._is_initialized:
            raise RuntimeError("Must call init_distributed() before save_cores_distributed()")
        
        # Step 1: Gather all partition info to rank 0
        # Each rank has its own local_cores in _contract_plan
        local_cores = self._contract_plan.local_cores if self._contract_plan else []
        
        # Prepare local weights dictionary
        local_weights = {}
        for name in local_cores:
            if name in self._local_qctn.cores_weights:
                weight = self._local_qctn.cores_weights[name]
                if isinstance(weight, TNTensor):
                    tensor = (weight.tensor * weight.scale).detach().cpu()
                else:
                    tensor = weight.detach().cpu()
                local_weights[name] = tensor
        
        # Step 2: Gather all weights to rank 0
        # Use all_gather_object for simplicity (works with any Python objects)
        if self.world_size > 1:
            # Gather core names from all ranks
            all_weights_list = [None] * self.world_size
            dist.all_gather_object(all_weights_list, local_weights)
        else:
            all_weights_list = [local_weights]
        
        # Step 3: Rank 0 merges and saves
        if self.rank == 0:
            try:
                from safetensors.numpy import save_file
            except ImportError as exc:
                raise ImportError(
                    "safetensors is required to save cores; "
                    "install it with `pip install safetensors`."
                ) from exc
            
            # Merge all weights
            merged_weights = {}
            for weights_dict in all_weights_list:
                for name, tensor in weights_dict.items():
                    merged_weights[name] = tensor
            
            # Convert to numpy and prepare for saving
            tensor_dict = {}
            for core_name, tensor in merged_weights.items():
                if hasattr(tensor, 'numpy'):
                    tensor_dict[f"core_{core_name}"] = tensor.numpy()
                else:
                    tensor_dict[f"core_{core_name}"] = tensor.cpu().numpy()
            
            # Prepare metadata
            metadata_dict = {} if metadata is None else {str(k): str(v) for k, v in metadata.items()}
            metadata_dict['world_size'] = str(self.world_size)
            metadata_dict['num_cores'] = str(len(tensor_dict))
            
            # Save to file
            save_file(tensor_dict, str(file_path), metadata=metadata_dict)
            print(f"[DistributedEngine] Saved {len(tensor_dict)} cores to {file_path}")
        
        # Synchronize all processes
        if self.world_size > 1:
            dist.barrier()
