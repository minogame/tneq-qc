import numpy as np
import itertools
import re, random
from pathlib import Path
from typing import Union, Any, Optional, Mapping
from ..config import Configuration
from ..backends.copteinsum import ContractorOptEinsum
from .tn_tensor import TNTensor
from .tn_graph import TNGraph

class QCTNHelper:
    """
    Helper class for Quantum Circuit Tensor Network (QCTN) operations.
    Provides methods for converting quantum circuit graphs to adjacency matrices and counting qubits.
    """

    @staticmethod
    def iter_symbols(extend=False):
        """
        Generate a sequence of symbols for quantum circuit cores.
        If extend is True, use a range of Chinese characters; otherwise, use uppercase letters
        """
 
        if extend:
            symbols = [chr(i) for i in range(0x4E00, 0x9FFF + 1)]
            random.shuffle(symbols)  # Shuffle the symbols for randomness
            symbols = "".join(symbols)
        else:
            symbols = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        for symbol in symbols:
            yield symbol

    @staticmethod
    def generate_example_graph(n=16, target=False, graph_type="any", dim_char=None):
        """Generate an example quantum circuit graph."""
        if target:
            return  "-2-A-5-----C-3-----E-2-\n" \
                    "-2-----B----4------E-2-\n" \
                    "-2-A-4-B-7-C-2-D-4-E-2-\n" \
                    "-2-----B-6-----D-----2-\n" \
                    "-2-A-3-----C-8-D-----2-"
        else:
            def generate_mps_graph(n, dim_char=None):
                graph = ""
                # char_list = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
                # char_list = [next(QCTNHelper.iter_symbols(True)) for _ in range(n)]
                import opt_einsum
                char_list = [opt_einsum.get_symbol(i) for i in range(n)]

                if dim_char is None:
                    dim_char = '3'

                for i in range(n):
                    cid = i - 1
                    nid = i
                    if i == 0:
                        line = f"-{dim_char}-" + char_list[i] + (n - 2) * 6 * "-" + f"-{dim_char}-"
                    elif i == n - 1:
                        line = f"-{dim_char}-" + (n - 2) * 6 * "-" + char_list[cid] + f"-{dim_char}-"
                    else:
                        line = f"-{dim_char}-"
                        line += cid * 6 * "-"
                        line += char_list[cid]
                        line += f"--{dim_char}--"
                        line += char_list[nid]
                        line += (n - nid - 2) * 6 * "-"
                        line += f"-{dim_char}-"
                    
                    graph += line + "\n"
                return graph
            
            def generate_tree_graph(n, dim_char='3'):
                "graph like a tree structure"
                """
                -3-------A-3-
                -3---B-3-A-3-
                -3---B-3-C-3-
                -3-------C-3-

                -3---------A-3-
                -3-----B-3-A-3-
                -3-C-3-B-----3-
                -3-C-3-D-----3-
                -3-----D-3-E-3-
                -3---------E-3-
                """
                graph = ""
                import opt_einsum
                char_list = [opt_einsum.get_symbol(i) for i in range(n)]

                if dim_char is None:
                    dim_char = '3'
                
                m = n // 2

                left = (m - 1) * 4
                right = 0
                for i in range(m):
                    if i == 0:
                        line = "-" * left
                        line += char_list[i]

                        left -= 4
                    else:
                        line = "-" * left
                        line += char_list[i] + f"-{dim_char}-" + char_list[i - 1]
                        line += '-' * right

                        left -= 4
                        right += 4

                    graph += '-' + dim_char + '-' + line + '-' + dim_char + '-' + "\n"
                
                if n % 2 == 1:
                    line = char_list[m - 1] + '-' * ((m - 1) * 4)

                    graph += '-' + dim_char + '-' + line + '-' + dim_char + '-' + "\n"

                left = 0
                right = (m - 2) * 4
                for i in range(m, m * 2):
                    if i < m * 2 - 1:
                        line = "-" * left
                        line += char_list[i - 1] + f"-{dim_char}-" + char_list[i]
                        line += '-' * right

                        left += 4
                        right -= 4
                    else:
                        line = "-" * left
                        line += char_list[i - 1]
                    graph += '-' + dim_char + '-' + line + '-' + dim_char + '-' + "\n"
                
                return graph
            
            def generate_wall_graph_col(n, L, dim_char='3'):
                """
                Generate a brick wall structure graph.
                n: number of qubits (rows)
                L: number of layers/columns
                dim_char: dimension character for physical indices
                
                Brick wall structure: alternating layers of two-qubit gates
                - Even layers (0, 2, 4, ...): gates on pairs (0,1), (2,3), (4,5), ...
                - Odd layers (1, 3, 5, ...): gates on pairs (1,2), (3,4), (5,6), ...
                
                Example with n=4, L=4:
                -3-A---3---B-----3-
                -3-A-3-C-3-B-3-D-3-
                -3-E-3-C-3-F-3-D-3-
                -3-E---3---F-----3-
                
                char indices are assigned in row-major order (by row, then by layer)
                """

                graph = ""
                import opt_einsum
                
                if dim_char is None:
                    dim_char = '3'
                
                # Calculate total number of chars needed
                # Each layer has floor(n/2) or ceil(n/2) interactions depending on parity
                total_chars = L * (n // 2)
                char_list = [opt_einsum.get_symbol(i) for i in range(total_chars)]
                
                # Create a 2D array to store which char connects which qubits
                # char_map[layer][pair_index] = char_symbol
                char_map = {}
                char_idx = 0
                
                for layer in range(L):
                    char_map[layer] = {}
                    if layer % 2 == 0:
                        # Even layer: pairs (0,1), (2,3), (4,5), ...
                        for pair_idx in range(n // 2):
                            char_map[layer][pair_idx] = char_list[char_idx]
                            char_idx += 1
                    else:
                        # Odd layer: pairs (1,2), (3,4), (5,6), ...
                        for pair_idx in range((n - 1) // 2):
                            char_map[layer][pair_idx] = char_list[char_idx]
                            char_idx += 1
                
                # Generate the graph string
                for row in range(n):
                    line = f"-{dim_char}-"
                    
                    for layer in range(L):
                        if layer % 2 == 0:
                            # Even layer: pairs (0,1), (2,3), (4,5), ...
                            pair_idx = row // 2
                            if row % 2 == 0 and pair_idx < n // 2:
                                # First qubit in pair
                                line += char_map[layer][pair_idx]
                                line += f"-{dim_char}-"
                            elif row % 2 == 1 and pair_idx < n // 2:
                                # Second qubit in pair
                                line += char_map[layer][pair_idx]
                                line += f"-{dim_char}-"
                            else:
                                # No gate for this qubit in this layer
                                line += f"---{dim_char}---"
                        else:
                            # Odd layer: pairs (1,2), (3,4), (5,6), ...
                            if row == 0:
                                # First qubit has no gate in odd layers
                                line += f"---{dim_char}---"
                            elif row == n - 1:
                                # Last qubit has no gate in odd layers (if n is even)
                                line += f"---{dim_char}---"
                            else:
                                # Middle qubits
                                pair_idx = (row - 1) // 2
                                if row % 2 == 1 and pair_idx < (n - 1) // 2:
                                    # First qubit in pair
                                    line += char_map[layer][pair_idx]
                                    line += f"-{dim_char}-"
                                elif row % 2 == 0 and pair_idx < (n - 1) // 2:
                                    # Second qubit in pair
                                    line += char_map[layer][pair_idx]
                                    line += f"-{dim_char}-"
                                else:
                                    # No gate
                                    line += f"---{dim_char}---"
                    
                    line += f"-{dim_char}-"
                    graph += line + "\n"
                
                return graph.rstrip()

            def generate_wall_graph(n, L, dim_char='3'):
                """
                Example with n=4, L=4:
                -3-A-3-----B-----3-
                -3-A-3-C-3-B-3-D-3-
                -3-E-3-C-3-F-3-D-3-
                -3-E-3-----F-----3-

                """

                graph = ""
                import opt_einsum
                
                if dim_char is None:
                    dim_char = '3'
                
                # Calculate total number of chars needed
                # Each layer has floor(n/2) or ceil(n/2) interactions depending on parity
                total_chars = L * (n // 2)
                char_list = [opt_einsum.get_symbol(i) for i in range(total_chars)]
                
                line_list = [['-' for i in range(4 * L)] for j in range(n)]

                for i in range(n):
                    line_list[i][-2] = dim_char

                idx = 0

                m = L // 2
                for i in range(n - 1):
                    for j in range(m):
                        offset = 0 if i % 2 == 0 else 4

                        line_list[i][offset + 8 * j] = char_list[idx]
                        line_list[i+1][offset + 8 * j] = char_list[idx]
                        if j < m - 1 or (j == m - 1 and i > 0):
                            line_list[i][offset + 8 * j + 2] = dim_char
                        if j < m - 1 or (j == m - 1 and i != n - 2):
                            line_list[i+1][offset + 8 * j + 2] = dim_char
                        
                        idx += 1
                
                for i in range(n):
                    graph += "-" + dim_char + "-" + ''.join(line_list[i]) + "\n"


                return graph.rstrip()

            def generate_circuit_graph(n, dim_char='2'):
                graph = ""
                import opt_einsum
                char_list = [opt_einsum.get_symbol(i) for i in range(n)]

                if dim_char is None:
                    dim_char = '2'
                
                for i in range(n):
                    graph += "-" + char_list[i] + "-" + dim_char + "-\n"
                return graph
                
            
            def generate_mx_graph(n, dim_char='2'):
                graph = ""
                import opt_einsum
                char_list = [opt_einsum.get_symbol(i) for i in range(n)]

                if dim_char is None:
                    dim_char = '2'
                
                for i in range(n):
                    graph += "-" + dim_char + "-" + char_list[i] + "-" + dim_char + "-\n"
                return graph


            if graph_type == "mps":
                return generate_mps_graph(n, dim_char)
            elif graph_type == "tree":
                return generate_tree_graph(n, dim_char)
            elif graph_type == "wall":
                # For wall graph, we need to determine L (number of layers)
                # Default to n layers if not specified
                L = 4
                return generate_wall_graph(n, L, dim_char)
            elif graph_type == "circuit":
                return generate_circuit_graph(n, dim_char)
            elif graph_type == "mx":
                return generate_mx_graph(n, dim_char)

            return generate_mps_graph(n, dim_char)

            
            # return  "-3-A-3-"
            # return  "-3-A-3-B-3-C-3-D-3-"

            # return  "-3-A-3-\n" \
            #         "-3-A-3-"

            # return  "-3-A-3-B-3-C-3-\n" \
            #         "-3-A-3-B-3-C-3-"

            # return  "-3-A-3-\n" \
            #         "-3-A-3-\n" \
            #         "-3-A-3-"

            # return  "-3-A-----3-\n" \
            #         "-3-A-3-B-3-\n" \
            #         "-3-----B-3-"

            # return  "-3-A-3-\n" \
            #         "-3-A-3-\n" \
            #         "-3-A-3-\n" \
            #         "-3-A-3-"


            # return  "-3-a-3-b-3-c---------3-\n" \
            #         "-3-a-3-b-------------3-\n" \
            #         "-3-a-3-b-3-c---------3-\n" \
            #         "-3---------c---------3-"
        
            # return  "-3-a-3-----c-3-d-3-e-3-\n" \
            #         "-3-a-3-b-3-----d-3-e-3-\n" \
            #         "-3-a-3-b-3-c-3-----e-3-\n" \
            #         "-3-----b-3-c-3-d-3-e-3-"
        
            # return  "-3-A--------3--C--3--D-3-\n" \
            #         "-3-A--3--B--------3--D-3-\n" \
            #         "-3-------B--3--C--3--D-3-"
        
            # return  "-3-a-------------3-e-3-\n" \
            #         "-3-a-3-b-----3-d-3-e-3-\n" \
            #         "-3-----b-3-c-3-d-----3-\n" \
            #         "-3---------c---------3-"
            
            return  "-3-a-------------3-e-3-\n" \
                    "-3-a-3-b-----3-d-3-e-3-\n" \
                    "-3-f-3-b-3-c-3-d-3-g-3-\n" \
                    "-3-f-3-----c-----3-g-3-"
        
            

            # return  "-3-A-------------------3-\n" \
            #         "-3-A--3--B-------------3-\n" \
            #         "-3-------B--3--C-------3-\n" \
            #         "-3-------------C--3--D-3-\n" \
            #         "-3-------------------D-3-"
        
            return  "-3-A-------------------------3-\n" \
                    "-3-A--3--B-------------------3-\n" \
                    "-3-------B--3--C-------------3-\n" \
                    "-3-------------C--3--D-------3-\n" \
                    "-3-------------------D--3--E-3-\n" \
                    "-3-------------------------E-3-"
            

            # circuit_states.
            # (1, K), 001 


            # return  "-2-A-2-"
            # return  "-2-A-3-B-4-C-3-D-2-"
        
            # return  "-2-A-2-\n" \
            #         "-2-A-2-"

            return  "-2-A-------------------4-\n" \
                    "-2-A--3--B-------------4-\n" \
                    "-2-------B--3--C-------4-\n" \
                    "-2-------------C--3--D-4-\n" \
                    "-2-------------------D-4-"


            return  "-2-A-5-----C-3-----E-2-\n" \
                    "-2-----B----4------E-2-\n" \
                    "-2-A-4-B-7-C-2-D-4-E-2-\n" \
                    "-2-----B-6-----D-----2-\n" \
                    "-2-A-3-----C-8-D-----2-"
        
            return  "-2-A--------5--C--5--D-5-\n" \
                    "-2-A--5--B--------5--D-5-\n" \
                    "-2-------B--5--C--5--D-5-"
        
            return  "-2-A--------2--C--2--D-2-\n" \
                    "-2-A--2--B--------2--D-2-\n" \
                    "-2-------B--2--C--2--D-2-"
        

            return  "-2-A--------5--C--7--D--9--E--9--F--------9-\n" \
                    "-2-A--3--B--------7--D--------9--F--7--G-9-\n" \
                    "-2-------B--5--C--7--D--9--E--------9--G-9-"
        

            return  "-2-A--------5--C--------9--E--9--F--7-----9--H-9-\n" \
                    "-2-A--3--B--5--C--7--D--9--E--9--F--7--G--9--H-9-\n" \
                    "-2-------B--------7--D--9--E--9--------G-------9-"


            return  "-2-A--------5--C-------7-\n" \
                    "-2-A--3--B--5--C--7--D-9-\n" \
                    "-2-------B--------7--D-9-"
        
            return  "-2-A--------5--C-------5-\n" \
                    "-2-A--5--B--5--C--5--D-5-\n" \
                    "-2-------B--------5--D-5-"

        
            

            # return  "-2-A-----------------------------------------A-2-\n" \
            #         "-2-A--2--B-----------------------------B--2--A-2-\n" \
            #         "-2-------B--2--C-----------------C--2--B-------2-\n" \
            #         "-2-------------C--2--D-----D--2--C-------------2-\n" \
            #         "-2-------------------D--2--D-------------------2-"

            return  "-2-----B-5-C-3-D-----2-\n" \
                    "-2-A-4---------D-----2-\n" \
                    "-2-A-4-B-7-C-2-D-4-E-2-\n" \
                    "-2-A-3-B-6---------E-2-\n" \
                    "-2---------C-8-----E-2-"
            """
            -2-A-------------------2-
            -2-A--2--B-------------2-
            -2-------B--2--C-------2-
            -2-------------C--2--D-2-
            -2-------------------D-2-
            """


    @staticmethod
    def generate_example_graph2():
        """Generate an example quantum circuit graph."""

    
    @staticmethod
    def generate_random_example_graph(nqubits=5, ncores=3):
        """Generate a random quantum circuit graph with specified number of qubits and cores."""

        cores = "".join([next(QCTNHelper.iter_symbols(True)) for _ in range(ncores)])
        graph = ""
        for i in range(nqubits):
            qubit = f"-{np.random.randint(2, 10)}-"
            for j in cores:
                if np.random.rand() > 0.5:
                    qubit += f"{j}-{np.random.randint(2, 10)}-"

            graph += f"{qubit}\n"

        return graph.strip()
    
    @staticmethod
    def triu_ndindex(n):
        """Generate indices for the upper triangular part of a square matrix."""
        for i in range(n):
            for j in range(i + 1, n):
                yield (i, j)

class QCTN:
    """
    Quantum Circuit Tensor Network (QCTN) class for quantum circuit simulation.
    
    Initialization Format:
        - A graph representing the quantum circuit, where open edges are qubits and marks are cores.
        - Each core is a tensor with a shape corresponding to the number of qubits it connects to.

    Example:
        -2-----B-5-C-3-D-----2-
        -2-A-4---------D-----2-
        -2-A-4-B-7-C-2-D-4-E-2-
        -2-A-3-B-6---------E-2-
        -2---------C-8-----E-2-

        where:
            - A, B, C, D, E are cores (tensors).
            - The numbers represent the rank of each core.

    Attributes:
        nqubits (int): Number of qubits in the quantum circuit.
        adjacency_matrix: (np.ndarray): Adjacency matrix representing the connection ranks with empty diagonal entries.
        circuit (tuple): (Input ranks, Connection ranks, Output ranks) for each core.
 
    """

    def __init__(self, graph, backend=None):
        """
        Initialize the QCTN with a quantum circuit graph.
        
        Args:
            graph (str): A string representation of the quantum circuit graph.
            backend (ComputeBackend): The backend to use for computation.
        """
        self.qubits = graph.strip().splitlines()
        self.nqubits = len(self.qubits)
        self.qubit_indices = list(range(self.nqubits))

        self.graph = graph
        self.tn_graph = TNGraph(graph, self.nqubits)

        import opt_einsum
        idx2core = [opt_einsum.get_symbol(i) for i in range(10000)]
        core2idx = {c: i for i, c in enumerate(idx2core)}

        full_cores = set([opt_einsum.get_symbol(i) for i in range(10000)])
        # full_cores = set([next(QCTNHelper.iter_symbols(True)) for _ in range(20000)])

        self.cores = list(set([c for c in graph if c in full_cores]))
        # 把self.cores按照在full_cores的相对顺序排序
        self.cores.sort(key=lambda x: core2idx[x])

        print(f"num cores: {len(self.cores)}")


        # self.cores = [opt_einsum.get_symbol(i) for i in range(self.nqubits-1)]


        # self.cores = list(set([c for c in graph if c.isupper()]))
        # if not self.cores:
        #     # If no uppercase core symbols found, try to find all chars in the CJK Unified Ideographs range
        #     self.cores = list(set([c for c in graph if 0x4E00 <= ord(c) <= 0x9FFF]))
        self.ncores = len(self.cores)

        # self.cores = sorted(self.cores)

        # This will build the attributes `self.circuit` and `self.adjacency_matrix`
        self._circuit_to_adjacency()

        self.backend = backend
        self._loaded_metadata: Optional[Mapping[str, str]] = None

        # Initialize the circuit with input ranks, adjacency matrix, and output ranks
        # self.initialize_random_key = jax.random.PRNGKey(0)

        # Initialize the cores with random values
        self.cores_weights = {}
        self._init_cores()

        # Placeholders for einsum expressions
        self.einsum_expr = None

    @classmethod
    def envolve_from_another_qctn(cls, qctn, strategies=None):
        """
        Create a new QCTN instance by evolving from another QCTN instance.
        
        Args:
            qctn (QCTN): The original QCTN instance to evolve from.
            strategies (list, optional): A list of strategies for evolution. Defaults to None.
        
        Returns:
            QCTN: A new QCTN instance evolved from the original.
        """
        if strategies is None \
           or (isinstance(strategies, list) and not strategies):
            # If no strategies are provided, simply copy the original QCTN
            # This is useful for cases where we want to create a new instance without any modifications.
            # The optimization can be skipped.
            if isinstance(qctn, cls):
                # If qctn is already an instance of QCTN, copy it
                return cls.copy(qctn)
            else:
                # If qctn is not an instance of QCTN, raise an error
                raise TypeError("qctn must be an instance of QCTN.")

        if isinstance(strategies, function):
            new_graph = strategies(qctn.graph)
            return cls(new_graph)
        elif isinstance(strategies, list):
            # Apply each strategy to the original graph
            new_graph = qctn.graph
            for strategy in strategies:
                if callable(strategy):
                    new_graph = strategy(new_graph)
                else:
                    raise TypeError("Each strategy must be a callable function.")
            return cls(new_graph)

    def __repr__(self):
        """
        String representation of the QCTN object.
        """
        # print(f"adjacency_table: {self.adjacency_table}")

        adjacency_matrix = np.empty((self.ncores, self.ncores), dtype=object)
        for i in range(self.ncores):
            for j in range(self.ncores):
                adjacency_matrix[i, j] = str(self.adjacency_matrix[i, j])
        
        circuit_input = [str(rank) for rank in self.circuit[0]]
        circuit_output = [str(rank) for rank in self.circuit[2]]

        return f"circuit_input = \n{circuit_input}\n adjacency_matrix = \n{adjacency_matrix}\n adjacency_table = \n{self.adjacency_table}\n circuit_output = \n{circuit_output}\n"

    def _circuit_to_adjacency(self,):
        """
        Convert the quantum circuit graph to adjacency table.
        
        This method builds self.adjacency_table, a list where each element corresponds to a core
        and contains a dict with:
        - 'core_idx': int, index of the core
        - 'core_name': str, name of the core
        - 'in_edge_list': list of dicts with keys:
            {'neighbor_idx', 'neighbor_name', 'edge_rank', 'qubit_idx'}
            For input edges (from circuit input), neighbor_idx = -1, neighbor_name = ""
        - 'out_edge_list': list of dicts with keys:
            {'neighbor_idx', 'neighbor_name', 'edge_rank', 'qubit_idx'}
            For output edges (to circuit output), neighbor_idx = -1, neighbor_name = ""
        - 'input_shape': list of input ranks (from in_edge_list)
        - 'output_shape': list of output ranks (from out_edge_list)
        - 'input_dim': int, product of input_shape
        - 'output_dim': int, product of output_shape
        """
        
        cores = "".join(self.cores)
        dict_core2idx = {core: idx for idx, core in enumerate(self.cores)}
        self.dict_core2idx = dict_core2idx  # Store for later use
        
        # Initialize adjacency_table
        self.adjacency_table = []
        for idx, core_name in enumerate(self.cores):
            self.adjacency_table.append({
                'core_idx': idx,
                'core_name': core_name,
                'in_edge_list': [],
                'out_edge_list': [],
                'input_shape': [],
                'output_shape': [],
                'input_dim': 1,
                'output_dim': 1,
            })
        
        input_pattern = re.compile(rf"^(\d+)([{cores}])")
        output_pattern = re.compile(rf"([{cores}])(\d+)$")
        connect_pattern = re.compile(rf"([{cores}])(\d+)(?=[{cores}])")

        for qubit_idx, line in enumerate(self.qubits):
            # print(f'qubit_idx: {qubit_idx}, line: {len(line)}, {line[-10:]}')
            # if qubit_idx == 2000:
            #     print(line)
            line = line.strip().replace("-", "")
            m_input = input_pattern.match(line)
            m_output = output_pattern.search(line)
            
            if m_input is None or m_output is None:
                # raise ValueError(
                #     f"Qubit {qubit_idx}: failed to parse line '{self.qubits[qubit_idx].strip()}' "
                #     f"(cleaned: '{line}'). "
                #     f"{'input pattern not matched' if m_input is None else 'output pattern not matched'}."
                # )

                return
            
            input_rank, input_core = m_input.groups() if m_input else (0, None)
            output_core, output_rank = m_output.groups() if m_output else (None, 0)
            input_rank, output_rank = int(input_rank), int(output_rank)
            input_core_idx = dict_core2idx[input_core]
            output_core_idx = dict_core2idx[output_core]
            
            # Add input edge: from circuit input (-1, "") to input_core
            self.adjacency_table[input_core_idx]['in_edge_list'].append({
                'neighbor_idx': -1,
                'neighbor_name': "",
                'edge_rank': input_rank,
                'qubit_idx': qubit_idx
            })
            
            # Add output edge: from output_core to circuit output (-1, "")
            self.adjacency_table[output_core_idx]['out_edge_list'].append({
                'neighbor_idx': -1,
                'neighbor_name': "",
                'edge_rank': output_rank,
                'qubit_idx': qubit_idx
            })
            
            for match in connect_pattern.finditer(line):
                end_pos = match.end()
                if end_pos >= len(line):
                    print(f"Warning: end_pos {end_pos} out of range for line '{line}'")
                    break

                core1, rank1 = match.groups()
                core2 = line[end_pos]

                core1_idx = dict_core2idx[core1]
                core2_idx = dict_core2idx[core2]
                rank1 = int(rank1)
                
                # Add to adjacency table
                # core1 -> core2: out_edge for core1, in_edge for core2
                self.adjacency_table[core1_idx]['out_edge_list'].append({
                    'neighbor_idx': core2_idx,
                    'neighbor_name': core2,
                    'edge_rank': rank1,
                    'qubit_idx': qubit_idx
                })
                self.adjacency_table[core2_idx]['in_edge_list'].append({
                    'neighbor_idx': core1_idx,
                    'neighbor_name': core1,
                    'edge_rank': rank1,
                    'qubit_idx': qubit_idx
                })

        # Compute input_shape, output_shape, input_dim, output_dim for each core
        for core_info in self.adjacency_table:
            core_info['input_shape'] = [edge['edge_rank'] for edge in core_info['in_edge_list']]
            core_info['output_shape'] = [edge['edge_rank'] for edge in core_info['out_edge_list']]
            core_info['input_dim'] = int(np.prod(core_info['input_shape'])) if core_info['input_shape'] else 1
            core_info['output_dim'] = int(np.prod(core_info['output_shape'])) if core_info['output_shape'] else 1

        # Build adjacency_matrix from adjacency_table for backward compatibility
        self.adjacency_matrix = np.empty((self.ncores, self.ncores), dtype=object)
        for i in range(self.ncores):
            for j in range(self.ncores):
                self.adjacency_matrix[i, j] = []
        
        for core_info in self.adjacency_table:
            core_idx = core_info['core_idx']
            for edge in core_info['out_edge_list']:
                if edge['neighbor_idx'] >= 0:  # Skip output edges (to circuit output)
                    self.adjacency_matrix[core_idx, edge['neighbor_idx']].append(edge['edge_rank'])
                    self.adjacency_matrix[edge['neighbor_idx'], core_idx].append(edge['edge_rank'])

        # Build circuit tuple for backward compatibility
        input_ranks = np.empty(self.ncores, dtype=object)
        output_ranks = np.empty(self.ncores, dtype=object)
        for i in range(self.ncores):
            input_ranks[i] = self.adjacency_table[i]['input_shape'].copy()
            output_ranks[i] = self.adjacency_table[i]['output_shape'].copy()
        self.circuit = (input_ranks, self.adjacency_matrix, output_ranks)

        # for debug, print adjacency_table
        # for core_info in self.adjacency_table:
        #     print(f"Core {core_info['core_name']} (idx {core_info['core_idx']}):")
        #     print(f"  input_shape: {core_info['input_shape']}, output_shape: {core_info['output_shape']}")
        #     print(f"  input_dim: {core_info['input_dim']}, output_dim: {core_info['output_dim']}")
        #     print(f"  In edges: {core_info['in_edge_list']}")
        #     print(f"  Out edges: {core_info['out_edge_list']}")

    def _init_cores(self):
        """
        Initialize the cores of the quantum circuit with random values.
        
        For each core, use the pre-computed values from adjacency_table:
        - input_shape: ranks from in_edge_list (already ordered by qubit_idx)
        - output_shape: ranks from out_edge_list (already ordered by qubit_idx)
        - input_dim: product of input_shape
        - output_dim: product of output_shape
        
        The core tensor is initialized with shape [input_dim, output_dim], 
        then reshaped to input_shape + output_shape.
        
        Returns:
            None: The cores are stored in the `cores_weights` attribute.
        """

        for idx, core_info in enumerate(self.adjacency_table):
            core_name = core_info['core_name']
            input_shape = core_info['input_shape']
            output_shape = core_info['output_shape']
            input_dim = core_info['input_dim']
            output_dim = core_info['output_dim']
            
            # print(f"_init_cores: {idx} {input_shape} {output_shape} {input_dim} {output_dim}")

            # Initialize core with shape [input_dim, output_dim]
            core = self.backend.init_random_core([input_dim, output_dim])
            
            # Reshape to input_shape + output_shape
            full_shape = input_shape + output_shape
            core = self.backend.reshape(core, full_shape)

            self.cores_weights[core_name] = core
            # self.cores_weights[core_name] = TNTensor(core)
            # self.cores_weights[core_name] = core
            # self.cores_weights[core_name].auto_scale()

    def set_cores(self, cores, strict: bool = True):
        """
        Set core tensors from a list or dict.

        Each supplied tensor is validated to have the **same total number of
        elements** (numel) as the corresponding existing core weight.  If the
        shapes differ but the sizes match, the tensor is reshaped to the
        target core weight's shape.

        Args:
            cores (list | dict):
                * **list** – tensors are matched to ``self.cores`` by
                  positional order.

                  - *strict=True*: ``len(cores)`` must equal ``self.ncores``.
                  - *strict=False*: only the first ``min(len(cores), ncores)``
                    cores are set; a warning is emitted if the lengths differ.

                * **dict** – keys are core names (single-character symbols).

                  - *strict=True*: the key set must exactly equal
                    ``set(self.cores)`` (no missing, no extra keys).
                  - *strict=False*: only the intersection of keys is used;
                    warnings list any missing or extra keys.

            strict (bool): Whether to require an exact one-to-one match.
                Defaults to ``True``.

        Raises:
            TypeError: If *cores* is neither a list nor a dict.
            ValueError: If *strict=True* and the sizes / keys do not match,
                or if any tensor's total element count differs from its
                target core weight.
        """
        import warnings

        if isinstance(cores, list):
            self._set_cores_from_list(cores, strict)
        elif isinstance(cores, dict):
            self._set_cores_from_dict(cores, strict)
        else:
            raise TypeError(
                f"cores must be a list or dict, got {type(cores).__name__}"
            )

    # ------------------------------------------------------------------
    # Internal helpers for set_cores
    # ------------------------------------------------------------------

    def _set_single_core(self, core_name: str, tensor):
        """
        Validate *tensor* against the existing weight for *core_name*,
        reshape if necessary, and store it.

        Raises:
            ValueError: If the total number of elements does not match.
        """
        target = self.cores_weights[core_name]
        target_shape = tuple(target.shape)
        target_numel = int(np.prod(target_shape))

        src_shape = tuple(tensor.shape)
        src_numel = int(np.prod(src_shape))

        if src_numel != target_numel:
            raise ValueError(
                f"Core '{core_name}': size mismatch — input has "
                f"{src_numel} elements (shape {src_shape}) but target "
                f"has {target_numel} elements (shape {target_shape})."
            )

        if src_shape != target_shape:
            tensor = self.backend.reshape(tensor, list(target_shape))

        self.cores_weights[core_name] = tensor

    def _set_cores_from_list(self, cores: list, strict: bool):
        import warnings

        if strict:
            if len(cores) != self.ncores:
                raise ValueError(
                    f"strict=True: expected {self.ncores} core tensors, "
                    f"got {len(cores)}."
                )
            for idx, tensor in enumerate(cores):
                self._set_single_core(self.cores[idx], tensor)
        else:
            n = min(len(cores), self.ncores)
            if len(cores) != self.ncores:
                warnings.warn(
                    f"strict=False: input list has {len(cores)} tensors but "
                    f"QCTN has {self.ncores} cores. Only the first {n} will "
                    f"be set.",
                    stacklevel=3,
                )
            for idx in range(n):
                self._set_single_core(self.cores[idx], cores[idx])

    def _set_cores_from_dict(self, cores: dict, strict: bool):
        import warnings

        input_keys = set(cores.keys())
        self_keys = set(self.cores)

        if strict:
            if input_keys != self_keys:
                missing = self_keys - input_keys
                extra = input_keys - self_keys
                parts = []
                if missing:
                    parts.append(f"missing keys ({len(missing)}): {missing}")
                if extra:
                    parts.append(f"extra keys ({len(extra)}): {extra}")
                raise ValueError(
                    f"strict=True: key mismatch — {'; '.join(parts)}."
                )
            for core_name in self.cores:
                self._set_single_core(core_name, cores[core_name])
        else:
            common = input_keys & self_keys
            missing = self_keys - input_keys
            extra = input_keys - self_keys
            if missing:
                warnings.warn(
                    f"strict=False: {len(missing)} core(s) missing from "
                    f"input dict and will keep their current weights: "
                    f"{missing}",
                    stacklevel=3,
                )
            if extra:
                warnings.warn(
                    f"strict=False: {len(extra)} extra key(s) in input dict "
                    f"will be ignored: {extra}",
                    stacklevel=3,
                )
            for core_name in self.cores:
                if core_name in common:
                    self._set_single_core(core_name, cores[core_name])

    def save_cores(self, file_path: Union[str, Path], metadata: Optional[Mapping[str, str]] = None):
        """Save all core tensors into a safetensors file."""

        if self.backend is None:
            raise RuntimeError("Backend must be initialized before saving cores.")

        try:
            from safetensors.numpy import save_file
        except ImportError as exc:
            raise ImportError("safetensors is required to save cores; install it with `pip install safetensors`.") from exc

        tensor_dict = {}
        for core_name, tensor in self.cores_weights.items():
            if isinstance(tensor, TNTensor):
                arr = self.backend.tensor_to_numpy(tensor.tensor * tensor.scale)
            else:
                arr = self.backend.tensor_to_numpy(tensor)
            if np.iscomplexobj(arr):
                tensor_dict[f"core_{core_name}_real"] = np.ascontiguousarray(arr.real)
                tensor_dict[f"core_{core_name}_imag"] = np.ascontiguousarray(arr.imag)
            else:
                tensor_dict[f"core_{core_name}"] = np.ascontiguousarray(arr)

        metadata_dict = {} if metadata is None else {str(k): str(v) for k, v in metadata.items()}
        save_file(tensor_dict, str(file_path), metadata=metadata_dict)

    def load_cores(self, file_path: Union[str, Path], strict: bool = True) -> Mapping[str, str]:
        """Load saved core tensors from a safetensors file."""

        if self.backend is None:
            raise RuntimeError("Backend must be initialized before loading cores.")

        try:
            from safetensors.numpy import load_file
        except ImportError as exc:
            raise ImportError("safetensors is required to load cores; install it with `pip install safetensors`.") from exc

        result = load_file(str(file_path))
        if isinstance(result, tuple) and len(result) == 2:
            tensor_dict, metadata = result
        else:
            tensor_dict = result
            metadata = {}

        for core_name in self.cores:
            key = f"core_{core_name}"
            key_real, key_imag = f"core_{core_name}_real", f"core_{core_name}_imag"
            if key_real in tensor_dict:
                array = tensor_dict[key_real] + 1j * tensor_dict[key_imag]
            elif key in tensor_dict:
                array = tensor_dict[key]
            else:
                if strict:
                    raise KeyError(f"Missing tensor for core {core_name} in {file_path}")
                continue
            tensor = self.backend.convert_to_tensor(array)
            tn_tensor = TNTensor(tensor)
            tn_tensor.auto_scale()
            self.cores_weights[core_name] = tn_tensor

        metadata_dict = {str(k): str(v) for k, v in metadata.items()}
        self._loaded_metadata = metadata_dict
        return metadata_dict

    @classmethod
    def from_pretrained(
        cls,
        graph: str,
        file_path: Union[str, Path],
        backend=None,
        strict: bool = True,
    ) -> "QCTN":
        """Create a QCTN instance loading core tensors from safetensors."""

        if backend is None:
            from ..backends.backend_factory import BackendFactory

            backend = BackendFactory.get_default_backend()

        instance = cls(graph, backend=backend)
        instance.load_cores(file_path, strict=strict)
        return instance


    def _contract_core_only(self, engine=ContractorOptEinsum):
        """
        Contract the quantum circuit tensor network without inputs.
        """

        return engine.contract_core_only(self)

    def _contract_with_inputs(self, inputs: Any = None, engine=ContractorOptEinsum):
        """
        Contract the quantum circuit tensor network with given inputs.

        Args:
            inputs (Any): The inputs for the contraction operation.
                It should be a tensor with the shape matching the input ranks of the circuit.

        Returns:
            The result of the contraction operation.
        """

        # Validate inputs
        if inputs is None:
            raise ValueError("Inputs must be provided for contraction.")
        if not isinstance(inputs, self.backend.get_tensor_type()):
            raise TypeError(f"Inputs must be a {self.backend.get_tensor_type()}.")
        if inputs.shape != tuple(itertools.chain.from_iterable(self.circuit[0])):
            raise ValueError(f"Input tensor shape {inputs.shape} does not match expected shape {tuple(itertools.chain.from_iterable(self.circuit[0]))}.")

        return engine.contract_with_inputs(self, inputs)

    def _contract_with_vector_inputs(self, inputs: list = None, engine=ContractorOptEinsum):
        """
        Contract the quantum circuit tensor network with given inputs.

        Args:
            inputs (list): The inputs for the contraction operation.
                It should be a tensor with the shape matching the input ranks of the circuit.

        Returns:
            The result of the contraction operation.
        """

        # Validate inputs
        if inputs is None:
            raise ValueError("Inputs must be provided for contraction.")
        if not all(isinstance(t, self.backend.get_tensor_type()) for t in inputs):
            raise TypeError(f"All elements in the list must be {self.backend.get_tensor_type()}.")
        if len(inputs) != self.nqubits:
            raise ValueError(f"Expected {self.nqubits} input vectors, got {len(inputs)}.")
        
        # Check dimensions of all inputed vectors
        if not all(t.ndim == 1 for t in inputs):
            raise ValueError("All input tensors must be 1-dimensional vectors.")

        if tuple(t.shape[0] for t in inputs) != tuple(itertools.chain.from_iterable(self.circuit[0])):
            raise ValueError(f"Input tensor shapes {tuple(t.shape for t in inputs)} do not match expected shape {tuple(itertools.chain.from_iterable(self.circuit[0]))}.")

        return engine.contract_with_vector_inputs(self, inputs)

    def _contract_with_QCTN(self, qctn, engine=ContractorOptEinsum):
        """
        Contract the quantum circuit tensor network with another QCTN instance.
        
        Args:
            qctn (QCTN): Another instance of QCTN to contract with.
        
        Returns:
            The result of the contraction operation.
        """

        if not list(itertools.chain.from_iterable(self.circuit[0])) == list(itertools.chain.from_iterable(qctn.circuit[0])):
            raise ValueError("Input ranks of the two QCTNs do not match.")
        if not list(itertools.chain.from_iterable(self.circuit[2])) == list(itertools.chain.from_iterable(qctn.circuit[2])):
            raise ValueError("Output ranks of the two QCTNs do not match.")
        
        return engine.contract_with_QCTN(self, qctn)
    
    def _contract_with_QCTN_for_gradient(self, qctn, engine=ContractorOptEinsum):
        """
        Contract the quantum circuit tensor network for a specific core gradient.
        
        Args:
            qctn : Another instance of QCTN to contract with.
        
        Returns:
            The result of the contraction operation for core gradient.
        """

        if not list(itertools.chain.from_iterable(self.circuit[0])) == list(itertools.chain.from_iterable(qctn.circuit[0])):
            raise ValueError("Input ranks of the two QCTNs do not match.")
        if not list(itertools.chain.from_iterable(self.circuit[2])) == list(itertools.chain.from_iterable(qctn.circuit[2])):
            raise ValueError("Output ranks of the two QCTNs do not match.")

        return engine.contract_with_QCTN_for_gradient(self, qctn)

    def contract(self, attach: Union[Any, 'QCTN', list] = None, engine=ContractorOptEinsum):
        """
        Contract the quantum circuit tensor network.

        Args:
            attach (Union[Any, list[Any], 'QCTN'], optional): The inputs for the contraction operation.
                If a tensor is provided, it should be a tensor with the shape matching the input ranks of the circuit.
                If a list of tensors is provided, it should contain vectors for each qubit.
                If a QCTN instance is provided, it will contract with that instance.
            engine (ContractorOptEinsum): The contraction engine to use. Default is ContractorOptEinsum.

        Returns:
            The result of the contraction operation.
        """

        if attach is None:
            return self._contract_core_only(engine)
        elif isinstance(attach, self.backend.get_tensor_type()):
            print('contract with tensor')
            return self._contract_with_inputs(attach, engine)
        elif isinstance(attach, list):
            print('contract with list of tensors')
            return self._contract_with_vector_inputs(attach, engine)
        elif isinstance(attach, QCTN):
            print('contract with QCTN')
            return self._contract_with_QCTN(attach, engine)
        else:
            raise TypeError(f"attach must be a {self.backend.get_tensor_type()}, a list of {self.backend.get_tensor_type()} or an instance of QCTN.")
    
    def _contract_with_self(self, engine=ContractorOptEinsum, circuit_array_input=None, circuit_list_input=None):
        """
        Contract the quantum circuit tensor network with itself.

        Args:
            engine (ContractorOptEinsum): The contraction engine to use. Default is ContractorOptEinsum.

        Returns:
            The result of the contraction operation.
        """

        return engine.contract_with_self(self, circuit_array_input, circuit_list_input)

    def _contract_with_self_for_gradient(self, engine=ContractorOptEinsum, circuit_array_input=None, circuit_list_input=None):
        """
        Contract the quantum circuit tensor network with itself.

        Args:
            engine (ContractorOptEinsum): The contraction engine to use. Default is ContractorOptEinsum.

        Returns:
            The result of the contraction operation.
        """

        return engine.contract_with_self_for_gradient(self, circuit_array_input, circuit_list_input)

    def contract_with_self(self, attach: Union[Any, 'QCTN', list] = None, engine=ContractorOptEinsum):
        """
        Contract the quantum circuit tensor network with itself.

        Args:
            engine (ContractorOptEinsum): The contraction engine to use. Default is ContractorOptEinsum.

        Returns:
            The result of the contraction operation.
        """
        if attach is None:
            return self._contract_with_self(engine)
        elif isinstance(attach, self.backend.get_tensor_type()):
            return self._contract_with_self(engine, circuit_array_input=attach)
        elif isinstance(attach, list):
            raise TypeError("attach must be None when contracting with self.")
        elif isinstance(attach, QCTN):
            raise TypeError("attach must be None when contracting with self.")
        else:
            raise TypeError("attach must be None when contracting with self.")
    
    def contract_with_self_for_gradient(self, attach: Union[Any, 'QCTN', list] = None, engine=ContractorOptEinsum):
        """
        Contract the quantum circuit tensor network with itself.

        Args:
            engine (ContractorOptEinsum): The contraction engine to use. Default is ContractorOptEinsum.

        Returns:
            The result of the contraction operation.
        """
        if attach is None:
            return self._contract_with_self_for_gradient(engine)
        elif isinstance(attach, self.backend.get_tensor_type()):
            return self._contract_with_self_for_gradient(engine, circuit_array_input=attach)
        elif isinstance(attach, list):
            raise TypeError("attach must be None when contracting with self.")
        elif isinstance(attach, QCTN):
            raise TypeError("attach must be None when contracting with self.")
        else:
            raise TypeError("attach must be None when contracting with self.")

    def contract_with_QCTN_for_gradient(self, attach, engine=ContractorOptEinsum):
        """
        Contract the quantum circuit tensor network, return the loss and the gradient for all cores,
        The attach must be a QCTN instance.
        The loss for gradient computation is (X @ Y^T - 1) ** 2, where X is self and Y is the attach QCTN.
        The gradient only computes the core gradients, not the input gradients.
        

        Args:
            attach ('QCTN'): The inputs for the contraction operation.
            engine (ContractorOptEinsum): The contraction engine to use. Default is ContractorOptEinsum.

        Returns:
            The result of value and grad.
        """
        if not isinstance(attach, QCTN):
            raise TypeError("attach must be an instance of QCTN.")
        return self._contract_with_QCTN_for_gradient(attach, engine)
    
    def optimize_contract_with_QCTN(self, attach, optimizer, engine=ContractorOptEinsum):
        """
        Optimize the contraction with another QCTN instance using a specified optimizer.

        Args:
            attach (QCTN): The QCTN instance to contract with.
            optimizer (Optimizer): The optimizer to use for the optimization process.
            engine (ContractorOptEinsum): The contraction engine to use. Default is ContractorOptEinsum.

        Returns:
            The optimized parameters after the contraction operation.
        """
        if not isinstance(attach, QCTN):
            raise TypeError("attach must be an instance of QCTN.")
        
        return optimizer.optimize(self.contract_with_QCTN_for_gradient, attach, engine=engine)

    # ================================================================
    # Graph manipulation helpers
    # ================================================================

    @staticmethod
    def _parse_qubit_line(line):
        """
        Parse a qubit line into a sequence of tokens.

        Given a graph line like ``-2-A-5-B-3-``, returns::

            [('dim', 2), ('core', 'A'), ('dim', 5), ('core', 'B'), ('dim', 3)]

        After stripping all dashes, the remaining characters are either
        digit-sequences (dimensions) or single non-digit characters (core
        symbols).  They always alternate ``dim, core, dim, core, …, dim``.

        Args:
            line (str): A single qubit line from the graph.

        Returns:
            list[tuple]: ``[(type, value), ...]`` where *type* is ``'dim'``
            or ``'core'``.
        """
        cleaned = line.strip().replace("-", "")
        result = []
        i = 0
        while i < len(cleaned):
            if cleaned[i].isdigit():
                j = i
                while j < len(cleaned) and cleaned[j].isdigit():
                    j += 1
                result.append(('dim', int(cleaned[i:j])))
                i = j
            else:
                result.append(('core', cleaned[i]))
                i += 1
        return result

    @staticmethod
    def _rebuild_qubit_line(tokens):
        """
        Rebuild a qubit line string from parsed tokens.

        Args:
            tokens: list of ``(type, value)`` tuples, e.g.
                ``[('dim', 2), ('core', 'A'), ('dim', 5)]``

        Returns:
            str: Rebuilt qubit line, e.g. ``-2-A-5-``
        """
        parts = [str(val) for _, val in tokens]
        return "-" + "-".join(parts) + "-"

    @staticmethod
    def _remap_graph(graph_lines, core_map):
        """
        Remap core symbols in graph lines according to *core_map*.

        Each character in every line is independently looked up in
        *core_map*; if found it is replaced, otherwise kept as-is.  This
        is safe because core symbols are single, non-digit, non-dash
        characters.

        Args:
            graph_lines (list[str]): Qubit line strings.
            core_map (dict[str, str]): ``{old_symbol: new_symbol}``.

        Returns:
            list[str]: Remapped qubit line strings.
        """
        new_lines = []
        for line in graph_lines:
            new_line = []
            for ch in line:
                new_line.append(core_map.get(ch, ch))
            new_lines.append("".join(new_line))
        return new_lines

    # ================================================================
    # Split / Merge operations
    # ================================================================

    def split(self, split_idx=None):
        """
        Split the QCTN into two QCTNs by core tensor index.

        Cores are divided into two groups:

        * **Group 1**: ``self.cores[:split_idx]``
        * **Group 2**: ``self.cores[split_idx:]``

        For each qubit line that contains cores from both groups, the bond
        dimension at the boundary becomes the output dimension for Group 1
        and the input dimension for Group 2.  Qubit lines that only
        contain cores from a single group are assigned entirely to that
        group's QCTN.

        Args:
            split_idx (int, optional): Index at which to split the core
                list.  Defaults to ``ncores // 2``.

        Returns:
            tuple[QCTN, QCTN]: Two new QCTN instances with the
            corresponding core weights copied.

        Raises:
            ValueError: If *split_idx* is out of range, or if cores from
                both groups are interleaved on any qubit line (i.e. a
                Group-1 core appears **after** a Group-2 core).
        """
        if split_idx is None:
            split_idx = self.ncores // 2

        if split_idx <= 0 or split_idx >= self.ncores:
            raise ValueError(
                f"split_idx must be between 1 and {self.ncores - 1}, "
                f"got {split_idx}"
            )

        cores_group1 = set(self.cores[:split_idx])
        cores_group2 = set(self.cores[split_idx:])

        lines_group1: list[str] = []
        lines_group2: list[str] = []

        for qubit_idx, line in enumerate(self.qubits):
            tokens = QCTN._parse_qubit_line(line)

            # Locate core tokens that belong to each group
            core_positions = [
                (i, tok[1])
                for i, tok in enumerate(tokens)
                if tok[0] == 'core'
            ]
            g1_pos = [(i, c) for i, c in core_positions if c in cores_group1]
            g2_pos = [(i, c) for i, c in core_positions if c in cores_group2]

            if g1_pos and g2_pos:
                last_g1 = max(i for i, _ in g1_pos)
                first_g2 = min(i for i, _ in g2_pos)

                if last_g1 >= first_g2:
                    raise ValueError(
                        f"Cannot split: cores from both groups are "
                        f"interleaved on qubit {qubit_idx}. Ensure that "
                        f"all Group-1 cores appear before Group-2 cores "
                        f"on every qubit line."
                    )

                # Group 1: [start … last_g1_core, dim_after_last_g1_core]
                g1_tokens = tokens[: last_g1 + 2]
                # Group 2: [dim_before_first_g2_core, first_g2_core … end]
                g2_tokens = tokens[first_g2 - 1 :]

                lines_group1.append(QCTN._rebuild_qubit_line(g1_tokens))
                lines_group2.append(QCTN._rebuild_qubit_line(g2_tokens))
            elif g1_pos:
                lines_group1.append(QCTN._rebuild_qubit_line(tokens))
            elif g2_pos:
                lines_group2.append(QCTN._rebuild_qubit_line(tokens))
            # else: qubit has no cores — skip (shouldn't happen)

        if not lines_group1:
            raise ValueError(
                "After split, Group 1 has no qubit lines. "
                "All qubits belong to Group 2."
            )
        if not lines_group2:
            raise ValueError(
                "After split, Group 2 has no qubit lines. "
                "All qubits belong to Group 1."
            )

        graph1 = "\n".join(lines_group1)
        graph2 = "\n".join(lines_group2)

        qctn1 = QCTN(graph1, backend=self.backend)
        qctn2 = QCTN(graph2, backend=self.backend)

        # Copy core weights (shapes are unchanged by the split)
        for core_name in self.cores[:split_idx]:
            if core_name in self.cores_weights:
                qctn1.cores_weights[core_name] = self.cores_weights[core_name]
        for core_name in self.cores[split_idx:]:
            if core_name in self.cores_weights:
                qctn2.cores_weights[core_name] = self.cores_weights[core_name]

        return qctn1, qctn2

    @staticmethod
    def merge(qctn1, qctn2):
        """
        Left-right merge of two QCTNs into a single new QCTN (static method).

        The merged QCTN places *qctn1*'s graph on the left and *qctn2*'s
        graph on the right, concatenating each qubit line horizontally.

        Rules:

        1. The resulting number of qubits is ``max(qctn1.nqubits, qctn2.nqubits)``.
        2. The QCTN with fewer qubits is padded at the bottom with
           dash-only lines so that both sides have the same number of rows.
        3. The right boundary (``-dim-`` at end) of *qctn1* and the left
           boundary (``-dim-`` at start) of *qctn2* overlap – only one
           copy is kept.  The boundary from the QCTN that originally has
           **more qubits** is preserved (if equal, *qctn1*'s is kept).
        4. Core tensors are renamed contiguously via
           ``opt_einsum.get_symbol(0, 1, 2, …)``.

        Args:
            qctn1 (QCTN): Left QCTN.
            qctn2 (QCTN): Right QCTN.

        Returns:
            QCTN: A new merged QCTN with renamed cores and copied weights.
        """
        import opt_einsum

        n1, n2 = qctn1.nqubits, qctn2.nqubits
        max_qubits = max(n1, n2)

        # ---- core symbol renaming ----
        total_cores = qctn1.ncores + qctn2.ncores
        new_symbols = [opt_einsum.get_symbol(i) for i in range(total_cores)]

        core_map1 = {
            old: new_symbols[i] for i, old in enumerate(qctn1.cores)
        }
        core_map2 = {
            old: new_symbols[qctn1.ncores + i]
            for i, old in enumerate(qctn2.cores)
        }

        remapped1 = QCTN._remap_graph(qctn1.qubits, core_map1)
        remapped2 = QCTN._remap_graph(qctn2.qubits, core_map2)

        # ---- determine padding widths ----
        # Use the max width of each side's real lines as the padding width
        # for the extra qubit rows added to the shorter side.
        # stripped_l1 = l1 without right boundary, stripped_l2 = l2 without left boundary
        pad_width1 = max(len(l) for l in remapped1) - 3
        pad_width2 = max(len(l) for l in remapped2) - 3

        # ---- horizontal merge ----
        new_lines = []
        for qi in range(max_qubits):
            has_l1 = qi < n1
            has_l2 = qi < n2

            l1 = remapped1[qi] if has_l1 else ("-" * pad_width1)
            l2 = remapped2[qi] if has_l2 else ("-" * pad_width2)

            # Extract 4 segments:
            #   stripped_l1: l1 with right boundary removed  (e.g. "-3-A-5-B")
            #   dim_l1:      right boundary of l1            (e.g. "-3-")
            #   dim_l2:      left boundary of l2             (e.g. "-3-")
            #   stripped_l2: l2 with left boundary removed   (e.g. "C-5-D-3-")
            m1 = re.search(r'-\d+-$', l1)
            dim_l1 = m1.group() if has_l1 else ""
            stripped_l1 = l1[:m1.start()] if has_l1 else l1

            m2 = re.match(r'^-\d+-', l2)
            dim_l2 = m2.group() if has_l2 else ""
            stripped_l2 = l2[m2.end():] if has_l2 else l2

            if has_l1 and has_l2:
                # Both exist: keep qctn1's right boundary as the shared dim
                merged = stripped_l1 + dim_l1 + stripped_l2
            elif has_l1:
                # Only l1 exists: pad the right side
                dim_l2 = '---'
                merged = stripped_l1 + stripped_l2 + dim_l1
            else:
                # Only l2 exists: pad the left side
                dim_l1 = '---'
                merged = dim_l2 + stripped_l1 + stripped_l2

            new_lines.append(merged)

        new_graph = "\n".join(new_lines)

        backend = qctn1.backend if qctn1.backend is not None else qctn2.backend
        new_qctn = QCTN(new_graph, backend=backend)

        # Copy core weights under their new names
        for old_name, new_name in core_map1.items():
            if old_name in qctn1.cores_weights:
                new_qctn.cores_weights[new_name] = qctn1.cores_weights[old_name]
        for old_name, new_name in core_map2.items():
            if old_name in qctn2.cores_weights:
                new_qctn.cores_weights[new_name] = qctn2.cores_weights[old_name]

        return new_qctn

    def merge_with(self, other):
        """
        Merge *self* with another QCTN and return a new QCTN.

        Equivalent to ``QCTN.merge(self, other)``.  The result has
        *self*'s cores first (preserving relative order), followed by
        *other*'s cores, with all core names reassigned contiguously.

        Args:
            other (QCTN): Another QCTN to merge with.

        Returns:
            QCTN: A new merged QCTN.
        """
        return QCTN.merge(self, other)
    
