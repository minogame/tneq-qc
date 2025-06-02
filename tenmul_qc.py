import jax
import numpy as np
import jax.numpy as jnp
import functools
import re
import random

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
    def generate_example_graph():
        """Generate an example quantum circuit graph."""
        return "-2-----B-5-C-3-D-----2-\n" \
               "-2-A-4---------D-----2-\n" \
               "-2-A-4-B-7-C-2-D-4-E-2-\n" \
               "-2-A-3-B-6---------E-2-\n" \
               "-2---------C-8-----E-2-"
    
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
 
    """

    def __init__(self, graph):
        """
        Initialize the QCTN with a quantum circuit graph.
        
        Args:
            graph (str): A string representation of the quantum circuit graph.
        """
        self.graph = graph
        self.qubits = graph.strip().splitlines()
        self.nqubits = len(self.qubits)
        self.cores = list(set([c for c in graph if c.isupper()]))
        if not self.cores:
            # If no uppercase core symbols found, try to find all chars in the CJK Unified Ideographs range
            self.cores = list(set([c for c in graph if 0x4E00 <= ord(c) <= 0x9FFF]))
        self.ncores = len(self.cores)
        self.adjacency_matrix = self._circuit_to_adjacency()

    def __repr__(self):
        """
        String representation of the QCTN object.
        """
        adjacency_matrix = np.empty((self.ncores, self.ncores), dtype=object)
        for i in range(self.ncores):
            for j in range(self.ncores):
                adjacency_matrix[i, j] = str(self.adjacency_matrix[i, j])
        return adjacency_matrix


    def _circuit_to_adjacency(self,):
        """
        Convert the quantum circuit graph to an adjacency matrix.
        
        Returns:
            np.ndarray: Adjacency matrix representing the quantum circuit.
        """
        self.adjacency_matrix = np.empty((self.ncores, self.ncores), dtype=object)
        for i in range(self.ncores):
            for j in range(self.ncores):
                self.adjacency_matrix[i, j] = []
        # Optionally, you can initialize the diagonal differently if needed
        for i in range(self.ncores):
            self.adjacency_matrix[i, i] = [[], []]

        cores = "".join(self.cores)
        dict_core2idx = {core: idx for idx, core in enumerate(self.cores)}
        input_pattern = re.compile(rf"^(\d+)([{cores}])")
        output_pattern = re.compile(rf"([{cores}])(\d+)$")
        connect_pattern = re.compile(rf"([{cores}])(\d+)([{cores}])")

        # print(f"Input Pattern: {input_pattern.pattern}")
        # print(f"Output Pattern: {output_pattern.pattern}")
        # print(f"Connect Pattern: {connect_pattern.pattern}")

        for line in self.qubits:
            line = line.strip().replace("-", "")
            # print(f"Processing line: {line}")
            input_rank, input_core = input_pattern.match(line).groups()
            # print(f"Input Core: {input_core}, Input Rank: {input_rank}")
            output_core, output_rank = output_pattern.search(line).groups()
            input_rank, output_rank = int(input_rank), int(output_rank)
            input_core_idx = dict_core2idx[input_core]
            output_core_idx = dict_core2idx[output_core]
            self.adjacency_matrix[input_core_idx, input_core_idx][0].append(input_rank)
            self.adjacency_matrix[output_core_idx, output_core_idx][1].append(output_rank)
            for match in connect_pattern.finditer(line):
                core1, rank1, core2 = match.groups()
                core1_idx = dict_core2idx[core1]
                core2_idx = dict_core2idx[core2]
                rank1 = int(rank1)
                self.adjacency_matrix[core1_idx, core2_idx].append(rank1)
                self.adjacency_matrix[core2_idx, core1_idx].append(rank1)
        return self.adjacency_matrix
            

if __name__ == "__main__":
    example_graph = QCTNHelper.generate_random_example_graph(30, 50)
    print(f"Example Graph: \n {example_graph}")

    # example_graph = QCTNHelper.generate_example_graph()
    # print(f"Example Graph: \n{example_graph}")
    qctn = QCTN(example_graph)
    print(f"QCTN Adjacency Matrix:\n{qctn.__repr__()}")
    print(f"Number of Qubits: {qctn.nqubits}")
    print(f"Number of Cores: {qctn.ncores}")
