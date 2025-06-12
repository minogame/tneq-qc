import jax
import numpy as np
import jax.numpy as jnp
import functools
import re
import random
from typing import Union
import opt_einsum

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

class ContractorOptEinsum:
    """
    ContractorOptEinsum class for optimized tensor contraction using opt_einsum.
    
    This class provides methods to contract tensors using the opt_einsum library,
    which is optimized for performance and memory efficiency.
    """

    @staticmethod
    def contract(tensors, equation, optimize='greedy'):
        """
        Contract tensors using opt_einsum.
        
        Args:
            tensors (list): List of tensors to be contracted.
            equation (str): The einsum equation specifying the contraction.
            optimize (str): Optimization strategy for contraction. Default is 'greedy'.
        
        Returns:
            jnp.ndarray: The result of the tensor contraction.
        """
        return opt_einsum.contract(equation, *tensors, optimize=optimize)

class ContractorQCTN:
    """
    ContractorQCTN class for contracting quantum circuit tensor networks.
    
    This class provides methods to contract quantum circuit tensor networks using JAX.
    It supports both contraction with inputs and contraction with another QCTN instance.
    """

    @staticmethod
    def contract(qctn, inputs=None):
        """
        Contract the quantum circuit tensor network with given inputs.
        
        Args:
            qctn (QCTN): The quantum circuit tensor network to contract.
            inputs (jnp.ndarray or dict, optional): The inputs for the contraction operation.
        
        Returns:
            jnp.ndarray: The result of the contraction operation.
        """
        return qctn.contract(attach=inputs)

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
        self._circuit_to_adjacency()

        # initialize the circuit with input ranks, adjacency matrix, and output ranks
        self.initialize_random_key = jax.random.PRNGKey(0)
        self._init_cores()

    def __repr__(self):
        """
        String representation of the QCTN object.
        """
        adjacency_matrix = np.empty((self.ncores, self.ncores), dtype=object)
        for i in range(self.ncores):
            for j in range(self.ncores):
                adjacency_matrix[i, j] = str(self.adjacency_matrix[i, j])
        
        circuit_input = [str(rank) for rank in self.circuit[0]]
        circuit_output = [str(rank) for rank in self.circuit[2]]

        return f"circuit_input = \n{circuit_input}\n adjacency_matrix = \n{adjacency_matrix}\n circuit_output = \n{circuit_output}\n"

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
        input_ranks = np.empty(self.ncores, dtype=object)
        output_ranks = np.empty(self.ncores, dtype=object)
        for i in range(self.ncores):
            input_ranks[i] = []
            output_ranks[i] = []

        cores = "".join(sorted(self.cores))
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
            input_ranks[input_core_idx].append(input_rank)
            output_ranks[output_core_idx].append(output_rank)
            for match in connect_pattern.finditer(line):
                core1, rank1, core2 = match.groups()
                core1_idx = dict_core2idx[core1]
                core2_idx = dict_core2idx[core2]
                rank1 = int(rank1)
                self.adjacency_matrix[core1_idx, core2_idx].append(rank1)
                self.adjacency_matrix[core2_idx, core1_idx].append(rank1)

        self.circuit = (input_ranks, self.adjacency_matrix, output_ranks)

    def _init_cores(self):
        """
        Initialize the cores of the quantum circuit with random values.
        
        Returns:
            None: The cores are stored in the `cores_weigts` attribute.
        """

        self.cores_weigts = {}
        for idx, core_name in enumerate(self.cores):
            # These ranks should be expressed as lists of integers ordered by the qubits.
            # Therefore we conduct "+" on all of these ranks to obtain the shape of the core.
            input_rank = self.circuit[0][idx]
            output_rank = self.circuit[2][idx]
            adjacency_ranks = self.adjacency_matrix[idx, :]

            core_shape = input_rank + adjacency_ranks + output_rank
            core = jax.random.normal(self.initialize_random_key, shape=core_shape) / 1e5            
            self.cores_weigts[core_name] = core

    def _contract_only(self, *args, **kwargs):
        """
        Contract the quantum circuit tensor network without inputs.
        
        Args:
            *args: Positional arguments for contraction.
            **kwargs: Keyword arguments for contraction.
        
        Returns:
            The result of the contraction operation.
        """
        # Placeholder for contraction logic
        raise NotImplementedError("Contraction logic is not implemented yet.")

    def _contract_with_inputs(self, inputs: Union[jnp.ndarray, dict] = None):
        """
        Contract the quantum circuit tensor network with given inputs.
        
        Args:
            inputs (jnp.ndarray or dict): The inputs for the contraction operation.
            If a dictionary is provided, it should map core names to their respective input tensors.
            If a jnp.ndarray is provided, it should be a tensor with the shape matching the input ranks of the circuit.
        
        Returns:
            The result of the contraction operation.
        """
        if inputs is None:
            raise ValueError("Inputs must be provided for contraction.")
        if isinstance(inputs, dict):
            # If inputs are provided as a dictionary, we need to ensure they match the core names
            for core_name in self.cores:
                if core_name not in inputs:
                    raise ValueError(f"Input for core '{core_name}' is missing.")
            # Convert the dictionary to a list of tensors ordered by core names
            inputs = [inputs[core_name] for core_name in self.cores]
        elif isinstance(inputs, jnp.ndarray):
            # If inputs are provided as a single tensor, we need to ensure it matches the input ranks
            if inputs.shape != tuple(self.circuit[0]):
                raise ValueError(f"Input tensor shape {inputs.shape} does not match expected shape {tuple(self.circuit[0])}.")
            # Convert the single tensor to a list of tensors ordered by core names
            inputs = [inputs] * self.ncores
        else:
            raise TypeError("Inputs must be a jnp.ndarray or a dictionary mapping core names to tensors.")
        # Here we would implement the contraction logic using JAX or other libraries.
        # For now, we will raise NotImplementedError as a placeholder.
        # This is a placeholder for the contraction logic.
        if len(inputs) != self.ncores:
            raise ValueError(f"Expected {self.ncores} inputs, but got {len(inputs)}.")
        if any(input_tensor.ndim != len(self.circuit[0][idx]) for idx, input_tensor in enumerate(inputs)):
            raise ValueError("Input tensors must have the same number of dimensions as their corresponding core input ranks.")
        # Here we would implement the contraction logic using JAX or other libraries.
        # For now, we will raise NotImplementedError as a placeholder.
        if any(core_name not in self.cores_weigts for core_name in self.cores):
            raise ValueError("Some core names in the inputs do not match the initialized cores.")
        # This is a placeholder for the contraction logic.
        # For now, we will raise NotImplementedError as a placeholder.

        # Placeholder for contraction logic
        raise NotImplementedError("Contraction logic is not implemented yet.")

    def _contract_with_QCTN(self, qctn):
        """
        Contract the quantum circuit tensor network with another QCTN instance.
        
        Args:
            qctn (QCTN): Another instance of QCTN to contract with.
        
        Returns:
            The result of the contraction operation.
        """
        if not isinstance(qctn, QCTN):
            raise TypeError("The argument must be an instance of QCTN.")
        # Here we would implement the contraction logic using JAX or other libraries.
        # For now, we will raise NotImplementedError as a placeholder.
        # This is a placeholder for the contraction logic.
        # For now, we will raise NotImplementedError as a placeholder.
        
        # Placeholder for contraction logic
        raise NotImplementedError("Contraction logic is not implemented yet.")
    
    def _contract_for_core_gradient(self, core_name, inputs=None):
        """
        Contract the quantum circuit tensor network for a specific core gradient.
        
        Args:
            core_name (str): The name of the core to contract for gradient computation.
            inputs (jnp.ndarray or dict, optional): The inputs for the contraction operation.
                If a dictionary is provided, it should map core names to their respective input tensors.
                If a jnp.ndarray is provided, it should be a tensor with the shape matching the input ranks of the circuit.
        
        Returns:
            The result of the contraction operation for the specified core.
        """
        if core_name not in self.cores:
            raise ValueError(f"Core '{core_name}' is not part of this QCTN.")
        # Here we would implement the contraction logic using JAX or other libraries.
        # For now, we will raise NotImplementedError as a placeholder.
        
        # Placeholder for contraction logic
        raise NotImplementedError("Contraction logic is not implemented yet.")

    def contract(self, attach:Union[jnp.ndarray, dict, 'QCTN']=None, *args, **kwargs):
        """
        Contract the quantum circuit tensor network.
        
        Args:
            *args: Positional arguments for contraction.
            **kwargs: Keyword arguments for contraction.
        
        Returns:
            The result of the contraction operation.
        """

        if attach is None:
            return self._contract_only(*args, **kwargs)
        elif isinstance(attach, Union[jnp.ndarray, dict]):
            return self._contract_with_inputs(attach, *args, **kwargs)
        elif isinstance(attach, 'QCTN'):
            return self._contract_with_QCTN(attach, *args, **kwargs)
        else:
            raise TypeError("attach must be a jnp.ndarray, a dictionary, or an instance of QCTN.")

if __name__ == "__main__":
    # example_graph = QCTNHelper.generate_random_example_graph(30, 50)
    # print(f"Example Graph: \n {example_graph}")

    example_graph = QCTNHelper.generate_example_graph()
    print(f"Example Graph: \n{example_graph}")
    qctn = QCTN(example_graph)
    print(f"QCTN Adjacency Matrix:\n{qctn.__repr__()}")
    print(f"Cores: {qctn.cores}")
    print(f"Number of Qubits: {qctn.nqubits}")
    print(f"Number of Cores: {qctn.ncores}")
