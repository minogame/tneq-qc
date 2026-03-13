import numpy as np
import random


class QCTNHelper:
    """
    Helper class for Quantum Circuit Tensor Network (QCTN) operations.
    Provides methods for generating quantum circuit graph strings.
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
    def mps(nqubits, bond_dim, phys_dim=2):
        """Generate an MPS graph with the same topology as ``generate_example_graph(..., "mps")``.

        Core placement follows the staggered/diagonal layout used by the
        legacy example-graph generator: the first and last qubit rows have one
        core, middle rows have two neighboring cores.

        Example — ``mps(5, bond_dim=4, phys_dim=2)``::

            -2-a-------------------2-
            -2-a--4--b-------------2-
            -2-------b--4--c-------2-
            -2-------------c--4--d-2-
            -2-------------------d-2-

        Args:
            nqubits: Number of qubits (rows in the graph).
            bond_dim: Bond dimension between adjacent cores.
            phys_dim: Physical (input/output boundary) dimension.

        Returns:
            str: Graph string suitable for ``QCTN`` construction.
        """
        if nqubits <= 0:
            return ""
        if nqubits == 1:
            import opt_einsum
            c = opt_einsum.get_symbol(0)
            return f"-{phys_dim}-{c}-{phys_dim}-"

        import opt_einsum
        char_list = [opt_einsum.get_symbol(i) for i in range(nqubits)]
        p, b = str(phys_dim), str(bond_dim)

        rows = []
        for i in range(nqubits):
            cid = i - 1
            nid = i
            if i == 0:
                # First row: only the first core.
                line = f"-{p}-" + char_list[i] + (nqubits - 2) * 6 * "-" + f"-{p}-"
            elif i == nqubits - 1:
                # Last row: only the last core in the staggered chain.
                line = f"-{p}-" + (nqubits - 2) * 6 * "-" + char_list[cid] + f"-{p}-"
            else:
                # Middle rows: two neighboring cores linked by bond_dim.
                line = f"-{p}-"
                line += cid * 6 * "-"
                line += char_list[cid]
                line += f"--{b}--"
                line += char_list[nid]
                line += (nqubits - nid - 2) * 6 * "-"
                line += f"-{p}-"
            rows.append(line)

        return "\n".join(rows)

    @staticmethod
    def brickwall(nqubits: int, n_layers: int, phys_dim: int = 2) -> str:
        """Generate a brickwall (alternating-layer) quantum circuit graph.

        Directly follows the ``generate_wall_graph`` layout: adjacent qubit-row
        pairs share cores in a staggered column pattern, with ``n_layers``
        controlling the total number of columns (``m = n_layers // 2`` cores
        per row-pair).

        Example — ``brickwall(4, n_layers=4, phys_dim=2)``::

            -2-a-2-----b-----2-
            -2-a-2-c-2-b-2-d-2-
            -2-e-2-c-2-f-2-d-2-
            -2-e-2-----f-----2-

        Args:
            nqubits:  Number of qubits (rows).
            n_layers: Total column width (``L``); ``m = n_layers // 2`` cores
                      per adjacent-row pair.
            phys_dim: Physical (boundary) dimension.

        Returns:
            str: Graph string suitable for ``QCTN`` construction.
        """
        import opt_einsum

        dim_char = str(phys_dim)
        n, L = nqubits, n_layers
        m = L // 2

        total_chars = L * (n // 2)
        char_list = [opt_einsum.get_symbol(i) for i in range(total_chars)]

        # 2-D char array: n rows × (4*L) columns, all '-'
        line_list = [['-'] * (4 * L) for _ in range(n)]

        # 最后一列（右边界 dim）
        for i in range(n):
            line_list[i][-2] = dim_char

        idx = 0
        for i in range(n - 1):
            offset = 0 if i % 2 == 0 else 4
            for j in range(m):
                col = offset + 8 * j
                # 同一 core 同时出现在第 i 行和第 i+1 行
                line_list[i][col]     = char_list[idx]
                line_list[i + 1][col] = char_list[idx]
                # 行内 bond dim（中间列和末尾列的条件）
                if j < m - 1 or (j == m - 1 and i > 0):
                    line_list[i][col + 2] = dim_char
                if j < m - 1 or (j == m - 1 and i != n - 2):
                    line_list[i + 1][col + 2] = dim_char
                idx += 1

        rows = [
            "-" + dim_char + "-" + "".join(line_list[i])
            for i in range(n)
        ]
        return "\n".join(rows)

    @staticmethod
    def circuit_state(nqubits, phys_dim=2):
        """Generate a circuit-state graph: each qubit has one core, no left input dim.

        Each qubit line has a single core with only a right (output) edge of
        ``phys_dim``.  This corresponds to a vector (ket) state.

        Example — ``circuit_state(3, phys_dim=2)``::

            -A-2-
            -B-2-
            -C-2-

        Args:
            nqubits: Number of qubits (one core per qubit).
            phys_dim: Physical (output) dimension.

        Returns:
            str: Graph string suitable for ``QCTN`` construction.
        """
        import opt_einsum
        char_list = [opt_einsum.get_symbol(i) for i in range(nqubits)]
        p = str(phys_dim)
        return '\n'.join(f'-{c}-{p}-' for c in char_list)

    @staticmethod
    def circuit_bra(nqubits, phys_dim=2):
        """Generate a circuit-bra graph: each qubit has one core, left input dim only.

        Each qubit line has a single core with only a left (input) edge of
        ``phys_dim`` and no right (output) edge.  This corresponds to a row
        vector (bra) state ``⟨ψ|`` that closes the right boundary of the
        preceding component.

        Example — ``circuit_bra(3, phys_dim=2)``::

            -2-A-
            -2-B-
            -2-C-

        Args:
            nqubits: Number of qubits (one core per qubit).
            phys_dim: Physical (input) dimension.

        Returns:
            str: Graph string suitable for ``QCTN`` construction.
        """
        import opt_einsum
        char_list = [opt_einsum.get_symbol(i) for i in range(nqubits)]
        p = str(phys_dim)
        return '\n'.join(f'-{p}-{c}-' for c in char_list)

    @staticmethod
    def measure_matrix(nqubits, phys_dim=2):
        """Generate a measurement-matrix graph: each qubit has one core with in/out dims.

        Each qubit line has a single core with one input edge and one output
        edge, both of dimension ``phys_dim``.  This corresponds to a matrix
        operator (e.g. an observable).

        Example — ``measure_matrix(3, phys_dim=2)``::

            -2-A-2-
            -2-B-2-
            -2-C-2-

        Args:
            nqubits: Number of qubits (one core per qubit).
            phys_dim: Physical (input and output) dimension.

        Returns:
            str: Graph string suitable for ``QCTN`` construction.
        """
        import opt_einsum
        char_list = [opt_einsum.get_symbol(i) for i in range(nqubits)]
        p = str(phys_dim)
        return '\n'.join(f'-{p}-{c}-{p}-' for c in char_list)

    @staticmethod
    def triu_ndindex(n):
        """Generate indices for the upper triangular part of a square matrix."""
        for i in range(n):
            for j in range(i + 1, n):
                yield (i, j)
