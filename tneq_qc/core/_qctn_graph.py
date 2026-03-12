"""Graph-parsing Mixin for QCTN.

Contains all methods that convert the ASCII graph string into the
``adjacency_table`` data structure, plus graph manipulation helpers.
"""
import warnings
import numpy as np
import re


class QCTNGraphMixin:
    """Mixin: ASCII graph → adjacency_table, and graph manipulation helpers."""

    @classmethod
    def envolve_from_another_qctn(cls, qctn, strategies=None):
        """Create a new QCTN instance by evolving from another QCTN instance.

        .. deprecated::
            This method will be moved to the ``genetic`` module in a future
            release.  Use ``genetic.evolve(qctn, strategies)`` instead.

        Args:
            qctn (QCTN): The original QCTN instance to evolve from.
            strategies (list, optional): Strategies for evolution.

        Returns:
            QCTN: A new QCTN instance evolved from the original.
        """
        warnings.warn(
            "QCTN.envolve_from_another_qctn is deprecated and will be moved "
            "to the genetic module in a future release.",
            DeprecationWarning,
            stacklevel=2,
        )
        if strategies is None \
           or (isinstance(strategies, list) and not strategies):
            if isinstance(qctn, cls):
                return cls.copy(qctn)
            else:
                raise TypeError("qctn must be an instance of QCTN.")

        if callable(strategies):
            new_graph = strategies(qctn.graph)
            return cls(new_graph)
        elif isinstance(strategies, list):
            new_graph = qctn.graph
            for strategy in strategies:
                if callable(strategy):
                    new_graph = strategy(new_graph)
                else:
                    raise TypeError("Each strategy must be a callable function.")
            return cls(new_graph)

    def _circuit_to_adjacency(self,):
        """
        Convert the quantum circuit graph to adjacency table.

        This method builds self.adjacency_table, a list where each element
        corresponds to a core and contains a dict with:
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
            line = line.strip().replace("-", "")
            m_input = input_pattern.match(line)
            m_output = output_pattern.search(line)

            # Skip lines that have neither input nor output
            if m_input is None and m_output is None:
                return

            # Handle input edge (if present)
            if m_input:
                input_rank, input_core = m_input.groups()
                input_rank = int(input_rank)
                input_core_idx = dict_core2idx[input_core]

                # Add input edge: from circuit input (-1, "") to input_core
                self.adjacency_table[input_core_idx]['in_edge_list'].append({
                    'neighbor_idx': -1,
                    'neighbor_name': "",
                    'edge_rank': input_rank,
                    'qubit_idx': qubit_idx
                })

            # Handle output edge (if present)
            if m_output:
                output_core, output_rank = m_output.groups()
                output_rank = int(output_rank)
                output_core_idx = dict_core2idx[output_core]

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
