    def __str__(self) -> str:
        """Pretty-print tensor network structure based on adjacency_table.

        Reconstructs the graph string representation from the adjacency table,
        ensuring consistency with the actual topology.

        Example output:
            -2-A-5-B-----3-
            -2-----B-6-C-2-
        """
        if not self.qubits:
            return f"QCTN(composite, submodules={list(self._submodules.keys())})"
        if not self.cores:
            return "QCTN(empty)"

        # Debug: Print adjacency table structure
        print("\n=== Adjacency Table Debug ===")
        for info in self.adjacency_table:
            print(f"Core {info['core_name']}:")
            print(f"  in_edges: {[(e['qubit_idx'], e['neighbor_name'] or 'boundary', e['edge_rank']) for e in info['in_edge_list']]}")
            print(f"  out_edges: {[(e['qubit_idx'], e['neighbor_name'] or 'boundary', e['edge_rank']) for e in info['out_edge_list']]}")
        print("=" * 40)
        print()

        # Build a map: qubit_idx -> [(core_name, position_in_line, left_dim, right_dim)]
        # We need to determine the order of cores on each qubit line
        qubit_cores = {}  # qubit_idx -> list of (core_name, left_dim, right_dim, position)

        for core_info in self.adjacency_table:
            core_name = core_info['core_name']

            # Collect all qubits this core touches
            qubits_touched = set()
            for edge in core_info['in_edge_list']:
                qubits_touched.add(edge['qubit_idx'])
            for edge in core_info['out_edge_list']:
                qubits_touched.add(edge['qubit_idx'])

            # For each qubit, determine the left and right dimensions
            for qubit_idx in qubits_touched:
                # Find left dimension (from in_edge on this qubit)
                left_dim = None
                for edge in core_info['in_edge_list']:
                    if edge['qubit_idx'] == qubit_idx:
                        left_dim = edge['edge_rank']
                        break

                # Find right dimension (from out_edge on this qubit)
                right_dim = None
                for edge in core_info['out_edge_list']:
                    if edge['qubit_idx'] == qubit_idx:
                        right_dim = edge['edge_rank']
                        break

                if qubit_idx not in qubit_cores:
                    qubit_cores[qubit_idx] = []

                qubit_cores[qubit_idx].append({
                    'core_name': core_name,
                    'left_dim': left_dim,
                    'right_dim': right_dim,
                })

        # Sort cores on each qubit by their original order in self.cores
        core_order = {name: idx for idx, name in enumerate(self.cores)}
        for qubit_idx in qubit_cores:
            qubit_cores[qubit_idx].sort(key=lambda x: core_order[x['core_name']])

        # Build output lines
        lines = []
        for qubit_idx in range(self.nqubits):
            if qubit_idx not in qubit_cores or not qubit_cores[qubit_idx]:
                # Empty qubit line
                lines.append("-2-")
                continue

            cores_on_line = qubit_cores[qubit_idx]
            parts = []

            # Left boundary: use the left_dim of the first core, or nothing if None
            first_core = cores_on_line[0]
            if first_core['left_dim'] is not None:
                parts.append(f"-{first_core['left_dim']}-")
            else:
                parts.append("-")

            # Add cores and their connecting dimensions
            for idx, core_info in enumerate(cores_on_line):
                parts.append(core_info['core_name'])

                if idx < len(cores_on_line) - 1:
                    # Bond between this core and next core
                    # Use right_dim of current core
                    if core_info['right_dim'] is not None:
                        parts.append(f"-{core_info['right_dim']}-")
                    else:
                        parts.append("-")

            # Right boundary: use the right_dim of the last core, or nothing if None
            last_core = cores_on_line[-1]
            if last_core['right_dim'] is not None:
                parts.append(f"-{last_core['right_dim']}-")
            else:
                parts.append("-")

            lines.append(''.join(parts))

        return '\n'.join(lines)
