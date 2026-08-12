import math
import opt_einsum

def generate_mps_brickwall_graph(total_qubits, block_qubits, overlap=3, phys_dim=2):
    """Generate an MPS-of-Brickwall graph string.

    Each block is a 2-layer brickwall:
    - Even layer: pairs (0,1), (2,3), (4,5), ...
    - Odd layer:  pairs (1,2), (3,4), (5,6), ...

    Adjacent blocks overlap by `overlap` qubits.

    Args:
        total_qubits: Minimum number of qubits (actual may be slightly more).
        block_qubits: Number of qubits per brickwall block.
        overlap: Number of overlapping qubits between adjacent blocks.
        phys_dim: Physical dimension (default 2).

    Returns:
        tuple: (graph_string, actual_qubits)
    """
    assert block_qubits > overlap, "block_qubits must be > overlap"
    assert block_qubits >= 3, "block_qubits must be >= 3"

    dim = str(phys_dim)
    stride = block_qubits - overlap

    # Number of blocks needed to cover total_qubits
    if total_qubits <= block_qubits:
        n_blocks = 1
    else:
        n_blocks = 1 + math.ceil((total_qubits - block_qubits) / stride)

    actual_qubits = block_qubits + (n_blocks - 1) * stride

    # Each block uses 2 column-slots (even layer + odd layer)
    total_slots = n_blocks * 2
    width = total_slots * 4  # 4 chars per slot

    # Initialize char grid
    line_list = [['-'] * width for _ in range(actual_qubits)]

    # Right boundary dim
    for i in range(actual_qubits):
        line_list[i][-2] = dim

    # Place cores and collect per-row core columns
    sym_idx = 0
    row_core_cols = [[] for _ in range(actual_qubits)]  # track which cols have cores

    for b in range(n_blocks):
        start = b * stride

        # Even layer: pairs (start+0, start+1), (start+2, start+3), ...
        even_col = (b * 2) * 4
        for p in range(block_qubits // 2):
            q1 = start + 2 * p
            q2 = start + 2 * p + 1
            sym = opt_einsum.get_symbol(sym_idx)
            sym_idx += 1

            line_list[q1][even_col] = sym
            line_list[q2][even_col] = sym
            row_core_cols[q1].append(even_col)
            row_core_cols[q2].append(even_col)

        # Odd layer: pairs (start+1, start+2), (start+3, start+4), ...
        odd_col = (b * 2 + 1) * 4
        for p in range((block_qubits - 1) // 2):
            q1 = start + 2 * p + 1
            q2 = start + 2 * p + 2
            sym = opt_einsum.get_symbol(sym_idx)
            sym_idx += 1

            line_list[q1][odd_col] = sym
            line_list[q2][odd_col] = sym
            row_core_cols[q1].append(odd_col)
            row_core_cols[q2].append(odd_col)

    # Place dim chars between consecutive cores on each row.
    # Do NOT place dim after the last core (unless it coincides with the
    # right boundary position), to avoid parsing two adjacent numbers as
    # a single multi-digit value after dash removal.
    right_boundary_col = width - 2
    for q in range(actual_qubits):
        cols = sorted(row_core_cols[q])
        for i, col in enumerate(cols):
            if i < len(cols) - 1:
                # Not the last core → always place dim
                line_list[q][col + 2] = dim
            else:
                # Last core → place dim only if it coincides with right boundary
                if col + 2 == right_boundary_col:
                    pass  # already set
                # else: skip (right boundary handles the edge)

    # Build graph string
    lines = ['-' + dim + '-' + ''.join(line_list[i]) for i in range(actual_qubits)]
    return '\n'.join(lines), actual_qubits
