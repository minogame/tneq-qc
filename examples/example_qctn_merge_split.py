"""
Example script demonstrating QCTN chunk / concat operations and simple visualization.
"""

import numpy as np
import matplotlib.pyplot as plt

from tneq_qc.backends.backend_factory import BackendFactory
from tneq_qc.core.qctn import QCTN, QCTNHelper


def _adjacency_to_array(adj):
    """
    Convert an adjacency matrix of rank lists to a numeric array.

    Each entry (i, j) stores the sum of bond dimensions between cores i and j.
    """
    n = adj.shape[0]
    arr = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(n):
            if isinstance(adj[i, j], (list, tuple)):
                arr[i, j] = float(sum(adj[i, j]))
            elif adj[i, j] is None:
                arr[i, j] = 0.0
            else:
                # Fallback: try to interpret as scalar
                try:
                    arr[i, j] = float(adj[i, j])
                except Exception:
                    arr[i, j] = 0.0
    return arr


def _adjacency_from_table(qctn):
    """
    Rebuild an adjacency matrix from ``qctn.adjacency_table``.

    Each matrix entry stores a list of bond dimensions between two cores.
    """
    ncores = qctn.ncores
    adj = np.empty((ncores, ncores), dtype=object)
    for i in range(ncores):
        for j in range(ncores):
            adj[i, j] = []

    for core_info in qctn.adjacency_table:
        core_idx = core_info["core_idx"]
        for edge in core_info.get("out_edge_list", []):
            neighbor_idx = edge["neighbor_idx"]
            if neighbor_idx >= 0:
                adj[core_idx, neighbor_idx].append(edge["edge_rank"])
                adj[neighbor_idx, core_idx].append(edge["edge_rank"])

    return adj


def main():
    # ------------------------------------------------------------------
    # 1. 构造一个简单的 MPS 图并创建 QCTN
    # ------------------------------------------------------------------
    backend = BackendFactory.create_backend("pytorch", device="cpu", dtype="float32")

    num_qubits = 8
    graph_type = "mps"
    dim_char = "2"

    graph = QCTNHelper.generate_example_graph(
        n=num_qubits,
        graph_type=graph_type,
        dim_char=dim_char,
    )
    qctn = QCTN(graph, backend=backend)
    print(f"Original QCTN: nqubits = {qctn.nqubits}, ncores = {qctn.ncores}")
    print(qctn)
    print(qctn.graph)

    # ------------------------------------------------------------------
    # 2. 对 QCTN 做 chunk 操作
    # ------------------------------------------------------------------
    left_qctn, right_qctn = qctn.chunk()
    print(f"Left  QCTN: nqubits = {left_qctn.nqubits}, ncores = {left_qctn.ncores}")
    print(left_qctn)
    print(left_qctn.graph)
    print(f"Right QCTN: nqubits = {right_qctn.nqubits}, ncores = {right_qctn.ncores}")
    print(right_qctn)
    print(right_qctn.graph)

    # ------------------------------------------------------------------
    # 3. 对两个子 QCTN 做 concat 操作
    # ------------------------------------------------------------------
    merged_qctn = QCTN.concat([("left", left_qctn), ("right", right_qctn)])
    print(
        f"Merged QCTN: nqubits = {merged_qctn.nqubits}, "
        f"ncores = {merged_qctn.ncores}",
    )
    print(merged_qctn)
    print(merged_qctn.graph)

    # ------------------------------------------------------------------
    # 5. 可视化：原始 / 子网络 / 合并后 的邻接矩阵
    # ------------------------------------------------------------------
    adj_orig = _adjacency_to_array(_adjacency_from_table(qctn))
    adj_left = _adjacency_to_array(_adjacency_from_table(left_qctn))
    adj_right = _adjacency_to_array(_adjacency_from_table(right_qctn))
    adj_merged = _adjacency_to_array(_adjacency_from_table(merged_qctn))

    fig, axes = plt.subplots(2, 2, figsize=(8, 8))
    ax00, ax01, ax10, ax11 = axes.flatten()

    im0 = ax00.imshow(adj_orig, cmap="viridis")
    ax00.set_title("Original adjacency")
    fig.colorbar(im0, ax=ax00, fraction=0.046, pad=0.04)

    im1 = ax01.imshow(adj_left, cmap="viridis")
    ax01.set_title("Left part adjacency")
    fig.colorbar(im1, ax=ax01, fraction=0.046, pad=0.04)

    im2 = ax10.imshow(adj_right, cmap="viridis")
    ax10.set_title("Right part adjacency")
    fig.colorbar(im2, ax=ax10, fraction=0.046, pad=0.04)

    im3 = ax11.imshow(adj_merged, cmap="viridis")
    ax11.set_title("Merged adjacency")
    fig.colorbar(im3, ax=ax11, fraction=0.046, pad=0.04)

    for ax in (ax00, ax01, ax10, ax11):
        ax.set_xlabel("core index")
        ax.set_ylabel("core index")

    plt.tight_layout()
    plt.show()

    # ------------------------------------------------------------------
    # 4. 初始化三种结构的 QCTN (mps / tree / wall)，两两左右 concat
    # ------------------------------------------------------------------
    n_mps, n_tree, n_wall = 5, 5, 4

    graph_mps = QCTNHelper.generate_example_graph(n=n_mps, graph_type="mps", dim_char="3")
    graph_tree = QCTNHelper.generate_example_graph(n=n_tree, graph_type="tree", dim_char="3")
    graph_wall = QCTNHelper.generate_example_graph(n=n_wall, graph_type="wall", dim_char="3")

    qctn_mps = QCTN(graph_mps, backend=backend)
    qctn_tree = QCTN(graph_tree, backend=backend)
    qctn_wall = QCTN(graph_wall, backend=backend)

    print("=" * 60)
    print("Step 4: Three QCTN structures and pairwise left-right concat")
    print("=" * 60)

    print(f"\n[MPS]  nqubits={qctn_mps.nqubits}, ncores={qctn_mps.ncores}")
    print(qctn_mps)
    print(qctn_mps.graph)

    print(f"\n[Tree] nqubits={qctn_tree.nqubits}, ncores={qctn_tree.ncores}")
    print(qctn_tree)
    print(qctn_tree.graph)

    print(f"\n[Wall] nqubits={qctn_wall.nqubits}, ncores={qctn_wall.ncores}")
    print(qctn_wall)
    print(qctn_wall.graph)

    # ---- MPS + Tree ----
    merged_mps_tree = QCTN.concat([("mps", qctn_mps), ("tree", qctn_tree)])
    print(f"\n--- Concat(MPS, Tree) ---")
    print(f"nqubits={merged_mps_tree.nqubits}, ncores={merged_mps_tree.ncores}")
    print(merged_mps_tree)
    print(merged_mps_tree.graph)

    # ---- MPS + Wall ----
    merged_mps_wall = QCTN.concat([("mps", qctn_mps), ("wall", qctn_wall)])
    print(f"\n--- Concat(MPS, Wall) ---")
    print(f"nqubits={merged_mps_wall.nqubits}, ncores={merged_mps_wall.ncores}")
    print(merged_mps_wall)
    print(merged_mps_wall.graph)

    # ---- Tree + Wall ----
    merged_tree_wall = QCTN.concat([("wall", qctn_wall), ("tree", qctn_tree)])
    print(f"\n--- Concat(Wall, Tree) ---")
    print(f"nqubits={merged_tree_wall.nqubits}, ncores={merged_tree_wall.ncores}")
    print(merged_tree_wall)
    print(merged_tree_wall.graph)

    # ------------------------------------------------------------------
    # 5. 三种结构 (MPS / Circuit / Mx) 各 5 qubits 按顺序 concat
    # ------------------------------------------------------------------
    n5 = 5

    graph_mps5 = QCTNHelper.generate_example_graph(n=n5, graph_type="mps", dim_char="3")
    graph_circuit5 = QCTNHelper.generate_example_graph(n=n5, graph_type="circuit", dim_char="3")
    graph_mx5 = QCTNHelper.generate_example_graph(n=n5, graph_type="mx", dim_char="3")

    qctn_mps5 = QCTN(graph_mps5, backend=backend)
    qctn_circuit5 = QCTN(graph_circuit5, backend=backend)
    qctn_mx5 = QCTN(graph_mx5, backend=backend)

    print("\n" + "=" * 60)
    print("Step 5: Sequential concat of MPS → Circuit → Mx")
    print("         All 5 qubits")
    print("=" * 60)

    print(f"\n[MPS]     nqubits={qctn_mps5.nqubits}, ncores={qctn_mps5.ncores}")
    print(qctn_mps5)
    print(qctn_mps5.graph)

    print(f"[Circuit] nqubits={qctn_circuit5.nqubits}, ncores={qctn_circuit5.ncores}")
    print(qctn_circuit5)
    print(qctn_circuit5.graph)

    print(f"[Mx]      nqubits={qctn_mx5.nqubits}, ncores={qctn_mx5.ncores}")
    print(qctn_mx5)
    print(qctn_mx5.graph)

    # Step A: concat MPS + Circuit
    merged_step1 = QCTN.concat([("circuit", qctn_circuit5), ("mps", qctn_mps5)])
    print(f"\n--- Concat(MPS, Circuit) ---")
    print(f"nqubits={merged_step1.nqubits}, ncores={merged_step1.ncores}")
    print(merged_step1)
    print(merged_step1.graph)

    # Step B: concat (MPS+Circuit) + Mx
    merged_step2 = QCTN.concat([merged_step1, ("mx", qctn_mx5)])
    print(f"\n--- Concat(MPS+Circuit, Mx) ---")
    print(f"nqubits={merged_step2.nqubits}, ncores={merged_step2.ncores}")
    print(merged_step2)
    print(merged_step2.graph)





if __name__ == "__main__":
    main()
