"""
Example script demonstrating QCTN split / merge operations and simple visualization.
"""

import numpy as np
import matplotlib.pyplot as plt

from tneq_qc.backends.backend_factory import BackendFactory
from tneq_qc.core.qctn import QCTN, QCTNHelper


def _adjacency_to_array(adj):
    """
    Convert QCTN.adjacency_matrix (list-of-ranks per entry) to a numeric array.

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
    print("Original QCTN graph:")
    print(graph)

    qctn = QCTN(graph, backend=backend)
    print(f"Original QCTN: nqubits = {qctn.nqubits}, ncores = {qctn.ncores}, graph: \n{qctn.tn_graph.to_string()}")

    # ------------------------------------------------------------------
    # 2. 对 QCTN 做 split 操作
    # ------------------------------------------------------------------
    left_qctn, right_qctn = qctn.split()
    print(f"Left  QCTN: nqubits = {left_qctn.nqubits}, ncores = {left_qctn.ncores}, qctn = {left_qctn} graph: \n{left_qctn.graph}")
    print(f"Right QCTN: nqubits = {right_qctn.nqubits}, ncores = {right_qctn.ncores}, qctn = {right_qctn} graph: \n{right_qctn.graph}")

    # ------------------------------------------------------------------
    # 3. 对两个子 QCTN 做 merge 操作
    # ------------------------------------------------------------------
    merged_qctn = QCTN.merge(left_qctn, right_qctn)
    print(
        f"Merged QCTN: nqubits = {merged_qctn.nqubits}, "
        f"ncores = {merged_qctn.ncores}"
        f'graph : \n{merged_qctn.graph}',
    )

    # ------------------------------------------------------------------
    # 5. 可视化：原始 / 子网络 / 合并后 的邻接矩阵
    # ------------------------------------------------------------------
    adj_orig = _adjacency_to_array(qctn.adjacency_matrix)
    adj_left = _adjacency_to_array(left_qctn.adjacency_matrix)
    adj_right = _adjacency_to_array(right_qctn.adjacency_matrix)
    adj_merged = _adjacency_to_array(merged_qctn.adjacency_matrix)

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
    # 4. 初始化三种结构的 QCTN (mps / tree / wall)，两两左右 merge
    # ------------------------------------------------------------------
    n_mps, n_tree, n_wall = 5, 5, 4

    graph_mps = QCTNHelper.generate_example_graph(n=n_mps, graph_type="mps", dim_char="3")
    graph_tree = QCTNHelper.generate_example_graph(n=n_tree, graph_type="tree", dim_char="3")
    graph_wall = QCTNHelper.generate_example_graph(n=n_wall, graph_type="wall", dim_char="3")

    qctn_mps = QCTN(graph_mps, backend=backend)
    qctn_tree = QCTN(graph_tree, backend=backend)
    qctn_wall = QCTN(graph_wall, backend=backend)

    print("=" * 60)
    print("Step 4: Three QCTN structures and pairwise left-right merge")
    print("=" * 60)

    print(f"\n[MPS]  nqubits={qctn_mps.nqubits}, ncores={qctn_mps.ncores}")
    print(graph_mps)

    print(f"\n[Tree] nqubits={qctn_tree.nqubits}, ncores={qctn_tree.ncores}")
    print(graph_tree)

    print(f"\n[Wall] nqubits={qctn_wall.nqubits}, ncores={qctn_wall.ncores}")
    print(graph_wall)

    # ---- MPS + Tree ----
    merged_mps_tree = QCTN.merge(qctn_mps, qctn_tree)
    print(f"\n--- Merge(MPS, Tree) ---")
    print(f"nqubits={merged_mps_tree.nqubits}, ncores={merged_mps_tree.ncores}")
    print(merged_mps_tree.graph)

    # ---- MPS + Wall ----
    merged_mps_wall = QCTN.merge(qctn_mps, qctn_wall)
    print(f"\n--- Merge(MPS, Wall) ---")
    print(f"nqubits={merged_mps_wall.nqubits}, ncores={merged_mps_wall.ncores}")
    print(merged_mps_wall.graph)

    # ---- Tree + Wall ----
    merged_tree_wall = QCTN.merge(qctn_wall, qctn_tree)
    print(f"\n--- Merge(Wall, Tree) ---")
    print(f"nqubits={merged_tree_wall.nqubits}, ncores={merged_tree_wall.ncores}")
    print(merged_tree_wall.graph)

    # ------------------------------------------------------------------
    # 5. 三种结构 (MPS / Wall-circuit / Tree-mx) 各 5 qubits 按顺序 merge
    # ------------------------------------------------------------------
    n5 = 5

    graph_mps5 = QCTNHelper.generate_example_graph(n=n5, graph_type="mps", dim_char="3")
    graph_circuit5 = QCTNHelper.generate_example_graph(n=n5, graph_type="circuit", dim_char="3")
    graph_mx5 = QCTNHelper.generate_example_graph(n=n5, graph_type="mx", dim_char="3")

    qctn_mps5 = QCTN(graph_mps5, backend=backend)
    qctn_circuit5 = QCTN(graph_circuit5, backend=backend)
    qctn_mx5 = QCTN(graph_mx5, backend=backend)

    print("\n" + "=" * 60)
    print("Step 5: Sequential merge of MPS → Circuit → Mx")
    print("         All 5 qubits")
    print("=" * 60)

    print(f"\n[MPS]     nqubits={qctn_mps5.nqubits}, ncores={qctn_mps5.ncores}")
    print(graph_mps5)

    print(f"[Circuit] nqubits={qctn_circuit5.nqubits}, ncores={qctn_circuit5.ncores}")
    print(graph_circuit5)

    print(f"[Mx]      nqubits={qctn_mx5.nqubits}, ncores={qctn_mx5.ncores}")
    print(graph_mx5)

    # Step A: merge MPS + Circuit
    merged_step1 = QCTN.merge(qctn_circuit5, qctn_mps5)
    print(f"\n--- Merge(MPS, Circuit) ---")
    print(f"nqubits={merged_step1.nqubits}, ncores={merged_step1.ncores}")
    print(merged_step1.graph)

    # Step B: merge (MPS+Circuit) + Mx
    merged_step2 = QCTN.merge(merged_step1, qctn_mx5)
    print(f"\n--- Merge(MPS+Circuit, Mx) ---")
    print(f"nqubits={merged_step2.nqubits}, ncores={merged_step2.ncores}")
    print(merged_step2.graph)





if __name__ == "__main__":
    main()

