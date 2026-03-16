"""
Minimal single-node training example for QCTN.

This script is a compact version of ``train.py``:

- 单机单进程训练
- 使用 EngineSiamese + Optimizer
- 命令行参数风格与 ``train.py`` 保持一致，并新增若干常用参数

默认参数：
- backend      = 'pytorch'
- device       = 'cpu'
- graph_type   = 'mps'
- dim_char     = '2'
- dtype        = 'float32'
- x_distribution = 'gaussian'
"""

import os
import time
import argparse
from typing import Union

import numpy as np
import torch

from tneq_qc.backends.backend_factory import BackendFactory
from tneq_qc.core.engine_siamese import EngineSiamese
from tneq_qc.core.qctn import QCTN, QCTNHelper
from tneq_qc.optim.optimizer import Optimizer


def _resolve_dtype(dtype: Union[str, torch.dtype]) -> torch.dtype:
    if isinstance(dtype, torch.dtype):
        return dtype
    mapping = {
        "float32": torch.float32,
        "float64": torch.float64,
        "complex64": torch.complex64,
        "complex128": torch.complex128,
        "complex": torch.complex64,
    }
    return mapping.get(str(dtype).lower(), torch.float32)


def generate_circuit_states_list(num_qubits, K, device="cpu", dtype="float32"):
    """
    与 train.py 中相同：为每个 qubit 生成初始状态向量。
    """
    torch_dtype = _resolve_dtype(dtype)
    states = [torch.zeros(K, device=device, dtype=torch_dtype) for _ in range(num_qubits)]
    for s in states:
        s[-1] = 1.0
    return states


def build_dataset(engine, D, K, N, B, device, x_distribution="gaussian"):
    """
    生成一批简单的训练数据：
    - x: (B, D)，高斯分布或其他分布
    - 使用 engine.generate_data 生成测量算符 Mx_list
    """
    data_list_for_optim = []

    for _ in range(N):
        if x_distribution == "gaussian":
            x = torch.empty((B, D), device=device).normal_(mean=0.0, std=1.0)
        else:
            # 其他分布可以自行扩展，目前简单地回退为高斯
            x = torch.empty((B, D), device=device).normal_(mean=0.0, std=1.0)

        Mx_list, _ = engine.generate_data(x, K=K)
        data_list_for_optim.append({"measure_input_list": Mx_list})

    return data_list_for_optim


def main():
    parser = argparse.ArgumentParser(
        description="Minimal single-node QCTN training example."
    )

    # 与 train.py 类似的通用训练参数
    parser.add_argument(
        "--save-every",
        type=int,
        default=200,
        help="每多少个 step 打印一次日志（示例脚本不做 checkpoint 保存）。",
    )
    parser.add_argument(
        "--num-step",
        type=int,
        default=1000,
        help="训练总步数（默认 1000）。",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="pytorch",
        choices=["pytorch", "jax"],
        help="后端类型（默认 pytorch）。",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="设备（例如 cpu, cuda, cuda:0；默认 cpu）。",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float32",
        help="数值类型（默认 float32）。",
    )

    # QCTN 结构相关参数
    parser.add_argument(
        "--graph-type",
        type=str,
        default="mps",
        choices=["mps", "tree", "wall"],
        help="QCTN 图类型（默认 mps）。",
    )
    parser.add_argument(
        "--num-qubits",
        type=int,
        default=2,
        help="量子比特数量（默认 2",
    )
    parser.add_argument(
        "--dim-char",
        type=str,
        default="2",
        help="物理维度字符（默认 '2'）。",
    )

    # 数据生成相关参数
    parser.add_argument(
        "--x-distribution",
        type=str,
        default="gaussian",
        help="输入 x 的分布类型（默认 gaussian）。",
    )
    parser.add_argument(
        "--num-data",
        type=int,
        default=20,
        help="生成多少批训练数据 N（默认 20）。",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=512,
        help="每批数据大小 B（默认 512）。",
    )
    parser.add_argument(
        "--K",
        type=int,
        default=2,
        help="每个 qubit 的局域维度 K（默认 2）。",
    )

    args = parser.parse_args()

    # ------------------------------------------------------------------
    # 1. Backend & Engine
    # ------------------------------------------------------------------
    torch.manual_seed(42)
    np.random.seed(42)

    backend = BackendFactory.create_backend(
        args.backend,
        device=args.device,
        dtype=args.dtype,
    )
    engine = EngineSiamese(backend=backend, strategy_mode="balanced", mx_K=args.K)

    device = backend.backend_info.device
    print(f"Using backend: {backend.get_backend_name()} on device: {device}")

    # ------------------------------------------------------------------
    # 2. 构造 QCTN
    # ------------------------------------------------------------------
    graph = QCTNHelper.generate_example_graph(
        n=args.num_qubits,
        graph_type=args.graph_type,
        dim_char=args.dim_char,
    )
    print("QCTN graph:")
    print(graph)

    qctn = QCTN(graph, backend=backend)
    print(f"QCTN: nqubits = {qctn.nqubits}, ncores = {qctn.ncores}")
    for c_name in qctn.cores:
        core_tensor = qctn.cores_weights[c_name]
        if isinstance(core_tensor, torch.Tensor):
            core_tensor.requires_grad_(True)
        else:
            core_tensor.tensor.requires_grad_(True)
    

    # ------------------------------------------------------------------
    # 3. 准备训练数据与电路输入态
    # ------------------------------------------------------------------
    D = qctn.nqubits
    data_list_for_optim = build_dataset(
        engine,
        D=D,
        K=args.K,
        N=args.num_data,
        B=args.batch_size,
        device=device,
        x_distribution=args.x_distribution,
    )

    circuit_states_list = generate_circuit_states_list(
        num_qubits=D,
        K=args.K,
        device=device,
        dtype=args.dtype,
    )

    # 核心张量设置 requires_grad
    for c_name in qctn.cores:
        core = qctn.cores_weights[c_name]
        if isinstance(core, torch.Tensor):
            core.requires_grad_(True)
        else:
            # 例如 TNTensor 情况
            try:
                core.tensor.requires_grad_(True)
            except AttributeError:
                pass

    # ------------------------------------------------------------------
    # 4. 构建 Optimizer 并启动优化
    # ------------------------------------------------------------------
    optimizer = Optimizer(
        method="sgdg",
        max_iter=args.num_step,
        tol=0.0,
        learning_rate=1e-3,
        beta1=0.9,
        beta2=0.95,
        epsilon=1e-8,
        engine=engine,
        momentum=0.9,
        stiefel=True,
    )

    tic = time.time()
    optimizer.optimize(
        qctn,
        data_list=data_list_for_optim,
        circuit_states_list=circuit_states_list,
    )
    toc = time.time()

    print(f"Training finished. Time elapsed: {toc - tic:.2f} seconds")


if __name__ == "__main__":
    main()

