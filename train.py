import os
import time
import argparse
import cmath
from typing import Any, Union

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import matplotlib.pyplot as plt

from tneq_qc.backends.backend_factory import BackendFactory
from tneq_qc.config import Configuration
# from tneq_qc.core.cqctn import ContractorQCTN
from tneq_qc.backends.copteinsum import ContractorOptEinsum
from tneq_qc.core.engine_siamese import EngineSiamese
from tneq_qc.core.qctn import QCTN, QCTNHelper
from tneq_qc.optim.optimizer import Optimizer

def _resolve_dtype(dtype: Union[str, torch.dtype]) -> torch.dtype:
    """将字符串或 torch.dtype 转为 torch.dtype。"""
    if isinstance(dtype, torch.dtype):
        return dtype
    mapping = {
        'float32': torch.float32,
        'float64': torch.float64,
        'complex64': torch.complex64,
        'complex128': torch.complex128,
        'complex': torch.complex64,
    }
    return mapping.get(str(dtype).lower(), torch.float32)


def generate_circuit_states_list(num_qubits, K, device='cuda', dtype='float32'):
    """
    Generate circuit states list for each qubit.

    - Real dtype: 每个 qubit 状态为 [0, 0, ..., 1]（最后一维为 1）。
    - Complex dtype: 参考 reference_code/main.py，每个 qubit 状态为均匀复振幅
      state[:] = (1+1j) / sqrt(2*K)。

    Parameters:
    - num_qubits: Number of qubits
    - K: Dimension of each qubit state (即每个 qubit 的 R/K 维)
    - device: 设备
    - dtype: 'float32' / 'complex64' 或 torch.dtype
    Returns:
    - circuit_states_list: List of tensors representing the circuit states for each qubit
    """
    torch_dtype = _resolve_dtype(dtype)
    circuit_states_list = [torch.zeros(K, device=device, dtype=torch_dtype) for _ in range(num_qubits)]

    if torch_dtype == torch.complex64 or torch_dtype == torch.complex128:
        # 与 reference main.py 一致：均匀复振幅 (1+1j)/sqrt(2*K)
        val = (1.0 + 1j) / cmath.sqrt(2 * K)
        for i in range(len(circuit_states_list)):
            # circuit_states_list[i][:] = val
            circuit_states_list[i][-1] = 1.0
    else:
        for i in range(len(circuit_states_list)):
            circuit_states_list[i][-1] = 1.0

    return circuit_states_list


def wasserstein_1d(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    计算一维离散分布的 EMD / Wasserstein-1 距离。
    假设 a, b 长度相同且权重均匀，则
        W1 = mean(|sort(a) - sort(b)|)
    """
    a_sorted, _ = torch.sort(a)
    b_sorted, _ = torch.sort(b)
    return torch.mean(torch.abs(a_sorted - b_sorted))


def compute_sampling_metrics(
    qctn,
    engine: EngineSiamese,
    circuit_states_list,
    x_test: torch.Tensor,
    D: int,
    K: int,
    device: torch.device,
    num_eval_samples: int = 2048,
    bounds = (-5.0, 5.0),
    grid_size: int = 100,
):
    """
    使用 engine.sample 从当前 QCTN 采样，并与测试集 x_test 比较 Wasserstein_1d。
    返回一个 dict，便于后续扩展更多 metric。
    """
    if x_test is None or x_test.numel() == 0:
        return {}
    
    print(f"circuit_states_list: {circuit_states_list}")
    # num_eval_samples=1

    with torch.no_grad():
        samples = engine.sample(
            qctn,
            circuit_states_list,
            num_eval_samples,
            K,
            bounds=[bounds[0], bounds[1]],
            grid_size=grid_size,
        )
        # 复数 backend 时采样结果取实部再算 Wasserstein（概率/数值本身为实）
        if engine.backend.is_complex(samples):
            # samples = engine.backend.real(samples)
            samples = engine.backend.abs_square(samples)

        # # 准备与 samples 同样数量的测试集点
        # x_test = x_test.to(samples.device)
        # if x_test.shape[0] >= num_eval_samples:
        #     idx = torch.randperm(x_test.shape[0], device=samples.device)[:num_eval_samples]
        #     target_points = x_test[idx]
        # else:
        #     repeat = (num_eval_samples + x_test.shape[0] - 1) // x_test.shape[0]
        #     target_points = x_test.repeat(repeat, 1)[:num_eval_samples]
        # if engine.backend.is_complex(target_points):
        #     # target_points = engine.backend.real(target_points)
        #     target_points = engine.backend.abs_square(target_points)

        print(f"samples: {samples.shape}")
        for i in range(10):
            print(f"samples: {samples[i, :]}")

        target_points = torch.empty(samples.shape, device=device).normal_(mean=0.0, std=1.0)
        for i in range(10):
            print(f"target_points: {target_points[i, :]}")

        emd_list = []
        for d in range(D):
            emd = wasserstein_1d(samples[:, d], target_points[:, d])
            emd_list.append(emd)

        emd_tensor = torch.stack(emd_list)  # (D,)

        metrics = {
            "wasserstein_mean": emd_tensor.mean().item(),
        }

        # 如有需要，也可以在这里增加更多 metric，例如方差、KL 等
        return metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train QCTN with step-based checkpointing, eval and TensorBoard logging.")
    parser.add_argument("--save-every", type=int, default=100, help="每多少个 step 保存一次 checkpoint（默认 100）")
    parser.add_argument("--eval-every", type=int, default=200, help="每多少个 step 做一次 eval（默认 200）")
    parser.add_argument("--num-eval-samples", type=int, default=2048, help="eval 时采样的样本数（默认 2048）")
    parser.add_argument("--checkpoint-root", type=str, default="./checkpoints", help="checkpoint 根目录（默认 ./checkpoints）")
    parser.add_argument("--exp-name", type=str, default="", help="自定义实验子目录名（默认自动生成）")
    parser.add_argument("--num-step", type=int, default=1000, help="训练总步数（默认 1000）")
    args = parser.parse_args()

    backend_type = 'pytorch'

    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    backend = BackendFactory.create_backend(backend_type, device='cpu', dtype="complex64")
    # backend = BackendFactory.create_backend(backend_type, device='cpu', dtype="float32")

    engine = EngineSiamese(backend=backend, strategy_mode="balanced", mx_K=100)
    # engine = EngineSiamese(backend=backend, strategy_mode="fast", mx_K=100)

    x_distribution = "gaussian"

    suffix = "_exp01"
    # graph_type = "tree"
    # graph_type = "wall"
    graph_type = "std"
    # qctn_graph = QCTNHelper.generate_example_graph(n=17, dim_char='3')
    # qctn_graph = QCTNHelper.generate_example_graph(n=17, graph_type="std", dim_char='3')
    # print(f"std qctn_graph: \n{qctn_graph}")

    qctn_graph = QCTNHelper.generate_example_graph(n=3, graph_type=graph_type, dim_char='3')
    # qctn_graph = QCTNHelper.generate_example_graph(n=17, graph_type=graph_type, dim_char='3')
    # qctn_graph = QCTNHelper.generate_example_graph(n=5, graph_type=graph_type, dim_char='3')
    # qctn_graph = QCTNHelper.generate_example_graph(n=5, graph_type=graph_type, dim_char='3')
    # qctn_graph = QCTNHelper.generate_example_graph(n=257, graph_type=graph_type, dim_char='3')
    print(f"{graph_type} qctn_graph: \n{qctn_graph}")

    qctn_graph = """
-3-a-3-
-3-a-3-
-------
    """
    print(f"{graph_type} qctn_graph: \n{qctn_graph}")
    
    
    qctn = QCTN(qctn_graph, backend=engine.backend)

    # qctn = QCTN.from_pretrained(qctn_graph, "assets/qctn_cores_3qubitswall_dist_00.safetensors", backend=engine.backend)
    # pretrained_qctn = QCTN.from_pretrained(qctn_graph, "assets/qctn_cores_3qubitswall_dist_00.safetensors", backend=engine.backend)
    # right_qctn = QCTN(qctn_graph, backend=engine.backend)
    N = 100
    B = 1024
    # B = 1
    D = qctn.nqubits
    K = 3
    num_step = args.num_step

    # ================================
    # 实验与日志配置（可通过命令行参数覆盖）
    # ================================
    save_every = args.save_every
    eval_every = args.eval_every
    num_eval_samples = args.num_eval_samples
    checkpoint_root = args.checkpoint_root

    device = backend.backend_info.device
    
    default_exp_name = f"{qctn.nqubits}qubits{graph_type}{suffix}_{int(time.time())}"
    exp_name = args.exp_name if args.exp_name else default_exp_name
    exp_dir = os.path.join(checkpoint_root, exp_name)
    os.makedirs(exp_dir, exist_ok=True)
    print(f"Experiment directory: {exp_dir}")

    tb_log_dir = os.path.join(exp_dir, "tensorboard")
    writer = SummaryWriter(log_dir=tb_log_dir)
    print(f"TensorBoard log dir: {tb_log_dir}")

    data_list = []
    x_list = []
    
    # ================================
    # 定义用于生成 x 的目标分布（高斯）
    # 这里参考 set_target_score 中的 Gaussian 设置，
    # 并通过字典形式保留 target_distribution['gaussian']。
    # ================================
    cov_matrix = torch.eye(D, dtype=torch.float64)
    if D > 1:
        indices = torch.arange(D - 1)
        cov_matrix[indices + 1, indices] = 0.2
        cov_matrix[indices, indices + 1] = 0.2
    cov_matrix.diagonal().fill_(1.0)

    target_distribution_dict = {
        'gaussian': torch.distributions.MultivariateNormal(
            loc=torch.zeros(D, dtype=torch.float64),
            covariance_matrix=cov_matrix,
        ),
    }
    
    for i in range(N):
        if x_distribution == "gaussian":
            # 保持原有 Gaussian 采样方式（独立标准正态）
            x = torch.empty((B, D), device=device).normal_(mean=0.0, std=1.0)
        else:
            # 其他 x_distribution 时，参考 set_target_score 代码：
            # 使用 target_distribution_dict['gaussian'] 这个高斯分布来生成 x。
            base_gaussian = target_distribution_dict['gaussian']
            # MultivariateNormal 的 sample(shape) 返回 shape + event_shape
            x = base_gaussian.sample(torch.Size([B]))  # 形状为 (B, D)
            x = x.to(device)

        # Mx_list, out = engine.generate_data(x, K=K, ret_type='TNTensor')
        Mx_list, out = engine.generate_data(x, K=K)

        data_list += [({"measure_input_list": Mx_list}, out)]
        x_list.append(x.detach().clone())
    
    # data_list structure is [(dict, out), ...]
    # We only need the dict part for optimizer
    data_list_for_optim = [x[0] for x in data_list]

    # 构造测试集点（可将来替换为真实测试集）
    x_all = torch.cat(x_list, dim=0)  # (N * B, D)
    num_train_points = int(0.8 * x_all.shape[0])
    x_test = x_all[num_train_points:]
    if x_test.numel() == 0:
        x_test = x_all
    
    # for i in range(len(data_list_for_optim)):
    #     x = data_list_for_optim[i]["measure_input_list"]
    #     data_list_for_optim[i]["measure_input_list"] = [torch.eye(tensor.shape[-1], device=tensor.device) for tensor in x]

    # print(data_list_for_optim[0]["measure_input_list"][0].shape)
    
    circuit_states_list = generate_circuit_states_list(num_qubits=D, K=K, device=backend.backend_info.device, dtype=backend.backend_info.dtype)

    for c_name in qctn.cores:
        core_tensor = qctn.cores_weights[c_name]
        if isinstance(core_tensor, torch.Tensor):
            core_tensor.requires_grad_(True)
        else:
            core_tensor.tensor.requires_grad_(True)

        # elif isinstance(core_tensor, TNTensor):
        #     core_tensor.tensor.requires_grad_(True)
        requires_grad = core_tensor.requires_grad
        # requires_grad = core_tensor.tensor.requires_grad
        print(f"core {c_name} requires_grad: {requires_grad}")
    

    # for c_name in right_qctn.cores:
    #     core_tensor = right_qctn.cores_weights[c_name]
    #     core_tensor.requires_grad_(True)

    #     requires_grad = core_tensor.requires_grad
    #     print(f"pretrained core {c_name} requires_grad: {requires_grad}")
    
    lr_schedule = [
        (0, 1e-2),
        (200, 5e-3),
        (600, 2e-3),
        (800, 1e-3),
    ]

    optimizer = Optimizer(
        method='sgdg', 
        max_iter=num_step, 
        # tol=1e-6, 
        tol=0.0, 
        # learning_rate=1e-2, 
        learning_rate=1e-3, 
        beta1=0.9, 
        beta2=0.95, 
        epsilon=1e-8,
        engine=engine,
        # lr_schedule=lr_schedule,

        momentum=0.9,            # 动量因子
        stiefel=True,            # 启用 Stiefel 流形优化
    )

    # 绑定训练过程中的 checkpoint/eval/TensorBoard 配置
    optimizer.save_every = save_every
    optimizer.eval_every = eval_every

    def checkpoint_fn(step: int, qctn_obj, loss_value: float):
        ckpt_dir = os.path.join(exp_dir, f"step_{step:06d}")
        os.makedirs(ckpt_dir, exist_ok=True)
        cores_file = os.path.join(
            ckpt_dir,
            f"qctn_cores_{qctn_obj.nqubits}qubits{graph_type}{suffix}_step{step:06d}.safetensors",
        )
        qctn_obj.save_cores(cores_file, metadata={"graph": graph_type, "step": step, "loss": loss_value})
        print(f"[Checkpoint] Saved cores to {cores_file}")

    def eval_fn(step: int, qctn_obj):
        # 1) 采样指标（如 Wasserstein）
        metrics = compute_sampling_metrics(
            qctn_obj,
            engine,
            circuit_states_list,
            x_test,
            D,
            K,
            device,
            num_eval_samples=num_eval_samples,
        )
        print(f"[Eval] step {step}, metrics: {metrics}")

        # 2) 参考 tests/test_probabilities.py::test_heatmap_marginal
        #    在 eval 时也绘制前两个比特 (q0, q1) 的边缘概率热力图
        try:
            edge_size = 100
            B = edge_size * edge_size

            x_grid = torch.empty((B, D), device=device)
            delta = 5.0 / edge_size
            step_val = 10.0 / edge_size
            for dx in range(edge_size):
                for dy in range(edge_size):
                    vals = [dx * step_val - 5.0 + delta / 2.0,
                            dy * step_val - 5.0 + delta / 2.0]
                    if D > 2:
                        vals += [0.0] * (D - 2)
                    x_grid[dx * edge_size + dy, :] = torch.tensor(vals, device=device)

            # 生成测量算符
            Mx_list, _ = engine.generate_data(x_grid, K=K)
            measure_input_list = [engine.backend.convert_to_tensor(m) for m in Mx_list]

            # 计算前两个比特的边缘概率
            print("Calculating marginal probability heatmap for qubits [0, 1] during eval...")
            marg = engine.calculate_marginal_probability(
                qctn_obj,
                circuit_states_list,
                [measure_input_list[0], measure_input_list[1]],
                [0, 1],
            )
            if engine.backend.is_complex(marg):
                marg = engine.backend.real(marg)

            heatmap = marg.reshape(edge_size, edge_size).detach().cpu().numpy()

            # 保存到当前 step 的子目录下
            eval_dir = os.path.join(exp_dir, f"step_{step:06d}")
            os.makedirs(eval_dir, exist_ok=True)
            heatmap_file = os.path.join(
                eval_dir,
                f"marginal_probability_heatmap_q0_q1_step{step:06d}.png",
            )

            plt.figure()
            plt.imshow(heatmap, cmap="hot", interpolation="nearest")
            plt.colorbar()
            plt.title(f"Marginal Probability Heatmap (q0, q1) - step {step}")
            plt.savefig(heatmap_file)
            plt.close()
            print(f"[Eval] marginal probability heatmap saved to {heatmap_file}")
        except Exception as e:
            print(f"[Eval] heatmap generation failed at step {step}: {e}")

        return metrics

    optimizer.checkpoint_fn = checkpoint_fn
    optimizer.eval_fn = eval_fn
    optimizer.summary_writer = writer

    torch.cuda.empty_cache()
    
    tic = time.time()

    optimizer.optimize(qctn, 
                       data_list=data_list_for_optim, 
                    #    circuit_states=circuit_states_list,
                       circuit_states_list=circuit_states_list,
                    #    circuit_states_list=None,
                    #    right_qctn=right_qctn,
                       )

    toc = time.time()
    
    print(f"已分配显存: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
    print(f"缓存显存: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")
    print(f"Optimization Time: {toc - tic:.2f} seconds")

    # save final cores
    final_cores_file = os.path.join(
        exp_dir,
        f"qctn_cores_{qctn.nqubits}qubits{graph_type}{suffix}_final.safetensors",
    )
    qctn.save_cores(final_cores_file, metadata={"graph": graph_type})
    
    print(f"Saved trained QCTN cores to {final_cores_file}")

    # 关闭 TensorBoard writer
    writer.close()

    exit()

    # Choose the first data of each batch for testing
    test_loss_list = []
    for i in range(N):
        data_slice = [x[0:1] for x in data_list[i]["measure_input_list"]]
        result = engine.contract_with_std_graph(qctn, 
                                                circuit_states_list=circuit_states_list,
                                                measure_input_list=data_slice, 
                                                )
        print(f"Test {i}, Result: {result.item()}")
        test_loss_list.append(result.item())

    print(f"Average Result: {sum(test_loss_list) / len(test_loss_list)}")
    print(f"Max Result: {max(test_loss_list)}")
    print(f"Min Result: {min(test_loss_list)}")

    
    # load pretrained qctn
    pretrained_qctn = QCTN.from_pretrained(qctn_graph, cores_file, backend=engine.backend)
    with torch.no_grad():
        pretrained_result = engine.contract_with_std_graph(
            pretrained_qctn,
            circuit_states_list=circuit_states_list,
            measure_input_list=data_list[0]["measure_input_list"],
        )
    print(f"Pretrained Result (std graph): {pretrained_result} {pretrained_result.shape}")

    # get cross entropy loss for all pretrained results (shape = 512)
    loss = - torch.log(pretrained_result)
    print(f"Cross Entropy Loss: {loss} {torch.mean(loss)}")
