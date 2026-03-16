import torch
import numpy as np
from PIL import Image
from torchvision import datasets, transforms
from tqdm import tqdm
from tneq_qc.backends.backend_factory import BackendFactory
from tneq_qc.core.qctn import QCTN
from tneq_qc.core.tn_tensor import TNTensor
from tneq_qc.utils.graph_generators import QCTNHelper
from tneq_qc.core.engine_common import EngineCommon


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_mnist_image(idx: int = 0, size: int = 32):
    dataset = datasets.MNIST(root="./data", train=True, download=True)
    img, label = dataset[idx]
    img.save("assets/mnist/mnist_image_view.png")

    """从 MNIST 加载第 idx 张图片并 resize 到 size×size。"""
    transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
    ])
    dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    img, label = dataset[idx]
    return img[0], label  # 单通道 (size, size)


def init_model1_from_image(graph: str, image_tensor, backend):
    """用图片数据初始化含共享 core 'A' 的 QCTN（model1，固定不训练）。"""
    qctn = QCTN(graph, backend=backend)
    core_name = 'A'
    if core_name not in qctn.cores_weights:
        return qctn

    shape = qctn.cores_weights[core_name].tensor.shape
    total = 1
    for d in shape:
        total *= d

    img_flat = image_tensor.flatten()
    if img_flat.numel() < total:
        repeats = (total // img_flat.numel()) + 1
        img_flat = img_flat.repeat(repeats)[:total]
    else:
        img_flat = img_flat[:total]

    core_data = img_flat.reshape(shape).to(dtype=torch.complex64)
    core_data = core_data / torch.norm(core_data)
    qctn.cores_weights[core_name] = backend.convert_to_tensor(core_data)
    return qctn


def print_qctn_info(qctn, label: str = "QCTN"):
    print(f"\n[{label}]")
    print(f"  nqubits={qctn.nqubits}  ncores={qctn.ncores}  cores={qctn.cores}")
    print(f"  Structure:\n{qctn}")
    print("  Core shapes:")
    for c in qctn.cores:
        t = qctn.cores_weights[c]
        rg = t.tensor.requires_grad
        print(f"    '{c}' shape={tuple(t.tensor.shape)}  requires_grad={rg}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    # ---- Hyper-parameters -----------------------------------------------
    N_QUBITS   = 5
    N_LAYERS   = 4    # brickwall 层数
    PHYS_DIM   = 2
    IMAGE_SIZE = 32
    N_EPOCHS   = 1000
    LR         = 0.01
    LOG_EVERY  = 10   # 每隔多少 epoch 打印一次
    # -----------------------------------------------------------------------

    torch.manual_seed(42)

    # backend = BackendFactory.create_backend('pytorch', device='cpu', dtype='complex64')
    backend = BackendFactory.create_backend('pytorch', device='cpu', dtype='float32')
    engine  = EngineCommon(backend=backend, strategy_mode="full")

    # ------------------------------------------------------------------
    # Model 1: 5 qubits，共享 core A，用 MNIST 图片初始化，不参与训练
    # ------------------------------------------------------------------
    print("=" * 60)
    print("Model 1  (fixed — from MNIST image)")
    print("=" * 60)

    graph1 = "\n".join(["-2-A-2-"] * N_QUBITS)
    mnist_image, mnist_label = load_mnist_image(idx=0, size=IMAGE_SIZE)
    print(f"  Image label: {mnist_label} {mnist_image.shape} {mnist_image.max()} {mnist_image.min()}")
    Image.fromarray((mnist_image.numpy() * 255).astype(np.uint8)).save("assets/mnist/mnist_image.png")

    model1 = init_model1_from_image(graph1, mnist_image, backend)
    print_qctn_info(model1, label="Model 1")
    # exit()

    # ------------------------------------------------------------------
    # Model 2: MPS 结构，自动正交初始化，全部 core 参与训练
    # ------------------------------------------------------------------
    print()
    print("=" * 60)
    print("Model 2  (trainable MPS)")
    print("=" * 60)

    graph2 = QCTNHelper.brickwall(N_QUBITS, n_layers=N_LAYERS, phys_dim=PHYS_DIM)
    
    graph2 = "-2-A-2-\n-2-A-2-\n-2-A-2-\n-2-A-2-\n-2-A-2-"
    print(f"graph2: \n{graph2}")

    model2 = QCTN(graph2, backend=backend)

    print(f"model2: \n{model2}")
    # exit()

    model2.auto_init()

    for c in model2.cores:
        core = model2.cores_weights[c]
        noise = torch.randn_like(core.tensor) * 0.01
        core.set(core.tensor + noise, core.scale)
        core.tensor.requires_grad_(True)
        # model2.cores_weights[c].tensor.requires_grad_(True)

    print_qctn_info(model2, label="Model 2")

    # ------------------------------------------------------------------
    # 收集 model2 的可训练参数（model1 固定，仅作 target 使用）
    # ------------------------------------------------------------------
    def _raw(t):
        return t.tensor if isinstance(t, TNTensor) else t

    trainable_cores = [c for c in model2.cores if _raw(model2.cores_weights[c]).requires_grad]
    params = [model2.cores_weights[c] for c in trainable_cores]
    print(f"\nTrainable cores: {trainable_cores}  ({len(params)} tensors)")

    # ------------------------------------------------------------------
    # 训练循环
    # ------------------------------------------------------------------
    print()
    print("=" * 60)
    print(f"Training  epochs={N_EPOCHS}  lr={LR}  optimizer=Adam")
    print("=" * 60)

    optimizer_state: dict = {}
    loss_history = []

    for epoch in tqdm(range(1, N_EPOCHS + 1)):

        # 前向 + 反向：以 model1 的 contract 结果为 target，用 MSE loss
        loss, grads = engine.contract_with_compiled_strategy_for_gradient(
            model2, target=model1, loss='mse'
        )

        # 参数更新（原地修改 TNTensor，combined.cores_weights 自动反映）
        params, optimizer_state = backend.optimizer_update(
            params, list(grads), optimizer_state,
            method='adam',
            hyperparams={'learning_rate': LR, 'iter': epoch - 1},
        )

        loss_val = loss.item() if hasattr(loss, 'item') else float(loss)
        loss_history.append(loss_val)

        if epoch % LOG_EVERY == 0 or epoch == 1:
            print(f"  Epoch {epoch:4d}/{N_EPOCHS}  loss={loss_val:.6f}")

    # ------------------------------------------------------------------
    # 训练结果
    # ------------------------------------------------------------------
    print()
    print("=" * 60)
    print("Training complete")
    print(f"  Initial loss : {loss_history[0]:.6f}")
    print(f"  Final   loss : {loss_history[-1]:.6f}")
    print(f"  Loss reduced : {loss_history[0] - loss_history[-1]:.6f}")

    print("\nFinal gradient norms (last backward pass):")
    for c in model2.cores:
        core = model2.cores_weights[c]
        raw = core.tensor if isinstance(core, TNTensor) else core
        if raw.grad is not None:
            print(f"  '{c}' grad_norm={raw.grad.norm().item():.6f}")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 推理：对 model2 单独做一次 contract，结果保存为 32×32 图片
    # ------------------------------------------------------------------
    print()
    print("=" * 60)
    print("Inference on model2 → save as image")
    print("=" * 60)

    with torch.no_grad():
        result2 = engine.contract_with_compiled_strategy(model2)

    # 还原有效值并取绝对值
    result_np = backend.tensor_to_numpy(result2)          # ndarray, complex or real
    if np.iscomplexobj(result_np):
        result_np = np.abs(result_np)                     # 取模

    result_np = result_np.flatten()

    # 归一化到 [0, 255]
    r_min, r_max = result_np.min(), result_np.max()
    if r_max > r_min:
        result_np = (result_np - r_min) / (r_max - r_min)
    else:
        result_np = np.zeros_like(result_np)

    # reshape 到 32×32，不足则 pad，超出则截断
    total = 32 * 32
    if result_np.size < total:
        result_np = np.pad(result_np, (0, total - result_np.size))
    else:
        result_np = result_np[:total]

    img_array = (result_np.reshape(32, 32) * 255).astype(np.uint8)
    out_path = "assets/mnist/model2_output.png"
    Image.fromarray(img_array, mode="L").save(out_path)
    print(f"  Saved: {out_path}  (shape=32×32, dtype=uint8)")
    print(f"  Value range before norm: [{r_min:.4f}, {r_max:.4f}]")
    print("=" * 60)
