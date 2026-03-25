"""MNIST image training example.

Train a TN (model2) to approximate another TN (model1) initialized from MNIST.

Usage:
    python examples/train_mnist.py
"""

import torch
import numpy as np
from PIL import Image
from torchvision import datasets, transforms
from tneq_qc import QCTN, EngineCommon, BackendFactory
from tneq_qc.core.tn_tensor import TNTensor
from tneq_qc.utils.graph_generators import QCTNHelper
from tneq_qc.optim import Adam


def load_mnist_image(idx: int = 0, size: int = 32):
    """Load the idx-th MNIST image and resize to size x size."""
    transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
    ])
    dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    img, label = dataset[idx]
    return img[0], label


def init_model1_from_image(graph: str, image_tensor, backend):
    """Initialize QCTN with shared core 'A' from image data."""
    qctn = QCTN(graph, backend=backend)
    core_name = 'A'
    if core_name not in qctn.cores_weights:
        return qctn

    shape = qctn.cores_weights[core_name].shape
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
        print(f"    '{c}' shape={tuple(t.shape)}  requires_grad={t.requires_grad}")


if __name__ == "__main__":

    N_QUBITS   = 5
    PHYS_DIM   = 2
    IMAGE_SIZE = 32
    N_EPOCHS   = 1000
    LR         = 0.01
    LOG_EVERY  = 10

    torch.manual_seed(42)

    backend = BackendFactory.create_backend('pytorch', device='cpu', dtype='float32')
    engine  = EngineCommon(backend=backend, strategy_mode="full")

    # Model 1: fixed, from MNIST image
    print("=" * 60)
    print("Model 1  (fixed — from MNIST image)")
    print("=" * 60)

    graph1 = "\n".join(["-2-A-2-"] * N_QUBITS)
    mnist_image, mnist_label = load_mnist_image(idx=0, size=IMAGE_SIZE)
    print(f"  Image label: {mnist_label} {mnist_image.shape}")

    model1 = init_model1_from_image(graph1, mnist_image, backend)
    print_qctn_info(model1, label="Model 1")

    # Model 2: trainable
    print()
    print("=" * 60)
    print("Model 2  (trainable)")
    print("=" * 60)

    graph2 = "-2-A-2-\n-2-A-2-\n-2-A-2-\n-2-A-2-\n-2-A-2-"
    model2 = QCTN(graph2, backend=backend)
    model2.auto_init()

    for c in model2.cores:
        core = model2.cores_weights[c]
        noise = torch.randn_like(core.tensor) * 0.01
        core.set(core.tensor + noise, core.scale)
        core.requires_grad_(True)

    print_qctn_info(model2, label="Model 2")
    print(f"\nTrainable cores: {len(model2.parameters())}")

    # Train
    print()
    print("=" * 60)
    print(f"Training  epochs={N_EPOCHS}  lr={LR}  optimizer=Adam")
    print("=" * 60)

    optimizer = Adam(model2.parameters(), backend, lr=LR)
    loss_history = []

    for step in range(1, N_EPOCHS + 1):
        loss_val, grads = engine.contract_for_gradient(model2, target=model1, loss='mse')
        optimizer.step(list(grads))
        lv = float(loss_val)
        loss_history.append(lv)
        if step % LOG_EVERY == 0 or step == 1:
            print(f"  Step {step:4d}/{N_EPOCHS}  loss={lv:.6f}")

    # Results
    print()
    print("=" * 60)
    print("Training complete")
    print(f"  Initial loss : {loss_history[0]:.6f}")
    print(f"  Final   loss : {loss_history[-1]:.6f}")
    print(f"  Loss reduced : {loss_history[0] - loss_history[-1]:.6f}")
    print("=" * 60)

    # Inference: contract model2 and save as image
    print()
    print("=" * 60)
    print("Inference on model2 -> save as image")
    print("=" * 60)

    with torch.no_grad():
        result2 = engine.contract(model2)

    result_np = backend.tensor_to_numpy(result2)
    if np.iscomplexobj(result_np):
        result_np = np.abs(result_np)

    result_np = result_np.flatten()

    r_min, r_max = result_np.min(), result_np.max()
    if r_max > r_min:
        result_np = (result_np - r_min) / (r_max - r_min)
    else:
        result_np = np.zeros_like(result_np)

    total = 32 * 32
    if result_np.size < total:
        result_np = np.pad(result_np, (0, total - result_np.size))
    else:
        result_np = result_np[:total]

    img_array = (result_np.reshape(32, 32) * 255).astype(np.uint8)
    out_path = "assets/mnist/model2_output.png"
    Image.fromarray(img_array, mode="L").save(out_path)
    print(f"  Saved: {out_path}  (shape=32x32, dtype=uint8)")
    print(f"  Value range before norm: [{r_min:.4f}, {r_max:.4f}]")
    print("=" * 60)
