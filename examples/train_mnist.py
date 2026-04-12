"""MNIST image training example.

Train a TN (model2) to approximate another TN (model1) initialized from MNIST.
After training: saves model, outputs comparison images (original vs learned).

Usage:
    python examples/train_mnist.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import datasets, transforms
from tneq_qc import QCTN, EngineCommon, BackendFactory, create_optimizer
from tneq_qc.core.tn_tensor import TNTensor
from tneq_qc.utils.graph_generators import QCTNHelper


N_QUBITS   = 5
PHYS_DIM   = 2
IMAGE_SIZE = 32
N_EPOCHS   = 1000
LR         = 0.01
LOG_EVERY  = 10
SAVE_DIR   = "checkpoints"
ASSETS_DIR = "assets/mnist"


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

    core_data = img_flat.reshape(shape).to(dtype=backend.default_dtype)
    core_data = core_data / torch.norm(core_data)
    qctn.cores_weights[core_name] = backend.convert_to_tensor(core_data)
    return qctn


def tensor_to_image(result_np, size=32):
    """Convert a flattened tensor result into a normalized image array."""
    if np.iscomplexobj(result_np):
        result_np = np.abs(result_np)
    result_np = result_np.flatten()

    r_min, r_max = result_np.min(), result_np.max()
    if r_max > r_min:
        result_np = (result_np - r_min) / (r_max - r_min)
    else:
        result_np = np.zeros_like(result_np)

    total = size * size
    if result_np.size < total:
        result_np = np.pad(result_np, (0, total - result_np.size))
    else:
        result_np = result_np[:total]

    return (result_np.reshape(size, size) * 255).astype(np.uint8)


def print_qctn_info(qctn, label: str = "QCTN"):
    print(f"\n[{label}]")
    print(f"  nqubits={qctn.nqubits}  ncores={qctn.ncores}  cores={qctn.cores}")
    print(f"  Structure:\n{qctn}")
    print("  Core shapes:")
    for c in qctn.cores:
        t = qctn.cores_weights[c]
        print(f"    '{c}' shape={tuple(t.shape)}  requires_grad={t.requires_grad}")


if __name__ == "__main__":

    torch.manual_seed(42)
    device = os.environ.get('TNEQ_DEVICE', 'cpu')

    backend = BackendFactory.create_backend('pytorch', device=device, dtype='float32')
    engine  = EngineCommon(backend=backend, strategy_mode="full")

    # Model 1: fixed, from MNIST image
    print("=" * 60)
    print("Model 1  (fixed — from MNIST image)")
    print("=" * 60)

    graph1 = "\n".join(["-2-A-2-"] * N_QUBITS)
    mnist_image, mnist_label = load_mnist_image(idx=0, size=IMAGE_SIZE)
    print(f"  Image label: {mnist_label} {mnist_image.shape}")
    print(f"  Device: {device}")

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

    optimizer = create_optimizer("adam", model2.parameters(), backend=backend, lr=LR)
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

    # Save model
    os.makedirs(SAVE_DIR, exist_ok=True)
    os.makedirs(ASSETS_DIR, exist_ok=True)
    save_path = os.path.join(SAVE_DIR, "mnist_model2.safetensors")
    model2.save_cores(save_path, metadata={
        'mnist_label': str(mnist_label),
        'n_epochs': str(N_EPOCHS),
        'final_loss': f"{loss_history[-1]:.6f}",
    })
    print(f"Model saved: {save_path}")

    # Inference: contract both models and save comparison images
    print()
    print("=" * 60)
    print("Validation: comparing original vs learned")
    print("=" * 60)

    with torch.no_grad():
        result1 = engine.contract(model1)
        result2 = engine.contract(model2)

    img1_np = tensor_to_image(backend.tensor_to_numpy(result1), IMAGE_SIZE)
    img2_np = tensor_to_image(backend.tensor_to_numpy(result2), IMAGE_SIZE)

    # Save individual images
    img1_path = os.path.join(ASSETS_DIR, "model1_target.png")
    img2_path = os.path.join(ASSETS_DIR, "model2_output.png")
    Image.fromarray(img1_np, mode="L").save(img1_path)
    Image.fromarray(img2_np, mode="L").save(img2_path)
    print(f"  Target image : {img1_path}")
    print(f"  Learned image: {img2_path}")

    # Side-by-side comparison
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0].imshow(img1_np, cmap='gray')
    axes[0].set_title('Target (Model 1)')
    axes[0].axis('off')

    axes[1].imshow(img2_np, cmap='gray')
    axes[1].set_title('Learned (Model 2)')
    axes[1].axis('off')

    diff = np.abs(img1_np.astype(float) - img2_np.astype(float))
    axes[2].imshow(diff, cmap='hot')
    axes[2].set_title(f'|Difference| (MSE={np.mean(diff**2):.1f})')
    axes[2].axis('off')

    comparison_path = os.path.join(ASSETS_DIR, "comparison.png")
    plt.tight_layout()
    plt.savefig(comparison_path, dpi=150)
    plt.close()
    print(f"  Comparison   : {comparison_path}")

    # Loss curve
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(loss_history)
    ax.set_xlabel('Step')
    ax.set_ylabel('MSE Loss')
    ax.set_title(f'Training Loss (MNIST digit {mnist_label})')
    ax.grid(True, alpha=0.3)
    loss_path = os.path.join(ASSETS_DIR, "loss_curve.png")
    plt.tight_layout()
    plt.savefig(loss_path, dpi=150)
    plt.close()
    print(f"  Loss curve   : {loss_path}")

    # Validation: reload model and verify
    print("\n=== Reload Validation ===")
    model2_loaded = QCTN(graph2, backend=backend)
    model2_loaded.auto_init()
    model2_loaded.load_cores(save_path)

    with torch.no_grad():
        result_loaded = engine.contract(model2_loaded)

    result2_np = backend.tensor_to_numpy(result2)
    result_loaded_np = backend.tensor_to_numpy(result_loaded)
    reload_err = np.max(np.abs(result2_np - result_loaded_np))
    print(f"  Max reload error: {reload_err:.2e}")
    print("=" * 60)
