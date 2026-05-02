"""MNIST batch training example.

One batched student model (model2) learns B MNIST images simultaneously.

- model1 (teacher): shared core 'A' with batch dimension (B images stacked).
- model2 (student): shared core 'A' also has batch dimension, so each batch
  slice learns its corresponding teacher image.
- Loss = MSE averaged over the B aligned teacher/student pairs.

After training: saves model, outputs comparison image grid and loss curve.

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


# ── Configuration ───────────────────────────────────────────────────────────
N_QUBITS   = 5
PHYS_DIM   = 2
IMAGE_SIZE = 32          # 32×32 = 1024 = 2^10 elements per image
BATCH_SIZE = 8           # number of MNIST images to learn simultaneously
N_EPOCHS   = 1000
LR         = 0.01
LOG_EVERY  = 50
SAVE_DIR   = "checkpoints"
ASSETS_DIR = "assets/mnist"


# ── Helpers ─────────────────────────────────────────────────────────────────

def load_mnist_images(indices, size: int = 32):
    """Load multiple MNIST images, resize, return stacked tensor + labels."""
    transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
    ])
    dataset = datasets.MNIST(root="./data", train=True, download=True,
                             transform=transform)
    images, labels = [], []
    for idx in indices:
        img, lbl = dataset[idx]
        images.append(img[0])          # [size, size]
        labels.append(lbl)
    return torch.stack(images, dim=0), labels   # [B, size, size]


def init_teacher_batch(graph: str, images: torch.Tensor, backend):
    """Create model1 with batched core A from B images.

    images: [B, H, W] float tensor.
    Core A shape becomes (B, 2, 2, ..., 2) with has_batch=True.
    """
    qctn = QCTN(graph, backend=backend)

    core_info = next(info for info in qctn.adjacency_table if info['core_name'] == 'A')
    core_shape = tuple(core_info['input_shape'] + core_info['output_shape'])
    n_elem = 1
    for d in core_shape:
        n_elem *= d

    B = images.shape[0]
    flat = images.reshape(B, -1).to(dtype=backend.default_dtype)   # [B, H*W]

    # Pad or truncate to match core element count
    if flat.shape[1] < n_elem:
        flat = torch.nn.functional.pad(flat, (0, n_elem - flat.shape[1]))
    else:
        flat = flat[:, :n_elem]

    # Normalize each image independently
    norms = flat.norm(dim=1, keepdim=True).clamp(min=1e-8)
    flat = flat / norms

    batched_core = flat.reshape(B, *core_shape)
    qctn.cores_weights['A'] = TNTensor(batched_core, has_batch=True)
    return qctn


def init_student_batch(graph: str, batch_size: int, backend):
    """Create model2 with batched trainable cores.

    Each batch slice owns an independent copy of the shared core tensor, so
    batch item ``b`` learns teacher image ``b`` directly.
    """
    qctn = QCTN(graph, backend=backend)
    qctn.add_core_batch_size(batch_size)
    qctn.auto_init(orthogonal=False)
    for c in qctn.cores:
        core = qctn.cores_weights[c]
        if getattr(core, "is_fixed", False):
            continue
        noise = torch.randn_like(core.tensor) * 0.01
        qctn.cores_weights[c] = TNTensor(
            core.tensor + noise,
            scale=core.scale,
            has_batch=core.has_batch,
            is_fixed=core.is_fixed,
            fixed_kind=core.fixed_kind,
        )
        qctn.cores_weights[c].requires_grad_(True)
    return qctn


def tensor_to_image(result_np, size=32):
    """Convert a flattened tensor result into a normalised uint8 image."""
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
    print(f"  nqubits={qctn.nqubits}  ncores={qctn.ncores}")
    print(f"  Core shapes:")
    for c in qctn.cores:
        t = qctn.cores_weights[c]
        print(f"    '{c}' shape={tuple(t.shape)}  has_batch={t.has_batch}"
              f"  requires_grad={t.tensor.requires_grad}")


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    torch.manual_seed(42)
    device = os.environ.get('TNEQ_DEVICE', 'cpu')

    backend = BackendFactory.create_backend('pytorch', device=device,
                                            dtype='float32')
    engine  = EngineCommon(backend=backend, strategy="row_priority")

    graph = "\n".join(["-2-A-2-"] * N_QUBITS)

    # ── Teacher (model1): B images in batch ──────────────────────────────
    print("=" * 60)
    print(f"Model 1  (teacher — {BATCH_SIZE} MNIST images, fixed)")
    print("=" * 60)

    indices = list(range(BATCH_SIZE))
    images, labels = load_mnist_images(indices, size=IMAGE_SIZE)
    print(f"  Labels : {labels}")
    print(f"  Images : {tuple(images.shape)}")
    print(f"  Device : {device}")

    model1 = init_teacher_batch(graph, images, backend)
    print_qctn_info(model1, label="Model 1 (teacher)")

    # ── Student (model2): batched trainable cores ────────────────────────
    print()
    print("=" * 60)
    print(f"Model 2  (student — {BATCH_SIZE} batched params, trainable)")
    print("=" * 60)

    model2 = init_student_batch(graph, BATCH_SIZE, backend)
    print_qctn_info(model2, label="Model 2 (student)")
    print(f"  Trainable cores: {len(model2.parameters())}")

    # ── Training ─────────────────────────────────────────────────────────
    print()
    print("=" * 60)
    print(f"Training  epochs={N_EPOCHS}  lr={LR}  batch={BATCH_SIZE}  "
          f"optimizer=Adam")
    print("=" * 60)

    optimizer = create_optimizer("adam", model2.parameters(),
                                backend=backend, lr=LR)
    loss_history = []

    for step in range(1, N_EPOCHS + 1):
        loss_val, grads = engine.contract_for_gradient(
            model2, target=model1, loss='mse')
        optimizer.step(list(grads))
        lv = float(loss_val)
        loss_history.append(lv)
        if step % LOG_EVERY == 0 or step == 1:
            print(f"  Step {step:4d}/{N_EPOCHS}  loss={lv:.6f}")

    # ── Results ──────────────────────────────────────────────────────────
    print()
    print("=" * 60)
    print("Training complete")
    print(f"  Initial loss : {loss_history[0]:.6f}")
    print(f"  Final   loss : {loss_history[-1]:.6f}")
    print(f"  Loss reduced : {loss_history[0] - loss_history[-1]:.6f}")
    print("=" * 60)

    # ── Save model ───────────────────────────────────────────────────────
    os.makedirs(SAVE_DIR, exist_ok=True)
    os.makedirs(ASSETS_DIR, exist_ok=True)
    save_path = os.path.join(SAVE_DIR, "mnist_model2.safetensors")
    model2.save_cores(save_path, metadata={
        'mnist_labels': str(labels),
        'batch_size': str(BATCH_SIZE),
        'n_epochs': str(N_EPOCHS),
        'final_loss': f"{loss_history[-1]:.6f}",
    })
    print(f"Model saved: {save_path}")

    # ── Validation: contract both, compare per-image ─────────────────────
    print()
    print("=" * 60)
    print("Validation: comparing teacher images vs student output")
    print("=" * 60)

    with torch.no_grad():
        result1 = engine.contract(model1)   # batched: (B, 2, 2, ...)
        result2 = engine.contract(model2)   # batched: (B, 2, 2, ...)

    result1_np = backend.tensor_to_numpy(result1)
    result2_np = backend.tensor_to_numpy(result2)

    # Per-image MSE
    B = BATCH_SIZE
    for i in range(B):
        teacher_flat = result1_np[i].flatten()
        student_flat = result2_np[i].flatten()
        mse_i = np.mean((student_flat - teacher_flat) ** 2)
        print(f"  Image {i} (digit {labels[i]}): MSE={mse_i:.6f}")

    # ── Save comparison grid ─────────────────────────────────────────────
    n_cols = 2
    n_rows = B
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6, 3 * n_rows))
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    for i in range(B):
        teacher_img = tensor_to_image(result1_np[i], IMAGE_SIZE)
        student_img = tensor_to_image(result2_np[i], IMAGE_SIZE)

        ax_t = axes[i, 0]
        ax_t.imshow(teacher_img, cmap='gray')
        ax_t.set_title(f'Teacher #{i}\n(digit {labels[i]})', fontsize=9)
        ax_t.axis('off')

        ax_s = axes[i, 1]
        ax_s.imshow(student_img, cmap='gray')
        ax_s.set_title(f'Student #{i}', fontsize=9)
        ax_s.axis('off')

    comparison_path = os.path.join(ASSETS_DIR, "comparison.png")
    plt.tight_layout()
    plt.savefig(comparison_path, dpi=150)
    plt.close()
    print(f"  Comparison grid: {comparison_path}")

    # Loss curve
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(loss_history)
    ax.set_xlabel('Step')
    ax.set_ylabel('MSE Loss')
    ax.set_title(f'Training Loss ({BATCH_SIZE} MNIST images)')
    ax.grid(True, alpha=0.3)
    loss_path = os.path.join(ASSETS_DIR, "loss_curve.png")
    plt.tight_layout()
    plt.savefig(loss_path, dpi=150)
    plt.close()
    print(f"  Loss curve     : {loss_path}")

    # ── Reload verification ──────────────────────────────────────────────
    print("\n=== Reload Validation ===")
    model2_loaded = QCTN(graph, backend=backend)
    model2_loaded.load_cores(save_path)

    with torch.no_grad():
        result_loaded = engine.contract(model2_loaded)

    reload_err = np.max(np.abs(result2_np - backend.tensor_to_numpy(result_loaded)))
    print(f"  Max reload error: {reload_err:.2e}")
    print("=" * 60)


if __name__ == "__main__":
    main()
