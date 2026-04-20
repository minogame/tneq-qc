import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from examples.train_dist import N_QUBITS
from tneq_qc import QCTN, EngineCommon, BackendFactory, create_optimizer
from tneq_qc.core.tn_tensor import TNTensor
from tneq_qc.utils.graph_generators import QCTNHelper

import torch
import numpy as np
from torchvision import datasets, transforms

# Parameters
NUM_QUBITS = 5
DIM_LOCAL = 2
NUM_EPOCHS = 1000
LR = 0.01
LOG_EVERY = 10
SAVE_DIR = "checkpoints"
ASSETS_DIR = "assets/mnist"
BATCH_SIZE = 4
TN_STRUCTURE = "brickwall"  # "brickwall" or "mps"

def load_mnist_image(batch_size: int = 1, size: int = 32):
    transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
    ])
    dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")

    batch_size = min(batch_size, len(dataset))
    imgs = []
    labels = []
    for i in range(batch_size):
        img, label = dataset[i]
        imgs.append(img[0])  # Get the single channel
        labels.append(label)

    imgs = torch.stack(imgs, dim=0)
    labels = torch.tensor(labels, dtype=torch.long)
    return imgs, labels

if __name__ == "__main__":
    # Load image
    dim_img = DIM_LOCAL ** NUM_QUBITS
    images, labels = load_mnist_image(batch_size=BATCH_SIZE, size=dim_img)
    print(f"Loaded {len(images)} images with shape {images.shape} and labels {labels}")

    # Create backend and engine
    backend = BackendFactory.create_backend("pytorch", device="cpu")
    engine  = EngineCommon(backend=backend, strategy_mode="full")

    # model 1
    print("=" * 60)
    print("Model 1  (fixed — from MNIST image)")
    print("=" * 60)
    graph1 = "\n".join(["-2-A-2-"] * NUM_QUBITS)
    qtn_mnist = QCTN(graph1, backend=backend).auto_init()
    core_names = list(qtn_mnist.cores_weights.keys())

    core_shape = tuple(qtn_mnist.cores_weights[core_names[0]].shape)
    batched = images.reshape(BATCH_SIZE, *core_shape)
    qtn_mnist.cores_weights[core_names[0]] = TNTensor(batched, has_batch=True)

    t_mnist = TNTensor(batched, has_batch=True)

    # model 2
    print()
    print("=" * 60)
    print("Model 2  (trainable)")
    print("=" * 60)

    # get the MPS from helper
    if TN_STRUCTURE == "brickwall":
        graph2 = QCTNHelper.brickwall(NUM_QUBITS, 10, DIM_LOCAL)
    else:
        graph2 = QCTNHelper.mps(NUM_QUBITS, DIM_LOCAL*5,DIM_LOCAL)
    # NOTE: auto_init has no power to initilize batched cores
    qtn = QCTN(graph2, backend=backend).auto_init() 
    core_names = list(qtn.cores_weights.keys())
    for core_name in core_names:
        shape = qtn.cores_weights[core_name].shape
        # Materialize a real batched tensor so each slice has independent storage.
        batched_core = qtn.cores_weights[core_name].expand(BATCH_SIZE, *shape).clone()
        qtn.cores_weights[core_name] = TNTensor(batched_core, has_batch=True)

    print(f"[debug] Model 2 cores: {qtn.cores_weights.keys()} with shapes {[qtn.cores_weights[c].shape for c in qtn.cores_weights.keys()]}")

    for c in qtn.cores:
        core = qtn.cores_weights[c]
        core.requires_grad_(True)

    # Train
    print()
    print("=" * 60)
    print(f"Training  epochs={NUM_EPOCHS}  lr={LR}  optimizer=Adam")
    print("=" * 60)

    opt = create_optimizer("adam", qtn.parameters(), backend=backend, lr=LR)
    loss_history = []

    for step in range(1, NUM_EPOCHS + 1):
        loss_val, grads = engine.contract_for_gradient(qtn, target=t_mnist, loss='mse')
        opt.step(list(grads))
        lv = float(loss_val)
        loss_history.append(lv)
        if step % LOG_EVERY == 0:
            print(f"Step {step}/{NUM_EPOCHS}  Loss: {lv:.6f}")

    # draw the loss curve with log scale in y-axis
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 5))
    plt.plot(loss_history, label="Training Loss")
    plt.yscale("log")
    plt.xlabel("Epoch")
    plt.ylabel("Loss (log scale)")
    plt.title("Training Loss Curve")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # draw the original and reconstructed images
    # The first rows are original images, the second rows are reconstructed images from the trained model
    with torch.no_grad():
        reconstructed = engine.contract(qtn).cpu().numpy()

    print(f"Reconstructed shape: {reconstructed.shape}")
    reconstructed = reconstructed.reshape(BATCH_SIZE, dim_img, dim_img)
    plt.figure(figsize=(10, 4))
    for i in range(BATCH_SIZE):
        plt.subplot(2, BATCH_SIZE, i + 1)
        plt.imshow(images[i].cpu(), cmap="gray")
        plt.title(f"Original (Label: {labels[i].item()})")
        plt.axis("off")

        plt.subplot(2, BATCH_SIZE, BATCH_SIZE + i + 1)
        plt.imshow(reconstructed[i], cmap="gray")
        plt.title("Reconstructed")
        plt.axis("off")
    plt.tight_layout()
    plt.show()

    # save the orignal and reconstructed images and loss_history in a folder,
    # The file names are "original_{i}.png", "reconstructed_{TN_STRUCTURE}_{i}.png" and "loss_history_{TN_STRUCTURE}.npy"
    os.makedirs(ASSETS_DIR, exist_ok=True)
    for i in range(BATCH_SIZE):
        plt.imsave(os.path.join(ASSETS_DIR, f"original_{i}.png"), images[i].cpu(), cmap="gray")
        plt.imsave(os.path.join(ASSETS_DIR, f"reconstructed_{TN_STRUCTURE}_{i}.png"), reconstructed[i], cmap="gray")
    np.save(os.path.join(ASSETS_DIR, f"loss_history_{TN_STRUCTURE}.npy"), np.array(loss_history))