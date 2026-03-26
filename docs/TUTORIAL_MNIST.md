# Tutorial: MNIST Image Approximation Training

> Train a trainable tensor network to approximate a fixed tensor network initialized from an MNIST image

---

## 1. Task Description

Train a tensor network (model2) to approximate another tensor network (model1) initialized from MNIST image data.

**Mathematical objective**:

$$\min_{\theta} \| \text{model2}(\theta) - \text{model1} \|^2$$

**Features**:
- No DataGenerator / measurement matrices needed
- Direct MSE loss between QCTNs
- model1's core is filled with MNIST image pixels
- model2 learns model1's structure via gradient descent

---

## 2. Complete Code

```python
import torch
import numpy as np
import os
from tneq_qc import QCTN, EngineCommon, BackendFactory
from tneq_qc.optim import Adam
from torchvision import datasets, transforms

# ====================== Configuration ======================
N_QUBITS   = 5         # number of qubits
PHYS_DIM   = 2         # physical dimension
IMAGE_SIZE = 32         # image size (after resize)
N_EPOCHS   = 1000       # number of training epochs
LR         = 0.01       # learning rate

# ====================== Data Loading ======================

def load_mnist_image(idx=0, size=32):
    """Load the idx-th MNIST image and resize to size x size."""
    transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
    ])
    dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    img, label = dataset[idx]
    return img[0], label   # (size, size) grayscale, label

# ====================== Initialization ======================

torch.manual_seed(42)
backend = BackendFactory.create_backend('pytorch', device='cpu', dtype='float32')
engine  = EngineCommon(backend=backend, strategy_mode='full')

# Graph definition: all qubits share a single core A
graph = "\n".join(["-2-A-2-"] * N_QUBITS)

# Model 1: initialized from MNIST image (fixed)
mnist_image, label = load_mnist_image(idx=0, size=IMAGE_SIZE)
print(f"MNIST digit: {label}")

model1 = QCTN(graph, backend=backend)
model1.auto_init()

# Fill core A with image data
core_A = model1.cores_weights['A']
shape = tuple(core_A.shape)
total = 1
for d in shape:
    total *= d

img_flat = mnist_image.flatten()
if img_flat.numel() < total:
    img_flat = img_flat.repeat((total // img_flat.numel()) + 1)[:total]
else:
    img_flat = img_flat[:total]

core_data = img_flat.reshape(shape).to(dtype=torch.complex64)
core_data = core_data / torch.norm(core_data)     # normalize
model1.cores_weights['A'] = backend.convert_to_tensor(core_data)

# Model 2: trainable
model2 = QCTN(graph, backend=backend)
model2.auto_init()

for c in model2.cores:
    core = model2.cores_weights[c]
    noise = torch.randn_like(core.tensor) * 0.01
    core.set(core.tensor + noise, core.scale)
    core.requires_grad_(True)

# ====================== Training ======================

optimizer = Adam(model2.parameters(), backend, lr=LR)
loss_history = []

for step in range(1, N_EPOCHS + 1):
    # target=model1: the engine first contracts model1 to get the target tensor
    loss_val, grads = engine.contract_for_gradient(
        model2,
        target=model1,      # another QCTN as target
        loss='mse',
    )
    optimizer.step(list(grads))
    lv = float(loss_val)
    loss_history.append(lv)

    if step % 100 == 0:
        print(f"Step {step:4d}/{N_EPOCHS}  loss={lv:.6f}")

print(f"\nInitial loss: {loss_history[0]:.6f}")
print(f"Final   loss: {loss_history[-1]:.6f}")

# ====================== Save Model ======================

os.makedirs("checkpoints", exist_ok=True)
model2.save_cores("checkpoints/mnist_model2.safetensors", metadata={
    'mnist_label': str(label),
    'n_epochs': str(N_EPOCHS),
    'final_loss': f"{loss_history[-1]:.6f}",
})
```

---

## 3. Code Walkthrough

### 3.1 target=QCTN

```python
loss_val, grads = engine.contract_for_gradient(model2, target=model1, loss='mse')
```

When `target` is a `QCTN` instance, `TargetResolver` first executes `engine.contract(model1)` to obtain the target tensor, then computes the MSE loss against the contraction result of model2.

This means:
- model1's parameters do not participate in gradient computation
- Gradients only flow to model2's trainable parameters

### 3.2 Image to Core

```python
core_data = img_flat.reshape(shape).to(dtype=torch.complex64)
core_data = core_data / torch.norm(core_data)
```

Image pixels are flattened and filled into the core tensor. Normalization ensures numerical stability. Since the core shape may have fewer (or more) elements than image pixels, truncation or repeated padding is applied.

### 3.3 Adam Optimizer

MNIST approximation is a non-convex optimization problem, and Adam's adaptive learning rate is better suited:

```python
optimizer = Adam(model2.parameters(), backend, lr=0.01)
```

---

## 4. Visualization

### 4.1 Converting Contraction Results to Images

```python
import matplotlib.pyplot as plt

with torch.no_grad():
    result1 = engine.contract(model1)
    result2 = engine.contract(model2)

def tensor_to_image(result_np, size=32):
    if np.iscomplexobj(result_np):
        result_np = np.abs(result_np)
    result_np = result_np.flatten()
    r_min, r_max = result_np.min(), result_np.max()
    if r_max > r_min:
        result_np = (result_np - r_min) / (r_max - r_min)
    total = size * size
    result_np = result_np[:total] if result_np.size >= total else \
                np.pad(result_np, (0, total - result_np.size))
    return (result_np.reshape(size, size) * 255).astype(np.uint8)

img1 = tensor_to_image(backend.tensor_to_numpy(result1), IMAGE_SIZE)
img2 = tensor_to_image(backend.tensor_to_numpy(result2), IMAGE_SIZE)
```

### 4.2 Comparison Plot

```python
fig, axes = plt.subplots(1, 3, figsize=(12, 4))

axes[0].imshow(img1, cmap='gray')
axes[0].set_title('Target (Model 1)')

axes[1].imshow(img2, cmap='gray')
axes[1].set_title('Learned (Model 2)')

diff = np.abs(img1.astype(float) - img2.astype(float))
axes[2].imshow(diff, cmap='hot')
axes[2].set_title(f'|Difference|')

plt.savefig("assets/mnist/comparison.png", dpi=150)
```

### 4.3 Loss Curve

```python
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(loss_history)
ax.set_xlabel('Step')
ax.set_ylabel('MSE Loss')
ax.set_title(f'Training Loss (MNIST digit {label})')
plt.savefig("assets/mnist/loss_curve.png", dpi=150)
```

---

## 5. Load and Verify

```python
# Recreate a QCTN with the same structure
model2_loaded = QCTN(graph, backend=backend).auto_init()
model2_loaded.load_cores("checkpoints/mnist_model2.safetensors")

with torch.no_grad():
    result_loaded = engine.contract(model2_loaded)

# Compare original result with loaded result
result2_np = backend.tensor_to_numpy(result2)
loaded_np  = backend.tensor_to_numpy(result_loaded)
max_err = np.max(np.abs(result2_np - loaded_np))
print(f"Max reload error: {max_err:.2e}")   # should be close to 0
```

---

## 6. Key Takeaways

| Element | Choice |
|---|---|
| Model structure | Simple independent cores (`-2-A-2-`) |
| Loss function | MSE (target is another QCTN) |
| Optimizer | Adam (suited for non-convex optimization) |
| Data | No external data flow; model1 is fixed as target |
| dtype | float32 (image data is real-valued) |
| strategy_mode | `'full'` (small-scale network) |
