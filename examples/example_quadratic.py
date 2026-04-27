import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tneq_qc.utils.graph_generators import QCTNHelper
import torch
import numpy as np
import matplotlib.pyplot as plt

from tneq_qc import (
    QCTN, TNTensor, EngineCommon, BackendFactory, Quadratic,
    DataGenerator, create_optimizer
)

# Parameters
DIM_RANDOM_VARIABLES = 3
DIM_LOCAL = 4
BOND_DIM = DIM_LOCAL
BATCH_SIZE = 128
NUM_EPOCHS = 1000
LR = 0.01
LOG_EVERY = 1

NUM_SAMPLES = 1000

def init_circuit_01(qctn: QCTN, backend) -> QCTN:
    """Fill each circuit core with an alternating 0/1 pattern."""
    for c in qctn.cores:
        core = qctn.cores_weights[c]
        shape = tuple(core.shape)
        n = 1
        for d in shape:
            n *= d
        flat = torch.zeros(n, dtype=backend.default_dtype)
        flat[-1] = 1
        qctn.cores_weights[c] = backend.convert_to_tensor(flat.reshape(shape))
    return qctn

def compute_marginal_heatmap(engine, combined, data_gen, mx_core_names,
                             grid_size=100, bounds=(-3, 3)):
    """Compute marginal probability P(x_i) on a 1D grid for each qubit."""
    K = DIM_LOCAL
    x_min, x_max = bounds
    grid = np.linspace(x_min, x_max, grid_size).astype(np.float32)

    # Identity for unsampled cores
    backend = engine.backend
    ident = backend.eye(K)
    if isinstance(ident, TNTensor):
        ident = TNTensor(ident)
    else:
        ident = TNTensor(ident)

    n_dims = len(mx_core_names)
    heatmap = np.zeros((n_dims, grid_size))

    for dim_idx, core_name in enumerate(mx_core_names):
        # Reset all mx cores to identity
        for name in mx_core_names:
            combined[name] = ident

        # Evaluate probability on grid for this dimension
        for gi, x_val in enumerate(grid):
            x_input = np.array([[x_val]], dtype=np.float32)
            Mx_list, _ = data_gen.generate(x_input, K=K, ret_type='TNTensor')
            combined[core_name] = Mx_list[0]

            prob = engine.calculate_probability(combined, {})
            heatmap[dim_idx, gi] = max(prob, 0.0)

        # Restore identity
        combined[core_name] = ident

    return heatmap, grid


def init_measure_identity(qctn: QCTN, backend) -> QCTN:
    """Fill each measure core with an identity matrix placeholder."""
    for core_info in qctn.adjacency_table:
        core_name = core_info['core_name']
        input_shape = core_info['input_shape']
        output_shape = core_info['output_shape']
        input_dim = core_info['input_dim']
        output_dim = core_info['output_dim']
        if input_dim != output_dim:
            raise ValueError(
                f"Measure core {core_name!r} must be square, got {input_dim} and {output_dim}."
            )
        core = backend.eye(input_dim)
        qctn.cores_weights[core_name] = backend.reshape(core, input_shape + output_shape)
    return qctn

def main():
    # build the model
    device = os.environ.get("TNEQ_DEVICE", "cpu")
    backend = BackendFactory.create_backend('pytorch', device=device, dtype='complex64')
    data_gen = DataGenerator(backend, mx_K=DIM_LOCAL)

    # build the model
    engine = EngineCommon(backend=backend, strategy_mode="balanced")

    graph = QCTNHelper.mps(DIM_RANDOM_VARIABLES, bond_dim=BOND_DIM, phys_dim=DIM_LOCAL)
    model  = Quadratic(
        graph,
        DIM_LOCAL,
        backend=backend
    )
    model._submodules['tn'].auto_init(orthogonal=True)
    model._submodules['circuit'].auto_init(orthogonal=False)
    init_circuit_01(model._submodules['circuit'], backend)
    init_measure_identity(model._submodules['mx'], backend)
    model._submodules['tn'].requires_grad_(True)
    combined = model.build()
    print(combined.graph)

    # train the model
    opt = create_optimizer("sgdg", combined.parameters(), backend=backend, lr=LR)
    mx_core_names = model.mx_core_names
    loss_history = []

    rng = np.random.default_rng(seed=42)

    for step in range(1, NUM_EPOCHS + 1):
        # Sample a batch of data
        # Example 1 Gamma distribution: shape=2.0, scale=0.7 (mean=1.4) - a skewed distribution with support on [0, inf)
        # x = rng.gamma(shape=2.0, scale=0.7, size=(BATCH_SIZE, DIM_RANDOM_VARIABLES)).astype(np.float32)

        # Example 2 Gaussian mixture: 2 components centered at -1 and +1
        # x1 = rng.normal(-1, 0.5, size=(BATCH_SIZE // 2, DIM_RANDOM_VARIABLES))
        # x2 = rng.normal(1, 0.5, size=(BATCH_SIZE // 2, DIM_RANDOM_VARIABLES))
        # x = np.vstack([x1, x2]).astype(np.float32)

        # Example 3 Exponential distribution: scale=1.0 (mean=1.0) - a simple distribution with support on [0, inf)
        x = rng.exponential(scale=1.0, size=(BATCH_SIZE, DIM_RANDOM_VARIABLES)).astype(np.float32)

        Mx_list, _ = data_gen.generate(x, K=DIM_LOCAL, ret_type="TNTensor")
        for i, name in enumerate(mx_core_names):
            combined[name] = Mx_list[i]

        loss_val, grads = engine.contract_for_gradient(combined, target=1, loss='nll')
        opt.step(list(grads))
        lv = float(loss_val)
        loss_history.append(lv)
        if step % LOG_EVERY == 0 or step == NUM_EPOCHS:
            print(f"Step {step}/{NUM_EPOCHS}, Loss: {lv:.4f}")


    # evaluate the model
    # 1. Marginal probability heatmap
    print("Computing marginal heatmap...")
    heatmap, grid = compute_marginal_heatmap(
        engine, combined, data_gen, mx_core_names,
        grid_size=100, bounds=(-3, 3)
    )

    fig, ax = plt.subplots(figsize=(10, 4))
    im = ax.imshow(heatmap, aspect='auto', origin='lower',
                   extent=[grid[0], grid[-1], 0, len(mx_core_names)],
                   cmap='hot')
    ax.set_xlabel('x')
    ax.set_ylabel('Qubit index')
    ax.set_title(f'Marginal Probability Heatmap ({DIM_RANDOM_VARIABLES} qubits)')
    plt.colorbar(im, ax=ax, label='P(x)')
    plt.tight_layout()
    plt.show()

    # 2. Sample from the model
    print(f"Sampling {NUM_SAMPLES} points...")
    with torch.no_grad():
        samples = engine.sample(
            combined, data_gen, mx_core_names,
            num_samples=NUM_SAMPLES, bounds=(-3, 3), grid_size=100
        )
    if isinstance(samples, TNTensor):
        samples = samples.tensor
    samples_np = samples.detach().cpu().numpy() if hasattr(samples, 'detach') else np.array(samples)

    # Scatter plot (first 2 dimensions if available) compared to true distribution

    # true_samples = rng.gamma(shape=2.0, scale=0.7, size=(NUM_SAMPLES, DIM_RANDOM_VARIABLES)).astype(np.float32)

    # For Gaussian mixture:
    # true_samples = rng.normal(-1, 0.5, size=(NUM_SAMPLES // 2, DIM_RANDOM_VARIABLES))
    # x2 = rng.normal(1, 0.5, size=(NUM_SAMPLES // 2, DIM_RANDOM_VARIABLES))
    # true_samples = np.vstack([true_samples, x2]).astype(np.float32)

    # For Exponential distribution:
    true_samples = rng.exponential(scale=1.0, size=(NUM_SAMPLES, DIM_RANDOM_VARIABLES)).astype(np.float32)

    

    plt.figure(figsize=(6, 5))
    if DIM_RANDOM_VARIABLES >= 2:
        plt.scatter(true_samples[:, 0], true_samples[:, 1], alpha=0.5, label='True Samples')
        plt.scatter(samples_np[:, 0], samples_np[:, 1], alpha=0.5, label='Model Samples')
        plt.xlabel('x_0')
        plt.ylabel('x_1')
        plt.title('Scatter Plot of Samples (First 2 Dimensions)')
    else:
        plt.hist(true_samples[:, 0], bins=50, density=True, alpha=0.7, label='True Samples')
        plt.hist(samples_np[:, 0], bins=50, density=True, alpha=0.7, label='Model Samples')
        plt.xlabel('x')
        plt.ylabel('Density')
        plt.title('Histogram of Samples (1D)')
    plt.legend()
    plt.tight_layout()
    plt.show()




    # n_plot_dims = min(samples_np.shape[1], 2)
    # fig, axes = plt.subplots(1, n_plot_dims, figsize=(6 * n_plot_dims, 5))
    # if n_plot_dims == 1:
    #     axes = [axes]
    # for d in range(n_plot_dims):
    #     axes[d].hist(samples_np[:, d], bins=50, density=True, alpha=0.7, label='Samples')
    #     axes[d].set_xlabel(f'x_{d}')
    #     axes[d].set_ylabel('Density')
    #     axes[d].set_title(f'Sampled distribution (dim {d})')
    #     axes[d].legend()

    # plt.tight_layout()
    # plt.show()


if __name__ == "__main__":
    main()