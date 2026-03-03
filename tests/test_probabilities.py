from tneq_qc.backends import backend_factory
import torch
import numpy as np
import time
import matplotlib.pyplot as plt
import os

from tneq_qc.core.engine_siamese import EngineSiamese
from tneq_qc.core.qctn import QCTN, QCTNHelper

# ==========================================
# Helper Functions
# ==========================================

def generate_circuit_states_list(num_qubits, K, device='cuda'):
    circuit_states_list = [torch.zeros(K, device=device) for _ in range(num_qubits)]
    for i in range(len(circuit_states_list)):
        circuit_states_list[i][-1] = 1.0
    return circuit_states_list

# ==========================================
# Test Functions
# ==========================================

def test_probabilities():
    print("\n=== Test 0: Simple Probabilities (v2) ===")
    # Setup
    backend_type = 'pytorch'
    backend = backend_factory.BackendFactory.create_backend(backend_type, device='cuda')
    engine = EngineSiamese(backend=backend)
    
    # Create a simple 2-qubit circuit
    graph = "-2-A-2-\n-2-B-2-"
    qctn = QCTN(graph, backend=engine.backend)
    
    # Circuit states: |0> for all qubits
    # shape (B, 2)
    batch_size = 4
    state_0 = torch.tensor([1.0, 0.0], dtype=torch.float32)
    state_0_batch = state_0.unsqueeze(0).expand(batch_size, -1) # (B, 2)
    circuit_states = [state_0_batch, state_0_batch]
    
    circuit_states = [engine.backend.convert_to_tensor(s) for s in circuit_states]

    # Create Projectors
    # |0><0|
    proj_0 = torch.tensor([[1.0, 0.0], [0.0, 0.0]], dtype=torch.float32)
    proj_0_batch = proj_0.unsqueeze(0).expand(batch_size, -1, -1) # (B, 2, 2)
    
    # 1. Test Full Probability P(00)
    measure_list_full = [proj_0_batch, proj_0_batch]

    measure_list_full = [engine.backend.convert_to_tensor(m) for m in measure_list_full]

    prob_00 = engine.calculate_full_probability(qctn, circuit_states, measure_list_full)
    print(f"P(00) shape: {prob_00.shape}")
    print(f"P(00): {prob_00[0]}")
    
    # 2. Test Marginal Probability P(q0=0)
    measure_list_marginal = [proj_0_batch]

    measure_list_marginal = [engine.backend.convert_to_tensor(m) for m in measure_list_marginal]

    prob_q0_0 = engine.calculate_marginal_probability(qctn, circuit_states, measure_list_marginal, [0])
    print(f"P(q0=0) shape: {prob_q0_0.shape}")
    print(f"P(q0=0): {prob_q0_0[0]}")
    
    measure_list_conditional = [proj_0_batch, proj_0_batch]

    measure_list_conditional = [engine.backend.convert_to_tensor(m) for m in measure_list_conditional]

    # 3. Test Conditional Probability P(q1=0 | q0=0)
    cond_prob = engine.calculate_conditional_probability(
        qctn, 
        circuit_states, 
        measure_input_list=measure_list_conditional, 
        qubit_indices=[0, 1], 
        target_indices=[1]
    )
    print(f"P(q1=0 | q0=0) shape: {cond_prob.shape}")
    print(f"P(q1=0 | q0=0): {cond_prob[0]}")
    
    # Expected: P(00) / P(q0=0)
    expected = prob_00 / (prob_q0_0 + 1e-10)
    print(f"Expected: {expected[0]}")
    
    assert torch.allclose(cond_prob, expected, atol=1e-5)
    print("Conditional probability test passed!")

def test_random_probabilities(qctn_cores_file="./assets/qctn_cores_3qubits_exp1.safetensors", device='cuda'):
    print("\n=== Test 1: Random Probabilities ===")
    backend_type = 'pytorch'
    backend = backend_factory.BackendFactory.create_backend(backend_type, device=device)
    engine = EngineSiamese(backend=backend)
    
    # Load QCTN
    if not os.path.exists(qctn_cores_file):
        print(f"Warning: {qctn_cores_file} not found. Using random initialization.")
        qctn_graph = QCTNHelper.generate_example_graph()
        qctn = QCTN(qctn_graph, backend=engine.backend)
    else:
        qctn_graph = QCTNHelper.generate_example_graph()
        qctn = QCTN.from_pretrained(qctn_graph, qctn_cores_file, backend=engine.backend)
    
    D = qctn.nqubits
    K = 3 # Dimension of state
    
    # Generate Data
    # batch_size=1, num_sample=1 (num_batch=1)
    
    # data = generate_Mx_phi_x_data(num_batch=1, batch_size=1, num_qubits=D, K=K, device=device)
    # measure_input_list = data[0][0] # List of Mx tensors
    
    x = torch.empty((1, D), device=device).normal_(mean=0.0, std=1.0)
    Mx_list, out = engine.generate_data(x, K=K)
    measure_input_list = Mx_list
    
    circuit_states = generate_circuit_states_list(D, K, device=device)
    
    circuit_states = [engine.backend.convert_to_tensor(s) for s in circuit_states]
    measure_input_list = [engine.backend.convert_to_tensor(m) for m in measure_input_list]

    # 1. Full Probability
    prob_full = engine.calculate_full_probability(qctn, circuit_states, measure_input_list)
    print(f"Full Probability: {prob_full.item()}")
    
    # tmp = measure_input_list[0]
    # measure_input_list[0] = None
    # prob_matrix = engine.contract_with_compiled_strategy(qctn, circuit_states, measure_input_list)
    # print(f"prob_matrix: {prob_matrix} {prob_matrix.shape}")

    # print(f"tmp shape: {tmp.shape}")

    # result = torch.einsum("aij,aji->a", tmp, prob_matrix)
    
    # print(f"Recomputed Full Probability: {result.item()}")

    # 2. Marginal Probability for each qubit
    print("\nMarginal Probabilities:")
    for i in range(D):
        # measure_input_list[i] is (B, K, K)
        prob_marg = engine.calculate_marginal_probability(
            qctn, 
            circuit_states, 
            [measure_input_list[i]], 
            [i]
        )
        print(f"Qubit {i}: {prob_marg.item()}")
        
    # 3. Conditional Probabilities
    print("\nConditional Probabilities:")
    
    if D >= 2:
        # Case 1: P(q0 | q1)
        prob_cond_1 = engine.calculate_conditional_probability(
            qctn, circuit_states, 
            [measure_input_list[0], measure_input_list[1]], 
            [0, 1], 
            [0]
        )
        print(f"P(q0 | q1): {prob_cond_1.item()}")
        
        # Case 2: P(q1 | q0)
        prob_cond_2 = engine.calculate_conditional_probability(
            qctn, circuit_states, 
            [measure_input_list[0], measure_input_list[1]], 
            [0, 1], 
            [1]
        )
        print(f"P(q1 | q0): {prob_cond_2.item()}")
        
        # Case 3: P(q0 | q1) with same inputs (sanity check)
        prob_cond_3 = engine.calculate_conditional_probability(
            qctn, circuit_states, 
            [measure_input_list[0], measure_input_list[1]], 
            [0, 1], 
            [0]
        )
        print(f"P(q0 | q1) (repeat): {prob_cond_3.item()}")

        # Case 4: P(q1 | q0) with same inputs (sanity check)
        prob_cond_4 = engine.calculate_conditional_probability(
            qctn, circuit_states, 
            [measure_input_list[0], measure_input_list[1]], 
            [0, 1], 
            [1]
        )
        print(f"P(q1 | q0) (repeat): {prob_cond_4.item()}")

        # Case 5: If D > 2, we could do more. For D=2, we are limited.
        # Let's try P(q0 | q1) but using a subset of measurements?
        # No, conditional requires measurements on both.
        print("Completed conditional probability tests.")
    else:
        print("Not enough qubits for conditional probability test.")

def test_heatmap_marginal(qctn_cores_file="./assets/qctn_cores_3qubits_exp1.safetensors",
                          output_file = './assets/marginal_probability_heatmap_3qubits01.png',
                          device='cuda'):
    print("\n=== Test 2: Heatmap Marginal ===")
    backend_type = 'pytorch'
    backend = backend_factory.BackendFactory.create_backend(backend_type, device=device)
    engine = EngineSiamese(backend=backend)
    
    if not os.path.exists(qctn_cores_file):
        print(f"Warning: {qctn_cores_file} not found. Using random initialization.")
        qctn_graph = QCTNHelper.generate_example_graph()
        qctn = QCTN(qctn_graph, backend=engine.backend)
    else:
        qubits_str = qctn_cores_file.split('qubits')[0].split('_')[-1]
        n_qubits = int(qubits_str)

        graph_type_str = qctn_cores_file.split('qubits')[-1].split('_')[0]
        if graph_type_str == '':
            graph_type_str = 'std'
        qctn_graph = QCTNHelper.generate_example_graph(n=n_qubits, graph_type=graph_type_str)
        qctn = QCTN.from_pretrained(qctn_graph, qctn_cores_file, backend=engine.backend)
        
    edge_size = 100
    N = 1
    B = edge_size * edge_size
    D = qctn.nqubits
    K = 3

    # data_list = generate_Mx_phi_x_uniform(num_batch=N, batch_size=B, num_qubits=D, K=K, edge_size=edge_size, device=device)
    # measure_input_list = data_list[0][0]
    
    x = torch.empty((B, D), device=device)
    delta = 5 / edge_size
    step = 10 / edge_size
    for dx in range(edge_size):
        for dy in range(edge_size):
            vals = [dx * step - 5 + delta / 2, dy * step - 5 + delta / 2]
            if D > 2:
                vals += [0.0] * (D - 2)
            x[dx * edge_size + dy, :] = torch.tensor(vals, device=device)
    
    Mx_list, out = engine.generate_data(x, K=K)
    measure_input_list = Mx_list
    
    
    # Generate Uniform Data
    # data_list = generate_Mx_phi_x_uniform(num_batch=N, batch_size=B, num_qubits=D, K=K, edge_size=edge_size, device=device)
    # measure_input_list = data_list[0][0]
    
    circuit_states = generate_circuit_states_list(D, K, device=device)

    circuit_states = [engine.backend.convert_to_tensor(s) for s in circuit_states]
    measure_input_list = [engine.backend.convert_to_tensor(m) for m in measure_input_list]
    
    # Calculate Marginal Probability for first 2 qubits
    print("Calculating marginal probability for qubits [0, 1]...")
    result = engine.calculate_marginal_probability(
        qctn,
        circuit_states,
        [measure_input_list[0], measure_input_list[1]],
        [0, 1]
    )
    
    print(f"Result shape: {result.shape}")
    
    # Plot Heatmap
    heatmap = result.reshape(edge_size, edge_size).cpu().numpy()
    plt.figure()
    plt.imshow(heatmap, cmap='hot', interpolation='nearest')
    plt.colorbar()
    plt.title('Marginal Probability Heatmap (q0, q1)')
    
    plt.savefig(output_file)
    print(f"Heatmap saved to {output_file}")

def test_sampling(qctn_cores_file="./assets/qctn_cores_3qubits_exp1.safetensors", output_file="./assets/samples_scatter.png", device='cuda'):
    print("\n=== Test 3: Sampling (Continuous) ===")
    backend_type = 'pytorch'
    backend = backend_factory.BackendFactory.create_backend(backend_type, device=device)
    engine = EngineSiamese(backend=backend)
    
    # Load QCTN
    if not os.path.exists(qctn_cores_file):
        print(f"Warning: {qctn_cores_file} not found. Using random initialization.")
        qctn_graph = QCTNHelper.generate_example_graph()
        qctn = QCTN(qctn_graph, backend=engine.backend)
    else:
        qubits_str = qctn_cores_file.split('qubits')[0].split('_')[-1]
        n_qubits = int(qubits_str)
        qctn_graph = QCTNHelper.generate_example_graph(n=n_qubits)
        qctn = QCTN.from_pretrained(qctn_graph, qctn_cores_file, backend=engine.backend)
    
    D = qctn.nqubits
    K = 3 # Dimension
    
    # Circuit states: |0> (or whatever initial state)
    circuit_states = generate_circuit_states_list(D, K, device=device)
    circuit_states = [engine.backend.convert_to_tensor(s) for s in circuit_states]
    
    num_samples = 1000  # Increased for better visualization
    
    print(f"Sampling {num_samples} samples from {D}-qubit circuit using Numerical Inverse CDF...")
    # Using bounds [-5, 5] and grid_size 1000 as per instructions
    t0 = time.time()
    samples = engine.sample(qctn, circuit_states, num_samples, K, bounds=[-5, 5], grid_size=100)
    t1 = time.time()
    
    print(f"Sampling finished in {t1-t0:.4f}s")
    print(f"Samples shape: {samples.shape}")
    print(f"First 5 samples:\n{samples[:5]}")

    # ==========================================
    # Compare with Random Reference Distributions using EMD/Wasserstein-1
    # ==========================================
    # 1) 生成与 samples 形状一致的高斯分布随机数据
    #    N(0, 1) 的各向同性高斯
    random_gaussian = torch.empty_like(samples).normal_(mean=0.0, std=1.0)

    # 2) 可选：生成与采样范围一致的均匀分布数据（[-5, 5]），作为另一种参考分布
    random_uniform = (torch.rand_like(samples) * 10.0) - 5.0

    def wasserstein_1d(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """
        计算一维离散分布的 EMD / Wasserstein-1 距离。
        假设 a, b 长度相同且权重均匀，则
            W1 = mean(|sort(a) - sort(b)|)
        """
        a_sorted, _ = torch.sort(a)
        b_sorted, _ = torch.sort(b)
        return torch.mean(torch.abs(a_sorted - b_sorted))

    # 对每一个维度分别计算 samples 与参考分布之间的 1D Wasserstein 距离
    emd_gaussian_list = []
    emd_uniform_list = []
    for d in range(D):
        emd_g = wasserstein_1d(samples[:, d], random_gaussian[:, d])
        emd_u = wasserstein_1d(samples[:, d], random_uniform[:, d])
        emd_gaussian_list.append(emd_g)
        emd_uniform_list.append(emd_u)

    emd_gaussian = torch.stack(emd_gaussian_list)  # (D,)
    emd_uniform = torch.stack(emd_uniform_list)    # (D,)

    print(f"Per-dimension Wasserstein-1 distance to Gaussian (N(0,1)): {emd_gaussian}")
    print(f"Mean Wasserstein-1 distance to Gaussian: {emd_gaussian.mean()}")
    print(f"Per-dimension Wasserstein-1 distance to Uniform[-5,5]: {emd_uniform}")
    print(f"Mean Wasserstein-1 distance to Uniform[-5,5]: {emd_uniform.mean()}")
    
    # Plotting
    if samples.shape[1] >= 2 and output_file:
        # Move to CPU for plotting
        samples_np = samples.to('cpu').detach().numpy()
        x_vals = samples_np[:, 0]
        y_vals = samples_np[:, 1]
        
        plt.figure(figsize=(8, 8))
        plt.scatter(x_vals, y_vals, alpha=0.6, s=10, c='blue', edgecolors='none')
        plt.xlim(-5, 5)
        plt.ylim(-5, 5)
        plt.title(f"Sampled Points (N={num_samples})\nFirst 2 Qubits")
        plt.xlabel("Qubit 0")
        plt.ylabel("Qubit 1")
        plt.axhline(0, color='grey', linewidth=0.5)
        plt.axvline(0, color='grey', linewidth=0.5)
        plt.grid(True, alpha=0.3)
        
        plt.savefig(output_file, dpi=150)
        print(f"Plot saved to {output_file}")
    
    # Basic checks
    assert samples.shape == (num_samples, D)
    # Check if samples are within bounds
    assert samples.min() >= -5.0
    assert samples.max() <= 5.0
    
    # Optional: Basic moment check
    mean = samples.mean(dim=0)
    std = samples.std(dim=0)
    print(f"Sample Mean per qubit: {mean}")
    print(f"Sample Std per qubit: {std}")
    
    print("Sampling test passed!")


if __name__ == "__main__":
    # test_random_probabilities(device='cpu')
    # test_heatmap_marginal(qctn_cores_file="./assets/qctn_cores_3qubits_exp2.safetensors",
    #                       output_file = './assets/marginal_probability_heatmap_3qubits_exp2_01.png',
    #                       device='cpu')
    # test_sampling(qctn_cores_file="./assets/qctn_cores_3qubits_exp2.safetensors", device='cpu')
    
    
    # test_heatmap_marginal(qctn_cores_file="./assets/qctn_cores.safetensors",
    #                       output_file = './assets/marginal_probability_heatmap_3qubits_exp2_01.png',
    #                       device='cpu')
    # test_sampling(qctn_cores_file="./assets/qctn_cores.safetensors", device='cpu')

    # test_heatmap_marginal(qctn_cores_file="./assets/qctn_cores_65qubits_exp1.safetensors",
    #                       output_file = './assets/marginal_probability_heatmap_65qubits_01.png',
    #                       device='cpu')

    # exit()

    # test_heatmap_marginal(qctn_cores_file="./assets/qctn_cores_1025qubits_exp1.safetensors",
    #                       output_file = './assets/marginal_probability_heatmap_1025qubits_01.png',
    #                       device='cpu')

    
    # test_heatmap_marginal(qctn_cores_file="./assets/qctn_cores_5qubits_exp01.safetensors",
    #                       output_file = './assets/marginal_probability_heatmap_5qubits_exp01.png',
    #                       device='cpu')


    test_sampling(qctn_cores_file="./assets/qctn_cores_17qubits_exp01.safetensors", 
                    output_file="./assets/samples_scatter_17qubits_exp01.png",
                    device='cpu')
    
    test_sampling(qctn_cores_file="./assets/qctn_cores_129qubitsstd_exp01.safetensors", 
                    output_file="./assets/samples_scatter_129qubitsstd_exp01.png",
                    device='cpu')

    test_heatmap_marginal(qctn_cores_file="./assets/qctn_cores_129qubitsstd_exp01.safetensors",
                          output_file = './assets/marginal_probability_heatmap_129qubitsstd_exp01.png',
                          device='cpu')
    
    exit()


    test_heatmap_marginal(qctn_cores_file="./assets/qctn_cores_257qubitsstd_dist_00.safetensors",
                          output_file = './assets/marginal_probability_heatmap_257qubitsstd_dist_00.png',
                          device='cpu')

    test_heatmap_marginal(qctn_cores_file="./assets/qctn_cores_16qubitstree_exp01.safetensors",
                          output_file = './assets/marginal_probability_heatmap_16qubitstree_exp01.png',
                          device='cpu')

    test_heatmap_marginal(qctn_cores_file="./assets/qctn_cores_17qubitstree_exp01.safetensors",
                          output_file = './assets/marginal_probability_heatmap_17qubitstree_exp01.png',
                          device='cpu')

    test_heatmap_marginal(qctn_cores_file="./assets/qctn_cores_17qubitswall_exp01.safetensors",
                          output_file = './assets/marginal_probability_heatmap_17qubitswall_exp01.png',
                          device='cpu')
    
    test_heatmap_marginal(qctn_cores_file="./assets/qctn_cores_17qubitstree_dist_00.safetensors",
                          output_file = './assets/marginal_probability_heatmap_17qubitstree_dist_00.png',
                          device='cpu')
    
    test_heatmap_marginal(qctn_cores_file="./assets/qctn_cores_17qubitswall_dist_00.safetensors",
                          output_file = './assets/marginal_probability_heatmap_17qubitswall_dist_00.png',
                          device='cpu')
    
    test_heatmap_marginal(qctn_cores_file="./assets/qctn_cores_3qubitswall_dist_00.safetensors",
                          output_file = './assets/marginal_probability_heatmap_3qubitswall_dist_00.png',
                          device='cpu')
    
    exit()
    
    test_heatmap_marginal(qctn_cores_file="./assets/qctn_cores_2qubits_exp01.safetensors",
                          output_file = './assets/marginal_probability_heatmap_2qubits_exp01.png',
                          device='cpu')
    test_heatmap_marginal(qctn_cores_file="./assets/qctn_cores_3qubits_exp01.safetensors",
                          output_file = './assets/marginal_probability_heatmap_3qubits_exp01.png',
                          device='cpu')
    test_heatmap_marginal(qctn_cores_file="./assets/qctn_cores_17qubits_exp01.safetensors",
                          output_file = './assets/marginal_probability_heatmap_17qubits_exp01.png',
                          device='cpu')
    
    test_heatmap_marginal(qctn_cores_file="./assets/qctn_cores_2qubits_dist_00.safetensors",
                          output_file = './assets/marginal_probability_heatmap_2qubits_dist_00.png',
                          device='cpu')
    test_heatmap_marginal(qctn_cores_file="./assets/qctn_cores_3qubits_dist_00.safetensors",
                          output_file = './assets/marginal_probability_heatmap_3qubits_dist_00.png',
                          device='cpu')
    test_heatmap_marginal(qctn_cores_file="./assets/qctn_cores_5qubits_dist_00.safetensors",
                          output_file = './assets/marginal_probability_heatmap_5qubits_dist_00.png',
                          device='cpu')

    test_heatmap_marginal(qctn_cores_file="./assets/qctn_cores_17qubits_dist_00.safetensors",
                          output_file = './assets/marginal_probability_heatmap_17qubits_dist_00.png',
                          device='cpu')
    exit()

    test_heatmap_marginal(qctn_cores_file="./assets/qctn_cores_5qubits_dist_00.safetensors",
                          output_file = './assets/marginal_probability_heatmap_5qubits_dist_00.png',
                          device='cpu')
    exit()
    

    test_heatmap_marginal(qctn_cores_file="./assets/qctn_cores_17qubits_exp11.safetensors",
                          output_file = './assets/marginal_probability_heatmap_17qubits_11.png',
                          device='cpu')
    exit()
    
    test_heatmap_marginal(qctn_cores_file="./assets/qctn_cores_9qubits_exp1.safetensors",
                          output_file = './assets/marginal_probability_heatmap_9qubits_01.png',
                          device='cpu')
    exit()

    test_heatmap_marginal(qctn_cores_file="./assets/qctn_cores_129qubits_exp1.safetensors",
                          output_file = './assets/marginal_probability_heatmap_129qubits_01.png',
                          device='cpu')

    test_heatmap_marginal(qctn_cores_file="./assets/qctn_cores_17qubits.safetensors",
                          output_file = './assets/marginal_probability_heatmap_17qubits_01.png',
                          device='cpu')
    # test_sampling(qctn_cores_file="./assets/qctn_cores_17qubits.safetensors", 
    #                 output_file="./assets/samples_scatter_17qubits.png",
    #                 device='cpu')

    exit()

    test_heatmap_marginal(qctn_cores_file="./assets/qctn_cores_9qubits.safetensors",
                          output_file = './assets/marginal_probability_heatmap_9qubits_01.png',
                          device='cpu')
    # test_sampling(qctn_cores_file="./assets/qctn_cores_9qubits.safetensors", 
    #                 output_file="./assets/samples_scatter_9qubits.png",
    #                 device='cpu')

    exit()


    test_heatmap_marginal(qctn_cores_file="./assets/qctn_cores_2qubits.safetensors",
                          output_file = './assets/marginal_probability_heatmap_2qubits_01.png',
                          device='cpu')
    test_sampling(qctn_cores_file="./assets/qctn_cores_2qubits.safetensors", 
                    output_file="./assets/samples_scatter_2qubits.png",
                    device='cpu')


    test_heatmap_marginal(qctn_cores_file="./assets/qctn_cores_5qubits.safetensors",
                          output_file = './assets/marginal_probability_heatmap_5qubits_01.png',
                          device='cpu')
    test_sampling(qctn_cores_file="./assets/qctn_cores_5qubits.safetensors", 
                    output_file="./assets/samples_scatter_5qubits.png",
                    device='cpu')


    test_heatmap_marginal(qctn_cores_file="./assets/qctn_cores_5qubits_dist.safetensors",
                          output_file = './assets/marginal_probability_heatmap_5qubits_dist_01.png',
                          device='cpu')
    test_sampling(qctn_cores_file="./assets/qctn_cores_5qubits_dist.safetensors", 
                    output_file="./assets/samples_scatter_5qubits_dist.png",
                    device='cpu')
