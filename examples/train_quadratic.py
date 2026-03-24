"""
Quadratic form training example.

Structure: circuit + mps + mx + mps† + circuit†

- circuit : circuit_state, fixed, initialized with 0/1 pattern
- mps     : MPS, trainable (bond_dim=2), optimized with SGDG
- mx      : measure_matrix, replaced each step with DataGenerator output
- mps†    : Hermitian conjugate of mps (shares cores with mps)
- circuit†: bra-structured conjugate of circuit (fixed)

Loss: cross-entropy -mean(log(P+ε)) on batch expectation value ⟨ψ|U†MU|ψ⟩.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np

from tneq_qc.backends.backend_factory import BackendFactory
from tneq_qc.core.qctn import QCTN
from tneq_qc.core.tn_tensor import TNTensor
from tneq_qc.core.engine_common import EngineCommon
from tneq_qc.utils.graph_generators import QCTNHelper
from tneq_qc.utils.data_generator import DataGenerator, make_data_fn
from tneq_qc.optim import SGDG
from tneq_qc.trainer import Trainer, TrainConfig

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------
N_QUBITS   = 4
BOND_DIM   = 2
PHYS_DIM   = 2       # K for DataGenerator (== physical dim of mx core)
BATCH_SIZE = 128     # number of samples per Mx update step
N_STEPS    = 1000
LR         = 0.01
LOG_EVERY  = 1

torch.manual_seed(42)
np.random.seed(42)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def init_circuit_01(qctn: QCTN, backend) -> QCTN:
    """Fill each circuit core with an alternating 0/1 pattern."""
    for c in qctn.cores:
        core = qctn.cores_weights[c]
        shape = tuple(core.shape)
        dtype = core.dtype
        n = 1
        for d in shape:
            n *= d
        flat = torch.zeros(n, dtype=dtype)
        for i in range(n):
            flat[i] = float(i % 2)
        qctn.cores_weights[c] = backend.convert_to_tensor(flat.reshape(shape))
    return qctn


def print_qctn_info(qctn: QCTN, label: str = "QCTN"):
    print(f"\n[{label}]  nqubits={qctn.nqubits}  ncores={qctn.ncores}")
    print("  Core shapes (requires_grad / is_leaf):")
    for c in qctn.cores:
        t = qctn.cores_weights[c]
        shape = tuple(t.shape)
        rg    = t.requires_grad
        leaf  = t.is_leaf
        print(f"    '{c}': {shape}  requires_grad={rg}  is_leaf={leaf}")


# ---------------------------------------------------------------------------
# Loss: binary cross-entropy on the scalar contraction result
# ---------------------------------------------------------------------------

def nll_loss(result, target, backend):
    """NLL loss: -mean(log(P) + log_scale).

    Key design choices:
    - Use result.tensor (raw, without scale) for Born rule → avoids
      scale² amplification that drove p→0 and caused 23.025850 spikes.
    - Add log_scale (detached) as a constant correction term so the
      full log-probability log(P·scale) is properly accounted for.
    - target (label) is ignored; task is generative (maximize P for all
      data), so target is always 1.
    """
    # ① raw tensor + log_scale (detached constant)
    if isinstance(result, TNTensor):
        raw_r     = result.tensor
        log_scale = result.log_scale          # float, detached
    else:
        raw_r     = result
        log_scale = 0.0

    # ② Born rule: complex → real non-negative P = |W|²
    if isinstance(raw_r, torch.Tensor) and raw_r.is_complex():
        p = (raw_r * raw_r.conj()).real
    else:
        p = raw_r

    # ③ clamp to avoid log(0), then compute log(P)
    p = p.clamp(min=1e-10)
    log_result = torch.log(p)

    # ④ target = 1 for all samples (generative objective)
    tgt = torch.ones_like(p)

    # ⑤ NLL: -mean(log(P) + log_scale)
    log_total = log_result + log_scale
    return -backend.mean(tgt * log_total)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    backend  = BackendFactory.create_backend('pytorch', device='cpu', dtype='complex64')
    # backend  = BackendFactory.create_backend('pytorch', device='cpu', dtype='float32')
    engine   = EngineCommon(backend=backend, strategy_mode="full")
    data_gen = DataGenerator(backend, mx_K=PHYS_DIM)

    # ------------------------------------------------------------------
    # 1. Circuit state (fixed, 0/1 init)
    # ------------------------------------------------------------------
    graph_circuit = QCTNHelper.circuit_state(N_QUBITS, phys_dim=PHYS_DIM)
    circuit = QCTN(graph_circuit, backend=backend).auto_init()
    circuit = init_circuit_01(circuit, backend)
    print_qctn_info(circuit, "Circuit (fixed, 0/1 init)")
    for core in circuit.cores:
        print(f"    {core}: {circuit.cores_weights[core].tensor}")

    # ------------------------------------------------------------------
    # 2. MPS (trainable, QR init)
    # ------------------------------------------------------------------
    graph_mps = QCTNHelper.mps(N_QUBITS, bond_dim=BOND_DIM, phys_dim=PHYS_DIM)
    mps = QCTN(graph_mps, backend=backend).auto_init()
    for c in mps.cores:
        mps.cores_weights[c].requires_grad_(True)
    print_qctn_info(mps, "MPS (trainable)")

    # ------------------------------------------------------------------
    # 3. Measurement matrix Mx (data-driven, replaced each step)
    # ------------------------------------------------------------------
    graph_mx = QCTNHelper.measure_matrix(N_QUBITS, phys_dim=PHYS_DIM)
    mx = QCTN(graph_mx, backend=backend).auto_init()
    print_qctn_info(mx, "Mx (initial random)")

    # ------------------------------------------------------------------
    # 4. Build combined: circuit + mps + mx + mps† + circuit†
    # ------------------------------------------------------------------
    mps_h = mps.hermit()

    graph_circ_bra = QCTNHelper.circuit_bra(N_QUBITS, phys_dim=PHYS_DIM)
    circ_bra = QCTN(graph_circ_bra, backend=backend).auto_init()
    for c in circuit.cores:
        t = circuit.cores_weights[c]
        circ_bra.cores_weights[c] = TNTensor(
            t.tensor.conj(),
            scale=t.scale,
        )
    for core in circ_bra.cores:
        print(f"    {core}: {circ_bra.cores_weights[core].tensor}")

    combined = QCTN.concat([
        ('cs', circuit),
        ('mps', mps),
        ('mx', mx),
        ('mps_h', mps_h),
        ('cs_t', circ_bra),
    ])
    print_qctn_info(combined, "Combined (circuit + mps + mx + mps† + circuit†)")

    print(f"combined.core_weights: {combined.cores_weights}")
    print(f"combined.core_names: {combined.core_names}")
    print(f"combined: {repr(combined)}")

    print(f"\nTrainable cores: {len(combined.parameters())}")

    # Pre-collect mx readable names for data_fn
    mx_readable_names = [
        combined.core_names[sym]
        for sym in combined.cores
        if combined.core_names[sym].startswith('mx.')
    ]

    # ------------------------------------------------------------------
    # 5. Verify one forward pass before training
    # ------------------------------------------------------------------
    print()
    print("=" * 60)
    print("Verifying one forward pass ...")
    print("=" * 60)
    result = engine.contract_with_compiled_strategy(combined)
    if isinstance(result, TNTensor):
        eff = result.tensor * result.scale
        print(f"  result shape={tuple(result.shape)}  eff={eff}")
    else:
        print(f"  result={result}")

    # ------------------------------------------------------------------
    # 6. Training
    # ------------------------------------------------------------------
    print()
    print("=" * 60)
    print(f"Training  steps={N_STEPS}  batch={BATCH_SIZE}  lr={LR}  optimizer=SGDG  loss=CE")
    print("=" * 60)

    optimizer = SGDG(combined.parameters(), backend, lr=LR)
    data_fn = make_data_fn(data_gen, combined, mx_readable_names,
                           batch_size=BATCH_SIZE, num_qubits=N_QUBITS, K=PHYS_DIM)

    save_path = f"checkpoints/quadratic_{N_QUBITS}q_K{PHYS_DIM}_B{BOND_DIM}.safetensors"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    trainer = Trainer(engine, combined, optimizer,
        TrainConfig(max_steps=N_STEPS, log_every=LOG_EVERY, save_path=save_path))
    loss_history = trainer.fit(target=1, loss=nll_loss, data_fn=data_fn)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print()
    print("=" * 60)
    print("Training complete")
    print(f"  Initial loss : {loss_history[0]:.6f}")
    print(f"  Final   loss : {loss_history[-1]:.6f}")
    print(f"  Loss reduced : {loss_history[0] - loss_history[-1]:.6f}")
    print("=" * 60)
    print(f"  Saved to: {save_path}")
    print(f"  Core names: {combined.core_names}")


if __name__ == "__main__":
    main()
