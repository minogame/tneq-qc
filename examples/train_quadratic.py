"""
Quadratic form training example.

Structure: circuit + mps + mx + mps† + circuit†

- circuit : circuit_state, fixed, initialized with 0/1 pattern
- mps     : MPS, trainable (bond_dim=2), optimized with SGDG
- mx      : measure_matrix, replaced each step with DataGenerator output
- mps†    : Hermitian conjugate of mps (shares cores with mps)
- circuit†: bra-structured conjugate of circuit (fixed)

Loss: binary cross-entropy on the scalar expectation value ⟨ψ|U†MU|ψ⟩.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from tqdm import tqdm

from tneq_qc.backends.backend_factory import BackendFactory
from tneq_qc.core.qctn import QCTN
from tneq_qc.core.tn_tensor import TNTensor
from tneq_qc.core.engine_common import EngineCommon
from tneq_qc.utils.graph_generators import QCTNHelper
from tneq_qc.utils.data_generator import DataGenerator

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------
N_QUBITS  = 4
BOND_DIM  = 2
PHYS_DIM  = 2       # K for DataGenerator (== physical dim of mx core)
N_STEPS   = 1000
LR        = 0.01
LOG_EVERY = 1

torch.manual_seed(42)
np.random.seed(42)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _raw(t):
    """Return the underlying backend tensor regardless of TNTensor wrapping."""
    return t.tensor if isinstance(t, TNTensor) else t


def _is_trainable_leaf(t):
    """True iff this core is a leaf tensor that requires gradient."""
    raw = _raw(t)
    return raw.requires_grad and (not hasattr(raw, 'is_leaf') or raw.is_leaf)


def init_circuit_01(qctn: QCTN, backend) -> QCTN:
    """Fill each circuit core with an alternating 0/1 pattern."""
    for c in qctn.cores:
        core = qctn.cores_weights[c]
        shape = tuple(core.tensor.shape) if isinstance(core, TNTensor) else tuple(core.shape)
        dtype = core.tensor.dtype if isinstance(core, TNTensor) else core.dtype
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
        raw = _raw(t)
        shape = tuple(raw.shape)
        rg    = raw.requires_grad
        leaf  = getattr(raw, 'is_leaf', '?')
        print(f"    '{c}': {shape}  requires_grad={rg}  is_leaf={leaf}")


# ---------------------------------------------------------------------------
# Loss: binary cross-entropy on the scalar contraction result
# ---------------------------------------------------------------------------

def nll_loss(result, target, backend):
    """NLL loss: -mean(log(P) + log_scale).

    Mirrors engine_siamese.py's commented-out NLL path:
        log_total = log_result + log_scale
        loss = -mean(target * log_total)

    Key design choices:
    - Use result.tensor (raw, without scale) for Born rule → avoids
      scale² amplification that drove p→0 and caused 23.025850 spikes.
    - Add log_scale (detached) as a constant correction term so the
      full log-probability log(P·scale) is properly accounted for.
    - target (label) is ignored; task is generative (maximize P for all
      data), so target is always 1 — identical to engine_siamese.py.
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
    # backend  = BackendFactory.create_backend('pytorch', device='cpu', dtype='complex64')
    backend  = BackendFactory.create_backend('pytorch', device='cpu', dtype='float32')
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
        _raw(mps.cores_weights[c]).requires_grad_(True)
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
            t.tensor.conj() if isinstance(t, TNTensor) else t.conj(),
            scale=t.scale if isinstance(t, TNTensor) else 1.0,
        )
    for core in circ_bra.cores:
        print(f"    {core}: {circ_bra.cores_weights[core].tensor}")

    combined = QCTN.concat([circuit, mps, mx, mps_h, circ_bra])
    print_qctn_info(combined, "Combined (circuit + mps + mx + mps† + circuit†)")

    # ------------------------------------------------------------------
    # 5. Collect trainable params from combined (leaf cores only)
    #    Must match the order the engine collects them.
    # ------------------------------------------------------------------
    params = [
        combined.cores_weights[c]
        for c in combined.cores
        if _is_trainable_leaf(combined.cores_weights[c])
    ]
    print(f"\nTrainable cores: {len(params)}")

    # Pre-collect mx core names (in original mx) for inplace update.
    # combined.cores_weights[new_name] shares the same TNTensor object as
    # mx.cores_weights[old_name], so .set() on mx propagates to combined.
    mx_core_names = list(mx.cores)

    # ------------------------------------------------------------------
    # 6. Verify one forward pass before training
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
    # 7. Training loop
    # ------------------------------------------------------------------
    print()
    print("=" * 60)
    print(f"Training  steps={N_STEPS}  lr={LR}  optimizer=SGDG  loss=NLL")
    print("=" * 60)

    optimizer_state: dict = {}
    loss_history = []

    for step in tqdm(range(1, N_STEPS + 1)):

        # ---- Generate training sample --------------------------------
        # x shape: [1, N_QUBITS]; simple binary label: mean(x) > 0
        x = np.random.uniform(-1.0, 1.0, size=(1, N_QUBITS)).astype(np.float32)
        y = 1 # float(x.mean() > 0)

        # ---- Generate Mx from x via DataGenerator -------------------
        # Mx_list: N_QUBITS tensors, each TNTensor of shape [1, K, K]
        Mx_list, _ = data_gen.generate(x, K=PHYS_DIM)

        # print(f"  Mx_list: {Mx_list}")

        # ---- Update mx cores inplace --------------------------------
        # combined shares the same TNTensor objects as mx, so .set() here
        # is immediately visible to the engine's next contraction.

        for i, c in enumerate(mx_core_names):
            mx_item = Mx_list[i]
            if isinstance(mx_item, TNTensor):
                data  = mx_item.tensor[0]   # [K, K]
                scale = mx_item.scale
            else:
                data  = mx_item[0]          # [K, K]
                scale = 1.0
            core = mx.cores_weights[c]
            if isinstance(core, TNTensor):
                core.tensor.data.copy_(data)
                core.scale = scale
            else:
                core.data.copy_(data)

        # ---- Forward + backward ------------------------------------
        loss_tensor, grads = engine.contract_with_compiled_strategy_for_gradient(
            combined,
            target=y,
            loss=nll_loss,
        )

        # ---- Parameter update (SGDG) --------------------------------
        params, optimizer_state = backend.optimizer_update(
            params, list(grads), optimizer_state,
            method='sgdg',
            hyperparams={'learning_rate': LR},
        )

        loss_val = loss_tensor.item() if hasattr(loss_tensor, 'item') else float(loss_tensor)
        loss_history.append(loss_val)

        if step % LOG_EVERY == 0 or step == 1:
            print(f"  Step {step:4d}/{N_STEPS}  loss={loss_val:.6f}  y={y:.1f}")

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


if __name__ == "__main__":
    main()
