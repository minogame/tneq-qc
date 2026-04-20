""" Using a brickwall to approximiate a MPS
Structure: combined = student * teacher_h, all qubits traced.

"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np

from tneq_qc import QCTN, EngineCommon, BackendFactory, create_optimizer
from tneq_qc.core.tn_tensor import TNTensor
from tneq_qc.utils.graph_generators import QCTNHelper

NUM_QUBITS = 4
PHYS_DIM = 2
NUM_EPOCHS  = 2000
LR         = 0.001
LOG_EVERY   = 10
ASSETS_DIR = "assets/distilled_tneq"
BOND_DIM_MPS = PHYS_DIM
NUM_LAYERS_BRICKWALL = 10

def main():
    device = os.environ.get('TNEQ_DEVICE', 'cpu')
    backend = BackendFactory.create_backend('pytorch', device=device, dtype='complex64')
    engine = EngineCommon(backend=backend, strategy_mode="full")

    # Teacher (MPS, fixed)
    # graph_teacher = QCTNHelper.mps(NUM_QUBITS, BOND_DIM_MPS, PHYS_DIM)
    graph_teacher = QCTNHelper.generate_random_circuit_graph(NUM_QUBITS, phy_dim=PHYS_DIM, ncores=6)
    print(f"MPS Teacher graph:\n{graph_teacher}")
    qtn_teacher = QCTN(graph_teacher, backend=backend).auto_init()

    # Student (brickwall, trainable)
    # graph_student = QCTNHelper.mps(NUM_QUBITS, BOND_DIM_MPS, PHYS_DIM)
    graph_student = QCTNHelper.brickwall(NUM_QUBITS, NUM_LAYERS_BRICKWALL, PHYS_DIM)
    print(f"Brickwall Student graph:\n{graph_student}")
    qtn_student = QCTN(graph_student, backend=backend).auto_init()
    qtn_student.requires_grad_(True)

    # Loss contruction: 1 - Tr(student * teacher)
    combined = QCTN.concat([('s', qtn_student), ('t', qtn_teacher)])
    combined.set_trace('all')

    print(f"The combined graph is \n{combined.graph}")

    # Training loop
    opt = create_optimizer("sgdg", combined.parameters(), backend=backend, lr=LR)
    loss_history = []

    for step in range(1, NUM_EPOCHS + 1):
        loss_val, grads = engine.contract_for_gradient(combined, target=float(PHYS_DIM ** NUM_QUBITS), loss='mse_tneq')
        opt.step(list(grads))
        lv = float(loss_val)
        loss_history.append(lv)
        if step % LOG_EVERY == 0 or step == 1:
            print(f"Step {step}/{NUM_EPOCHS}, Loss: {lv:.6f}")

    # final fidelity
    ff = engine.contract(combined).numpy() 

    print(f"Tr(student * teacher) = {ff:.6f}")
    final_fidelity = np.abs(ff) ** 2 / (float(PHYS_DIM ** NUM_QUBITS) ** 2)
    print(f"Final fidelity (Tr(student * teacher)) = {final_fidelity:.6f}")



if __name__ == "__main__":
    main()