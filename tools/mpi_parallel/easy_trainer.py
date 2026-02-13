import os
import time
import argparse
from typing import Union

import numpy as np

from tneq_qc.backends.backend_factory import BackendFactory
from tneq_qc.core.engine_siamese import EngineSiamese
from tneq_qc.core.qctn import QCTN, QCTNHelper
from tneq_qc.optim.optimizer import Optimizer
from tneq_qc.core.tn_tensor import TNTensor

import logging
import time

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

class Trainer:
    """Worker: 负责训练"""
    def __init__(self):
        self._backend = BackendFactory.create_backend(
            "pytorch",
            device="cpu",
            dtype="float32",
        )

        graph = "-2-a-2-\n" \
                "-2-a-2-\n" \
                "-2-a-2-\n" \
                "-2-a-2-\n" \
                "-2-a-2-\n" \
                "-2-a-2-"

        # graph = "-2-a-2-\n" \
        #         "-2-a-2-"
        self._target_qctn = QCTN(graph, backend=self._backend)

    def run(self, tn_struct, weights):
        
        engine = EngineSiamese(backend=self._backend, strategy_mode="balanced", mx_K=2)

        qctn = QCTN(tn_struct, backend=self._backend)
        logger.info(f"QCTN: nqubits = {qctn.nqubits}, ncores = {qctn.ncores}")

        qctn.set_cores(weights)

        for c_name in qctn.cores:
            core_tensor = qctn.cores_weights[c_name]
            if isinstance(core_tensor, TNTensor):
                core_tensor.tensor.requires_grad_(True)
            else:
                core_tensor.requires_grad_(True)
        
        optimizer = Optimizer(
            method="sgdg",
            max_iter=100,
            tol=0.0,
            learning_rate=1e-3,
            beta1=0.9,
            beta2=0.95,
            epsilon=1e-8,
            engine=engine,
            momentum=0.9,
            stiefel=True,
        )

        data_list_for_optim = [
            {
                "measure_input_list": [self._backend.eye(2) for i in range(qctn.nqubits)],
            }
        ]

        tic = time.time()
        final_loss = optimizer.optimize(
            self._target_qctn,
            data_list=data_list_for_optim,
            circuit_states_list=None,
            right_qctn=qctn,
        )
        toc = time.time()

        new_weights = qctn.cores_weights

        logger.info(f"Training finished. Time elapsed: {toc - tic:.2f} seconds")

        return final_loss, new_weights



if __name__ == "__main__":
    import torch
    trainer = Trainer()

    n = 6

    graph = QCTNHelper.generate_example_graph(
        n=n,
        # n=2,
        graph_type="mps",
        dim_char="2",
    )

    weights = [torch.eye(4) for _ in range(n-1)]

    trainer.run(graph, weights=weights)