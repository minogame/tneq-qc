import torch
import logging
import time
from tneq_qc.core.qctn import QCTNHelper

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

class Manager:
    """Manager: 负责生成结构"""
    def _create_mock_tensors(self):
        return [torch.eye(4) for _ in range(5)]

    def generate(self, iteration, prev_results=None):
        new_tasks = []
        if prev_results is None:
            logger.info(f"[Manager] Iteration {iteration}: Initializing population...")
            for i in range(5):
                tn_struct = QCTNHelper.generate_example_graph(
                                n=6,
                                graph_type="mps",
                                dim_char="2",
                            )
                weights = self._create_mock_tensors()
                new_tasks.append({'struct': tn_struct, 'weights': weights})
        else:
            logger.info(f"[Manager] Iteration {iteration}: Evolving...")
            for res in prev_results:
                # 简单逻辑：继承上一代
                new_tasks.append({'struct': res['struct'], 'weights': res['weights']})
        return new_tasks


if __name__ == "__main__":
    from easy_trainer import Trainer
    manager = Manager()
    trainer = Trainer()
    
    tasks = manager.generate(iteration=0)

    for task in tasks:
        loss, weight = trainer.run(task['struct'], task['weights'])

        logger.info(f"Task {task['struct']} finished with loss {loss} and new weights {weight}.")