import os
import torch
import torch.distributed as dist
import argparse
from datetime import timedelta
import logging
import time

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# --- 1. 业务逻辑类 (Mock) ---

class Trainer:
    """Worker: 负责训练"""
    def run(self, tn_struct, weights):
        # 模拟训练耗时
        time.sleep(1.0)
        
        # 简单模拟：让参数发生一点变化，证明训练了
        new_weights = [w + 0.01 for w in weights]
        loss = torch.tensor(0.1) # Mock loss
        return new_weights, loss

class Manager:
    """Master: 负责生成结构"""
    def _create_mock_tensors(self):
        return [torch.eye(5) for _ in range(10)]

    def generate(self, iteration, prev_results=None):
        new_tasks = []
        if prev_results is None:
            logger.info(f"[Master] Iteration {iteration}: Initializing population...")
            for i in range(5):
                tn_struct = f"Gen{iteration}_Struct{i}"
                weights = self._create_mock_tensors()
                new_tasks.append({'struct': tn_struct, 'weights': weights})
        else:
            logger.info(f"[Master] Iteration {iteration}: Evolving...")
            for res in prev_results:
                # 简单逻辑：继承上一代
                new_tasks.append({'struct': res['struct'], 'weights': res['weights']})
        return new_tasks

# --- 2. 分布式通信封装 ---

def _send_task(task_idx, task, worker_rank):
    """向指定 Worker 发送一个任务"""
    logger.info(f"[Master] Sending Task {task_idx} to Worker {worker_rank}.")
    # flag=1 表示有任务
    dist.send(torch.tensor([1]), dst=worker_rank)
    dist.send_object_list([task], dst=worker_rank)


def _recv_result_from_any(active_worker_ranks):
    """
    从任意一个活跃 Worker 接收结果。
    由于 torch.distributed (gloo) 不支持 recv(source=ANY_SOURCE)，
    这里用轮询 irecv 的方式模拟。
    返回 (result, source_rank)。
    """
    # 为每个活跃 Worker 发起 非阻塞接收
    # 使用 recv_object_list 是阻塞的，所以这里按顺序逐个尝试
    # 实际上 gloo 的 P2P 没有 ANY_SOURCE，所以退化为按顺序等待第一个返回的
    # 折中方案：按 rank 顺序阻塞接收（保证不会死锁，因为所有 Worker 都一定会发回结果）
    for worker_rank in active_worker_ranks:
        res_list = [None]
        dist.recv_object_list(res_list, src=worker_rank)
        return res_list[0], worker_rank
    return None, -1


def _distribute_and_collect(tasks, world_size):
    """
    动态调度：维护任务队列和空闲 Worker 池。
    Worker 完成一个任务后，立即分配下一个，直到任务队列耗尽。
    """
    num_workers = world_size - 1
    task_queue = list(enumerate(tasks))  # [(task_idx, task), ...]
    results = []
    active_workers = {}  # {worker_rank: task_idx}  正在工作的 Worker
    idle_workers = list(range(1, world_size))  # 初始全部空闲

    # 持续调度，直到任务全部完成
    while task_queue or active_workers:
        # A. 将任务分发给所有空闲 Worker
        while idle_workers and task_queue:
            worker_rank = idle_workers.pop(0)
            task_idx, task = task_queue.pop(0)
            _send_task(task_idx, task, worker_rank)
            active_workers[worker_rank] = task_idx

        # B. 如果没有活跃 Worker，跳出（不应发生）
        if not active_workers:
            break

        # C. 等待任意一个活跃 Worker 返回结果
        active_ranks = sorted(active_workers.keys())
        result, source_rank = _recv_result_from_any(active_ranks)

        if result is not None:
            finished_task_idx = active_workers.pop(source_rank)
            result['task_id'] = finished_task_idx
            results.append(result)
            idle_workers.append(source_rank)
            logger.info(f"[Master] Received result for Task {finished_task_idx} from Worker {source_rank}.")

    # 按原始任务顺序排序
    results.sort(key=lambda x: x['task_id'])
    return results


def run_master(world_size):
    """Rank 0 的逻辑"""
    manager = Manager()
    total_iterations = 10
    num_workers = world_size - 1
    
    logger.info(f"[Master] Started. Managing {num_workers} workers.")
    
    prev_results = None
    
    for iteration in range(total_iterations):
        logger.info(f"[Master] Starting Iteration {iteration}...")
        # 1. 生成任务
        tasks = manager.generate(iteration, prev_results)
        
        # 2. 分发任务并收集结果 (动态调度：Worker 完成即分配下一个任务)
        results = _distribute_and_collect(tasks, world_size)
            
        prev_results = results
        logger.info(f"[Master] Iteration {iteration} finished. Collected {len(results)} results.")

    # 3. 结束所有 Worker
    for worker_rank in range(1, world_size):
        dist.send(torch.tensor([0]), dst=worker_rank) # Flag 0 = Exit
    
    logger.info("[Master] All done.")

def run_worker(rank):
    """Rank > 0 的逻辑"""
    trainer = Trainer()
    logger.info(f"[Worker {rank}] Ready.")
    
    while True:
        # 1. 接收命令 Flag
        flag = torch.tensor([0])
        dist.recv(flag, src=0)
        
        logger.info(f"[Worker {rank}] Received Flag: {flag.item()}")

        if flag.item() == 0:
            logger.info(f"[Worker {rank}] Received Exit Signal.")
            break
        
        # 2. 接收任务对象
        task_list = [None] # 必须包裹在 list 中
        dist.recv_object_list(task_list, src=0)
        task = task_list[0]

        logger.info(f"[Worker {rank}] Received Task: {task['struct']}")
        
        # 3. 训练
        new_weights, loss = trainer.run(task['struct'], task['weights'])
        
        # 4. 发送结果
        result = {
            'struct': task['struct'],
            'weights': new_weights,
            'loss': loss
        }
        dist.send_object_list([result], dst=0)

# --- 3. 启动入口 ---

def init_distributed():
    # 从环境变量读取配置 (torchrun 会自动设置这些)
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    
    # 初始化进程组，使用 GLOO 后端 (CPU 通信首选，且支持点对点操作)
    # 如果是 GPU 且需要 NCCL，注意 NCCL 不支持 send_object_list (P2P 对象传输)
    dist.init_process_group(backend="gloo", timeout=timedelta(minutes=5))
    
    return rank, world_size, local_rank

def main():
    rank, world_size, local_rank = init_distributed()
    
    logger.info(f"[Rank {rank}] Initialized. World Size: {world_size}, Local Rank: {local_rank}")

    if rank == 0:
        run_master(world_size)
    else:
        run_worker(rank)
        
    dist.destroy_process_group()

if __name__ == "__main__":
    main()