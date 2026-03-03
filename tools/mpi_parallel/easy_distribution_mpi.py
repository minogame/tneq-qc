#!/usr/bin/env python
# -*- coding: utf-8 -*-

from mpi4py import MPI
import torch
import time
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# --- 配置与常量 ---
TAG_TASK = 1      # 标识：发送的是任务
TAG_RESULT = 2    # 标识：发送的是结果
TAG_KILL = 9      # 标识：告诉 Worker 退出

# --- 1. 核心业务逻辑类 (Mock实现) ---

class Trainer:
    """
    消费者持有的训练器。
    负责实际的 Tensor Network 训练。
    """
    def __init__(self):
        pass

    def run(self, tn_struct, weights):
        """
        输入:
            tn_struct: TN 结构字符串
            weights: list of torch.Tensor (core tensors)
        输出:
            trained_weights: 训练后的 weights
            loss: float
        """
        # Mock 训练过程：模拟计算耗时
        time.sleep(1.0) 
        
        # 留空实现：直接返回输入的 weights，loss 设为 0.0
        trained_weights = weights
        loss = 0.0
        
        return trained_weights, loss

class Manager:
    """
    生产者持有的管理器。
    负责 TN 结构的生成和筛选。
    """
    def __init__(self):
        self.tn_id = 0

    def _create_mock_tensors(self):
        # 创建 10 个 5x5 的单位矩阵
        return [torch.eye(5) for _ in range(10)]

    def generate(self, iteration, prev_results=None):
        """
        输入:
            iteration: 当前迭代次数 (int)
            prev_results: list, 每个元素包含 (tn_struct, weights, loss)。
                          如果是第一次迭代，为 None。
        输出:
            new_tasks: list, 每个元素包含 (tn_struct, weights) 用于分发给 Worker。
        """
        new_tasks = []
        
        # 逻辑：
        # 如果是 None (初始状态)，生成一批初始结构。
        # 如果不是 None，根据 prev_results (也就是上一代的 loss) 进行进化/筛选。
        
        if prev_results is None:
            logger.info(f"[Manager] Iteration {iteration}: Initializing population...")
            # 假设初始种群大小为 5
            for i in range(5):
                tn_struct = "" # 初始为空字符串
                weights = self._create_mock_tensors()
                new_tasks.append((self.tn_id, tn_struct, weights))
                self.tn_id += 1
        else:
            logger.info(f"[Manager] Iteration {iteration}: Evolving from {len(prev_results)} results...")
            # Mock 逻辑：直接沿用上一代的参数，不做修改，继续训练
            # 在这里你可以实现遗传算法、结构剪枝等逻辑
            for item in prev_results:
                struct, w, loss = item
                # 假设这里基于 loss 修改了 struct (此处留空)
                new_struct = struct 
                new_tasks.append((self.tn_id, new_struct, w))
                self.tn_id += 1

        return new_tasks

# --- 2. MPI 进程封装类 ---

class Consumer:
    """
    Worker 进程逻辑
    """
    def __init__(self, comm, rank):
        self.comm = comm
        self.rank = rank
        self.trainer = Trainer()

    def start(self):
        logger.info(f"[Worker {self.rank}] Started.")
        status = MPI.Status()
        
        while True:
            # 1. 告知 Master 我准备好了 (向 Master 发送一个空信号，或者请求)
            # 为了简单，我们让 Master 被动监听，或者 Master 主动发给空闲 Worker。
            # 这里采用 Master 维护状态，Worker 等待接收的模式。
            
            # 阻塞接收消息
            data = self.comm.recv(source=0, tag=MPI.ANY_TAG, status=status)
            tag = status.Get_tag()

            if tag == TAG_KILL:
                logger.info(f"[Worker {self.rank}] Received KILL signal. Exiting.")
                break
            
            elif tag == TAG_TASK:
                # data 结构: (task_id, tn_struct, weights)
                task_id, tn_id, tn_struct, weights = data

                logger.info(f"[Worker {self.rank}] Received Task {task_id} (TN ID: {tn_id}).")
                
                # print(f"[Worker {self.rank}] Processing Task {task_id}...")
                
                # 执行训练
                new_weights, loss = self.trainer.run(tn_struct, weights)
                
                # 发送结果回 Master (Result Queue)
                result_data = {
                    'task_id': task_id,
                    'tn_id': tn_id,
                    'tn_struct': tn_struct, # 结构可能没变，也可能需要传回
                    'weights': new_weights,
                    'loss': loss
                }
                self.comm.send(result_data, dest=0, tag=TAG_RESULT)
            
            else:
                logger.warning(f"[Worker {self.rank}] Unknown tag {tag}.")

class Producer:
    """
    Master 进程逻辑
    """
    def __init__(self, comm, num_workers):
        self.comm = comm
        self.num_workers = num_workers # 除去 Master 之外的 Worker 数量
        self.manager = Manager()

    def start(self, total_iterations=3):
        logger.info(f"[Master] Started with {self.num_workers} workers.")
        
        prev_results = None
        
        for iteration in range(total_iterations):
            # 1. 调用 Manager 生成本轮任务
            # tasks 结构: List[(tn_struct, weights)]
            tasks = self.manager.generate(iteration, prev_results)
            
            # 2. 分发任务并收集结果 (模拟跨节点队列)
            # 这里实现这一轮的调度循环
            iteration_results = self.distribute_and_collect(tasks)
            
            # 3. 准备下一轮数据
            # 这里的 iteration_results 需要整理成 list 传给 generate
            # 结果结构重组为 Manager 需要的格式: (tn_struct, weights, loss)
            prev_results = []
            for res in iteration_results:
                prev_results.append((res['tn_struct'], res['weights'], res['loss']))
            
            logger.info(f"[Master] Iteration {iteration} finished. Avg Loss: {sum(r['loss'] for r in iteration_results)/len(iteration_results)}")

        # 任务全部结束，发送终止信号
        self.kill_workers()

    def distribute_and_collect(self, tasks):
        """
        这是你要求的“跨节点队列”的核心实现。
        Master 充当调度器，维护任务池和空闲 Worker 池。
        """
        task_queue = list(enumerate(tasks)) # 加上索引: [(0, task0), (1, task1)...]
        results = []
        
        # 追踪正在工作的 Worker 数量
        active_workers = 0
        
        # 初始时，所有 Worker (Rank 1 到 num_workers) 都是空闲的
        idle_workers = list(range(1, self.num_workers + 1))
        
        status = MPI.Status()

        # 当 任务队列不为空 或者 还有 Worker 在干活时，循环继续
        while task_queue or active_workers > 0:
            
            # A. 填充任务 (Push to Task Queue / Dispatch)
            # 只要有空闲 Worker 且 有任务，就发送
            while idle_workers and task_queue:
                worker_rank = idle_workers.pop(0)
                task_id, (tn_id, tn_struct, weights) = task_queue.pop(0)
                
                # 发送任务
                self.comm.send((task_id, tn_id, tn_struct, weights), dest=worker_rank, tag=TAG_TASK)
                active_workers += 1
            
            # B. 收集结果 (Pop from Result Queue)
            # Master 阻塞等待任意 Worker 的返回
            result_data = self.comm.recv(source=MPI.ANY_SOURCE, tag=TAG_RESULT, status=status)
            source_rank = status.Get_source()
            
            # 保存结果
            results.append(result_data)
            
            # 该 Worker 变为空闲
            active_workers -= 1
            idle_workers.append(source_rank)
        
        # 为了保证 Manager 拿到的列表顺序和 generate 输出的一致（如果需要的话），可以按 task_id 排序
        results.sort(key=lambda x: x['task_id'])
        return results

    def kill_workers(self):
        for rank in range(1, self.num_workers + 1):
            self.comm.send(None, dest=rank, tag=TAG_KILL)
        logger.info("[Master] Sent KILL signal to all workers.")

# --- 主程序入口 ---

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if size < 2:
        logger.error("This program requires at least 2 processes (1 Master + 1 Worker).")
        logger.error("Run with: mpiexec -n <num_procs> python script.py")
        return

    if rank == 0:
        # Master 进程
        producer = Producer(comm, num_workers=size - 1)
        producer.start(total_iterations=3) # 运行 3 代
    else:
        # Worker 进程
        consumer = Consumer(comm, rank)
        consumer.start()

if __name__ == "__main__":
    main()