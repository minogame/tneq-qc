from mpi4py import MPI
import torch
import time
import os
import socket
import pickle
import struct
import logging

# --- 日志配置 ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

# --- 配置与常量 ---
TAG_TASK = 1      # MPI 标识：发送的是任务
TAG_RESULT = 2    # MPI 标识：发送的是结果
TAG_KILL = 9      # MPI 标识：告诉 Worker 退出


# --- TCP 通信工具 ---

def tcp_send(sock, data):
    """通过 TCP 发送 pickle 序列化的数据（带长度前缀）"""
    payload = pickle.dumps(data)
    header = struct.pack('!I', len(payload))
    sock.sendall(header + payload)

def tcp_recv(sock):
    """从 TCP 接收 pickle 序列化的数据"""
    header = _recv_exact(sock, 4)
    if header is None:
        return None
    length = struct.unpack('!I', header)[0]
    payload = _recv_exact(sock, length)
    if payload is None:
        return None
    return pickle.loads(payload)

def _recv_exact(sock, n):
    """确保精确接收 n 字节"""
    data = b''
    while len(data) < n:
        chunk = sock.recv(n - len(data))
        if not chunk:
            return None
        data += chunk
    return data


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
        time.sleep(2.0) 

        logger.info(f"[Trainer] Training TN Struct: {tn_struct} with {len(weights)} tensors shape {[w.shape for w in weights]} ...")
        
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
        new_tasks = []
        
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


# --- 2. 节点内 MPI Worker ---

class Consumer:
    """
    节点内 MPI Worker 进程。
    通过 MPI 从本节点的 local_rank 0 (Producer 或 NodeAgent) 接收任务。
    """
    def __init__(self, comm, local_rank):
        self.comm = comm
        self.local_rank = local_rank
        self.trainer = Trainer()

    def start(self):
        logger.info(f"[Worker local_rank={self.local_rank}] Started.")
        status = MPI.Status()

        while True:
            data = self.comm.recv(source=0, tag=MPI.ANY_TAG, status=status)
            tag = status.Get_tag()

            if tag == TAG_KILL:
                logger.info(f"[Worker local_rank={self.local_rank}] Received KILL signal. Exiting.")
                break

            elif tag == TAG_TASK:
                task_id, tn_id, tn_struct, weights = data
                logger.info(f"[Worker local_rank={self.local_rank}] Received TASK {task_id} (TN ID: {tn_id}).")

                new_weights, loss = self.trainer.run(tn_struct, weights)

                result_data = {
                    'task_id': task_id,
                    'tn_id': tn_id,
                    'tn_struct': tn_struct,
                    'weights': new_weights,
                    'loss': loss,
                }
                self.comm.send(result_data, dest=0, tag=TAG_RESULT)

            else:
                logger.warning(f"[Worker local_rank={self.local_rank}] Unknown tag {tag}.")


class NodeAgent:
    """
    Worker 节点上的代理进程 (local_rank 0)。
    - TCP 连接 Master 获取任务
    - MPI 分发给本节点内的 Worker
    - 收集结果后通过 TCP 返回 Master
    """
    def __init__(self, comm, num_local_workers, master_addr, master_port, node_rank):
        self.comm = comm
        self.num_local_workers = num_local_workers
        self.master_addr = master_addr
        self.master_port = master_port
        self.node_rank = node_rank

    def start(self):
        logger.info(f"[NodeAgent node={self.node_rank}] Connecting to Master {self.master_addr}:{self.master_port} ...")

        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect((self.master_addr, self.master_port))
        logger.info(f"[NodeAgent node={self.node_rank}] Connected to Master.")

        # 握手：报告本节点可用 worker 数量
        tcp_send(sock, {'type': 'handshake', 'node_rank': self.node_rank, 'num_workers': self.num_local_workers})

        idle_workers = list(range(1, self.num_local_workers + 1))
        active_workers = 0
        status = MPI.Status()

        try:
            while True:
                msg = tcp_recv(sock)
                if msg is None:
                    break

                if msg['type'] == 'kill':
                    self._kill_local_workers()
                    break

                elif msg['type'] == 'task':
                    task_data = msg['data']

                    # 如果没有空闲 worker，先等一个结果回来
                    if not idle_workers:
                        result = self.comm.recv(source=MPI.ANY_SOURCE, tag=TAG_RESULT, status=status)
                        source_rank = status.Get_source()
                        idle_workers.append(source_rank)
                        active_workers -= 1
                        tcp_send(sock, {'type': 'result', 'data': result})

                    worker_rank = idle_workers.pop(0)
                    self.comm.send(task_data, dest=worker_rank, tag=TAG_TASK)
                    active_workers += 1

                elif msg['type'] == 'flush':
                    # Master 要求收回所有还在工作的结果
                    while active_workers > 0:
                        result = self.comm.recv(source=MPI.ANY_SOURCE, tag=TAG_RESULT, status=status)
                        source_rank = status.Get_source()
                        idle_workers.append(source_rank)
                        active_workers -= 1
                        tcp_send(sock, {'type': 'result', 'data': result})
                    tcp_send(sock, {'type': 'flush_done'})

        finally:
            sock.close()
            logger.info(f"[NodeAgent node={self.node_rank}] Disconnected from Master.")

    def _kill_local_workers(self):
        for rank in range(1, self.num_local_workers + 1):
            self.comm.send(None, dest=rank, tag=TAG_KILL)
        logger.info(f"[NodeAgent node={self.node_rank}] Sent KILL to all local workers.")


class Producer:
    """
    Master 进程逻辑 (NODE_RANK=0, local_rank=0)。
    - MPI 管理本节点内的 Worker
    - TCP server 管理远程 Worker 节点
    """
    def __init__(self, comm, num_local_workers, master_port, num_nodes):
        self.comm = comm
        self.num_local_workers = num_local_workers
        self.master_port = master_port
        self.num_nodes = num_nodes
        self.manager = Manager()
        self.remote_conns = {}  # node_rank -> {'sock': sock, 'num_workers': n}

    def start(self, total_iterations=3):
        if self.num_nodes > 1:
            self._accept_remote_nodes()

        total_workers = self.num_local_workers + sum(
            c['num_workers'] for c in self.remote_conns.values()
        )
        logger.info(f"[Master] Started with {total_workers} workers "
                     f"({self.num_local_workers} local + "
                     f"{total_workers - self.num_local_workers} remote).")

        prev_results = None

        for iteration in range(total_iterations):
            tasks = self.manager.generate(iteration, prev_results)
            iteration_results = self._distribute_and_collect(tasks)

            prev_results = []
            for res in iteration_results:
                prev_results.append((res['tn_struct'], res['weights'], res['loss']))

            avg_loss = sum(r['loss'] for r in iteration_results) / len(iteration_results)
            logger.info(f"[Master] Iteration {iteration} finished. Avg Loss: {avg_loss}")

        self._kill_all()

    def _accept_remote_nodes(self):
        """TCP server 等待所有远程 Worker 节点连接"""
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind(('0.0.0.0', self.master_port))
        server.listen(self.num_nodes)
        logger.info(f"[Master] TCP server listening on port {self.master_port}, "
                     f"waiting for {self.num_nodes - 1} remote node(s)...")

        expected = self.num_nodes - 1
        while len(self.remote_conns) < expected:
            conn, addr = server.accept()
            handshake = tcp_recv(conn)
            node_rank = handshake['node_rank']
            num_workers = handshake['num_workers']
            self.remote_conns[node_rank] = {'sock': conn, 'num_workers': num_workers}
            logger.info(f"[Master] Remote node {node_rank} connected from {addr}, "
                         f"{num_workers} workers available.")

        server.close()
        logger.info(f"[Master] All {expected} remote node(s) connected.")

    def _distribute_and_collect(self, tasks):
        """
        调度核心：同时向本地 MPI Worker 和远程 TCP Worker 分发任务。
        """
        task_queue = list(enumerate(tasks))
        results = []

        # 本地 MPI worker 管理
        local_idle = list(range(1, self.num_local_workers + 1))
        local_active = 0

        # 远程 worker 管理
        remote_idle = {}
        remote_active = {}
        for nk, info in self.remote_conns.items():
            remote_idle[nk] = info['num_workers']
            remote_active[nk] = 0

        mpi_status = MPI.Status()

        while task_queue or local_active > 0 or any(v > 0 for v in remote_active.values()):

            # --- A. 分发任务 ---
            dispatched = True
            while dispatched and task_queue:
                dispatched = False

                # 优先本地
                if local_idle and task_queue:
                    worker_rank = local_idle.pop(0)
                    task_id, (tn_id, tn_struct, weights) = task_queue.pop(0)
                    logger.info(f"[Master] Sending Task {task_id} to local worker {worker_rank}...")
                    self.comm.send((task_id, tn_id, tn_struct, weights), dest=worker_rank, tag=TAG_TASK)
                    local_active += 1
                    dispatched = True

                # 然后远程
                for nk in list(remote_idle.keys()):
                    if remote_idle[nk] > 0 and task_queue:
                        task_id, (tn_id, tn_struct, weights) = task_queue.pop(0)
                        logger.info(f"[Master] Sending Task {task_id} to remote node {nk}...")
                        tcp_send(self.remote_conns[nk]['sock'],
                                 {'type': 'task', 'data': (task_id, tn_id, tn_struct, weights)})
                        remote_idle[nk] -= 1
                        remote_active[nk] += 1
                        dispatched = True

            # --- B. 收集结果 (非阻塞轮询) ---
            collected = False

            # 本地 MPI 结果 (非阻塞)
            if local_active > 0:
                probe = self.comm.iprobe(source=MPI.ANY_SOURCE, tag=TAG_RESULT, status=mpi_status)
                if probe:
                    result = self.comm.recv(source=mpi_status.Get_source(), tag=TAG_RESULT)
                    source_rank = mpi_status.Get_source()
                    results.append(result)
                    local_active -= 1
                    local_idle.append(source_rank)
                    collected = True

            # 远程 TCP 结果 (非阻塞)
            for nk, info in self.remote_conns.items():
                if remote_active[nk] > 0:
                    info['sock'].setblocking(False)
                    try:
                        msg = tcp_recv(info['sock'])
                        if msg and msg['type'] == 'result':
                            results.append(msg['data'])
                            remote_active[nk] -= 1
                            remote_idle[nk] += 1
                            collected = True
                    except (BlockingIOError, OSError):
                        pass
                    finally:
                        info['sock'].setblocking(True)

            # 避免忙等
            if not collected and not task_queue:
                time.sleep(0.01)

        results.sort(key=lambda x: x['task_id'])
        return results

    def _kill_all(self):
        # 终止本地 worker
        for rank in range(1, self.num_local_workers + 1):
            self.comm.send(None, dest=rank, tag=TAG_KILL)
        logger.info("[Master] Sent KILL to all local workers.")

        # 终止远程节点
        for nk, info in self.remote_conns.items():
            tcp_send(info['sock'], {'type': 'kill'})
            info['sock'].close()
        logger.info("[Master] Sent KILL to all remote nodes.")


def main():
    comm = MPI.COMM_WORLD
    local_rank = comm.Get_rank()
    local_size = comm.Get_size()

    # 从环境变量读取分布式配置
    master_addr = os.environ.get('MASTER_ADDR', '127.0.0.1')
    master_port = int(os.environ.get('MASTER_PORT', '29500'))
    node_rank = int(os.environ.get('NODE_RANK', '0'))
    num_nodes = int(os.environ.get('NUM_NODES', '1'))

    num_local_workers = local_size - 1  # local_rank 0 是管理进程

    logger.info(f"[Node {node_rank}, local_rank {local_rank}] Starting up... "
                f"(local_size={local_size}, num_nodes={num_nodes})")

    if local_size < 2:
        logger.error("Each node needs at least 2 MPI processes (1 manager + 1 worker).")
        logger.error("Run with: NPROC_PER_NODE=3 bash dist.sh")
        return

    if local_rank == 0:
        if node_rank == 0:
            # Master 节点的 local_rank 0 → Producer
            producer = Producer(comm, num_local_workers, master_port, num_nodes)
            producer.start(total_iterations=10)
        else:
            # Worker 节点的 local_rank 0 → NodeAgent
            agent = NodeAgent(comm, num_local_workers, master_addr, master_port, node_rank)
            agent.start()
    else:
        # 所有节点的 local_rank >= 1 → Consumer (本地 MPI Worker)
        consumer = Consumer(comm, local_rank)
        consumer.start()


if __name__ == "__main__":
    main()