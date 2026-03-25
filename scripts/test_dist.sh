
echo "pjsub_node_list: " $(pjsub_node_list)
echo "PJM_NODE_RANK: " $PJM_NODE_RANK

MASTER_ADDR=$(hostname)
if [ "$PJM_NODE_RANK" -ne 0 ]; then
    # 非 0 号节点需要从 PJM 环境中获取 0 号节点的名称
    # 具体的获取方式取决于集群配置，通常可以使用以下技巧：
    MASTER_ADDR=$(pjsub_node_list | head -n 1)
fi
echo "Master Address: " $MASTER_ADDR
export MASTER_ADDR

echo "Master Address after: " $MASTER_ADDR


export MASTER_PORT=29500

# 3. 设置 Node Rank
export NODE_RANK=$PJM_NODE_RANK

# 4. 每个节点的进程数 (对应 4 个 CMG)
export NPROC_PER_NODE=4

# 5. 计算线程数 (每个 CMG 12 核心)
export OMP_NUM_THREADS=12


export GLOO_SOCKET_IFNAME=eno1
export NCCL_SOCKET_IFNAME=eno1
export TP_SOCKET_IFNAME=eno1


export RANK=$PJM_NODE_RANK

torchrun \
    --nnodes=$PJM_NODE_ITR \
    --nproc_per_node=2 \
    --node_rank=$RANK \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    examples/train_dist.py

echo ""
echo "Done!"
