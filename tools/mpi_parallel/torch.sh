#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

net_name=tofu1

# MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_ADDR=`ip -o -4 addr list ${net_name} | awk '{print $4}' | cut -d/ -f1`
MASTER_PORT="${MASTER_PORT:-29500}"
NODE_RANK="${NODE_RANK:-0}"
NUM_NODES="${NUM_NODES:-1}"

export GLOO_SOCKET_IFNAME=${net_name}
export NCCL_SOCKET_IFNAME=${net_name}
export TP_SOCKET_IFNAME=${net_name}

echo "=============================================="
echo " Torch 分布式训练启动"
echo "  MASTER_ADDR:    ${MASTER_ADDR}"
echo "  MASTER_PORT:    ${MASTER_PORT}"
echo "  NODE_RANK:      ${NODE_RANK}"
echo "=============================================="

torchrun \
    --nnodes=1 \
    --nproc_per_node=4 \
    --node_rank="${NODE_RANK}" \
    --master_addr="${MASTER_ADDR}" \
    --master_port="${MASTER_PORT}" \
    "${SCRIPT_DIR}/easy_distribution_torch.py"


# torchrun \
#     --nnodes=${NUM_NODES} \
#     --nproc_per_node=2 \
#     --node_rank="${NODE_RANK}" \
#     --master_addr="${MASTER_ADDR}" \
#     --master_port="${MASTER_PORT}" \
#     "${SCRIPT_DIR}/easy_distribution_torch.py"


# MASTER_ADDR=192.168.100.118 NODE_RANK=0 NUM_NODES=1 bash tools/mpi_parallel/torch.sh

# MASTER_ADDR=192.168.100.118 NODE_RANK=0 NUM_NODES=2 bash tools/mpi_parallel/torch.sh
# MASTER_ADDR=192.168.100.118 NODE_RANK=1 NUM_NODES=2 bash tools/mpi_parallel/torch.sh