#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29500}"
NODE_RANK="${PMIX_RANK:-0}"
NUM_NODES="2"

export GLOO_SOCKET_IFNAME=tofu1
export NCCL_SOCKET_IFNAME=tofu1
export TP_SOCKET_IFNAME=tofu1

echo "=============================================="
echo " Torch 分布式训练启动"
echo "  MASTER_ADDR:    ${MASTER_ADDR}"
echo "  MASTER_PORT:    ${MASTER_PORT}"
echo "  NODE_RANK:      ${NODE_RANK}"
echo "  NUM_NODES:      ${NUM_NODES}"
echo "=============================================="

export PYTHONPATH=.

torchrun \
    --nnodes=${NUM_NODES} \
    --nproc_per_node=2 \
    --node_rank="${NODE_RANK}" \
    --master_addr="${MASTER_ADDR}" \
    --master_port="${MASTER_PORT}" \
    "${SCRIPT_DIR}/easy_distribution_torch.py"
