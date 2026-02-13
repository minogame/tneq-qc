#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export MASTER_ADDR=`ip -o -4 addr list tofu1 | awk '{print $4}' | cut -d/ -f1`
MASTER_PORT="${MASTER_PORT:-29500}"
NODE_RANK="${NODE_RANK:-0}"

export GLOO_SOCKET_IFNAME=tofu1
export NCCL_SOCKET_IFNAME=tofu1
export TP_SOCKET_IFNAME=tofu1

echo "=============================================="
echo " Torch 单节点训练启动"
echo "  MASTER_ADDR:    ${MASTER_ADDR}"
echo "  MASTER_PORT:    ${MASTER_PORT}"
echo "  NODE_RANK:      ${NODE_RANK}"
echo "=============================================="

export PYTHONPATH=.

torchrun \
    --nnodes=1 \
    --nproc_per_node=4 \
    --node_rank="${NODE_RANK}" \
    --master_addr="${MASTER_ADDR}" \
    --master_port="${MASTER_PORT}" \
    "${SCRIPT_DIR}/easy_distribution_torch.py"
