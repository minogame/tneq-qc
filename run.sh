# python train.py \
#   --num-step 5000 \
#   --save-every 1000 \
#   --eval-every 1000 \
#   --num-eval-samples 4096 \
#   --checkpoint-root ./checkpoints \
#   --exp-name debug_w1_test

# exit;


#!/bin/bash
# Run distributed training example with 4 processes using PyTorch CPU backend

# set -e

# cd "$(dirname "$0")"

# Fix IPv6 resolution issues on macOS
# export TP_SOCKET_IFNAME=lo0

# export MASTER_ADDR=192.168.100.120
# export MASTER_PORT=29500

# export GLOO_SOCKET_IFNAME=eno1
# export NCCL_SOCKET_IFNAME=eno1
# export TP_SOCKET_IFNAME=eno1

export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29500

export GLOO_SOCKET_IFNAME=lo
export NCCL_SOCKET_IFNAME=lo
export TP_SOCKET_IFNAME=lo

# export NCCL_SOCKET_IFNAME=lo0
# export TORCH_DISTRIBUTED_DEBUG=INFO

# echo "Starting distributed training with 4 processes..."
# echo "Backend: PyTorch CPU (gloo)"
# echo "Master: $MASTER_ADDR:$MASTER_PORT"
# echo ""

if [ $# -gt 0 ]; then
    RANK=$1
else
    RANK=0
fi


torchrun \
    --nnodes=1 \
    --nproc_per_node=4 \
    --node_rank=$RANK \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    examples/example_distributed_training.py

echo ""
echo "Done!"
