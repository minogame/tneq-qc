#!/bin/bash
# Run distributed training example with 4 processes using PyTorch CPU backend

export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29500

export GLOO_SOCKET_IFNAME=lo
export NCCL_SOCKET_IFNAME=lo
export TP_SOCKET_IFNAME=lo


if [ $# -gt 0 ]; then
    RANK=$1
else
    RANK=0
fi

export OMP_NUM_THREADS=12

torchrun \
    --nnodes=1 \
    --nproc_per_node=4 \
    --node_rank=$RANK \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    bind_cmg.py examples/example_distributed_training.py


echo ""
echo "Done!"
