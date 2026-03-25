# export CUDA_VISIBLE_DEVICES=1

# mpiexec -n 2 python tneq_qc/genetic_tensor_network.py



# mpiexec -n 3 python tneq_qc/genetic_tensor_network.py --distributed --info


# mpiexec -n 2 python tneq_qc/mpi_agent.py

# 日志同步保存到run.log中，但同时不影响终端输出
# mpiexec -n 2 python test_mpi_agent.py
# mpiexec -n 2 python test_mpi_overlord.py
# mpiexec -n 2 python test_mpi_overlord.py > run.log 2>&1 # & tail -f run.log



#!/bin/bash
# Run distributed training example with 4 processes using PyTorch CPU backend

# set -e

# cd "$(dirname "$0")"

# Fix IPv6 resolution issues on macOS
# export TP_SOCKET_IFNAME=lo0

export MASTER_ADDR=192.168.100.120
export MASTER_PORT=29500

export GLOO_SOCKET_IFNAME=eno1
export NCCL_SOCKET_IFNAME=eno1
export TP_SOCKET_IFNAME=eno1

# export GLOO_SOCKET_IFNAME=lo0
# export NCCL_SOCKET_IFNAME=lo0
# export TP_SOCKET_IFNAME=lo0

# export NCCL_SOCKET_IFNAME=lo0
# export TORCH_DISTRIBUTED_DEBUG=INFO

# echo "Starting distributed training with 4 processes..."
# echo "Backend: PyTorch CPU (gloo)"
# echo "Master: $MASTER_ADDR:$MASTER_PORT"
# echo ""

# if [ $# -gt 0 ]; then
#     RANK=$1
# else
#     RANK=0
# fi

torchrun --standalone --nnodes=1 --nproc_per_node=2 \
  --master_addr=127.0.0.1 \
  --master_port=29500 \
  examples/train_dist.py

# torchrun \
#     --nnodes=2 \
#     --nproc_per_node=4 \
#     --node_rank=$RANK \
#     --master_addr=$MASTER_ADDR \
#     --master_port=$MASTER_PORT \
#     examples/train_dist.py

echo ""
echo "Done!"
