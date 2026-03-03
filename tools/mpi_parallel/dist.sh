#!/bin/bash
# ==============================================================
# MPI 分布式启动脚本 (无 SSH，每台机器独立运行)
# 用于启动 easy_distribution.py 的 Master-Worker 训练框架
#
# 用法:
#   机器1 (Master):
#     MASTER_ADDR=192.168.100.118 NODE_RANK=0 NUM_NODES=2 bash tools/mpi_parallel/dist.sh
#
#   机器2 (Worker):
#     MASTER_ADDR=192.168.100.118 NODE_RANK=1 NUM_NODES=2 bash tools/mpi_parallel/dist.sh
#
#   单机模式 (不设置环境变量即可):
#     bash tools/mpi_parallel/dist.sh [nproc_per_node]
# ==============================================================

# --- 可配置参数 (环境变量 或 默认值) ---
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}   # Master 节点 IP
MASTER_PORT=${MASTER_PORT:-29500}          # Master 节点通信端口
NODE_RANK=${NODE_RANK:-0}                  # 当前节点编号 (0=Master, 1,2,...=Worker)
NUM_NODES=${NUM_NODES:-1}                  # 总节点数
NPROC_PER_NODE=${1:-${NPROC_PER_NODE:-3}} # 每节点 MPI 进程数，默认 3

SCRIPT_DIR="tools/mpi_parallel"
PYTHON_SCRIPT="${SCRIPT_DIR}/easy_distribution_mpi.py"

# --- 检查环境 ---
if ! command -v mpiexec &> /dev/null; then
    echo "[ERROR] mpiexec 未找到，请先安装 MPI (如 OpenMPI 或 MPICH)。"
    exit 1
fi

if ! python -c "import mpi4py" &> /dev/null 2>&1; then
    echo "[ERROR] mpi4py 未安装，请先安装: pip install mpi4py"
    exit 1
fi

if [ ! -f "${PYTHON_SCRIPT}" ]; then
    echo "[ERROR] 找不到 Python 脚本: ${PYTHON_SCRIPT}"
    exit 1
fi

# --- 启动 ---
echo "=============================================="
echo " MPI 分布式训练启动"
echo "  MASTER_ADDR:    ${MASTER_ADDR}"
echo "  MASTER_PORT:    ${MASTER_PORT}"
echo "  NODE_RANK:      ${NODE_RANK}"
echo "  NUM_NODES:      ${NUM_NODES}"
echo "  NPROC_PER_NODE: ${NPROC_PER_NODE}"
echo "  脚本:           ${PYTHON_SCRIPT}"
echo "=============================================="

# 通过环境变量传递配置给 Python 进程
export MASTER_ADDR
export MASTER_PORT
export NODE_RANK
export NUM_NODES
export NPROC_PER_NODE

mpiexec -n "${NPROC_PER_NODE}" \
        python "${PYTHON_SCRIPT}"

EXIT_CODE=$?

if [ ${EXIT_CODE} -eq 0 ]; then
    echo "=============================================="
    echo " [Node ${NODE_RANK}] 训练完成 (exit code: 0)"
    echo "=============================================="
else
    echo "=============================================="
    echo " [Node ${NODE_RANK}] 运行异常 (exit code: ${EXIT_CODE})"
    echo "=============================================="
fi

exit ${EXIT_CODE}





exit;



#!/bin/bash
#PJM -L "node=2"
#PJM -L "rscgrp=small"
#PJM -L "elapse=24:00:00"
#PJM -g ra010014
#PJM -x PJM_LLIO_GFSCACHE=/vol0002:/vol0003:/vol0004
#PJM -S                # Direction of output statistic information file (-s or -S)

# spack load python
. /vol0004/apps/oss/spack/share/spack/setup-env.sh
spack load /qhm66vh
spack load /ce7iuxj

# activate python virtual environment 
export WORK=/vol0206/data/ra010014/u13028
source $WORK/tneq/bin/activate

cd $WORK/codes/tneq-qc/
bash test_dist.sh > test_dist.txt



#!/bin/bash
#PJM -L "node=2"
#PJM -L "rscgrp=small"
#PJM -L "elapse=24:00:00"
#PJM -g ra010014
#PJM -x PJM_LLIO_GFSCACHE=/vol0002:/vol0003:/vol0004
#PJM -S

# spack load python
. /vol0004/apps/oss/spack/share/spack/setup-env.sh
spack load /qhm66vh
spack load /ce7iuxj

# activate python virtual environment 
export WORK=/vol0206/data/ra010014/u13028
source $WORK/tneq/bin/activate

cd $WORK/codes/tneq-qc/

mpiexec -n 2 bash test_dist.sh > test_dist.txt




login5$ cat run_tohoku.sh.46861434.out 
1: lo: <LOOPBACK,UP,LOWER_UP> mtu 65536 qdisc noqueue state UNKNOWN group default qlen 1000
    link/loopback 00:00:00:00:00:00 brd 00:00:00:00:00:00
    inet 127.0.0.1/8 scope host lo
       valid_lft forever preferred_lft forever
    inet6 ::1/128 scope host 
       valid_lft forever preferred_lft forever
2: tofu1: <BROADCAST,UP,LOWER_UP> mtu 1500 qdisc fq_codel state UNKNOWN group default qlen 1000
    link/ether 02:00:10:0a:00:01 brd ff:ff:ff:ff:ff:ff
    inet 10.241.65.2/12 brd 10.255.255.255 scope global noprefixroute tofu1
       valid_lft forever preferred_lft forever
    inet6 fe80::10ff:fe0a:1/64 scope link 
       valid_lft forever preferred_lft forever
3: tofu0: <BROADCAST,UP,LOWER_UP> mtu 1500 qdisc fq_codel state UNKNOWN group default qlen 1000
    link/ether 02:00:10:0a:00:01 brd ff:ff:ff:ff:ff:ff
    inet 10.225.65.2/12 brd 10.239.255.255 scope global dynamic tofu0
       valid_lft 376583508sec preferred_lft 376583508sec
    inet6 fe80::10ff:fe0a:1/64 scope link 
       valid_lft forever preferred_lft forever
4: virbr0: <NO-CARRIER,BROADCAST,MULTICAST,UP> mtu 1500 qdisc noqueue state DOWN group default qlen 1000
    link/ether 52:54:00:6e:e7:bc brd ff:ff:ff:ff:ff:ff
    inet 192.168.122.1/24 brd 192.168.122.255 scope global virbr0
       valid_lft forever preferred_lft forever
5: krm-virbr0: <NO-CARRIER,BROADCAST,MULTICAST,UP> mtu 1500 qdisc noqueue state DOWN group default qlen 1000
    link/ether 52:54:00:ee:64:f9 brd ff:ff:ff:ff:ff:ff
    inet 192.168.120.1/24 brd 192.168.120.255 scope global krm-virbr0
       valid_lft forever preferred_lft forever




login5$ cat  output.46861434/0/1/stdout.1.1 
pjsub_node_list: 
PJM_NODE_RANK: 
PJM_O_NODEINF: 
PJM_NODE_ITR: 
OMPI_COMM_WORLD_RANK: 
OMPI_COMM_WORLD_SIZE: 
PMIX_RANK:  1
PMIX_SIZE: 
MASTER_ADDR:  c25-3201c
MASTER_PORT:  29500
Hostname: s c25-3009c
login5$ cat output.46861434/0/1/std
stderr.1.0  stderr.1.1  stdout.1.0  stdout.1.1  
login5$ cat output.46861434/0/1/std
stderr.1.0  stderr.1.1  stdout.1.0  stdout.1.1  
login5$ cat output.46861434/0/1/stdout.1.0 
pjsub_node_list: 
PJM_NODE_RANK: 
PJM_O_NODEINF: 
PJM_NODE_ITR: 
OMPI_COMM_WORLD_RANK: 
OMPI_COMM_WORLD_SIZE: 
PMIX_RANK:  0
PMIX_SIZE: 
MASTER_ADDR:  c25-3201c
MASTER_PORT:  29500
Hostname: s c25-3201c




login5$ cat  output.46861434/0/1/stdout.1.1 
pjsub_node_list: 
PJM_NODE_RANK: 
PJM_O_NODEINF: 
PJM_NODE_ITR: 
OMPI_COMM_WORLD_RANK: 
OMPI_COMM_WORLD_SIZE: 
PMIX_RANK:  1
PMIX_SIZE: 
MASTER_ADDR:  c25-3201c
MASTER_PORT:  29500
Hostname: s c25-3009c
login5$ cat output.46861434/0/1/std
stderr.1.0  stderr.1.1  stdout.1.0  stdout.1.1  
login5$ cat output.46861434/0/1/std
stderr.1.0  stderr.1.1  stdout.1.0  stdout.1.1  
login5$ cat output.46861434/0/1/stdout.1.0 
pjsub_node_list: 
PJM_NODE_RANK: 
PJM_O_NODEINF: 
PJM_NODE_ITR: 
OMPI_COMM_WORLD_RANK: 
OMPI_COMM_WORLD_SIZE: 
PMIX_RANK:  0
PMIX_SIZE: 
MASTER_ADDR:  c25-3201c
MASTER_PORT:  29500
Hostname: s c25-3201c
login5$ cat output.46861434/0/1/stdout.1.1 
pjsub_node_list: 
PJM_NODE_RANK: 
PJM_O_NODEINF: 
PJM_NODE_ITR: 
OMPI_COMM_WORLD_RANK: 
OMPI_COMM_WORLD_SIZE: 
PMIX_RANK:  1
PMIX_SIZE: 
MASTER_ADDR:  c25-3201c
MASTER_PORT:  29500
Hostname: s c25-3009c









echo "pjsub_node_list: " $(pjsub_node_list)
echo "PJM_NODE_RANK: " $PJM_NODE_RANK
echo "PJM_O_NODEINF: " $PJM_O_NODEINF
echo "PJM_NODE_ITR: " $PJM_NODE_ITR
echo "OMPI_COMM_WORLD_RANK: " $OMPI_COMM_WORLD_RANK
echo "OMPI_COMM_WORLD_SIZE: " $OMPI_COMM_WORLD_SIZE
echo "PMIX_RANK: " $PMIX_RANK
echo "PMIX_SIZE: " $PMIX_SIZE
echo "MASTER_ADDR: " $MASTER_ADDR
echo "MASTER_PORT: " $MASTER_PORT

# MASTER_ADDR=$(hostname)
# if [ "$PJM_NODE_RANK" -ne 0 ]; then
#     # 非 0 号节点需要从 PJM 环境中获取 0 号节点的名称
#     # 具体的获取方式取决于集群配置，通常可以使用以下技巧：
#     MASTER_ADDR=$(pjsub_node_list | head -n 1)
# fi
# echo "Master Address: " $MASTER_ADDR
# export MASTER_ADDR

# echo "Master Address after: " $MASTER_ADDR
echo "Hostname: " $(hostname -i)

export MASTER_PORT=29500

# 3. 设置 Node Rank
export NODE_RANK=$PJM_NODE_RANK

# 4. 每个节点的进程数 (对应 4 个 CMG)
export NPROC_PER_NODE=4

# 5. 计算线程数 (每个 CMG 12 核心)
export OMP_NUM_THREADS=12


export GLOO_SOCKET_IFNAME=tofu1
export NCCL_SOCKET_IFNAME=tofu1
export TP_SOCKET_IFNAME=tofu1


export RANK=$PMIX_RANK

torchrun \
    --nnodes=2 \
    --nproc_per_node=2 \
    --node_rank=$RANK \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    examples/example_distributed_training.py

echo ""
echo "Done!"













#!/bin/bash
#PJM -L "node=2"
#PJM -L "rscgrp=small"
#PJM -L "elapse=24:00:00"
#PJM -g ra010014
#PJM -x PJM_LLIO_GFSCACHE=/vol0002:/vol0003:/vol0004
#PJM -S

# spack load python
. /vol0004/apps/oss/spack/share/spack/setup-env.sh
spack load /qhm66vh
spack load /ce7iuxj

# activate python virtual environment 
export WORK=/vol0206/data/ra010014/u13028
source $WORK/tneq/bin/activate

cd $WORK/codes/tneq-qc/

# export MASTER_ADDR=$(hostname)
export MASTER_PORT=29500  # 随便选个空闲端口

echo "====ip address===="
ip address

echo "====ip address tofu1===="
ip -o -4 addr list tofu1

echo "====hostname -i===="
hostname -i


export MASTER_ADDR=`ip -o -4 addr list tofu1 | awk '{print $4}' | cut -d/ -f1`
echo "ip address: " $MASTER_ADDR



mpiexec -n 2 -x MASTER_ADDR -x MASTER_PORT bash test_dist.sh



