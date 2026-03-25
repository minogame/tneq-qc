import os
import sys
import subprocess

def main():
    # 获取由 torchrun 设置的本地 rank (0-3)
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    print(f'bind_cmg {local_rank}')

    # CMG 映射表 (基于 Processors Programming Guide)
    # 格式: LOCAL_RANK: (Core Range, Memory Node)
    # 注意：核心编号可能因系统/OS配置略有不同，以下为文档推荐配置
    cmg_mapping = {
        0: ("12-23", "4"),  # CMG0
        1: ("24-35", "5"),  # CMG1
        2: ("36-47", "6"),  # CMG2
        3: ("48-59", "7")   # CMG3
    }

    """
    numactl --hardware
    available: 2 nodes (0-1)
    node 0 cpus: 0 1 2 3 8 9 10 11
    node 0 size: 31801 MB
    node 0 free: 4056 MB
    node 1 cpus: 4 5 6 7 12 13 14 15
    node 1 size: 32252 MB
    node 1 free: 3264 MB
    node distances:
    node   0   1 
    0:  10  21 
    1:  21  10
    """
    # cmg_mapping = {
    #     0: ("0-3", "0"),
    #     1: ("8-11", "0"),
    #     2: ("4-7", "1"),
    #     3: ("12-15", "1")
    # }

    if local_rank in cmg_mapping:
        cores, mem = cmg_mapping[local_rank]
        # 构建 numactl 命令
        # numactl -C <核心范围> -m <内存节点> python <原脚本> <参数>
        cmd = ["numactl", "-C", cores, "-m", mem, "python"] + sys.argv[1:]
    else:
        # 如果 rank 超出范围（例如意外启动了更多进程），仅运行 python
        cmd = ["python"] + sys.argv[1:]

    # 使用 execvp 替换当前进程，保持 PID 不变，利于信号传递
    os.execvp("numactl", cmd)

if __name__ == "__main__":
    main()