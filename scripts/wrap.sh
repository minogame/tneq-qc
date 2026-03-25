#!/bin/bash
# 根据 torchrun 设置的 LOCAL_RANK 绑定 CMG
case $LOCAL_RANK in
    0) exec numactl --cpunodebind=0 --membind=0 "$@" ;;
    1) exec numactl --cpunodebind=1 --membind=1 "$@" ;;
    2) exec numactl --cpunodebind=2 --membind=2 "$@" ;;
    3) exec numactl --cpunodebind=3 --membind=3 "$@" ;;
esac