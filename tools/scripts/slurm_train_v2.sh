#!/usr/bin/env bash

set -x

PARTITION=batch_ugrad
JOB_NAME="UniTR-Train"
GPUS=$1
PY_ARGS=${@:2}

GPUS_PER_NODE=$1
SRUN_ARGS=${SRUN_ARGS:-""}

while true
do
    PORT=$(( ((RANDOM<<15)|RANDOM) % 49152 + 10000 ))
    status="$(nc -z 127.0.0.1 $PORT < /dev/null &>/dev/null; echo $?)"
    if [ "${status}" != "0" ]; then
        break;
    fi
done
echo $PORT

srun -p ${PARTITION} \
    --job-name=${JOB_NAME} \
    --gres=gpu:${GPUS_PER_NODE} \
    --cpus-per-gpu=8 \
    --mem-per-gpu=32G \
    --kill-on-bad-exit=1 \
    ${SRUN_ARGS} \
    python -m torch.distributed.launch --nproc_per_node=${GPUS_PER_NODE} train.py --launcher pytorch --tcp_port ${PORT} ${PY_ARGS}
