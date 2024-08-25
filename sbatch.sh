#!/usr/bin/bash

#SBATCH -J UniTR-Train-temporal_deformable
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=32G
#SBATCH -p batch_ugrad
#SBATCH -t 3-0
#SBATCH -o logs/slurm-%A.outs

cat $0
pwd
which python
hostname

. /data/sw/spack/share/spack/setup-env.sh
spack find
spack load cuda@11.3.0
nvcc -V

# python -m pcdet.datasets.nuscenes.nuscenes_dataset --func create_nuscenes_infos \
#     --cfg_file tools/cfgs/dataset_configs/nuscenes_dataset.yaml \
#     --version v1.0-trainval \
#     --with_cam \
#     --with_cam_gt

# multi-gpu training
# note that we don't use image pretrain in BEV Map Segmentation
cd tools

## default
# bash scripts/dist_train.sh 2 --cfg_file ./cfgs/nuscenes_models/unitr_map.yaml --sync_bn --eval_map --logger_iter_interval 1000 \
    # --extra_tag default --ckpt ../output/cfgs/nuscenes_models/unitr_map/default/ckpt/latest_model.pth

## temporal
# bash scripts/dist_train.sh 2 --cfg_file ./cfgs/nuscenes_models/unitr_map.yaml --sync_bn --eval_map --logger_iter_interval 1000 \
#     --extra_tag temporal --ckpt ../output/cfgs/nuscenes_models/unitr_map/temporal/ckpt/latest_model.pth

## temporal deformable
bash scripts/dist_train.sh 2 --cfg_file ./cfgs/nuscenes_models/unitr_map.yaml --sync_bn --eval_map --logger_iter_interval 1000 --use_amp --extra_tag temporal_deformable

## add lss
# bash scripts/dist_train.sh 2 --cfg_file ./cfgs/nuscenes_models/unitr_map+lss.yaml --sync_bn --eval_map --logger_iter_interval 1000

# # multi-gpu testing
# ## normal
# bash scripts/dist_test.sh 2 --cfg_file ./cfgs/nuscenes_models/unitr_map.yaml --ckpt <CHECKPOINT_FILE> --eval_map

# ## add LSS
# bash scripts/dist_test.sh 2 --cfg_file ./cfgs/nuscenes_models/unitr_map+lss.yaml --ckpt <CHECKPOINT_FILE> --eval_map
# # NOTE: evaluation results will not be logged in *.log, only be printed in the teminal

exit 0
