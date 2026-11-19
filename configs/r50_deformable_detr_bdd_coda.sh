#!/usr/bin/env bash

set -x

EXP_DIR=${EXP_DIR:-output_bdd_coda}
PY_ARGS=${@:1}

python -u main.py \
    --dataset bdd_coda \
    --dataset_file bdd_coda \
    --data_root ./data/BDD \
    --ad_train_ann ./data/BDD/annotations/bdd100k_labels_images_train.json \
    --ad_train_img_dir ./data/BDD/train/images \
    --ad_val_ann ./data/CODA/annotations/annotations.json \
    --ad_val_img_dir ./data/CODA/val/images \
    --train_set bdd_train \
    --test_set coda_val \
    --PREV_INTRODUCED_CLS 0 \
    --CUR_INTRODUCED_CLS 5 \
    --num_known_classes 5 \
    --num_classes 6 \
    --epochs 50 \
    --lr_drop_epochs 25 \
    --output_dir "${EXP_DIR}" \
    --resume "./output_bdd_coda/checkpoint.pth" \
    ${PY_ARGS}
