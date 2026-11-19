#!/usr/bin/env bash

set -x

EXP_DIR=${EXP_DIR:-output_soda_coda}
PY_ARGS=${@:1}

python -u main.py \
    --dataset soda_coda \
    --dataset_file soda_coda \
    --data_root ./data/CODA \
    --ad_train_ann ./data/CODA/annotations/train_all.json \
    --ad_train_img_dir ./data/CODA/train \
    --ad_val_ann ./data/CODA/annotations/annotations.json \
    --ad_val_img_dir ./data/CODA/val/images \
    --train_set soda_train \
    --test_set coda_val \
    --PREV_INTRODUCED_CLS 0 \
    --CUR_INTRODUCED_CLS 6 \
    --num_known_classes 6 \
    --num_classes 7 \
    --epochs 50 \
    --lr_drop_epochs 25 \
    --output_dir "${EXP_DIR}" \
    --resume "./output_soda_coda/checkpoint0035.pth" \
    ${PY_ARGS}

