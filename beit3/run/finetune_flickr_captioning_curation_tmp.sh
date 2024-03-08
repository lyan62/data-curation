#!/bin/bash

export OUTPUT_DIR= # Path to the output directory
export RUN_NAME=flickr30k_remove_2std

nvidia-smi
# conda install pytorch torchvision cudatoolkit=11.6 -c pytorch
python -c "import torch; print(torch.__version__)"

# torchrun launch configuration
OMP_NUM_THREADS=1 torchrun --standalone --nnodes=1 --nproc_per_node=8 beit3/run_beit3_finetuning_al.py \
    --model beit3_base_patch16_480 \
    --input_size 480 \
    --task flickr30k_captioning \
    --batch_size 32 \
    --update_freq 1 \
    --layer_decay 1.0 \
    --lr 4e-5 \
    --randaug \
    --epochs 10 \
    --warmup_epochs 1 \
    --drop_path 0.1 \
    --sentencepiece_model PATH/beit3.spm \
    --finetune  PATH/beit3_base_patch16_224.pth \
    --data_path PATH/data \
    --output_dir ${OUTPUT_DIR}/${RUN_NAME} \
    --log_dir ${OUTPUT_DIR}/${RUN_NAME} \
    --weight_decay 0.05 \
    --seed 42 \
    --save_ckpt_freq 5 \
    --num_max_bpe_tokens 32 \
    --captioning_mask_prob 0.7 \
    --drop_worst_after 12000 \
    --dist_eval \
    --checkpoint_activations \
    --enable_deepspeed \
    --train_eval \
    --curation_method remove \
    --dynamic 
