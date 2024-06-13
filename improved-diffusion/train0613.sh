#!/bin/bash
# 设置CUDA_VISIBLE_DEVICES环境变量为1

conda activate laydi 
cd /home/ubuntu/ygq/LayOutDiffusion/improved-diffusion
# 使用mpiexec启动python程序train.py
export CUDA_VISIBLE_DEVICES=1
mpiexec -n 1  python scripts/train.py --checkpoint_path ../results/checkpoint/rico_v9 --model_arch transformer --modality e2e-tgt --save_interval 5000 --lr 2e-5 --batch_size 32 --diffusion_steps 200 --noise_schedule gaussian_refine_pow2.5 --use_kl False --learn_sigma False --aux_loss True --rescale_timesteps False --seq_length 121 --num_channels 128 --seed 102 --dropout 0.1 --padding_mode pad --experiment random --lr_anneal_steps 240000 --weight_decay 0.0 --predict_xstart True --training_mode discrete1 --vocab_size 186 --submit False --e2e_train ../data/processed_datasets/RICO_ltrb_lex --alignment_loss False  > output.log 2>&1 &

# 等待脚本执行完毕
# wait
