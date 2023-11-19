#!/usr/bin/bash -x
#SBATCH --account=laion
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=12
#SBATCH --time=48:00:00
#SBATCH --partition=g40x
#SBATCH --job-name=dcgan
#SBATCH --output=logs/dcgan-%j.out
#SBATCH --error=logs/dcgan-%j.err
#SBATCH --exclusive
#SBATCH --exclude=ip-26-0-157-217,ip-26-0-144-140,ip-26-0-147-120

LOGS=/fsx/home-wendlerc/models/dcgan


export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
srun python models/dcgan.py --url "pipe:aws s3 cp s3://stability-west/laioco-latents-split-full{/,/aug/001/}{000..500}/{000000..000100}.tar -" \
                       --batch_size 256 \
                       --num_workers 12 \
                       --n_feats 256 \
                       --latent_dim 512 \
                       --log_every_n_steps 1000 \
                       --checkpoint_every_n_examples 200000000 \
                       --checkpoint_path $LOGS \
                       --accelerator auto \

