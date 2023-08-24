
eval "$(/zangzelin/conda/bin/conda shell.bash hook)"; conda activate env_san

CUDA_VISIBLE_DEVICES=0 wandb agent zangzelin/OVANET_DMT/ee29z0t2 &
CUDA_VISIBLE_DEVICES=1 wandb agent zangzelin/OVANET_DMT/ee29z0t2 &
CUDA_VISIBLE_DEVICES=2 wandb agent zangzelin/OVANET_DMT/ee29z0t2 &
CUDA_VISIBLE_DEVICES=3 wandb agent zangzelin/OVANET_DMT/ee29z0t2 