
eval "$(/zangzelin/conda/bin/conda shell.bash hook)"; conda activate env_san

CUDA_VISIBLE_DEVICES=0 wandb agent zangzelin/OVANET_DMT/yu5k29s9 &
CUDA_VISIBLE_DEVICES=1 wandb agent zangzelin/OVANET_DMT/yu5k29s9 &
CUDA_VISIBLE_DEVICES=2 wandb agent zangzelin/OVANET_DMT/yu5k29s9 &
CUDA_VISIBLE_DEVICES=2 wandb agent zangzelin/OVANET_DMT/yu5k29s9 &
CUDA_VISIBLE_DEVICES=3 wandb agent zangzelin/OVANET_DMT/yu5k29s9 &
CUDA_VISIBLE_DEVICES=4 wandb agent zangzelin/OVANET_DMT/yu5k29s9 &
CUDA_VISIBLE_DEVICES=5 wandb agent zangzelin/OVANET_DMT/yu5k29s9 &
CUDA_VISIBLE_DEVICES=7 wandb agent zangzelin/OVANET_DMT/yu5k29s9 &
CUDA_VISIBLE_DEVICES=6 wandb agent zangzelin/OVANET_DMT/yu5k29s9 