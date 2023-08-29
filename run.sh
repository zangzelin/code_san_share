
eval "$(/zangzelin/conda/bin/conda shell.bash hook)"; conda activate env_san

CUDA_VISIBLE_DEVICES=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 taskset -c 0-7 wandb agent zangzelin/OVANET_DMT/ujfhi2x2 &
CUDA_VISIBLE_DEVICES=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 taskset -c 8-15 wandb agent zangzelin/OVANET_DMT/ujfhi2x2 &
CUDA_VISIBLE_DEVICES=2 CUBLAS_WORKSPACE_CONFIG=:4096:8 taskset -c 16-23 wandb agent zangzelin/OVANET_DMT/ujfhi2x2 &
CUDA_VISIBLE_DEVICES=3 CUBLAS_WORKSPACE_CONFIG=:4096:8 taskset -c 24-31 wandb agent zangzelin/OVANET_DMT/ujfhi2x2 &
CUDA_VISIBLE_DEVICES=4 CUBLAS_WORKSPACE_CONFIG=:4096:8 taskset -c 32-39 wandb agent zangzelin/OVANET_DMT/ujfhi2x2 &
CUDA_VISIBLE_DEVICES=5 CUBLAS_WORKSPACE_CONFIG=:4096:8 taskset -c 40-47 wandb agent zangzelin/OVANET_DMT/ujfhi2x2 &
CUDA_VISIBLE_DEVICES=7 CUBLAS_WORKSPACE_CONFIG=:4096:8 taskset -c 48-55 wandb agent zangzelin/OVANET_DMT/ujfhi2x2 &
CUDA_VISIBLE_DEVICES=6 CUBLAS_WORKSPACE_CONFIG=:4096:8 taskset -c 56-63 wandb agent zangzelin/OVANET_DMT/ujfhi2x2 