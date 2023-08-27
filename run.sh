
eval "$(/zangzelin/conda/bin/conda shell.bash hook)"; conda activate env_san

CUDA_VISIBLE_DEVICES=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 taskset -c 0-9 wandb agent zangzelin/OVANET_DMT/yu5k29s9 &
CUDA_VISIBLE_DEVICES=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 taskset -c 10-19 wandb agent zangzelin/OVANET_DMT/yu5k29s9 &
CUDA_VISIBLE_DEVICES=2 CUBLAS_WORKSPACE_CONFIG=:4096:8 taskset -c 20-29 wandb agent zangzelin/OVANET_DMT/yu5k29s9 &
CUDA_VISIBLE_DEVICES=2 CUBLAS_WORKSPACE_CONFIG=:4096:8 taskset -c 30-39 wandb agent zangzelin/OVANET_DMT/yu5k29s9 &
CUDA_VISIBLE_DEVICES=3 CUBLAS_WORKSPACE_CONFIG=:4096:8 taskset -c 40-49 wandb agent zangzelin/OVANET_DMT/yu5k29s9 &
CUDA_VISIBLE_DEVICES=4 CUBLAS_WORKSPACE_CONFIG=:4096:8 taskset -c 50-59 wandb agent zangzelin/OVANET_DMT/yu5k29s9 &
CUDA_VISIBLE_DEVICES=5 CUBLAS_WORKSPACE_CONFIG=:4096:8 taskset -c 60-69 wandb agent zangzelin/OVANET_DMT/yu5k29s9 &
CUDA_VISIBLE_DEVICES=7 CUBLAS_WORKSPACE_CONFIG=:4096:8 taskset -c 70-79 wandb agent zangzelin/OVANET_DMT/yu5k29s9 &
CUDA_VISIBLE_DEVICES=6 CUBLAS_WORKSPACE_CONFIG=:4096:8 taskset -c 80-89 wandb agent zangzelin/OVANET_DMT/yu5k29s9 