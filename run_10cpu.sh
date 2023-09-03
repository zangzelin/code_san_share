ln -s /zangzelin/data /root/data
eval "$(/zangzelin/conda/bin/conda shell.bash hook)"; conda activate env_san
wandb login --host=https://api.wandb.ai 46a45f51484e3cce8ee9d0d270ea18b98ef9151d


CUDA_VISIBLE_DEVICES=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 taskset -c 0-11 wandb agent zangzelin_hotmail/OVANET_DMT/n0k1s38j &
sleep 60
CUDA_VISIBLE_DEVICES=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 taskset -c 12-23 wandb agent zangzelin_hotmail/OVANET_DMT/n0k1s38j &
sleep 60
CUDA_VISIBLE_DEVICES=2 CUBLAS_WORKSPACE_CONFIG=:4096:8 taskset -c 24-35 wandb agent zangzelin_hotmail/OVANET_DMT/n0k1s38j &
sleep 60
CUDA_VISIBLE_DEVICES=3 CUBLAS_WORKSPACE_CONFIG=:4096:8 taskset -c 36-47 wandb agent zangzelin_hotmail/OVANET_DMT/n0k1s38j &
sleep 60
CUDA_VISIBLE_DEVICES=4 CUBLAS_WORKSPACE_CONFIG=:4096:8 taskset -c 48-59 wandb agent zangzelin_hotmail/OVANET_DMT/n0k1s38j &
sleep 60
CUDA_VISIBLE_DEVICES=5 CUBLAS_WORKSPACE_CONFIG=:4096:8 taskset -c 60-71 wandb agent zangzelin_hotmail/OVANET_DMT/n0k1s38j &
sleep 60
CUDA_VISIBLE_DEVICES=7 CUBLAS_WORKSPACE_CONFIG=:4096:8 taskset -c 72-83 wandb agent zangzelin_hotmail/OVANET_DMT/n0k1s38j &
sleep 60
CUDA_VISIBLE_DEVICES=6 CUBLAS_WORKSPACE_CONFIG=:4096:8 taskset -c 84-95 wandb agent zangzelin_hotmail/OVANET_DMT/n0k1s38j 