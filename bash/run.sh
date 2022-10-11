source /zangzelin/.bashrc;
# <<< conda initialize <<<
export http_proxy=http://192.168.105.204:3128
export https_proxy=http://192.168.105.204:3128



echo 'dfsdfsdfsdfdsf'

echo 'dfsdfsdfsdfdsf'

cd /zangzelin/project/ovanet_-dmt/
pwd

if [ -f "/opt/anaconda3/etc/profile.d/conda.sh" ]; then
    . "/opt/anaconda3/etc/profile.d/conda.sh"
    CONDA_CHANGEPS1=false conda activate torch1.8
fi

rm /tmp/pymp* -r

CONID='zangzelin@gmail.com' 
CONKEY='lly19960925' 

cd ./
expect -c "spawn git pull origin; expect \"*Username*\" { send \"${CONID}\n\"; exp_continue } \"*Password*\" { send \"${CONKEY}\n\" }; interact"

conda-env list



export LC_ALL=C.UTF-8
/opt/anaconda3/envs/torch1.8/bin/python -m pip install wandb -U

wandb login --host=https://api.wandb.ai 795ecdb0233cb17d995270e1d732b27dafd8efaa

# bash ./bash/wandb.sh
# CUDA_VISIBLE_DEVICES=0 bash ./bash/wandb.sh &
# CUDA_VISIBLE_DEVICES=1 bash ./bash/wandb.sh &
# CUDA_VISIBLE_DEVICES=0 bash ./bash/wandb.sh &
# CUDA_VISIBLE_DEVICES=1 bash ./bash/wandb.sh &
# CUDA_VISIBLE_DEVICES=0 bash ./bash/wandb.sh &
CUDA_VISIBLE_DEVICES=0 bash ./bash/wandb.sh


# CUDA_VISIBLE_DEVICES=1 wandb agent zangzelin/PatEmb/fxmn1939 &
# CUDA_VISIBLE_DEVICES=2 wandb agent zangzelin/PatEmb/fxmn1939 &
# sleep 30s
# CUDA_VISIBLE_DEVICES=3 wandb agent zangzelin/PatEmb/fxmn1939 

# /opt/anaconda3/envs/torch1.8/bin/python /zangzelin/project/otn/bash/check_mem.py

sleep 1200m

sleep 30m
