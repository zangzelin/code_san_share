
eval "$(/zangzelin/conda/bin/conda shell.bash hook)"; conda activate env_san

CUDA_VISIBLE_DEVICES=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 python train_san.py --alpha=0.9 --augNearRate=10000 --aug_type=1 --batch_size=80 --beta=1.4 --config=configs/dnet-train-config_OPDA.yaml --data_aug_crop=0.85 --ent_open_scale=0.6 --gamma=0.4 --lr=0.012 --min_step=15000 --mlp_weight_decay=0.0002 --multi=0.2 --sigmaP=50 --source_data=./txt/source_dpainting_univ_zzl.txt --target_data=./txt/target_dreal_univ_zzl.txt --v_latent=10 & 

CUDA_VISIBLE_DEVICES=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 python train_san.py --alpha=0.9 --augNearRate=10000 --aug_type=1 --batch_size=80 --beta=1.4 --config=configs/dnet-train-config_OPDA.yaml --data_aug_crop=0.85 --ent_open_scale=0.6 --gamma=0.4 --lr=0.012 --min_step=15000 --mlp_weight_decay=0.0002 --multi=0.2 --sigmaP=50 --source_data=./txt/source_dsketch_univ_zzl.txt --target_data=./txt/target_dreal_univ_zzl.txt --v_latent=10 & 

CUDA_VISIBLE_DEVICES=2 CUBLAS_WORKSPACE_CONFIG=:4096:8 python train_san.py --alpha=0.9 --augNearRate=10000 --aug_type=1 --batch_size=80 --beta=1.4 --config=configs/dnet-train-config_OPDA.yaml --data_aug_crop=0.85 --ent_open_scale=0.6 --gamma=0.4 --lr=0.012 --min_step=15000 --mlp_weight_decay=0.0002 --multi=0.2 --sigmaP=50 --source_data=./txt/source_dreal_univ_zzl.txt --target_data=./txt/target_dpainting_univ_zzl.txt --v_latent=10 &

CUDA_VISIBLE_DEVICES=3 CUBLAS_WORKSPACE_CONFIG=:4096:8 python train_san.py --alpha=0.9 --augNearRate=10000 --aug_type=1 --batch_size=80 --beta=1.4 --config=configs/dnet-train-config_OPDA.yaml --data_aug_crop=0.85 --ent_open_scale=0.6 --gamma=0.4 --lr=0.012 --min_step=15000 --mlp_weight_decay=0.0002 --multi=0.2 --sigmaP=50 --source_data=./txt/source_dsketch_univ_zzl.txt --target_data=./txt/target_dpainting_univ_zzl.txt --v_latent=10 

sleep 60s

CUDA_VISIBLE_DEVICES=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 python train_san.py --alpha=0.9 --augNearRate=10000 --aug_type=1 --batch_size=80 --beta=1.4 --config=configs/dnet-train-config_OPDA.yaml --data_aug_crop=0.85 --ent_open_scale=0.6 --gamma=0.4 --lr=0.012 --min_step=15000 --mlp_weight_decay=0.0002 --multi=0.2 --sigmaP=50 --source_data=./txt/source_dpainting_univ_zzl.txt --target_data=./txt/target_dsketch_univ_zzl.txt --v_latent=10 & 

CUDA_VISIBLE_DEVICES=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 python train_san.py --alpha=0.9 --augNearRate=10000 --aug_type=1 --batch_size=80 --beta=1.4 --config=configs/dnet-train-config_OPDA.yaml --data_aug_crop=0.85 --ent_open_scale=0.6 --gamma=0.4 --lr=0.012 --min_step=15000 --mlp_weight_decay=0.0002 --multi=0.2 --sigmaP=50 --source_data=./txt/source_dreal_univ_zzl.txt --target_data=./txt/target_dsketch_univ_zzl.txt --v_latent=10 & 

