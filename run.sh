
eval "$(/zangzelin/conda/bin/conda shell.bash hook)"; conda activate env_san

CUDA_VISIBLE_DEVICES=0 python train_san.py --alpha=0.4 --augNearRate=500 --aug_type=1 --batch_size=36 --beta=0.8 --config=configs/office-train-config_ODA.yaml --data_aug_crop=0.6 --ent_open_scale=0.2 --gamma=0.1 --lr=0.03 --min_step=10000 --mlp_weight_decay=0.0002 --scheduler_gamma=10 --sigmaP=10 --source_data=./txt/source_webcam_obda_zzl.txt --target_data=./txt/target_dslr_obda_zzl.txt --top_k=50 --v_latent=10

CUDA_VISIBLE_DEVICES=0 python train_scl_topk_v2.py --alpha=0.9 --augNearRate=10000 --aug_type=1 --batch_size=80 --beta=1.4 --config=configs/dnet-train-config_OPDA.yaml --data_aug_crop=0.85 --ent_open_scale=0.5 --gamma=0.35 --lr=0.012 --min_step=15000 --mlp_weight_decay=0.0002 --multi=0.2 --sigmaP=50 --source_data=./txt/source_dpainting_univ_zzl.txt --target_data=./txt/target_dsketch_univ_zzl.txt --v_latent=10