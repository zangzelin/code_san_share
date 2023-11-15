## Code for SAN

This repository provides code for SAN.

### Environment

```
pytorch-lightning            1.4.8
torch                        1.11.0
scikit-learn                 1.0
torchvision                  0.12.0
pandas                       1.3.0
numpy                        1.20.2
wandb                        0.12.5
```

### how to run

```
mkdir file_save
mkdir img
python train_san.py --alpha=0.7 --augNearRate=10000 --aug_type=1 --beta=1.4 --config=configs/office-train-config_OPDA.yaml --data_aug_crop=0.8 --ent_open_scale=0.4 --gamma=0.6 --source_data=webcam_opda_zzl.txt --target_data=dslr_opda_zzl.txt
```

# opda 
## domainet

```
wandb sweep sweep/new_opda/domainnet_v1.yaml
wandb agent */OVANET_DMT/***
``` 

``` bash
CUDA_VISIBLE_DEVICES=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 python train_san.py --alpha=0.9 --augNearRate=10000 --aug_type=1 --batch_size=80 --beta=1.4 --config=configs/dnet-train-config_OPDA.yaml --data_aug_crop=0.85 --ent_open_scale=0.6 --gamma=0.4 --lr=0.012 --min_step=15000 --mlp_weight_decay=0.0002 --multi=0.2 --sigmaP=50 --source_data=./txt/source_dpainting_univ_zzl.txt --target_data=./txt/target_dreal_univ_zzl.txt --v_latent=10 & 

CUDA_VISIBLE_DEVICES=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 python train_san.py --alpha=0.9 --augNearRate=10000 --aug_type=1 --batch_size=80 --beta=1.4 --config=configs/dnet-train-config_OPDA.yaml --data_aug_crop=0.85 --ent_open_scale=0.6 --gamma=0.4 --lr=0.012 --min_step=15000 --mlp_weight_decay=0.0002 --multi=0.2 --sigmaP=50 --source_data=./txt/source_dsketch_univ_zzl.txt --target_data=./txt/target_dreal_univ_zzl.txt --v_latent=10 & 

CUDA_VISIBLE_DEVICES=2 CUBLAS_WORKSPACE_CONFIG=:4096:8 python train_san.py --alpha=0.9 --augNearRate=10000 --aug_type=1 --batch_size=80 --beta=1.4 --config=configs/dnet-train-config_OPDA.yaml --data_aug_crop=0.85 --ent_open_scale=0.6 --gamma=0.4 --lr=0.012 --min_step=15000 --mlp_weight_decay=0.0002 --multi=0.2 --sigmaP=50 --source_data=./txt/source_dreal_univ_zzl.txt --target_data=./txt/target_dpainting_univ_zzl.txt --v_latent=10 &

CUDA_VISIBLE_DEVICES=3 CUBLAS_WORKSPACE_CONFIG=:4096:8 python train_san.py --alpha=0.9 --augNearRate=10000 --aug_type=1 --batch_size=80 --beta=1.4 --config=configs/dnet-train-config_OPDA.yaml --data_aug_crop=0.85 --ent_open_scale=0.6 --gamma=0.4 --lr=0.012 --min_step=15000 --mlp_weight_decay=0.0002 --multi=0.2 --sigmaP=50 --source_data=./txt/source_dsketch_univ_zzl.txt --target_data=./txt/target_dpainting_univ_zzl.txt --v_latent=10 

sleep 60s

CUDA_VISIBLE_DEVICES=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 python train_san.py --alpha=0.9 --augNearRate=10000 --aug_type=1 --batch_size=80 --beta=1.4 --config=configs/dnet-train-config_OPDA.yaml --data_aug_crop=0.85 --ent_open_scale=0.6 --gamma=0.4 --lr=0.012 --min_step=15000 --mlp_weight_decay=0.0002 --multi=0.2 --sigmaP=50 --source_data=./txt/source_dpainting_univ_zzl.txt --target_data=./txt/target_dsketch_univ_zzl.txt --v_latent=10 & 

CUDA_VISIBLE_DEVICES=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 python train_san.py --alpha=0.9 --augNearRate=10000 --aug_type=1 --batch_size=80 --beta=1.4 --config=configs/dnet-train-config_OPDA.yaml --data_aug_crop=0.85 --ent_open_scale=0.6 --gamma=0.4 --lr=0.012 --min_step=15000 --mlp_weight_decay=0.0002 --multi=0.2 --sigmaP=50 --source_data=./txt/source_dreal_univ_zzl.txt --target_data=./txt/target_dsketch_univ_zzl.txt --v_latent=10 & 
```

the results 

``` bash
run 1:
dpainting -> dreal : 0.5838
dsketch -> dreal : 0.5793
dreal -> dpainting : 0.5293
dsketch -> dpainting : 0.4791
dpainting -> dsketch : 0.4758
dreal -> dsketch : 0.4671
mean: 51.9

run 2:
dpainting -> dreal: 0.58
dsketch -> dreal: 0.5823
dreal -> dpainting: 0.5283
dsketch -> dpainting: 0.4822
dpainting -> dsketch: 0.4738
dreal -> dsketch: 0.4647
mean: 51.9

run 3:
dpainting -> _dsketch: 0.4707
dreal -> _dsketch: 0.4651
dreal -> _dpainting: 0.5298
dpainting -> _dreal: 0.5781
dsketch -> _dreal: 0.5886
dsketch -> _dpainting: 0.4904
mean: 52.0
```

## visda


