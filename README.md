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

# UNDA 
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

``` bash
train_san.py --alpha=0.9 --augNearRate=100 --aug_type=1 --batch_size=30 --beta=1.4 --config=configs/visda-train-config_UDA.yaml --data_aug_crop=0.55 --ent_open_scale=0.08 --gamma=0.9 --lr=0.006 --min_step=5000 --mlp_weight_decay=0.0002 --multi=0.2 --sigmaP=50 --source_data=./txt/source_visda_list_univ_zzl.txt --target_data=./txt/target_visda_list_univ_zzl.txt --v_latent=10
```

the results 

``` bash
run 1:
hscore: 61.2
```

## officehome

``` bash
train_san.py --alpha=0.5 --augNearRate=10000 --aug_type=1 --batch_size=50 --beta=1.2 --config=configs/officehome-train-config_OPDA.yaml --data_aug_crop=0.7 --ent_open_scale=0.1 --gamma=0.6 --lr=0.01 --min_step=10000 --mlp_weight_decay=0.0005 --multi=0.2 --sigmaP=50 --source_data=./txt/source_Art_univ_zzl.txt --target_data=./txt/target_Real_univ_zzl.txt --v_latent=10

train_san.py --alpha=0.5 --augNearRate=10000 --aug_type=1 --batch_size=50 --beta=1.2 --config=configs/officehome-train-config_OPDA.yaml --data_aug_crop=0.7 --ent_open_scale=0.1 --gamma=0.6 --lr=0.01 --min_step=10000 --mlp_weight_decay=0.0005 --multi=0.2 --sigmaP=50 --source_data=./txt/source_Product_univ_zzl.txt --target_data=./txt/target_Real_univ_zzl.txt --v_latent=10

train_san.py --alpha=0.5 --augNearRate=10000 --aug_type=1 --batch_size=50 --beta=1.2 --config=configs/officehome-train-config_OPDA.yaml --data_aug_crop=0.7 --ent_open_scale=0.1 --gamma=0.6 --lr=0.01 --min_step=10000 --mlp_weight_decay=0.0005 --multi=0.2 --sigmaP=50 --source_data=./txt/source_Real_univ_zzl.txt --target_data=./txt/target_Art_univ_zzl.txt --v_latent=10

train_san.py --alpha=0.5 --augNearRate=10000 --aug_type=1 --batch_size=50 --beta=1.2 --config=configs/officehome-train-config_OPDA.yaml --data_aug_crop=0.7 --ent_open_scale=0.1 --gamma=0.6 --lr=0.01 --min_step=10000 --mlp_weight_decay=0.0005 --multi=0.2 --sigmaP=50 --source_data=./txt/source_Art_univ_zzl.txt --target_data=./txt/target_Product_univ_zzl.txt --v_latent=10

train_san.py --alpha=0.5 --augNearRate=10000 --aug_type=1 --batch_size=50 --beta=1.2 --config=configs/officehome-train-config_OPDA.yaml --data_aug_crop=0.7 --ent_open_scale=0.1 --gamma=0.6 --lr=0.01 --min_step=10000 --mlp_weight_decay=0.0005 --multi=0.2 --sigmaP=50 --source_data=./txt/source_Clipart_univ_zzl.txt --target_data=./txt/target_Real_univ_zzl.txt --v_latent=10

train_san.py --alpha=0.5 --augNearRate=10000 --aug_type=1 --batch_size=50 --beta=1.2 --config=configs/officehome-train-config_OPDA.yaml --data_aug_crop=0.7 --ent_open_scale=0.1 --gamma=0.6 --lr=0.01 --min_step=10000 --mlp_weight_decay=0.0005 --multi=0.2 --sigmaP=50 --source_data=./txt/source_Real_univ_zzl.txt --target_data=./txt/target_Product_univ_zzl.txt --v_latent=10

train_san.py --alpha=0.5 --augNearRate=10000 --aug_type=1 --batch_size=50 --beta=1.2 --config=configs/officehome-train-config_OPDA.yaml --data_aug_crop=0.7 --ent_open_scale=0.1 --gamma=0.6 --lr=0.01 --min_step=10000 --mlp_weight_decay=0.0005 --multi=0.2 --sigmaP=50 --source_data=./txt/source_Product_univ_zzl.txt --target_data=./txt/target_Art_univ_zzl.txt --v_latent=10

train_san.py --alpha=0.5 --augNearRate=10000 --aug_type=1 --batch_size=50 --beta=1.2 --config=configs/officehome-train-config_OPDA.yaml --data_aug_crop=0.7 --ent_open_scale=0.1 --gamma=0.6 --lr=0.01 --min_step=10000 --mlp_weight_decay=0.0005 --multi=0.2 --sigmaP=50 --source_data=./txt/source_Clipart_univ_zzl.txt --target_data=./txt/target_Art_univ_zzl.txt --v_latent=10

train_san.py --alpha=0.5 --augNearRate=10000 --aug_type=1 --batch_size=50 --beta=1.2 --config=configs/officehome-train-config_OPDA.yaml --data_aug_crop=0.7 --ent_open_scale=0.1 --gamma=0.6 --lr=0.01 --min_step=10000 --mlp_weight_decay=0.0005 --multi=0.2 --sigmaP=50 --source_data=./txt/source_Clipart_univ_zzl.txt --target_data=./txt/target_Product_univ_zzl.txt --v_latent=10

train_san.py --alpha=0.5 --augNearRate=10000 --aug_type=1 --batch_size=50 --beta=1.2 --config=configs/officehome-train-config_OPDA.yaml --data_aug_crop=0.7 --ent_open_scale=0.1 --gamma=0.6 --lr=0.01 --min_step=10000 --mlp_weight_decay=0.0005 --multi=0.2 --sigmaP=50 --source_data=./txt/source_Real_univ_zzl.txt --target_data=./txt/target_Clipart_univ_zzl.txt --v_latent=10

train_san.py --alpha=0.5 --augNearRate=10000 --aug_type=1 --batch_size=50 --beta=1.2 --config=configs/officehome-train-config_OPDA.yaml --data_aug_crop=0.7 --ent_open_scale=0.1 --gamma=0.6 --lr=0.01 --min_step=10000 --mlp_weight_decay=0.0005 --multi=0.2 --sigmaP=50 --source_data=./txt/source_Art_univ_zzl.txt --target_data=./txt/target_Clipart_univ_zzl.txt --v_latent=10

train_san.py --alpha=0.5 --augNearRate=10000 --aug_type=1 --batch_size=50 --beta=1.2 --config=configs/officehome-train-config_OPDA.yaml --data_aug_crop=0.7 --ent_open_scale=0.1 --gamma=0.6 --lr=0.01 --min_step=10000 --mlp_weight_decay=0.0005 --multi=0.2 --sigmaP=50 --source_data=./txt/source_Product_univ_zzl.txt --target_data=./txt/target_Clipart_univ_zzl.txt --v_latent=10
```

the results 

``` bash

Art -> Real : 86.93
Product -> Real : 82.78
Real ->  Art : 81.07
Art -> duct : 80.97
Clipart -> Real : 80.65
Real -> duct : 80.51
Product ->  Art : 75.7
Clipart ->  Art : 73.81
Clipart -> duct : 72.54
Real -> part : 66.27
Art -> part : 65.96
Product -> part : 63.07
mean hscore: 75.9
```




