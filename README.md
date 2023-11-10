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
python train_san.py --alpha=0.7 --augNearRate=10000 --aug_type=1 --beta=1.4 --config=configs/office-train-config_OPDA.yaml --data_aug_crop=0.8 --ent_open_scale=0.4 --gamma=0.6 --source_data=./txt/source_webcam_opda_zzl.txt --target_data=./txt/target_dslr_opda_zzl.txt
```

# opda 
## domainet

```
wandb sweep sweep/new_opda/domainnet_v1.yaml
wandb agent */OVANET_DMT/***
```
The sweep results is in http://www.zangzelin.fun:4080/zangzelin/OVANET_DMT/sweeps/yu5k29s9?workspace=user-zangzelin

run 1:
dpainting -> dreal : 0.5838
dsketch -> dreal : 0.5793
dreal -> dpainting : 0.5293
dsketch -> dpainting : 0.4791
dpainting -> dsketch : 0.4758
dreal -> dsketch : 0.4671

run 2:
dpainting -> dreal: 0.58
dsketch -> dreal: 0.5823
dreal -> dpainting: 0.5283
dsketch -> dpainting: 0.4822
dpainting -> dsketch: 0.4738
dreal -> dsketch: 0.4647
