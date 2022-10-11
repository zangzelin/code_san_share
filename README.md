## [RSCL: Noise-Resistant for Universal Domain Adaptation]

This repository provides code for RSCL.

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
python train_rscl.py --alpha=0.7 --augNearRate=10000 --aug_type=1 --beta=1.4 --config=configs/office-train-config_OPDA.yaml --data_aug_crop=0.8 --ent_open_scale=0.4 --gamma=0.6 --source_data=./txt/source_webcam_opda_zzl.txt --target_data=./txt/target_dslr_opda_zzl.txt
```