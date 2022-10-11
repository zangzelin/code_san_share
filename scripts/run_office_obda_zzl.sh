#!/bin/bash
python train.py  --config configs/office-train-config_ODA.yaml --source ./txt/source_amazon_obda_zzl.txt --target ./txt/target_dslr_obda_zzl.txt --gpu 1 &
python train.py  --config configs/office-train-config_ODA.yaml --source ./txt/source_amazon_obda_zzl.txt --target ./txt/target_webcam_obda_zzl.txt --gpu 2&
python train.py  --config configs/office-train-config_ODA.yaml --source ./txt/source_webcam_obda_zzl.txt --target ./txt/target_amazon_obda_zzl.txt --gpu 3&
python train.py  --config configs/office-train-config_ODA.yaml --source ./txt/source_dslr_obda_zzl.txt --target ./txt/target_amazon_obda_zzl.txt --gpu 4&
python train.py  --config configs/office-train-config_ODA.yaml --source ./txt/source_dslr_obda_zzl.txt --target ./txt/target_webcam_obda_zzl.txt --gpu 5&
python train.py  --config configs/office-train-config_ODA.yaml --source ./txt/source_webcam_obda_zzl.txt --target ./txt/target_dslr_obda_zzl.txt --gpu 6
