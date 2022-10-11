#!/bin/bash
python train.py  --config configs/visda-train-config_UDA.yaml --source txt/source_list_univ_zzl.txt --target txt/target_list_univ_zzl.txt --gpu 0
