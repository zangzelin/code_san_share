#!/bin/bash
python train.py  --v_latent 1 --sigmaP 10 --config configs/officehome-train-config_ODA.yaml --source ./txt/source_Real_obda_zzl.txt --target ./txt/target_Clipart_obda_zzl.txt --gpu 0 &  
python train.py  --v_latent 1 --sigmaP 10 --config configs/officehome-train-config_ODA.yaml --source ./txt/source_Real_obda_zzl.txt --target ./txt/target_Art_obda_zzl.txt --gpu 1 & 
python train.py  --v_latent 1 --sigmaP 10 --config configs/officehome-train-config_ODA.yaml --source ./txt/source_Real_obda_zzl.txt --target ./txt/target_Product_obda_zzl.txt --gpu 2 & 
python train.py  --v_latent 1 --sigmaP 10 --config configs/officehome-train-config_ODA.yaml --source ./txt/source_Product_obda_zzl.txt --target ./txt/target_Art_obda_zzl.txt --gpu 3 & 
python train.py  --v_latent 1 --sigmaP 10 --config configs/officehome-train-config_ODA.yaml --source ./txt/source_Product_obda_zzl.txt --target ./txt/target_Clipart_obda_zzl.txt --gpu 4 & 
python train.py  --v_latent 1 --sigmaP 10 --config configs/officehome-train-config_ODA.yaml --source ./txt/source_Product_obda_zzl.txt --target ./txt/target_Real_obda_zzl.txt --gpu 5 & 
python train.py  --v_latent 1 --sigmaP 10 --config configs/officehome-train-config_ODA.yaml --source ./txt/source_Clipart_obda_zzl.txt --target ./txt/target_Real_obda_zzl.txt --gpu 6 & 
python train.py  --v_latent 1 --sigmaP 10 --config configs/officehome-train-config_ODA.yaml --source ./txt/source_Clipart_obda_zzl.txt --target ./txt/target_Art_obda_zzl.txt --gpu 7 & 
python train.py  --v_latent 1 --sigmaP 10 --config configs/officehome-train-config_ODA.yaml --source ./txt/source_Clipart_obda_zzl.txt --target ./txt/target_Product_obda_zzl.txt --gpu 0 & 
python train.py  --v_latent 1 --sigmaP 10 --config configs/officehome-train-config_ODA.yaml --source ./txt/source_Art_obda_zzl.txt --target ./txt/target_Product_obda_zzl.txt --gpu 1 & 
python train.py  --v_latent 1 --sigmaP 10 --config configs/officehome-train-config_ODA.yaml --source ./txt/source_Art_obda_zzl.txt --target ./txt/target_Real_obda_zzl.txt --gpu 2 & 
python train.py  --v_latent 1 --sigmaP 10 --config configs/officehome-train-config_ODA.yaml --source ./txt/source_Art_obda_zzl.txt --target ./txt/target_Clipart_obda_zzl.txt --gpu 3

python train.py  --v_latent 2 --sigmaP 10 --config configs/officehome-train-config_ODA.yaml --source ./txt/source_Real_obda_zzl.txt --target ./txt/target_Clipart_obda_zzl.txt gpu 0 &  
python train.py  --v_latent 2 --sigmaP 10 --config configs/officehome-train-config_ODA.yaml --source ./txt/source_Real_obda_zzl.txt --target ./txt/target_Art_obda_zzl.txt --gpu 1 &
python train.py  --v_latent 2 --sigmaP 10 --config configs/officehome-train-config_ODA.yaml --source ./txt/source_Real_obda_zzl.txt --target ./txt/target_Product_obda_zzl.txt --gpu 2 &
python train.py  --v_latent 2 --sigmaP 10 --config configs/officehome-train-config_ODA.yaml --source ./txt/source_Product_obda_zzl.txt --target ./txt/target_Art_obda_zzl.txt --gpu 3 &
python train.py  --v_latent 2 --sigmaP 10 --config configs/officehome-train-config_ODA.yaml --source ./txt/source_Product_obda_zzl.txt --target ./txt/target_Clipart_obda_zzl.txt --gpu 4 &
python train.py  --v_latent 2 --sigmaP 10 --config configs/officehome-train-config_ODA.yaml --source ./txt/source_Product_obda_zzl.txt --target ./txt/target_Real_obda_zzl.txt --gpu 5 &
python train.py  --v_latent 2 --sigmaP 10 --config configs/officehome-train-config_ODA.yaml --source ./txt/source_Clipart_obda_zzl.txt --target ./txt/target_Real_obda_zzl.txt --gpu 6 &
python train.py  --v_latent 2 --sigmaP 10 --config configs/officehome-train-config_ODA.yaml --source ./txt/source_Clipart_obda_zzl.txt --target ./txt/target_Art_obda_zzl.txt --gpu 7 &
python train.py  --v_latent 2 --sigmaP 10 --config configs/officehome-train-config_ODA.yaml --source ./txt/source_Clipart_obda_zzl.txt --target ./txt/target_Product_obda_zzl.txt --gpu 0 &
python train.py  --v_latent 2 --sigmaP 10 --config configs/officehome-train-config_ODA.yaml --source ./txt/source_Art_obda_zzl.txt --target ./txt/target_Product_obda_zzl.txt --gpu 1 &
python train.py  --v_latent 2 --sigmaP 10 --config configs/officehome-train-config_ODA.yaml --source ./txt/source_Art_obda_zzl.txt --target ./txt/target_Real_obda_zzl.txt --gpu 2 &
python train.py  --v_latent 2 --sigmaP 10 --config configs/officehome-train-config_ODA.yaml --source ./txt/source_Art_obda_zzl.txt --target ./txt/target_Clipart_obda_zzl.txt --gpu 0

python train.py  --v_latent 5 --sigmaP 10 --config configs/officehome-train-config_ODA.yaml --source ./txt/source_Real_obda_zzl.txt --target ./txt/target_Clipart_obda_zzl.txt gpu 0 &  
python train.py  --v_latent 5 --sigmaP 10 --config configs/officehome-train-config_ODA.yaml --source ./txt/source_Real_obda_zzl.txt --target ./txt/target_Art_obda_zzl.txt --gpu 1 &
python train.py  --v_latent 5 --sigmaP 10 --config configs/officehome-train-config_ODA.yaml --source ./txt/source_Real_obda_zzl.txt --target ./txt/target_Product_obda_zzl.txt --gpu 2 &
python train.py  --v_latent 5 --sigmaP 10 --config configs/officehome-train-config_ODA.yaml --source ./txt/source_Product_obda_zzl.txt --target ./txt/target_Art_obda_zzl.txt --gpu 3 &
python train.py  --v_latent 5 --sigmaP 10 --config configs/officehome-train-config_ODA.yaml --source ./txt/source_Product_obda_zzl.txt --target ./txt/target_Clipart_obda_zzl.txt --gpu 4 &
python train.py  --v_latent 5 --sigmaP 10 --config configs/officehome-train-config_ODA.yaml --source ./txt/source_Product_obda_zzl.txt --target ./txt/target_Real_obda_zzl.txt --gpu 5 &
python train.py  --v_latent 5 --sigmaP 10 --config configs/officehome-train-config_ODA.yaml --source ./txt/source_Clipart_obda_zzl.txt --target ./txt/target_Real_obda_zzl.txt --gpu 6 &
python train.py  --v_latent 5 --sigmaP 10 --config configs/officehome-train-config_ODA.yaml --source ./txt/source_Clipart_obda_zzl.txt --target ./txt/target_Art_obda_zzl.txt --gpu 7 &
python train.py  --v_latent 5 --sigmaP 10 --config configs/officehome-train-config_ODA.yaml --source ./txt/source_Clipart_obda_zzl.txt --target ./txt/target_Product_obda_zzl.txt --gpu 0 &
python train.py  --v_latent 5 --sigmaP 10 --config configs/officehome-train-config_ODA.yaml --source ./txt/source_Art_obda_zzl.txt --target ./txt/target_Product_obda_zzl.txt --gpu 1 &
python train.py  --v_latent 5 --sigmaP 10 --config configs/officehome-train-config_ODA.yaml --source ./txt/source_Art_obda_zzl.txt --target ./txt/target_Real_obda_zzl.txt --gpu 2 &
python train.py  --v_latent 5 --sigmaP 10 --config configs/officehome-train-config_ODA.yaml --source ./txt/source_Art_obda_zzl.txt --target ./txt/target_Clipart_obda_zzl.txt --gpu 0

python train.py  --v_latent 0.5 --sigmaP 10 --config configs/officehome-train-config_ODA.yaml --source ./txt/source_Real_obda_zzl.txt --target ./txt/target_Clipart_obda_zzl.txt gpu 0 &  
python train.py  --v_latent 0.5 --sigmaP 10 --config configs/officehome-train-config_ODA.yaml --source ./txt/source_Real_obda_zzl.txt --target ./txt/target_Art_obda_zzl.txt --gpu 1 &
python train.py  --v_latent 0.5 --sigmaP 10 --config configs/officehome-train-config_ODA.yaml --source ./txt/source_Real_obda_zzl.txt --target ./txt/target_Product_obda_zzl.txt --gpu 2 &
python train.py  --v_latent 0.5 --sigmaP 10 --config configs/officehome-train-config_ODA.yaml --source ./txt/source_Product_obda_zzl.txt --target ./txt/target_Art_obda_zzl.txt --gpu 3 &
python train.py  --v_latent 0.5 --sigmaP 10 --config configs/officehome-train-config_ODA.yaml --source ./txt/source_Product_obda_zzl.txt --target ./txt/target_Clipart_obda_zzl.txt --gpu 4 &
python train.py  --v_latent 0.5 --sigmaP 10 --config configs/officehome-train-config_ODA.yaml --source ./txt/source_Product_obda_zzl.txt --target ./txt/target_Real_obda_zzl.txt --gpu 5 &
python train.py  --v_latent 0.5 --sigmaP 10 --config configs/officehome-train-config_ODA.yaml --source ./txt/source_Clipart_obda_zzl.txt --target ./txt/target_Real_obda_zzl.txt --gpu 6 &
python train.py  --v_latent 0.5 --sigmaP 10 --config configs/officehome-train-config_ODA.yaml --source ./txt/source_Clipart_obda_zzl.txt --target ./txt/target_Art_obda_zzl.txt --gpu 7 &
python train.py  --v_latent 0.5 --sigmaP 10 --config configs/officehome-train-config_ODA.yaml --source ./txt/source_Clipart_obda_zzl.txt --target ./txt/target_Product_obda_zzl.txt --gpu 0 &
python train.py  --v_latent 0.5 --sigmaP 10 --config configs/officehome-train-config_ODA.yaml --source ./txt/source_Art_obda_zzl.txt --target ./txt/target_Product_obda_zzl.txt --gpu 1 &
python train.py  --v_latent 0.5 --sigmaP 10 --config configs/officehome-train-config_ODA.yaml --source ./txt/source_Art_obda_zzl.txt --target ./txt/target_Real_obda_zzl.txt --gpu 2 &
python train.py  --v_latent 0.5 --sigmaP 10 --config configs/officehome-train-config_ODA.yaml --source ./txt/source_Art_obda_zzl.txt --target ./txt/target_Clipart_obda_zzl.txt --gpu 0

python train.py  --v_latent 0.1 --sigmaP 10 --config configs/officehome-train-config_ODA.yaml --source ./txt/source_Real_obda_zzl.txt --target ./txt/target_Clipart_obda_zzl.txt --gpu 0 &
python train.py  --v_latent 0.1 --sigmaP 10 --config configs/officehome-train-config_ODA.yaml --source ./txt/source_Real_obda_zzl.txt --target ./txt/target_Art_obda_zzl.txt --gpu 1 &
python train.py  --v_latent 0.1 --sigmaP 10 --config configs/officehome-train-config_ODA.yaml --source ./txt/source_Real_obda_zzl.txt --target ./txt/target_Product_obda_zzl.txt --gpu 2 &
python train.py  --v_latent 0.1 --sigmaP 10 --config configs/officehome-train-config_ODA.yaml --source ./txt/source_Product_obda_zzl.txt --target ./txt/target_Art_obda_zzl.txt --gpu 3 &
python train.py  --v_latent 0.1 --sigmaP 10 --config configs/officehome-train-config_ODA.yaml --source ./txt/source_Product_obda_zzl.txt --target ./txt/target_Clipart_obda_zzl.txt --gpu 4 &
python train.py  --v_latent 0.1 --sigmaP 10 --config configs/officehome-train-config_ODA.yaml --source ./txt/source_Product_obda_zzl.txt --target ./txt/target_Real_obda_zzl.txt --gpu 5 &
python train.py  --v_latent 0.1 --sigmaP 10 --config configs/officehome-train-config_ODA.yaml --source ./txt/source_Clipart_obda_zzl.txt --target ./txt/target_Real_obda_zzl.txt --gpu 6 &
python train.py  --v_latent 0.1 --sigmaP 10 --config configs/officehome-train-config_ODA.yaml --source ./txt/source_Clipart_obda_zzl.txt --target ./txt/target_Art_obda_zzl.txt --gpu 7 & 
python train.py  --v_latent 0.1 --sigmaP 10 --config configs/officehome-train-config_ODA.yaml --source ./txt/source_Clipart_obda_zzl.txt --target ./txt/target_Product_obda_zzl.txt --gpu 0 &
python train.py  --v_latent 0.1 --sigmaP 10 --config configs/officehome-train-config_ODA.yaml --source ./txt/source_Art_obda_zzl.txt --target ./txt/target_Product_obda_zzl.txt --gpu 1 &
python train.py  --v_latent 0.1 --sigmaP 10 --config configs/officehome-train-config_ODA.yaml --source ./txt/source_Art_obda_zzl.txt --target ./txt/target_Real_obda_zzl.txt --gpu 2 &
python train.py  --v_latent 0.1 --sigmaP 10 --config configs/officehome-train-config_ODA.yaml --source ./txt/source_Art_obda_zzl.txt --target ./txt/target_Clipart_obda_zzl.txt --gpu 0
