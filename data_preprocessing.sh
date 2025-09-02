#!/bin/bash

echo "Data preprocessing started..."

cd data

python3 01pre_processing_siloracle.py
python3 02pre_processing_active.py

rm siloactive_F5.csv siloactive_KHK.csv

cd ..

echo "Data preprocessing completed"