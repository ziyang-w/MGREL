#!/bin/bash
# python src/model/load_data.py
# python src/model/feature_extraction.py

# sh OpenNE/src/run_model.sh

# python src/ml.py
# echo hello world

for i in {0..9}
do 
    python src/model/main.py -ah 512 64 -id 1 -md findNew -fo 5 -se $i -sk True -skae True -skopenne True -om SDNE HOPE -maskType dis -savePath uk_gd
done

