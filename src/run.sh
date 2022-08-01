#!/bin/bash

for i in {0..9}
do 
    python src/model/main.py -ah 512 64 -id 1 -md findNew -fo 5 -se $i -sk True -skae True -skopenne True -om SDNE HOPE -maskType dis -savePath uk_gd
done

