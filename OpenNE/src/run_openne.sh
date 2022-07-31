#!/bin/bash
# gcn lap gf tadw, need feature graph
# grarep has some bugs

# TODO:os.system()不能运行conda activate 命令
# conda activate openne
for model in deepwalk line node2vec  hope sdne;do 
    echo "====== running ${model} ======"
    python -m openne --model ${model} --local-dataset --root-dir  OpenNE/GeneDis --adjfile adj_gg.adjlist --dim 128
    python -m openne --model ${model} --local-dataset --root-dir  OpenNE/GeneDis --adjfile adj_dd.adjlist --dim 128
done

