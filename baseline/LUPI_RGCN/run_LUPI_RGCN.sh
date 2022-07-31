#! bin/bash
# 生成0到10，作为-se参数调用LUPI_RGCN_wzy.py文件
for i in {0..9}
do
    # python baseline/LUPI_RGCN/LUPI_RGCN_wzy.py -se $i -md train 
    python baseline/LUPI_RGCN/LUPI_RGCN_wzy.py -se $i -md findNew -maskType gene -savePath uk_gd
    # python baseline/LUPI_RGCN/LUPI_RGCN_wzy.py -se $i -md findNew -maskType dis -savePath uk_gd
    # echo $i
done
