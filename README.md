# gene_dis_predict_by_ML

该项目主要是论文的代码存储库

## 运行项目

首先cd到目标该项目目录下，然后运行`run.sh`

```SH
cd YOUR_PAHT/gene_dis_predict_by_ML
python src/model/main.py
```

## 项目主要包含如下文件夹
* baseline：其中存放我们论文中提及的baseline模型的代码，包括对数据的处理和建模。
* data：主要存放我们自己提取的数据，在应用之前，需要从[Disease_gene_prioritization_GCN](https://github.com/liyu95/Disease_gene_prioritization_GCN/tree/af763c0ea291406da89edbe92525edb79a03c69a/data_prioritization)或者[Disease-Gene-Prioritization-with-Privileged-Information-and-Heteroscedastic-Dropout](https://github.com/juanshu30/Disease-Gene-Prioritization-with-Privileged-Information-and-Heteroscedastic-Dropout)项目中的data处链接获取剩余的三个文件`clinicalfeatures_tfidf.mat`，`genes_phenes.mat`，`GeneFeatures.mat`

    * 我们在data中提供了基于CTD数据库的基因疾病关联文件`geneDisAdj_sum.pkl`，文件以crs矩阵格式存储。在论文的研究中，我们用新的基因疾病关联（22504条链接）替换了原有的关联（3954条链接），在数据对齐当中，有的极少部分的数据并没有对齐，为了避免冲突，我们在原始数据的基础上，仅替换了对齐的数据，对于没有对齐的数据则以原始数据替换，从而保持了原有数据的形状(12331*3215)
    * 我们还提供了源数据中OMIM的疾病ID`diseaseID.txt`和NCBI的基因ID`EntrezID.txt`文件
    * 最后我们提供了我们合并之后的数据`gg.npz`，`gd.npz`，`dd.npz`，以及特征数据
    * 具体的细节可以参见论文（）

* src：主要存放论文中提到的模型方法
    * model
        * utils.py
        * args.py
        * load_data.py
        * feature_extration.py
        * machineLearning.py
        * main.py
        * main_analysis.py
    * result_analysi.ipynb
    * run.sh
* OpenNE：原项目见[OpenNE](https://github.com/thunlp/OpenNE),我们在其中添加了便于运行的shell脚本。
* baseline：该文件夹内保存着相关baseline的实现代码

# 更多的内容










