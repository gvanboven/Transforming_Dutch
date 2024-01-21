## wl-coref preprocessing steps

This dir contains the code used to preprocess the SoNaR-1 dataset for the wl-coref model. 

To preprocess the data before training the wl-coref model, take the following steps (in order): 
```
python mmaxconll.py <inputdir> <outputdir> #transform the MMAX data to the CONLL format
python truncate_data.py <inputdir> # we had to partition the largest files due to memory constraints
python exclude_singletons.py <inputdir> # exclude singletons from the data
python data_transformation.py <inputdir> [settings] #optional: transform the data for debiasing/ evaluation
python convert_to_jsonlines.py <inputdir>  #transform the CONLL formatted data to the required jsonlines format
python convert_to_heads.py <inputdir> #extract the heads from the jsonlines files
```

We use code by Poot and van Cranenburgh (2020) (`mmaxconll.py` MMAX -> CONLL) and  Dobrovolskii (2021) (`convert_to_jsonlines.py` and `convert_to_heads.py` CONLL -> jsonlines) to transform the SoNaR-1 corpus, which originally is MMAX format into the jsonlines format required by the wl-coref model.

The 25G Quadro RTX 6000 GPU0 used for this project encounters memory constraints for the longest documents. We therefore partition all documents (`truncate_data.py`) exceeding 3,500 tokens into files of uniform sizes, each containing fewer than 3,500 tokens. In total, 54 out of 861 documents require partitioning, with 43 documents divided into two segments, seven documents divided into three segments, and four documents divided into four or more segments. 
For some mention-antecedent pairs, the partitioning of documents results in a division of the pair over two separate documents. This likely affects the LEA performance scores, as some of the clusters have changed. But, as the data is partitioned in exactly the same way across all the models considered within this study, all models are affected by the partitioning equally.

The construction of pronoun-specific data, as well as our debiasing data are performed through `data_transformation.py`. 