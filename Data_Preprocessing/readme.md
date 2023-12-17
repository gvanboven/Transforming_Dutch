## wl-coref preprocessing steps

This dir contains the code used to preprocess the SoNaR-1 dataset for the wl-coref model. 

To preprocess the data before training the wl-coref model, take the following steps: 
```
python mmaxconll.py <inputdir> <outputdir> #transform the MMAX data to the CONLL format
python truncate_data.py <inputdir> # I had to partition the largest files due to memory constraints
python exclude_singletons.py <inputdir> # exclude singletons from the data
python data_transformation.py <inputdir> [settings] #optional: transform the data for debiasing
python convert_to_jsonlines.py <inputdir>  #transform the CONLL formatted data to the required jsonlines format
python convert_to_heads.py <inputdir> #extract the heads from the jsonlines files
```