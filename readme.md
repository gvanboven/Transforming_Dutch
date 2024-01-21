## Transforming Dutch: Debiasing Dutch Corefence Resolution Systems for Non-Binary Pronouns

This repository countains the code for our paper "Transforming Dutch: Debiasing Dutch Corefence Resolution Systems for Non-Binary Pronouns", submitted to FAccT 2024.

### Abstract
Gender-neutral pronouns are increasingly being introduced across Western languages, and are continuously more frequently being adopted by non-binary individuals. Recent evaluations have however demonstrated that English language models and coreference resolution systems are unable to correctly process gender-neutral pronouns (Cao and Daumé III, 2021; Baumler and Rudinger, 2022; Dev et al., 2021), which carries the risk of causing harmful consequences such as erasing and misgendering non-binary individuals (Dev et al.,2021). This thesis pioneers an examination of a Dutch language model’s performance on gender-neutral pronouns, specifically hen and die. In the Dutch context, additional challenges arise from the relative novelty of these pronouns, introduced in 2016, compared to the longstanding existence of the singular they in English. To carry out this evaluation, a novel Dutch neural coreference model is published, and an innovative evaluation metric, a pronoun score, is introduced, which directly represents the percentage of correctly processed pronouns. The results reveal diminished performance on gender-neutral pronouns compared to gendered counterparts. In response to these challenges, this study compares, as a first of its kind, the usage of two debiasing techniques in non-binary contexts: Counterfactual Data Augmentation (CDA) and delexicalisation (Lauscher et al., 2022). Although delexicalisation fails to yield improvement, CDA significantly diminishes the
performance gap between gendered and gender-neutral pronouns. A noteworthy contribution is the demonstration that debiasing remains effective even in low-resource settings, where only a limited set of debiasing documents is applied. This efficacy extends to previously unseen neopronouns, which are currently infrequently used but may gain popularity in the future. This underscores the viability of effective debiasing with minimal resources and low computational costs.

### Repository structure 
The `wl-coref` directory contains the code for the wl-coref model by Dobrovolskii (2021) : https://github.com/vdobrovolskii/wl-coref. The code was directly copied, with some small adaptations for the Dutch data marked in the files. 

The `pronouns_score.py` provides the implementation for the **pronoun score**. The code should work for documents in the CoNLL-2012 format. 

The `Data_Preprocessing` directory contains the preprocessing code used for the SoNaR-1 corpus, in order to use this corpus for training the wl-coref model. Moreover, this dir contains the data transformation code used to create the CDA and delexicalisation debiasing data.

The `Test suite ` dir contains an additional test suite we created for pronoun-related behaviour.

Finally, the `notebooks` dir contains the files in which we analysed the results. 
