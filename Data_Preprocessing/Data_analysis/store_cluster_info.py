"""
This code can be used to extract frequency information about the clusters in a CoNLL-2021 corpus -- in order to do a further analysis.
The information is stored in 1 JSON file

The code is based on a script by as Dobrovolskii (2021), see https://github.com/gvanboven/wl-coref/tree/main

Use this code as follows:
python get_pos_data.py path/to/coref/data /path/to/output.json

In the output json file, 3 dictionaries are stored
- "refs_per_sent"  : {filename : number_of_referents_in_file}
- "clusters_per_doc"   : {filename: number_of_unique_clusters}
- "refs_per_cluster"   : {filename: {cluster : number_of_referents}}
"""

import argparse
import os
import re
import json
from typing import Dict, Generator, List, Tuple
from collections import defaultdict

DATA_SPLITS = ["dev", "test", "train"]
SENT_PATTERN = re.compile(r"(?:^(?!#).+$\n?)+", flags=re.M) # changed to fit the SoNaR data


def get_filenames(path: str) -> Generator[str, None, None]:
    """
    Yields all filenames in a directory in a recursive manner.
    """
    for filename in sorted(os.listdir(path)):

        full_filename = os.path.join(path, filename)
        if os.path.isdir(full_filename):
            yield from get_filenames(full_filename)
        else:
            yield full_filename

def get_conll_filenames(data_dir: str) -> Dict[str, List[str]]:
    """
    Returns a dictionary {data_split: [filename, ...], ...}, where data_split
    is one of "development", "test", "train" and filename is
    a full path to _gold_conll file
    """
    conll_filenames = {}
    for data_split in DATA_SPLITS:
        data_split_dir = os.path.join(data_dir, data_split)# adapted for SoNaR
        conll_filenames[data_split] = [
            filename for filename in get_filenames(data_split_dir)
            ]
    return conll_filenames

def store_doc_lengths(conll_filenames : List[str], outpath : str) -> None:
    """
    Obtains the cluster data from the corpus and stores this information in the output JSON file
    """
    referents_per_sent = defaultdict(lambda : 0)
    clusters_per_doc = {}
    refs_per_cluster = defaultdict(lambda : 0)

    for split, filelist in conll_filenames.items():
        for filename in filelist:
            with open(filename, mode="r", encoding="utf-8") as f:
                sents = re.findall(SENT_PATTERN, f.read())

            clusters = []
            for sent_id, sent in enumerate(sents):
                n_referents = 0 
                
                for word in sent.splitlines():
                    corefinfo = word.split()[-1]

                    refs = re.findall(re.compile(r"\([0-9]+", flags=re.M), corefinfo)
                    n_referents += len(refs)

                    for ref in refs:
                        refs_per_cluster[f"{filename}_{ref.replace('(','')}"] += 1 

                    cluster_info = corefinfo.translate({ord(i): None for i in '()-'}).split('|')
                    if cluster_info != ['']:
                        clusters.extend([int(c) for c in cluster_info])
                
                referents_per_sent[f'{filename}_{sent_id}'] = n_referents
                
            unique_clusters = set(clusters)
            clusters_per_doc[filename] = len(unique_clusters)

    with open(outpath, "w") as outfile:
        json.dump({'refs_per_sent' : referents_per_sent, 
                   'clusters_per_doc' : clusters_per_doc,
                   'refs_per_cluster' : refs_per_cluster}, outfile)
    
            

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(
        description="Converts conll-formatted files to json.")
    argparser.add_argument("conll_dir", help="The root directory of"
                           " conll-formatted OntoNotes corpus.")
    argparser.add_argument("outpath", default="lassy", help="The path"
                           " the the .json file where the doc lengths should be stored")

    args = argparser.parse_args()
    
    
    #open coref and dependency data
    conll_filenames = get_conll_filenames(args.conll_dir)
   
    #extract and store document length data
    store_doc_lengths(conll_filenames, args.outpath)
