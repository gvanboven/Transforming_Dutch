"""
This code can be used to extract document length information CoNLL-2021 corpus -- in order to do a further analysis.
The information is stored in 1 JSON file

The code is based on a script by as Dobrovolskii (2021), see https://github.com/gvanboven/wl-coref/tree/main

Use this code as follows:
python get_pos_data.py path/to/coref/data /path/to/output.json

In the output JSON file, information is stores in a dictionary as: {fileame : number_of_sentences}
"""

import argparse
import os
import re
import json
from typing import Dict, Generator, List, Tuple

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
    Computes the number of sentences per documents and stores these values in 
    the output .json file, as a dictionary with {filename : str, n_sents : int}
    """
    doc_lens = {}
    for split, filelist in conll_filenames.items():
        for filename in filelist:
            with open(filename, mode="r", encoding="utf-8") as f:
                sents = re.findall(SENT_PATTERN, f.read())
                n_sents = len(sents)
            doc_lens[filename] = n_sents
    with open(outpath, "w") as outfile:
        json.dump(doc_lens, outfile)
    
            

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
