"""
This code can be used to extract frequency information about the POStags in a CoNLL-2021 corpus -- in order to do a further analysis.
The information is stored in 6 jsonfiles

The code is based on a script by as Dobrovolskii (2021), see https://github.com/gvanboven/wl-coref/tree/main

Use this code as follows:
python get_pos_data.py path/to/coref/data /path/to/syntactic/data

It stores the following documents:
- "poscounts.json"  : {POS : freq}
- "postoken.json"   : {POS: {token :freq}}
- "tokenpos.json"   : {token: {POS : freq}}
- "tokenposdata"    : {token : {POS: {POStag: freq}}}
- "postokendata"    : {POS : {token: {POStag: freq}}}
- "posdata"         : {POS : {POStag: freq}}}
"""

import argparse
import os
import re
import json
from typing import Dict, Generator, List, Tuple
from collections import defaultdict
  

DATA_SPLITS = ["dev", "test", "train"]
DEPS_FILENAME = "deps.conllu"
DEPS_IDX_FILENAME = "deps.index"
DEP_SENT_PATTERN = re.compile(r"(?:^\d.+$\n?)+", flags=re.M)
SENT_PATTERN = re.compile(r"(?:^(?!#).+$\n?)+", flags=re.M) # changed to fit the SoNaR data
WORD_PATTERN = re.compile(r"(?:^(?!#).+$\n?)", flags=re.M) 


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

def get_lassy_filenames(lassy_dir: str, conll_filenames: Dict) -> Tuple[Dict[str, List[str]], Dict[str, str]]:
    """
    Returns 
    1. a dictionary {data_split: [filename, ...], ...}, where data_split
    is one of "development", "test", "train" and filename is
    a full path to a lassy file
    2. a dictionary {lassy_filename : conll_filename, ...} where lassy_filename 
    is the path to a lassy file and conll_filename is the path to its corresponding
    conll file with coref annotation
    """
    lassy_filenames, coref_lassy_alignment = {}, {}
    for data_split, filenames in conll_filenames.items():
        data_split_files = []
        for filename in filenames:
            fname = os.path.basename(filename) + 'u'
            lassy_filename = os.path.join(lassy_dir, fname)
            coref_lassy_alignment[filename] = lassy_filename
            data_split_files.append(lassy_filename)
        lassy_filenames[data_split] = data_split_files
    return lassy_filenames, coref_lassy_alignment

def extract_data(conll_filenames : List[str], coref_lassy_alignment : Dict [str, str]) -> None:
    """
    Creates 6 frequency dictionaries, to count POS-related frequencies.
    Goes through the corpus to obtain frequency counts
    Then stores the dicts in JSON files.

    """
    doc_len_error_counter = 0
    
    POStoken = defaultdict(lambda: defaultdict(lambda : 0))
    tokenPOS = defaultdict(lambda: defaultdict(lambda : 0))
    POScounts = defaultdict(lambda : 0)
    tokenPOSdata = defaultdict(lambda: defaultdict(lambda : defaultdict(lambda : 0)))
    POStokendata = defaultdict(lambda: defaultdict(lambda : defaultdict(lambda : 0)))
    POSdata = defaultdict(lambda : defaultdict(lambda : 0))

    for split, filelist in conll_filenames.items():
        for filename in filelist:
            with open(filename, mode="r", encoding="utf-8") as f:
                sents = re.findall(SENT_PATTERN, f.read())
            with open(coref_lassy_alignment[filename], mode="r", encoding="utf-8") as f:
                dep_sents = re.findall(DEP_SENT_PATTERN, f.read())
            with open(filename, mode="r", encoding="utf-8") as f:
                words = re.findall(WORD_PATTERN, f.read())

            if len(dep_sents) == len(sents):
                for sent_id, sources in enumerate(zip(sents, dep_sents)):
                    sent, parsed_sent = [s.splitlines() for s in sources]
                    try: 
                        assert len(sent) == len(parsed_sent)
                    except:
                        print(f"ASSERTIONERROR sent length for file {filename}, skipping this file")
                        return {}

                    #obtain document frequencies
                    for s_word, p_word in zip(sent, parsed_sent):
                        s_cols = s_word.split()
                        p_cols = p_word.split('\t')

                        word = s_cols[3].lower()
                        pos = p_cols[3] 
                        postag = p_cols[4]

                        POScounts[pos] += 1
                        POStoken[pos][word] += 1
                        tokenPOS[word][pos] += 1
                        tokenPOSdata[word][pos][postag] += 1
                        POStokendata[pos][word][postag] += 1
                        POSdata[pos][postag] += 1
            else:
                doc_len_error_counter += 1
                print(filename, len(dep_sents), len(sents), len(words))
    print(f"doc len error for {doc_len_error_counter} documents")

    #store dicts in JSON files
    with open("poscounts.json", "w") as outfile:
        json.dump(POScounts, outfile)
    with open("postoken.json", "w") as outfile:
        json.dump(POStoken, outfile)
    with open("tokenpos.json", "w") as outfile:
        json.dump(tokenPOS, outfile)
    with open("tokenposdata.json", "w") as outfile:
        json.dump(tokenPOSdata, outfile)
    with open("postokendata.json", "w") as outfile:
        json.dump(POStokendata, outfile)
    with open("posdata.json", "w") as outfile:
        json.dump(POSdata, outfile)        

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(
        description="Converts conll-formatted files to json.")
    argparser.add_argument("conll_dir", help="The root directory of"
                           " conll-formatted OntoNotes corpus.")
    argparser.add_argument("lassy_dir", default="lassy", help="The root directory of"
                           " lassy dependency files in conllu format.")

    args = argparser.parse_args()
    
    #open coref and dependency data
    conll_filenames = get_conll_filenames(args.conll_dir)
    lassy_filenames, coref_lassy_alignment = get_lassy_filenames(args.lassy_dir, conll_filenames) #added to align lassy and coref data
    
    #extract POS data
    extract_data(conll_filenames, coref_lassy_alignment)
