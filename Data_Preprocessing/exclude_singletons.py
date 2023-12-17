"""
This code can be used to remove all singleton annotations from a coreference resolution corpus in CoNLL-2012 format.
The code is based on a script by as Dobrovolskii (2021), see https://github.com/gvanboven/wl-coref/tree/main
Use this code as follows:
python exclude_singletons.py <corpus-path>
The new dataset (without singleton annotations) will be stored at <corpus_path>_singletons_excld.conll
"""

import argparse
import os
import re
import sys
import shutil
from typing import Dict, Generator, List
from collections import defaultdict

import re

DATA_SPLITS = ["dev", "test", "train"]

SENT_PATTERN = re.compile(r"(?:^(?!#).+$\n?)+", flags=re.M) 
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

def extract_singletons(words: List[str]) -> List[str]:
    """
    Goes through all rows in a CoNLL file and stores the number of referents of each cluster in the document.
    Returns a list of singleton clusters, i.e. clusters with only 1 mention.
    """
    cluster_counts =  defaultdict(lambda: 0)
    stack = []
    for word in words:
        cluster = word.split()[11]
        if not cluster == '-':
            #regocognise 1-word mentions
            full_mention = re.findall('\([0-9]+\)', cluster)
            for mention in full_mention:
                cluster_counts[re.findall('[0-9]+', mention)[0]] += 1
                cluster = cluster.replace(mention, '')

            #recognise multi-word clusters; of which this is the first word
            mention_opening = re.findall('\([0-9]+', cluster)
            for mention in mention_opening:
                #add to stack; do not count the mention yet
                stack.append(re.findall('[0-9]+', mention)[0])
                cluster = cluster.replace(mention, '')

            #recognise multi-word clusters; of which this is the first word
            mention_closing = re.findall('[0-9]+\)', cluster)
            for mention in mention_closing:
                #remove from strack and count the mention
                cluster_counts[stack.pop(-1)] += 1
    #return a list of singleton clusters only
    return [cluster for cluster, count in cluster_counts.items() if count == 1]

def remove_singletons(coref : str, singletons_list : List[str])-> str:
    """
    Takes a coreference annotation, e.g. '(21)|22)|23)' and removes the singleton clusters from this annotation. 
    For example, if 22 and 23 are singletons, the annotation becomes '(21)'
    """
    new_coref = []
    coref_info = coref.split("|")
    for ci in coref_info:
        #only keep those clusters that are not singletons
        if re.findall('[0-9]+', ci)[0] not in singletons_list:
            new_coref.append(ci)
    if new_coref == []:
        return '-'
    else:
        return '|'.join(new_coref)
    
def store_processed_sents(sents: List[str], outdir : str, split : str, filename : str) -> None:
    """
    Saves the updated files to the new output dir
    """
    basename = os.path.basename(filename)
    #create new filename
    outpath =  os.path.join(outdir, split, basename)
    #store
    fout = open(outpath, mode="w", encoding="utf-8")
    fout.write("\n".join(sents))

def exclude_singletons(conll_filenames : List[str], outdir : str) -> None:
    """
    Extracts all singletons in a document and subsequently removes them.
    Stores the updated datafile at the new datapath
    """
    for split, filelist in conll_filenames.items():
        for filename in filelist:
            with open(filename, mode="r", encoding="utf-8") as f:
                words = re.findall(WORD_PATTERN, f.read())
            
            singletons_list = extract_singletons(words)

            with open(filename, mode="r", encoding="utf-8") as f:
                sents = re.findall(SENT_PATTERN, f.read())
            for i, sent in enumerate(sents):
                tmp_sent= ''
                for word in sent.split('\n'):
                    if word == "":
                        continue
                    word_data = word.split()
                    coref = word_data[11]
                    if not coref == '-':
                        word_data[11] = remove_singletons(coref, singletons_list)
                    tmp_sent += '\t'.join(word_data) + '\n'
                sents[i] = tmp_sent
            store_processed_sents(sents, outdir, split, filename)
            

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(
        description="Converts conll-formatted files to json.")
    argparser.add_argument("conll_dir", help="The root directory of"
                           " conll-formatted OntoNotes corpus.")

    args = argparser.parse_args()

    outdir = os.path.normpath(args.conll_dir) +  "_singletons_excld"
    print(f"files will be exported to {outdir}")

    if os.path.exists(outdir):
        response = input(f"{outdir} already exists!"
                        f" Enter 'yes' to delete it or anything to exit: ")
        if response != "yes":
            sys.exit()
        shutil.rmtree(outdir)

    os.makedirs(outdir)
    for split in DATA_SPLITS:
        os.makedirs(os.path.join(outdir, split))
    
    #open coref and dependency data
    conll_filenames = get_conll_filenames(args.conll_dir)
    
    exclude_singletons(conll_filenames, outdir)
