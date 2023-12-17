"""
This code can be used to truncate a set of documents in CoNLL-2012 format, that exceed a set number of words [MAXDOCLENGTH].
The documents will be split into files of equals sentence lengths. 
The number of new files will be x, with x= [doc_length] / [MAXDOCLENGTH]

The code is based on a script by as Dobrovolskii (2021), see https://github.com/gvanboven/wl-coref/tree/main

Use this code as follows:
python truncate_data.py /path/to/corpus /path/to/syntactic/data
The new dataset (without singleton annotations) will be stored at [/path/to/corpus]_truncated
"""

import argparse
import os
import re
import math
import sys
import shutil
from typing import Dict, Generator, List, Tuple
from collections import defaultdict

DATA_SPLITS = ["dev", "test", "train"]
DEPS_FILENAME = "deps.conllu"
DEPS_IDX_FILENAME = "deps.index"
DEP_SENT_PATTERN = re.compile(r"(?:^\d.+$\n?)+", flags=re.M)
SENT_PATTERN = re.compile(r"(?:^(?!#).+$\n?)+", flags=re.M) # changed to fit the SoNaR data
WORD_PATTERN = re.compile(r"(?:^(?!#).+$\n?)", flags=re.M) 
MAXDOCLEN = 3500 #number of words

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

def copy_files(outdir: str, filename: str, sents: List[str], outpath_addition: str ='', split: str = '') -> None:
    """
    Copies the given sentences to a new file with the location:
    '[outdir]/[split]/[filename]/[outpath_additon].conll' for coref files (which are stored in separate datasplit directories)
    '[outdir]/[filename]/[outpath_additon].conll' for lassy files (which are all stored in the same directory)
    """
    basename, ext = os.path.splitext(os.path.basename(filename))
    new_filename = basename + outpath_addition + ext
    if split != '':
        outpath =  os.path.join(outdir, split, new_filename)
    else:
        outpath =  os.path.join(outdir, new_filename)
    fout = open(outpath, mode="w", encoding="utf-8")
    fout.write("\n".join(sents))

def truncate(conll_filenames : List[str], coref_lassy_alignment : Dict [str, str], 
                                outdir_conll : str, outdir_lassy: str) -> None:
    """
    Copies the coref and lassy files simultaneously from the data directories to the new directories
    Files that exceed the max. word count, are split into files of the same size and and are stored 
    into a file with with the name "[filename]_[i].conll" with i = the split index
    """
    doc_len_error_counter = 0
    truncate_counter = 0
    truncate_dict = defaultdict(lambda:0)
    for split, filelist in conll_filenames.items():
        for filename in filelist:
            with open(filename, mode="r", encoding="utf-8") as f:
                sents = re.findall(SENT_PATTERN, f.read())
                n_sents = len(sents)
            with open(coref_lassy_alignment[filename], mode="r", encoding="utf-8") as f:
                dep_sents = re.findall(DEP_SENT_PATTERN, f.read())
            with open(filename, mode="r", encoding="utf-8") as f:
                words = re.findall(WORD_PATTERN, f.read())
                n_words = len(words)

            if len(dep_sents) == len(sents):
                #if document does not exceed the max. length, copy the files as they are
                if n_words < MAXDOCLEN: 
                    copy_files(outdir_conll, filename, sents, '', split)
                    copy_files(outdir_lassy, coref_lassy_alignment[filename], dep_sents)
                else:
                    #split doc into n_step files
                    n_steps = math.ceil(n_words/MAXDOCLEN)
                    step = math.ceil(n_sents/n_steps)

                    print(filename, n_steps)
                    truncate_counter += 1
                    truncate_dict[n_steps] += 1
                    for i in range(n_steps):
                        ext = '_' + str(i)

                        lower_bound = i * step
                        higher_bound = n_sents if i == n_steps else (i+1) * step

                        copy_files(outdir_conll, filename, sents[lower_bound:higher_bound], ext, split)
                        copy_files(outdir_lassy, coref_lassy_alignment[filename], dep_sents[lower_bound:higher_bound], ext)
            #in case the syntax and coref data is not the same length
            else:
                doc_len_error_counter += 1
                print(filename, len(dep_sents), len(sents), len(words))

    print(f"doc len error for {doc_len_error_counter} documents")
    print(f"a total of {truncate_counter} documents truncated")
    print(truncate_dict)

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(
        description="Converts conll-formatted files to json.")
    argparser.add_argument("conll_dir", help="The root directory of"
                           " conll-formatted OntoNotes corpus.")
    argparser.add_argument("lassy_dir", default="lassy", help="The root directory of"
                           " lassy dependency files in conllu format.")

    args = argparser.parse_args()

    outdir_conll = os.path.normpath(args.conll_dir) +  "_truncated"
    outdir_lassy = os.path.normpath(args.lassy_dir) +  "_truncated"

    for path in [outdir_conll, outdir_lassy]:
        if os.path.exists(path):
            response = input(f"{path} already exists!"
                         f" Enter 'yes' to delete it or anything to exit: ")
            if response != "yes":
                sys.exit()
            shutil.rmtree(path)

    os.makedirs(outdir_conll)
    os.makedirs(outdir_lassy)
    for split in DATA_SPLITS:
        os.makedirs(os.path.join(outdir_conll, split))
    
    #open coref and dependency data
    conll_filenames = get_conll_filenames(args.conll_dir)
    lassy_filenames, coref_lassy_alignment = get_lassy_filenames(args.lassy_dir, conll_filenames) #added to align lassy and coref data
    
    #truncate documents
    truncate(conll_filenames, coref_lassy_alignment, outdir_conll, outdir_lassy)