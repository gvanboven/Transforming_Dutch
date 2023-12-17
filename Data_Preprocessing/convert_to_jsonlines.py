""""
FILE TAKEN FROM Dobrovolskii (2021) https://github.com/gvanboven/wl-coref/tree/main
SOME SMALL ADAPTATIONS WERE MADE TO FIT THE SoNaR-1 DATA -- THOSE ARE MARKED WITH COMMENTS
-------------------
Use this file to change the format from CoNLL2012 to jsonlines, which is the expected format of the wl-coref model.
Run the file as follows:
python conver_to_jsonlines.py <conll2021-coref-data> <jsonlines-output-path> --lassy-dir <conll-2012-syntactic-data-path> 
"""

import argparse
from collections import defaultdict
from itertools import chain
import os
import re
import shutil
import subprocess
import sys
from typing import Dict, Generator, List, TextIO, Union, Tuple

import jsonlines
from tqdm import tqdm

DATA_SPLITS = ["dev", "test", "train"]
DEPS_FILENAME = "deps.conllu"
DEPS_IDX_FILENAME = "deps.index"
DEP_SENT_PATTERN = re.compile(r"(?:^\d.+$\n?)+", flags=re.M)
SENT_PATTERN = re.compile(r"(?:^(?!#).+$\n?)+", flags=re.M) # changed to fit the SoNaR data

class CorefSpansHolder:
    """
    A simple container to process coreferent spans line by line
    (as previous information might be needed)

    self.starts contains word indices of span starts
    self.spans contains spans that have been built

    Both dictionaries use entity indices as keys.
    """
    def __init__(self):
        self.starts = defaultdict(lambda: [])
        self.spans = defaultdict(lambda: [])

    def __iter__(self):
        for start_lst in self.starts.values():
            assert len(start_lst) == 0
        return iter(self.spans.values())

    def add(self, coref_info: str, word_id: int):
        """
        Examples of coref_info: "(50)", "(50", "50)", "(50)|(80" etc
        """
        coref_info = coref_info.split("|")
        for ci in coref_info:
            self._add_one(ci, word_id)

    def _add_one(self, coref_info: str, word_id: int):
        if coref_info[0] == "(":
            if coref_info[-1] == ")":
                entity_id = int(coref_info[1:-1])
                self.spans[entity_id].append([word_id, word_id + 1])
            else:
                entity_id = int(coref_info[1:])
                self.starts[entity_id].append(word_id)
        elif coref_info[-1] == ")":
            entity_id = int(coref_info[:-1])
            self.spans[entity_id].append(
                [self.starts[entity_id].pop(), word_id + 1])
        else:
            raise ValueError(f"Invalid coref_info: {coref_info}")


def build_jsonlines(data_dir: str,
                    out_dir: str,
                    tmp_dir: str) -> None:
    """
    Builds a file for each data split where each line corresponds
    to a document.
    """
    print("Building jsonlines...")
    data_dir = os.path.normpath(data_dir)

    fidx = open(os.path.join(tmp_dir, DEPS_IDX_FILENAME),
                mode="r", encoding="utf8")
    out = {split_type: jsonlines.open(
            os.path.join(out_dir, f"sonar_{split_type}.jsonlines"),
            mode="w", compact=True
            ) for split_type in DATA_SPLITS}

    # This here is memory-unfriendly, but should be fine for most
    with open(os.path.join(tmp_dir, DEPS_FILENAME),
              mode="r", encoding="utf8") as fgold:
        gold_sents_gen = re.finditer(DEP_SENT_PATTERN, fgold.read())

    assertions = 0
    for line in fidx:
        n_sents, filename = line.rstrip().split("\t")
        n_sents = int(n_sents)
        sents = [next(gold_sents_gen).group(0) for _ in range(n_sents)]
        data = build_one_jsonline(filename, sents)
        if data != {}:
            out[get_split_type(data_dir, filename)].write(data)
        else: 
            assertions += 1
    print(f"in total there occured {assertions} assertion errors")

    fidx.close()
    for fileobj in out.values():
        fileobj.close()


def build_one_jsonline(filename: str,
                       parsed_sents: List[str]) -> Dict[str, Union[list, str]]:
    """
    Returns a dictionary of the following structure:

    document_id:    str,
    cased_words:    [str, ...]                # words
    sent_id:        [int, ...]                # word id to sent id
    part_id:        [int, ...]                # word id to part id
    speaker:        [str, ...]                # word id to speaker
    pos:            [str, ...]                # word id to POS
    deprel:         [str, ...]                # word id to dep. relation
    head:           [int, ...]                # word id to head, None for root
    clusters:       [[[int, int], ...], ...]  # list of clusters, where each
                                                cluster is
                                                a list of spans of words

    """
    with open(filename, mode="r", encoding="utf8") as f:
        sents = re.findall(SENT_PATTERN, f.read())
        try:
            assert len(sents) == len(parsed_sents)
        except:
            print(f"ASSERTION doc length error for file {filename} : {len(sents)} {len(parsed_sents)}, skipping this file")
            return {}

    data = {
        "document_id":      None,
        "cased_words":      [],
        "sent_id":          [],
        "part_id":          [],
        "speaker":          [],
        "pos":              [],
        "postag":           [],
        "deprel":           [],
        "head":             [],
        "clusters":         []
    }
    coref_spans = CorefSpansHolder()
    total_words = 0
    for sent_id, sources in enumerate(zip(sents, parsed_sents)):
        sent, parsed_sent = [s.splitlines() for s in sources]
        try: 
            assert len(sent) == len(parsed_sent)
        except:
            print(f"ASSERTIONERROR sent length for file {filename}, skipping this file")
            return {}

        for s_word, p_word in zip(sent, parsed_sent):
            s_cols = s_word.split()
            p_cols = p_word.split('\t')

            document_id = s_cols[0]
            part_id = int(s_cols[1])
            word_id = total_words + int(s_cols[2])  # continuous word_id
            word = s_cols[3]

            speaker = 0 # SONAR: missing
            coref_info = s_cols[-1]  

            pos = p_cols[3] 
            postag = p_cols[4]
            deprel = p_cols[7] 

            # DS indexing starts with 1, zero is reserved for root
            # Converting word_id to continuous id, setting root head to None
            head = int(p_cols[6]) - 1 
            head = None if head < 0 else total_words + head

            if coref_info != "-":
                coref_spans.add(coref_info, word_id)

            if data["document_id"] is None:
                data["document_id"] = document_id
            else:
                assert data["document_id"] == document_id
            data["cased_words"].append(word)
            data["part_id"].append(part_id)
            data["sent_id"].append(sent_id)
            data["speaker"].append(speaker)
            data["pos"].append(pos)
            data["postag"].append(postag)
            data["deprel"].append(deprel)
            data["head"].append(head)

        total_words += len(sent)

    data["clusters"] = list(coref_spans)
    if data["document_id"] == None:
        print(filename)
    return data


def convert_con_to_dep(temp_dir: str, filenames: Dict[str, List[str]]) -> None:
    """
    Runs stanford parser on filenames in temp_dir to convert
    consituency trees to Universal Dependencies.
    """
    print("Converting constituents to dependencies...")
    cmd = ("java -cp downloads/stanford-parser.jar"
           " edu.stanford.nlp.trees.EnglishGrammaticalStructure"
           " -basic -keepPunct -conllx -treeFile"
           " FILENAME").split()
    for data_split, filelist in filenames.items():
        for filename in tqdm(filelist, ncols=0, desc=data_split, unit="docs"):
            fname = os.path.join(os.path.basename(os.path.dirname(filename)), os.path.basename(filename))
            temp_filename = os.path.join(temp_dir, fname)
            cmd[-1] = temp_filename
            with open(temp_filename + "_dep", mode="w") as out:
                subprocess.run(cmd, check=True, stdout=out)
    print()


def extract_trees_from_file(fileobj: TextIO) -> Generator[str, None, None]:
    """
    Yields constituency trees from a conll file.
    """
    current_parse = []
    for line in fileobj:
        line = line.lstrip()
        if not line or line[0] == "#":
            continue
        columns = line.split()
        word_number, word, postag, parse_bit = columns[2:6] # SONAR: pos is postag! 
        if int(word_number) == 0 and current_parse:
            yield "".join(current_parse)
            current_parse = []
        pos, _ = postag.split("[")
        i = parse_bit.index("*")
        new_parse_bit = parse_bit[:i] + f"({pos} {word})" + parse_bit[i + 1:]
        current_parse.append(new_parse_bit)
    yield "".join(current_parse)


def extract_trees_to_files(dest_dir: str,
                           filenames: Dict[str, List[str]]) -> None:
    """
    Creates files names like filenames in dest_dir, writing to each file
    constituency trees line by line.
    """
    for filelist in filenames.values():
        for filename in filelist:
            with open(filename, mode="r", encoding="utf8") as f_to_read:
                fname = os.path.join(os.path.basename(os.path.dirname(filename)), os.path.basename(filename))
                temp_path = os.path.join(dest_dir, fname) #edited
                assert not os.path.isfile(temp_path)
                temp_dir = os.path.split(temp_path)[0]
                os.makedirs(temp_dir, exist_ok=True)
                with open(temp_path, mode="w", encoding="utf8") as f_to_write:
                    for tree in extract_trees_from_file(f_to_read):
                        f_to_write.write(tree + "\n")


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
    lassy_filenames, lassy_coref_alignment = {}, {}
    for data_split, filenames in conll_filenames.items():
        data_split_files = []
        for filename in filenames:
            fname = os.path.basename(filename) + 'u'
            lassy_filename = os.path.join(lassy_dir, fname)
            lassy_coref_alignment[lassy_filename] = filename
            data_split_files.append(lassy_filename)
        lassy_filenames[data_split] = data_split_files
    return lassy_filenames, lassy_coref_alignment



def get_split_type(data_dir: str, query_path: str) -> str:
    """
    Returns the split type of query path, where it is one of the types
    listed in DATA_SPLITS. Raises ValueError if no type could be determined.
    """
    query_path = os.path.normpath(query_path)
    for split_type in DATA_SPLITS:
        if query_path[len(data_dir) + 1:].startswith(split_type):
            return split_type
    raise ValueError("Query path does not contain split type information!")


def merge_dep_files(temp_dir: str, filenames: Dict[str, List[str]], lassy_coref_alignment: Dict[str, str]) -> None:
    """
    Writes the contents of all files in filenames into one file,
    builds its index in a separate file.
    """
    fout = open(os.path.join(temp_dir, DEPS_FILENAME), mode="w", encoding="utf-8")
    fidx = open(os.path.join(temp_dir, DEPS_IDX_FILENAME), mode="w")


    for filelist in filenames.values():
        
        for filename in filelist:
            #adapted to open lassy files directly
            with open(filename, mode="r", encoding="utf-8") as f:
                sents = re.findall(DEP_SENT_PATTERN, f.read())
            # store the location of the coref data, to create a link between lassy and coref data
            fidx.write(f"{len(sents)}\t{lassy_coref_alignment[filename]}\n") 
            fout.write("\n".join(sents))
            fout.write("\n")

    fout.close()
    fidx.close()


def split_jsonlines(out_dir: str,
                    tmp_dir: str,
                    language: str = "english") -> None:
    """ Splits jsonlines located in tmp_dir and writes them to out_dir.
    Splitting means separating different parts of the same document into
    multiple jsonlines. """
    to_split = {split_type: jsonlines.open(
                os.path.join(tmp_dir, f"sonar_{split_type}.jsonlines"),
                mode="r") for split_type in DATA_SPLITS}
    out = {split_type: jsonlines.open(
            os.path.join(out_dir, f"sonar_{split_type}.jsonlines"),
            mode="w", compact=True
            ) for split_type in DATA_SPLITS}

    for split_type, jsonlines_to_split in to_split.items():
        for doc in jsonlines_to_split:
            if doc["document_id"] == None:
                continue
            for part in split_one_jsonline(doc):
                out[split_type].write(part)

    for f in chain(to_split.values(), out.values()):
        f.close()


def split_one_jsonline(doc: dict):
    if doc["part_id"][0] == doc["part_id"][-1]:
        doc["part_id"] = doc["part_id"][0]
        return [doc]

    part_starts = [0]
    parts = [doc["part_id"][0]]
    for i, part_id in enumerate(doc["part_id"]):
        if part_id != parts[-1]:
            part_starts.append(i)
            parts.append(part_id)

    split_docs = []
    for i in range(len(parts)):
        start = part_starts[i]
        if i < len(parts) - 1:
            end = part_starts[i + 1]
        else:
            end = len(doc["cased_words"])
        sent_start = doc["sent_id"][start]

        split_doc = {
            "document_id": doc["document_id"],
            "cased_words": doc["cased_words"][start:end],
            "sent_id": [s - sent_start for s in doc["sent_id"][start:end]],
            "part_id": doc["part_id"][start:end][0],
            "speaker": doc["speaker"][start:end],
            "pos": doc["pos"][start:end],
            "deprel": doc["deprel"][start:end],
            "head": [(h - start) if h is not None else h
                     for h in doc["head"][start:end]],
            "clusters": []
        }
        for cluster in doc["clusters"]:
            split_cluster = []
            for span_start, span_end in cluster:
                if span_start >= start and span_start < end:
                    assert span_end > span_start and span_end <= end
                    split_cluster.append([span_start - start,
                                          span_end - start])
            if split_cluster:
                split_doc["clusters"].append(split_cluster)
        split_docs.append(split_doc)
    return split_docs


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(
        description="Converts conll-formatted files to json.")
    argparser.add_argument("conll_dir", help="The root directory of"
                           " conll-formatted OntoNotes corpus.")
    argparser.add_argument("out_dir", help="The directory where"
                           " the output jsonlines will be written.")
    argparser.add_argument("--tmp-dir", default="temp", help="A directory to"
                           " keep temporary files in."
                           " Defaults to 'temp'.")
    argparser.add_argument("--lassy-dir", default="lassy", help="The root directory of"
                           " lassy dependency files in conllu format.")
    argparser.add_argument("--keep-tmp-dir", action="store_true", help="If set"
                           ", the temporary directory will not be deleted.")

    args = argparser.parse_args()

    if os.path.exists(args.tmp_dir):
        response = input(f"{args.tmp_dir} already exists!"
                         f" Enter 'yes' to delete it or anything to exit: ")
        if response != "yes":
            sys.exit()
        shutil.rmtree(args.tmp_dir)

    os.makedirs(args.tmp_dir)
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)


    data_dir = args.conll_dir #changed to fit SoNaR data
    conll_filenames = get_conll_filenames(data_dir)


    # the following two steps are skipped because the lassy data is already in the correct format
    #extract_trees_to_files(args.tmp_dir, conll_filenames)
    #convert_con_to_dep(args.tmp_dir, conll_filenames) 
    
    lassy_filenames, lassy_coref_alignment = get_lassy_filenames(args.lassy_dir, conll_filenames) #added to align lassy and coref data
    merge_dep_files(args.tmp_dir, lassy_filenames, lassy_coref_alignment) # adapted to process lassy data

    #the rest of the steps are followed normally
    build_jsonlines(data_dir, args.tmp_dir, args.tmp_dir)
    split_jsonlines(args.out_dir, args.tmp_dir)
    if not args.keep_tmp_dir:
        shutil.rmtree(args.tmp_dir)