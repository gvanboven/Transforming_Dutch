"""
This code takes out gender clues from a corpus, by taking 3 steps:
    1. replacing gendered pronouns by gender neutral pronouns. What the novel pronoun should be depends on a defined setting [setting].
    the possible settings are defined in 'rewrite_setting_rules.json'
    2. rewriting gendered nouns to gendered nouns, following the rules defined in 'rewrite_nouns.json'
    3. anonymising first names, by replacing them with a tag 'ANON_[x]', where x is an integer. 
    The same value of [x] is used for different occurrences of the same name within 1 document.

The code is based on a script by as Dobrovolskii (2021), see https://github.com/gvanboven/wl-coref/tree/main
Furthermore, the anonymisation step is based on https://github.com/uclanlp/corefBias/tree/master/WinoBias/wino -- 
the pronoun and noun steps are also based on this paper, Zhao et al. (2018)

Use this code as follows:
python data_transformation.py <coref_inputdir> <syntax_inputdir> <pronoun_settings_path> <nouns_rules_path> <setting> 
The coreference output dir will be <coref_outputdir>_<setting>
The syntactic output dir will be <syntax_inputdir>_<setting>
"""

import argparse
from collections import defaultdict
import os
import re
import shutil
import sys
from typing import Dict, Generator, List, Tuple
import random

import json

DATA_SPLITS = ["dev", "test", "train"]
DEPS_FILENAME = "deps.conllu"
DEPS_IDX_FILENAME = "deps.index"
DEP_SENT_PATTERN = re.compile(r"(?:^\d.+$\n?)+", flags=re.M)
SENT_PATTERN = re.compile(r"(?:^(?!#).+$\n?)+", flags=re.M) # changed to fit the SoNaR data



def transform_data(data_dir: str,
                    coref_out_dir: str, lassy_out_dir :str,
                    tmp_dir: str) -> None:
    """
    Opens all the files in data corpus, replaces the gender clues, and stores the data in the output directories
    """
    print("Start transforming the data...")
    #open data
    data_dir = os.path.normpath(data_dir)

    fidx = open(os.path.join(tmp_dir, DEPS_IDX_FILENAME),
                mode="r", encoding="utf8")
    # This here is memory-unfriendly, but should be fine for most
    with open(os.path.join(tmp_dir, DEPS_FILENAME),
              mode="r", encoding="utf8") as fgold:
        gold_sents_gen = re.finditer(DEP_SENT_PATTERN, fgold.read())

    assertions = 0
    for doc_id, line in enumerate(fidx): #go through documents
        n_sents, filename = line.rstrip().split("\t")
        n_sents = int(n_sents)
        
        sents = [next(gold_sents_gen).group(0) for _ in range(n_sents)]
        
        sent_data, parsed_sent_data = transform_document(filename, sents, doc_id) # replace gender clues
        #print(sent_data)
        #break
        write_data(coref_out_dir, lassy_out_dir, filename, sent_data, parsed_sent_data) # store the transformed data
        #break
    print(f"in total there occured {assertions} assertion errors")
    
    fidx.close()

def write_data(coref_out_dir: str, lassy_out_dir: str, filename: str, 
               sent_data : List[str], parsed_sent_data: List[str]) -> None:
    """
    Copies the given sentences to a new file with the location:
    '[outdir]/[split]/[filename].conll' for coref files (which are stored in separate datasplit directories)
    '[outdir]/[filename].conllu' for lassy files (which are all stored in the same directory)
    """
    split =  os.path.basename(os.path.dirname(filename))
    basename = os.path.basename(filename)

    sent_outpath = os.path.join(coref_out_dir,split, basename)
    p_basename = basename + 'u'
    parsed_sent_outpath = os.path.join(lassy_out_dir, p_basename)

    fouts = open(sent_outpath, mode="w", encoding="utf-8")
    fouts.write("\n".join(sent_data))

    foutp = open(parsed_sent_outpath, mode="w", encoding="utf-8")
    foutp.write("\n".join(parsed_sent_data))

    with open(r".\Data_analysis\noun_freqs.json", "w+") as outfile:
        json.dump(noun_rewrite_rules, outfile)


def extract_pronouns(word : str, postag: str) -> str:
    """
    Applied filters to words with a PRON POS label, to identify whether this is a 3rd-person pronoun.
    Replaces the pronoun with a tag indicating its gramatical function, i.e. <SUBJ>, <OBJ> or <POSS>
    """
    if word[0].isupper():
        cap = '1'
    else:
        cap = '0'
    if 'excl' not in postag and 'onbep' not in postag and 'betr' not in postag and 'aanw' not in postag and "vb" not in postag:
        if 'mv' not in postag :
            try: 
                number = re.search(re.compile(r"[1-3]", flags=re.M), postag)[0]   
            except:
                return word      
            if number == '3':
                gender = postag[-4:].replace("|","")
                if gender not in ['fem', 'masc']:
                    if word.lower() == 'haar' : 
                        gender = 'fem'
                    elif word.lower() == 'zijn' or word.lower() == "z'n":
                        gender = 'masc'
                    else:
                        return word
                    
                if 'bez' in postag:
                    return f'<POSS>{cap}'
                elif 'nomin' in postag:
                    if word.lower() != 'men':
                        return f'<SUBJ>{cap}'
                else:
                    if word.lower() == 'ze':
                        return f'<SUBJ>{cap}' 
                    else:
                        return f'<OBJ>{cap}'
            
    return word

def recognise_pronouns(sent :List[str] , parsed_sent: List[str], name_count:Dict[str,int], cur_order: int) -> List[str]:
    """
    Replaces all 3rd person pronouns in the sentence by a tag. This is done so that all pronouns in the sentence can be rewritten at once.
    Continuing, nouns are replaced and first names are anonymised during this step. 
    """
    sent_tmp = sent.copy()
    open_PER_entity = False
    for word_id, (s_word, p_word) in enumerate(zip(sent, parsed_sent)): #iterate over words
        s_cols = s_word.split()
        p_cols = p_word.split('\t')

        word = s_cols[3]
        pos = p_cols[3] 
        postag = p_cols[4]
        NE = s_cols[10]

        if 'PER' in NE: #anonymise names
            if word not in name_count:
                name_count[word] = cur_order 
                cur_order += 1
            new_word = "ANON_" + str(name_count[word])

            if ')' not in NE:
                open_PER_entity = True

        elif open_PER_entity:
            if word not in name_count:
                name_count[word] = cur_order 
                cur_order += 1
            new_word = "ANON_" + str(name_count[word])
            if ')' in NE:
                open_PER_entity = False

        elif pos == 'PRON': #replace only 3rd person pronouns by a tag
            new_word = extract_pronouns(word, postag)
                
        elif word.lower() in noun_rewrite_rules: # rewrite nouns
            new_word = noun_rewrite_rules[word.lower()]["rewrite"]
            noun_rewrite_rules[word.lower()]["freq"] += 1
            if word[0].isupper():
                new_word = new_word.capitalize()
        else:
            continue
        s_cols[3] = new_word
        sent_tmp[word_id] = '\t'.join(s_cols)

    return sent_tmp, name_count, cur_order

def replace_pronouns(sent_tmp : List[str], parsed_sent: List[str], doc_id : int) -> (List[str], List[str]):
    """
    Replaces all 3rd person pronouns in a sentence at once, for each setting included in the current settings. 
    Creates a new sentence instance for each defined pronoun setting.
    For instance, if in the settings the pronoun 'hen' and 'die' are selected, 
    for the example sentence "Anna is jarig en zij doet boodschappen met haar vriendin",  following two sentences will be added to the data:
    1. "Anna is jarig en hen doet boodschappen met hun vriendin"
    2. "Anna is jarig en die doet boodschappen met diens vriendin"
    """
    sent_data, parsed_sent_data = [], []
    changed = False
    random.shuffle(settings)

    for i, setting in enumerate(settings): #create a new instance for all settings
        sent_copy = sent_tmp.copy()
        parsed_sent_copy = parsed_sent.copy()

        for word_id, (s_word, p_word) in enumerate(zip(sent_tmp, parsed_sent)):
            s_cols = s_word.split()
            p_cols = p_word.split('\t')

            word = s_cols[3]
                
            if word[:-1] in ['<POSS>', '<SUBJ>', '<OBJ>']: #replace pronouns with the matching word in the current setting
                changed = True
                tag = word[:-1]
                cap = word[-1:]
                if setting == 'unseen': 
                    new_word = rules[setting][tag][random.randint(0, len(rules[setting][tag]) - 1)]
                    print(new_word)
                elif setting == 'gender-neutral':
                    if (doc_id % 2) == 0 :
                        new_word = rules["die"][tag]
                    else:
                        new_word = rules["hen"][tag]
                else:
                    new_word = rules[setting][tag]
                if cap == '1' and setting != 'delex':
                    new_word = new_word.capitalize()

                s_cols[3] = new_word
                p_cols[1] = new_word
                p_cols[2] = new_word

                sent_copy[word_id] = '\t'.join(s_cols) 
                parsed_sent_copy[word_id] = '\t'.join(p_cols)              

        if changed:
            if i != len(settings) -1 :
                sent_copy[-1] += '\n'
                parsed_sent_copy[-1] += '\n'
            sent_data.extend(sent_copy )
            parsed_sent_data.extend(parsed_sent_copy)

    if not changed :
        return sent_tmp, parsed_sent
    else:
        return sent_data, parsed_sent_data


def transform_document(filename: str, parsed_sents: List[str], doc_id : int) -> (List[str], List[str]):
    """
    Opens a document and iterates over its sentences. 
    In each sentence the gender clues are replaced, and the new sentences are returned
    """
    #open sentences
    with open(filename, mode="r", encoding="utf8") as f:
        sents = re.findall(SENT_PATTERN, f.read())
        try:
            assert len(sents) == len(parsed_sents)
        except:
            print(f"ASSERTION doc length error for file {filename} : {len(sents)} {len(parsed_sents)}, skipping this file")
            return [],[]
    sents_out, parsed_sents_out = [], []

    name_count = defaultdict(lambda:0)
    current_order = 0
    #iterate over sentences
    for sent_id, sources in enumerate(zip(sents, parsed_sents)):
        sent, parsed_sent = [s.splitlines() for s in sources]
        try: 
            assert len(sent) == len(parsed_sent)
        except:
            print(f"ASSERTIONERROR sent length for file {filename}, skipping this file")
            return [],[]
        
        #First ALL pronouns in a document are recognised, so that they can then be replaced all at once.
        sent_tmp, name_count, current_order = recognise_pronouns(sent, parsed_sent, name_count, current_order)
        sent_data, parsed_sent_data,  = replace_pronouns(sent_tmp, parsed_sent, doc_id)

        sent_data[-1] += '\n'
        parsed_sent_data[-1] += '\n'
        sents_out.extend(sent_data)
        parsed_sents_out.extend(parsed_sent_data)
    return sents_out, parsed_sents_out


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
    is one of "dev", "test", "train" and filename is
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


def merge_dep_files(temp_dir: str, filenames: Dict[str, List[str]], lassy_coref_alignment: Dict[str, str]) -> None:
    """
    Writes the contents of all files in filenames into one file,
    builds its index in a separate file.
    """
    fout = open(os.path.join(temp_dir, DEPS_FILENAME), mode="w", encoding="utf-8")
    fidx = open(os.path.join(temp_dir, DEPS_IDX_FILENAME), mode="w")


    for filelist in filenames.values():
        
        for filename in filelist:
            with open(filename, mode="r", encoding="utf-8") as f:
                sents = re.findall(DEP_SENT_PATTERN, f.read())
            # store the location of the coref data, to create a link between lassy and coref data
            fidx.write(f"{len(sents)}\t{lassy_coref_alignment[filename]}\n") 
            fout.write("\n".join(sents))
            fout.write("\n")

    fout.close()
    fidx.close()

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(
        description="Converts conll-formatted files to json.")
    argparser.add_argument("conll_dir", help="The root directory of"
                           " conll-formatted OntoNotes corpus.")
    argparser.add_argument("lassy_dir", default="lassy", help="The root directory of"
                           " lassy dependency files in conllu format.")
    argparser.add_argument("rewrite_rules", help="path to JSON file containing"
                           " the pronoun rewriting rules.")
    argparser.add_argument("noun_rewrite_rules", help="path to JSON file containing"
                           " the noun rewriting rules.")
    argparser.add_argument("rewrite_setting", help="rewriting setting")
    argparser.add_argument("--tmp-dir", default="temp", help="A directory to"
                           " keep temporary files in."
                           " Defaults to 'temp'.")
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

    ### open noun and pronoun rewrite rules
    with open(args.rewrite_rules) as json_file:
        rewrite_rules = json.load(json_file)
    rules = rewrite_rules["rules"]
    print(rules)

    global noun_rewrite_rules

    with open(args.noun_rewrite_rules) as json_file:
        noun_rewrite_rules = json.load(json_file)
    for k in noun_rewrite_rules.keys():
        noun_rewrite_rules[k]['freq'] = 0

    setting_name = args.rewrite_setting
    settings = rewrite_rules["settings"][args.rewrite_setting]

    #create output dirs
    coref_outdir = os.path.join('..', 'Data',f'Sonar_conll_splits_{setting_name}')
    lassy_outdir = os.path.join('..', 'Data',f'CONLLU_{setting_name}')
    if os.path.exists(coref_outdir):
        response = input(f"{coref_outdir} already exists!"
                         f" Enter 'yes' to delete it or anything to exit: ")
        if response != "yes":
            sys.exit()
        shutil.rmtree(coref_outdir)
        shutil.rmtree(lassy_outdir)
    os.makedirs(coref_outdir)
    for split in DATA_SPLITS:
        os.makedirs(os.path.join(coref_outdir, split))
    os.makedirs(lassy_outdir)

    #open data
    conll_filenames = get_conll_filenames(args.conll_dir)
    lassy_filenames, lassy_coref_alignment = get_lassy_filenames(args.lassy_dir, conll_filenames) #align lassy and coref data
    merge_dep_files(args.tmp_dir, lassy_filenames, lassy_coref_alignment) # process lassy data

    #data transformation
    transform_data(args.conll_dir, coref_outdir, lassy_outdir, args.tmp_dir)
 
    if not args.keep_tmp_dir:
        shutil.rmtree(args.tmp_dir)