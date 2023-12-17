import re
import os
from collections import defaultdict
import json

from typing import Dict, Generator, List
import argparse
from run import eval

WORD_PATTERN = re.compile(r"(?:^(?!#).+$\n?)", flags=re.M) 
COREF_PATTERN = re.compile(r"\([0-9]*\)",flags=re.M)


setting_dict = {
                'all' : ['hij', 'hem', 'zijn', 'zij', 'haar', 'hen', 'hun', 'die', 'diens', "dee", "dij", "nij", "vij", "zhij", "zem", "dem", "ner", "vijn", "zhaar", "zeer", "dijr", "nijr", "vijns", "zhaar", "zeer"],
                'fem' : ['zij', 'haar'],
                'masc': ['hij', 'hem', 'zijn'] ,
                'gi'  : ['hen', 'hun', 'die', 'diens']
               }

def third_person_pronoun(word,postag, pronoun_list):
    if 'excl' not in postag and 'onbep' not in postag and 'betr' not in postag and 'aanw' not in postag and "vb" not in postag:
        if 'mv' not in postag :
            try: 
                number = re.search(re.compile(r"[1-3]", flags=re.M), postag)[0]

                if number == '3':
                    gender = postag[-4:].replace("|","")
                    if word.lower()  == 'men':
                        return False
                    if gender not in ['fem', 'masc']:
                        if word.lower() not in pronoun_list:
                            return False
                    return True
                else:
                    return False
            except:
                    print(f"FAil for {word, postag}")   

def split_mention(mention : str) -> str:
    return mention.replace('(','').replace(')','')

def extract_clusters(wordlist : List[str] ) -> List:
    clusters =  defaultdict(lambda: defaultdict(lambda : []))
    stack = {}
    for word_id, word in enumerate(wordlist):
        cluster = word.split()[11]
        filename = word.split()[0]

        if cluster != '-':
            #regocognise 1-word mentions
            single_word_mention = re.findall('\([0-9]+\)', cluster)

            for mention in single_word_mention:
                clusters[filename][split_mention(mention)].append(str(word_id))
                cluster = cluster.replace(mention, '')

            #recognise multi-word clusters; of which this is the first word
            mention_opening = re.findall('\([0-9]+', cluster)
            for mention in mention_opening:
                #add to stack; do not count the mention yet
                stack[split_mention(mention)] = word_id
                cluster = cluster.replace(mention, '')

            #recognise multi-word clusters; of which this is the first word
            mention_closing = re.findall('[0-9]+\)', cluster)
            for mention in mention_closing:
                #remove from strack and count the mention
                try:
                    begin_id = stack.pop(split_mention(mention))#stack.pop(-1)
                except:
                    continue
                clusters[filename][split_mention(mention)].append(f"{begin_id}-{word_id}")
    clusterlist = {filename : cluster.values() for filename, cluster in clusters.items()}
    return clusterlist

def extract_pronouns(wordlist : List[str], setting : str = 'all') -> List:
    pronouns =  defaultdict(lambda: [])

    for word_id, word in enumerate(wordlist):
        instance = word.split()
        filename = instance[0]
        pos = instance[4]
        postag = instance[5]
        token = instance[3]

        if pos == "PRON":
            if third_person_pronoun(token.lower(), postag, setting_dict[setting]):
                if token.lower() in setting_dict[setting]:
                    pronouns[filename].append(str(word_id))
    return pronouns

def antecedent_check(mention, reference):
    if '-' in reference:
        reference = reference.split('-')[1]
    return int(reference) < int(mention)

def antecedent_overlap(gold_cluster, predicted_antecedent):
    overlap = [antecedent for antecedent in predicted_antecedent if antecedent in gold_cluster]
    return 1 if len(overlap) > 0 else 0

def compare_antecedents(gold_clusters: Dict[str, str], pred_clusters: Dict[str,str],
                        pronoun_indices: Dict[str,int]) -> List [bool]:
    correct_antecedent_predictions = []

    for filename, gold_clusters in gold_clusters.items():
        for gold_cluster in gold_clusters:
            for mention in gold_cluster:

                if mention in pronoun_indices[filename]:
                    try:
                        relevant_pred_clusters = [cluster for cluster in pred_clusters[filename] if mention in cluster]
                    except KeyError:
                        print(f"KeyError for {filename}")
                        continue
                    pred_antecedents = [reference for cluster in relevant_pred_clusters for reference in cluster \
                                       if antecedent_check(mention, reference)]
                    correct_antecedent_predictions.append(antecedent_overlap(gold_cluster, pred_antecedents))

    return correct_antecedent_predictions

def compute_score(pronoun_results):
    return sum(pronoun_results)/len(pronoun_results)*100

def compute_pronoun_score(goldwords, predwords, setting='all'):
    #extract indices of 3p pronouns
    pronoun_indices = extract_pronouns(goldwords, setting) #{filename : [indices]}

    #extract gold clusters 
    gold_clusters = extract_clusters(goldwords) #{filename : [[clusters],.. []]
    #extract pred clusters
    pred_clusters = extract_clusters(predwords)
    #compare the antedecents for pronouns
    pronoun_results = compare_antecedents(gold_clusters, pred_clusters, pronoun_indices)
    pronoun_score = compute_score(pronoun_results)
    print(f"number of pronouns: {len(pronoun_results)}")
    print(f"pronoun score : {pronoun_score}")
    return pronoun_score

if __name__ == "__main__":

    argparser = argparse.ArgumentParser()
    argparser.add_argument("modelname")
    argparser.add_argument("--config-file", default="config.toml")
    argparser.add_argument("--data-split", choices=("train", "dev", "test"),
                           default="test",
                           help="Data split to be used for evaluation."
                                " Defaults to 'test'."
                                " Ignored in 'train' mode.")
    argparser.add_argument("--weights",
                           help="Path to file with weights to load."
                                " If not supplied, in 'eval' mode the latest"
                                " weights of the experiment will be loaded;"
                                " in 'train' mode no weights will be loaded.")
    argparser.add_argument("--word-level", action="store_true",
                           help="If set, output word-level conll-formatted"
                                " files in evaluation modes. Ignored in"
                                " 'train' mode.")
    argparser.add_argument("--epochs", type=int,
                          help="Adjust to override the number of epochs config value")
    argparser.add_argument("--lr", type=float, default=3e-4,
                          help="Adjust to override the learning rate config value")
    argparser.add_argument("--bertlr", type=float, default=1e-5,
                          help="Adjust to override the bert learning rate config value")
    argparser.add_argument("--seed", type=int, default=2020,
                          help="Adjust to override the bert learning rate config value")
    argparser.add_argument("--devdata", type=str, 
                          help="Adjust to override the path to the dev dataset")
    argparser.add_argument("--testdata", type=str, 
                          help="Adjust to override the path to the test dataset") 
    args = argparser.parse_args()

    # provide the list of data files that should be evaluated
    data_list = [f'hij_test_head.jsonlines',
                 f'zij_test_head.jsonlines',
                 f'hen_test_head.jsonlines',
                 f'die_test_head.jsonlines',
                 ]

    logs_file = os.path.join('data/train_logs', (args.modelname + '.json'))

    for file in data_list:
        if args.data_split == "test":
            eval(args.modelname, args.data_split, args.weights, testdata=file)

        elif args.data_split == "dev":
            eval(args.modelname, args.data_split, args.weights, devdata=file)


        data_type = os.path.splitext(os.path.basename(file))[0]
        eval_basename = f"./data/conll_logs/{args.modelname}_{data_type}"

        goldfile = f"{eval_basename}.gold.conll"
        predfile = f"{eval_basename}.pred.conll"

        with open(goldfile, mode="r", encoding="utf-8") as f:
            goldwords = re.findall(WORD_PATTERN, f.read())

        with open(predfile, mode="r", encoding="utf-8") as f:
            predwords = re.findall(WORD_PATTERN, f.read())

        print(eval_basename)
        pronoun_score = compute_pronoun_score(goldwords, predwords)

        with open(logs_file, "r+") as outfile:
            data = json.load(outfile)
            data[f'{file}_pronoun_score'] = pronoun_score

        with open(logs_file, "w") as outfile:
            json.dump(data, outfile, indent=2)
