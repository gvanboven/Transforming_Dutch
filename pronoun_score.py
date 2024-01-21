import re
from collections import defaultdict
from typing import Dict, List, Bool

WORD_PATTERN = re.compile(r"(?:^(?!#).+$\n?)", flags=re.M) 
COREF_PATTERN = re.compile(r"\([0-9]*\)",flags=re.M)

#different settings distinguish between which pronouns should be considered in the evaluation
setting_dict = {
                'all' : ['hij', 'hem', 'zijn', 'zij', 'haar', 'hen', 'hun', 'die', 'diens', "dee", "dij", "nij", "vij", "zhij", "zem", "dem", "ner", "vijn", "zhaar", "zeer", "dijr", "nijr", "vijns", "zhaar", "zeer"],
                'fem' : ['zij', 'haar'],
                'masc': ['hij', 'hem', 'zijn'] ,
                'gi'  : ['hen', 'hun', 'die', 'diens']
               }

def third_person_pronoun(word: str,postag: str, pronoun_list: List) -> Bool:
    '''
    This function is languge-specific and should be adapted for languages other than Dutch.
    The purpose of this function is to check whether a word+postag is a Thrird person pronoun (True) or not (False)
    '''
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
    '''
    Extract all clusters from each file in the gold/predicted data
    Returns a dict {filename : [[cluster1],[]...]}
    ''' 
    clusters =  defaultdict(lambda: defaultdict(lambda : []))
    stack = {}
    #iterate through the mentions
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
                clusters[filename][split_mention(mention)].append(f"{begin_id}-{word_id}")#store cluster
    clusterlist = {filename : cluster.values() for filename, cluster in clusters.items()}
    return clusterlist

def extract_pronouns(wordlist : List[str], setting : str = 'all') -> Dict:
    '''
    Extract the indices of all third-persion pronouns per file from the gold annotations
    returns a dict {filename: [third-person pronoun indices]}
    '''
    pronouns =  defaultdict(lambda: [])

    #iterate through the mentions
    for word_id, word in enumerate(wordlist):
        instance = word.split()
        filename = instance[0]
        pos = instance[4]
        postag = instance[5]
        token = instance[3]

        if pos == "PRON":
            #check if the mention is a third person pronoun
            if third_person_pronoun(token.lower(), postag, setting_dict[setting]):
                if token.lower() in setting_dict[setting]: #check if this pronoun should be evaluated in the current setting
                    pronouns[filename].append(str(word_id))#if so, strore the pronoun index
    return pronouns

def antecedent_check(mention, reference):
    #check if the reference mention preceeds the current mention (i.e. whether the reference is an antecedent)
    if '-' in reference:
        reference = reference.split('-')[1]
    return int(reference) < int(mention)

def antecedent_overlap(gold_cluster, predicted_antecedent):
    overlap = [antecedent for antecedent in predicted_antecedent if antecedent in gold_cluster]
    return 1 if len(overlap) > 0 else 0

def compare_antecedents(gold_clusters: Dict[str, str], pred_clusters: Dict[str,str],
                        pronoun_indices: Dict[str,int]) -> List [bool]:
    '''
    Compare the gold antecedents of third-person singular pronouns with the predicted antecedents
    Returns a list of bools, representing whether this pronoun has a correct antecedent or not
    '''
    correct_antecedent_predictions = []

    for filename, gold_clusters in gold_clusters.items():
        for gold_cluster in gold_clusters:
            for mention in gold_cluster:

                if mention in pronoun_indices[filename]:#check if the mention is a third-person singular pronoun
                    try:
                        relevant_pred_clusters = [cluster for cluster in pred_clusters[filename] if mention in cluster]
                    except KeyError:
                        print(f"KeyError for {filename}")
                        continue
                    pred_antecedents = [reference for cluster in relevant_pred_clusters for reference in cluster \
                                       if antecedent_check(mention, reference)] # extrect predicted ants
                    correct_antecedent_predictions.append(antecedent_overlap(gold_cluster, pred_antecedents)) # check if >= antecedent is correct


    return correct_antecedent_predictions

def compute_score(pronoun_results: List[bool]) -> float:
    return sum(pronoun_results)/len(pronoun_results)*100

def compute_pronoun_score(goldwords: List[str], predwords: List[str], setting: str='all') -> float:
    #extract indices of 3p pronouns
    pronoun_indices = extract_pronouns(goldwords, setting) #{filename : [indices]}

    #extract gold clusters 
    gold_clusters = extract_clusters(goldwords) #{filename : [[clusters],.. []]
    #extract pred clusters
    pred_clusters = extract_clusters(predwords)

    #compare the antedecents for mentions that are third persion singular pronouns
    pronoun_results = compare_antecedents(gold_clusters, pred_clusters, pronoun_indices)
    pronoun_score = compute_score(pronoun_results) #compute the percentage of correctly resolved pronouns
    print(f"number of pronouns: {len(pronoun_results)}")
    print(f"pronoun score : {pronoun_score}")
    return pronoun_score

eval_basename = "" #insert filename

goldfile = f"{eval_basename}.gold.conll"
predfile = f"{eval_basename}.pred.conll"

#The current code should work for files in the CONLL-2012 format
with open(goldfile, mode="r", encoding="utf-8") as f:
    goldwords = re.findall(WORD_PATTERN, f.read())

with open(predfile, mode="r", encoding="utf-8") as f:
    predwords = re.findall(WORD_PATTERN, f.read())

print(eval_basename)
pronoun_score = compute_pronoun_score(goldwords, predwords)