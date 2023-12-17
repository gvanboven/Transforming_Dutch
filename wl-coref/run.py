""" Runs experiments with CorefModel.

Try 'python run.py -h' for more details.
"""

import argparse
from contextlib import contextmanager
import datetime
import random
import sys
import time
import os
import shutil
import psutil

import numpy as np  # type: ignore
import torch        # type: ignore
import json

from coref import CorefModel


@contextmanager
def output_running_time():
    """ Prints the time elapsed in the context """
    start = int(time.time())
    try:
        yield
    finally:
        end = int(time.time())
        delta = datetime.timedelta(seconds=end - start)
        print(f"Total running time: {delta}")


def seed(value: int) -> None:
    """ Seed random number generators to get reproducible results """
    print(f"set seed to {value}")
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    torch.cuda.manual_seed_all(value)           # type: ignore
    torch.backends.cudnn.deterministic = True   # type: ignore
    torch.backends.cudnn.benchmark = False      # type: ignore

#ADDED FOR EVALUATION
def eval(modelname, data_split, weights, config_file='config.toml', lr=3e-4, bertlr=1e-5, devdata = None, testdata = None):
    print("start to create model")
    model = CorefModel(config_file, 'xlm-roberta', lr, bertlr)
    print("created model")

    model.config.model_name = modelname
    model.config.logs_file = os.path.join(model.config.logs_dir, (model.config.model_name + '.json'))

    if devdata:
        model.config.dev_data = devdata
        print(f"path to dev data set to : {model.config.dev_data}")

    if testdata:
        model.config.test_data = testdata
        print(f"path to test data set to : {model.config.test_data}")    

    model.config.data_type = os.path.splitext(os.path.basename(model.config.__dict__[f"{data_split}_data"]))[0]
    model.load_weights(path=weights, map_location="cpu",
                           ignore={"bert_optimizer", "general_optimizer",
                                   "bert_scheduler", "general_scheduler"})

    model.evaluate(data_split,
                       word_level_conll=False)#args.word_level)
    with open(model.config.logs_file, "r+") as outfile:
        data = json.load(outfile)
        data[f'{model.config.test_data}_eval'] = model.train_logs[f'{data_split}_eval']

    with open(model.config.logs_file, "w") as outfile:
            json.dump(data, outfile, indent=2)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("mode", choices=("train", "eval"))
    argparser.add_argument("experiment")
    argparser.add_argument("modelname")
    argparser.add_argument("--config-file", default="config.toml")
    argparser.add_argument("--data-split", choices=("train", "dev", "test"),
                           default="test",
                           help="Data split to be used for evaluation."
                                " Defaults to 'test'."
                                " Ignored in 'train' mode.")
    argparser.add_argument("--batch-size", type=int,
                           help="Adjust to override the config value if you're"
                                " experiencing out-of-memory issues")
    argparser.add_argument("--warm-start", action="store_true",
                           help="If set, the training will resume from the"
                                " last checkpoint saved if any. Ignored in"
                                " evaluation modes."
                                " Incompatible with '--weights'.")
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


    if args.warm_start and args.weights is not None:
        print("The following options are incompatible:"
              " '--warm_start' and '--weights'", file=sys.stderr)
        sys.exit(1)

    seed(args.seed)
    print("start to create model")
    model = CorefModel(args.config_file, args.experiment, args.lr, args.bertlr)
    print("created model")
    
    ## THE BELOW IS ADDED, to store the different model versions efficiently, in order to compare them
    model.config.model_name = args.modelname

    if args.devdata:
        model.config.dev_data = args.devdata
        print(f"path to dev data set to : {model.config.dev_data}")
    if args.testdata:
        model.config.test_data = args.testdata
        print(f"path to test data set to : {model.config.test_data}") 
    
    model.config.data_type = os.path.splitext(os.path.basename(model.config.__dict__[f"{args.data_split}_data"]))[0]
    model.config.logs_file = os.path.join(model.config.logs_dir, (model.config.model_name + '.json'))
    model_path = os.path.join(model.config.model_dir, model.config.model_name)
    )
    if os.path.exists(model.config.logs_file) and args.mode == "train":
        response = input(f"a model with the name {model.config.model_name} already exists!"
                         f" Enter 'yes' to delete it or anything to exit: ")
        if response != "yes":
            sys.exit()
        os.remove(model.config.logs_file)
        shutil.rmtree(model_path)

    model_path = os.path.join(model.config.model_dir, model.config.model_name)
    if not os.path.exists(model_path):
        os.mkdir(model_path)

    if args.batch_size:
        model.config.a_scoring_batch_size = args.batch_size
    if args.epochs:
        model.config.train_epochs = args.epochs
    if args.devdata:
        model.config.dev_data = args.devdata
        print(f"path to dev data set to : {model.config.dev_data}")
    if args.testdata:
        model.config.test_data = args.testdata
        print(f"path to test data set to : {model.config.test_data}")    


    if args.mode == "train":
        model.train_logs['train-data'] = model.config.train_data
        model.train_logs['dev-data'] = model.config.dev_data
        model.train_logs['batch-size'] = model.config.a_scoring_batch_size
        model.train_logs['epochs'] = model.config.train_epochs
        model.train_logs['learning-rate'] = model.config.learning_rate
        model.train_logs['bert-learning-rate'] = model.config.bert_learning_rate
        model.train_logs['seed'] = args.seed


        if args.weights is not None or args.warm_start:
            model.load_weights(path=args.weights, map_location="cpu",
                               noexception=args.warm_start)
        with output_running_time():
            print("start training model")
            model.train()

        dev_f1s = [epoch['sl_f1'] for epoch in model.train_logs['dev_eval']]
        best_epoch = dev_f1s.index(max(dev_f1s))
        model.train_logs['best_epoch_path'] =  model.train_logs["checkpoint_paths"][best_epoch]

        with open(model.config.logs_file, "w") as outfile: # store model details in log file (ADDED)
            json.dump(model.train_logs, outfile, indent=2)
    else:
        model.load_weights(path=args.weights, map_location="cpu",
                           ignore={"bert_optimizer", "general_optimizer",
                                   "bert_scheduler", "general_scheduler"})
        model.evaluate(data_split=args.data_split,
                       word_level_conll=args.word_level)

        with open(model.config.logs_file, "r+") as outfile: # store additional eval results in logs file (ADDED)
            data = json.load(outfile)
            data[f'{model.config.test_data}_eval'] = model.train_logs[f'{args.data_split}_eval']
        
        with open(model.config.logs_file, "w") as outfile:
            json.dump(data, outfile, indent=2)
