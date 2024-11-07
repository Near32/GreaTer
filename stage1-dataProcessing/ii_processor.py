import json
import os
import re
import random
import pandas as pd


def read_json(path: str):
    with open(path) as fh:
        #return [json.loads(line) for line in fh.readlines() if line]
        return json.load(open(path))


def get_examples(split, task):
    path = os.path.join("instruction_induction/data/raw/",f"{split}/", f"{task}.json", )
    examples = read_json(path)['examples']
    loaded_json = read_json(path)
    examples = loaded_json['examples']
    num_examples = loaded_json['metadata']['num_examples']
    loaded_data = []

    for ind in range(int(num_examples)):
        currdict = {}
        currdict['goal'] = examples[str(ind+1)]['input']
        currdict['target'] = examples[str(ind+1)]['output']
        loaded_data.append(currdict)

    print(f"{len(loaded_data)} {split} {task} examples")
    return loaded_data

def get_train(task, samples=None):
    data = get_examples('induce', task)
    if (samples):
        samples = min(len(data), samples)
        data_sampled = random.sample(data, samples)
        print(f"picked {len(data_sampled)} samples")
        return data_sampled

    return data

def get_test(task):
    data = get_examples('execute', task)
    return data

def process_task(task):
    train = get_train(task, samples=100)
    test = get_test(task)
    train_path = "instruction_induction/data/processed/"+ str(task) + "/" + "train.csv"
    test_path = "instruction_induction/data/processed/"+ str(task) + "/" + "test.csv"

    df_train = pd.DataFrame(train)
    df_test = pd.DataFrame(test)

    os.makedirs(os.path.dirname(train_path), exist_ok=True)
    os.makedirs(os.path.dirname(test_path), exist_ok=True)

    df_train.to_csv(train_path, index=False)
    df_test.to_csv(test_path, index=False)

if __name__ == "__main__":
    tasks = ['antonyms', 'informal_to_formal', 'negation', 'orthography_starts_with', 'rhymes', 'second_word_letter',
             'sentence_similarity', 'sentiment', 'synonyms', 'taxonomy_animal', 'translation_en-de', 'translation_en-fr', 'translation_en-es',
             'word_in_context']

    for t in tasks:
        process_task(t)

    print("Task successful")
