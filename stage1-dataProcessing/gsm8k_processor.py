import json
import os
import re
import pandas as pd

# TODO later

def read_jsonl(path: str):
    with open(path) as fh:
        return [json.loads(line) for line in fh.readlines() if line]


def get_examples(split):
    path = os.path.join("grade_school_math/data/", f"{split}.jsonl")
    examples = read_jsonl(path)

    for ex in examples:
        ex.update(question=ex["question"])
        ex.update(answer=ex["answer"].split("####")[0])

    print(f"{len(examples)} {split} examples")
    return examples


ANS_RE = re.compile(r"#### (\-?[0-9\.\,]+)")
INVALID_ANS = "[invalid]"


def extract_answer(completion):
    match = ANS_RE.search(completion)
    if match:
        match_str = match.group(1).strip()
        match_str = match_str.replace(",", "")
        return match_str
    else:
        return INVALID_ANS

def is_correct(model_completion, gt_example):
    gt_answer = extract_answer(gt_example["answer"])
    assert gt_answer != INVALID_ANS
    return extract_answer(model_completion) == gt_answer

def convert_ds(path):
    print('here')
    examples = get_examples('train')
    print(ex for t, ex in enumerate(examples) if t<10)
    for t,ex in enumerate(examples):
        if t< 10:
            print(ex)

    return examples

task = 'train'
ex = convert_ds(task)
ex = pd.DataFrame(ex)
ex.rename(columns={'question': 'goal', 'answer': 'target'}, inplace=True)
path = 'grade_school_math/data/' + task + '.csv'
ex.to_csv(path, index=False)