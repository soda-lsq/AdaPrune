import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import argparse
import torch
import pandas as pd
import numpy as np
from transformers import AutoTokenizer


TASKS = [
        'abstract_algebra',
        'anatomy',
        'astronomy',
        'business_ethics',
        'clinical_knowledge',
        'college_biology',
        'college_chemistry',
        'college_computer_science',
        'college_mathematics',
        'college_medicine',
        'college_physics',
        'computer_security',
        'conceptual_physics',
        'econometrics',
        'electrical_engineering',
        'elementary_mathematics',
        'formal_logic',
        'global_facts',
        'high_school_biology',
        'high_school_chemistry',
        'high_school_computer_science',
        'high_school_european_history',
        'high_school_geography',
        'high_school_government_and_politics',
        'high_school_macroeconomics',
        'high_school_mathematics',
        'high_school_microeconomics',
        'high_school_physics',
        'high_school_psychology',
        'high_school_statistics',
        'high_school_us_history',
        'high_school_world_history',
        'human_aging',
        'human_sexuality',
        'international_law',
        'jurisprudence',
        'logical_fallacies',
        'machine_learning',
        'management',
        'marketing',
        'medical_genetics',
        'miscellaneous',
        'moral_disputes',
        'moral_scenarios',
        'nutrition',
        'philosophy',
        'prehistory',
        'professional_accounting',
        'professional_law',
        'professional_medicine',
        'professional_psychology',
        'public_relations',
        'security_studies', 
        'sociology',
        'us_foreign_policy',
        'virology',
        'world_religions']

choices = ["A", "B", "C", "D"]


def format_subject(subject):
    l = subject.split("_")
    s = ""
    for entry in l:
        s += " " + entry
    return s


def format_example(df, idx, include_answer=True):
    prompt = df.iloc[idx, 0]
    k = df.shape[1] - 2
    for j in range(k):
        prompt += "\n{}. {}".format(choices[j], df.iloc[idx, j+1])
    prompt += "\nAnswer:"
    if include_answer:
        prompt += " {}\n\n".format(df.iloc[idx, k + 1])
    return prompt


def gen_prompt(train_df, subject, k=-1):
    prompt = "The following are multiple choice questions (with answers) about {}.\n\n".format(format_subject(subject))
    if k == -1:
        k = train_df.shape[0]
    for i in range(k):
        prompt += format_example(train_df, i)
    return prompt


def get_mmlu(args, tokenizer):
    trainloader = []
    sample_data = []
    for task in TASKS[0:args.seq_len]:    
        dev_df = pd.read_csv(os.path.join(args.data_dir, "dev", task + "_dev.csv"), header=None)[:args.ncal]
        train_prompt = gen_prompt(dev_df, task, args.ncal)
        sample_data.append(train_prompt)

    for sample in sample_data:
        tokenized_sample = tokenizer(sample, padding='max_length', max_length=args.seq_len, return_tensors='pt')
        inp = tokenized_sample.input_ids[:, 0:args.seq_len]
        tar = inp.clone()
        tar[:, :-1] = -100
        inp_tar_pair = (inp, tar)
        trainloader.append(inp_tar_pair)

    return trainloader, sample_data


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_path', type=str)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', type=int, default="cuda")
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--ncal', type=int, default=1)
    parser.add_argument('--ntrain', type=int, default=5)
    parser.add_argument('--nsamples', type=int, default=10)
    parser.add_argument('--seq_len', type=int, default=512)

    args = parser.parse_args()

    # Setting seeds for reproducibility
    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(
        args.save_path,
        use_fast=False,
        padding_side="left",
    )
    tokenizer.pad_token_id = 0 if tokenizer.pad_token_id is None else tokenizer.pad_token_id
    tokenizer.bos_token_id = 1

    # Load calibdation data
    calibdation_data = dict()
    print("loading calibdation data")

    trainloader, tokenized_samples = get_mmlu(args, tokenizer)
    example_prompts = torch.cat(tokenized_samples, dim=0).to(args.device) 

    calibdation_data["trainloader"] = trainloader
    calibdation_data["example_prompts"] = example_prompts

    if not os.path.exists(f'./calibdation_data'):
        os.makedirs(f'./calibdation_data')
    with open(f'./calibdation_data/mmlu_data.pth', 'wb') as f:
        torch.save(calibdation_data, f)
