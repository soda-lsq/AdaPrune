import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import argparse
import random
import torch
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer


def get_c4(args, tokenizer):
    # Load train and validation datasets
    traindata = load_dataset('allenai/c4', data_files={'train': 'en/c4-train.00000-of-01024.json.gz'}, split='train')
    samples = []
    
    # Generate samples from training set
    random.seed(args.seed)
    trainloader = []
    for _ in range(args.nsamples):
        while True:
            i = random.randint(0, len(traindata) - 1)
            samples.append(traindata[i]['text'])
            trainenc = tokenizer(traindata[i]['text'], return_tensors='pt')
            if trainenc.input_ids.shape[1] > args.seq_len:
                break
        i = random.randint(0, trainenc.input_ids.shape[1] - args.seq_len - 1)
        j = i + args.seq_len
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    return trainloader, samples


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_path', type=str)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', type=int, default="cuda")
    parser.add_argument('--nsamples', type=int, default=128)
    parser.add_argument('--seq_len', type=int, default=2048)

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

    trainloader, tokenized_samples = get_c4(args, tokenizer)
    example_prompts = torch.cat(tokenized_samples, dim=0).to(args.device) 

    calibdation_data["trainloader"] = trainloader
    calibdation_data["example_prompts"] = example_prompts

    if not os.path.exists(f'./calibdation_data'):
        os.makedirs(f'./calibdation_data')
    with open(f'./calibdation_data/c4_data.pth', 'wb') as f:
        torch.save(calibdation_data, f)




