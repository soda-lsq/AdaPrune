import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import re
import time
import torch
import numpy as np
from tqdm import tqdm



def prepare_input(tokenizer, prompts):
    input_tokens = tokenizer.batch_encode_plus(prompts, return_tensors="pt", padding=True)
    for t in input_tokens:
        if torch.is_tensor(input_tokens[t]):
            input_tokens[t] = input_tokens[t].to('cuda')

    return input_tokens


def batch_split(prompts, batch_num):
    batch_prompts = []
    mini_batch = []
    for prompt in prompts:
        mini_batch.append(prompt)
        if len(mini_batch) == batch_num:
            batch_prompts.append(mini_batch)
            mini_batch = []
    if len(mini_batch) != 0:
        batch_prompts.append(mini_batch)
    return batch_prompts


def batch_infer(model, tokenizer, prompts):
    batch_size = 2
    answers = []
    for batch_input in tqdm(batch_split(prompts, batch_size)):
        encode_inputs = prepare_input(tokenizer, batch_input)
        outputs = model.generate(**encode_inputs, max_new_tokens=256, pad_token_id=tokenizer.eos_token_id)
        answers.extend(tokenizer.batch_decode(outputs, skip_special_tokens=True))
    return answers


def test_answer(pred_str, ans_str, question):
    pred_str = pred_str.lower()
    question = question.strip().lower()
    preds = pred_str.split("\n\n")
    max_cover_num = 0
    for idx, pred_str in enumerate(preds):
        pred_str_words = pred_str.split()
        question_words = question.split()
        cover_num = len([word for word in question_words if word in pred_str_words])
        if cover_num > max_cover_num:
            max_cover_num = cover_num
            max_idx = idx
    pred_str = preds[max_idx]
    pattern = '\d*\.?\d+'
    pred = re.findall(pattern, pred_str)
    if(len(pred) >= 1):
        # print(ans_str)
        gold = re.findall(pattern, ans_str)
        gold = gold[-1]
        # print(pred_str)
        pred = pred[-1]
        return pred == gold
    else: 
        return False


def eval_acc(validation_data, prompt, model, tokenizer):

    print("evaluating on gsm8k")

    questions = validation_data["question"]
    answers = validation_data["answer"]
    prompts = [prompt + '\nQuestion: ' + q + '\n' for q in questions]

    pred_answers = batch_infer(model, tokenizer, prompts)

    count = 0

    for index in range(len(validation_data)):
        q = questions[index]
        a = answers[index]
        ans_ = pred_answers[index]

        if(test_answer(ans_, a, q)):
            count += 1
    predict_acc = float(count / len(validation_data))

    print("gsm8k_acc:", predict_acc*100)

    return predict_acc

