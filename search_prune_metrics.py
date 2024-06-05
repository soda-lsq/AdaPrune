import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import argparse
import torch
import torch.nn as nn
import optuna
import numpy as np
from layerwrapper import WrappedGPT
from transformers import AutoTokenizer, AutoModelForCausalLM
from operator_candidates import W_ABS_UNARY_KEYS, X_NORM_UNARY_KEYS, W_ABS_SCALAR_UNARY_KEYS, unary_operation


def prepare_calibration_input(args, model, trainloader, device):
    layers = model.model.layers
    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros((args.nsamples, args.seq_len, model.config.hidden_size), dtype=dtype, device=device)   
    inps.requires_grad = False
    cache = {'i': 0, 'attention_mask': None, "position_ids": None}
    
    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            cache['position_ids'] = kwargs['position_ids']
            raise ValueError
    layers[0] = Catcher(layers[0])
    for batch in trainloader:
        try:
            model(batch[0].to(device))
        except ValueError:
            pass 

    layers[0] = layers[0].module
    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']

    return inps, outs, attention_mask, position_ids 


def find_layers(module, layers=[nn.Linear], name=''):
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(
            child, layers=layers, name=name + '.' + name1 if name != '' else name1
        ))
    return res


def get_model_output(args, example_prompts, tokenizer, model):
    outputs = []
    for sample in example_prompts:
        tokenized_sample = tokenizer(sample, padding='max_length', max_length=args.seq_len, truncation=True, return_tensors='pt')
        with torch.no_grad():
            output = model(**tokenized_sample.to(args.device))[0]
            outputs.append(output.squeeze())
    return torch.stack(outputs)


def run_prune_metric(args, metric_ops, model, trainloader):
    
    with torch.no_grad():
        inps, outs, attention_mask, position_ids = prepare_calibration_input(model, trainloader, args.device)
    
    # Strart Pruning
    layers = model.model.layers
    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)

        # get input X
        wrapped_layers = {}
        for name in subset:
            wrapped_layers[name] = WrappedGPT(subset[name])

        def add_batch(name):
            def tmp(_, inp, out):
                wrapped_layers[name].add_batch(inp[0].data, out.data)
            return tmp

        handles = []
        for name in wrapped_layers:
            handles.append(subset[name].register_forward_hook(add_batch(name)))
        for j in range(args.nsamples):
            with torch.no_grad():
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]    
        for h in handles:
            h.remove()

        for name in subset:
            sub_layer = subset[name]
            indexed_name = f"{name}_layer_{i}"
            print(f"pruning layer {i} name {name}")
            # get wight
            W = sub_layer.weight.data
            # get activation
            X = wrapped_layers[name].scaler_row.reshape((1,-1))

            WX_W_ops = unary_operation(torch.abs(W), metric_ops["WX_W_u1"])
            WX_X_ops = unary_operation(torch.sqrt(X), metric_ops["WX_X_u1"])

            WX_W_alpha = unary_operation(torch.abs(W), metric_ops["WX_W_c1"])
            WX_X_alpha = unary_operation(X, metric_ops["WX_X_c1"])

            W_metric = (WX_W_alpha * WX_W_ops) * (WX_X_alpha * WX_X_ops)

            W_mask = (torch.zeros_like(W_metric) == 1)  ## initialize a mask to be all False

            sort_res = torch.sort(W_metric, dim=-1, stable=True)

            # unstructured pruning
            indices = sort_res[1][:,:int(W_metric.shape[1]*args.sparsity_ratio)]
            W_mask.scatter_(1, indices, True)
            subset[name].weight.data[W_mask] = 0  ## set weights to zero 
    
        for j in range(args.nsamples):
            with torch.no_grad():
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]   
        inps, outs = outs, inps
    
    torch.cuda.empty_cache()


def objective(trial):
    torch.cuda.empty_cache()

    # load model: load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(
        args.save_path,
        use_fast=False,
        padding_side="left",
    )
    tokenizer.pad_token_id = 0 if tokenizer.pad_token_id is None else tokenizer.pad_token_id
    tokenizer.bos_token_id = 1

    model = AutoModelForCausalLM.from_pretrained(args.save_path, return_dict=True, 
    low_cpu_mem_usage=True, 
    torch_dtype=torch.float16,
    )
    model.seqlen = 2048
    model.to(args.device)
    model.eval()

    # define expression operators
    metric_ops = dict()
    metric_ops["WX_W_u1"] = trial.suggest_categorical("WX_W_u1", W_ABS_UNARY_KEYS)
    metric_ops["WX_X_u1"] = trial.suggest_categorical("WX_X_u1", X_NORM_UNARY_KEYS)
    metric_ops["WX_W_c1"] = trial.suggest_categorical("WX_W_c1", W_ABS_SCALAR_UNARY_KEYS)
    metric_ops["WX_X_c1"] = trial.suggest_categorical("WX_X_c1", W_ABS_SCALAR_UNARY_KEYS)
    
    original_outputs = get_model_output(args, example_prompts, tokenizer, model)

    # run sampled pruning metric
    run_prune_metric(args, metric_ops, model, trainloader)
    pruned_outputs = get_model_output(args, example_prompts, tokenizer, model)

    # calculate final layer output difference
    reconstruction_norm = torch.norm(pruned_outputs-original_outputs, p="fro")
    eval_result = reconstruction_norm.item()

    return eval_result




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_path', type=str)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', type=int, default="cuda")
    parser.add_argument('--calibdation_data_path', type=str)
    parser.add_argument('--nsamples', type=int, default=128)
    parser.add_argument('--seq_len', type=int, default=2048)
    parser.add_argument('--sparsity_ratio', type=float, default=0.5)
    parser.add_argument('--ntrials', type=int, default=300)
    

    args = parser.parse_args()

    # Setting seeds for reproducibility
    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)

    # load calibdation_data
    with open(args.calibdation_data_path, 'rb') as file:
        calibdation_data = torch.load(args.calibdation_data_path, map_location=torch.device('cpu')) 
    trainloader = calibdation_data["trainloader"] 
    example_prompts = calibdation_data["example_prompts"]

    # run pruning parameter search 
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.NSGAIIISampler())

    print(f"Sampler is {study.sampler.__class__.__name__}")
    study.optimize(objective, n_trials=args.ntrials)

    # trial results
    trial = study.best_trial
    print('ppl: {}'.format(trial.value))
    print("Best hyperparameters: {}".format(trial.params))