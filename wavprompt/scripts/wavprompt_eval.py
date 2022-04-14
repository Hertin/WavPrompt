import os
import numpy as np
import torch
import json
import itertools
import matplotlib.pyplot as plt
import random
from tqdm import tqdm
from easydict import EasyDict as edict
from collections import Counter
from fairseq import checkpoint_utils, options, scoring, tasks, utils
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
from fairseq.logging import progress_bar
from fairseq.tasks.wavprompt_evaluation import WavPromptEvaluationConfig as TaskConfig
from fairseq.tasks.wavprompt_evaluation import WavPromptEvaluation as TaskClass
from collections import defaultdict
from itertools import product, combinations
import joblib
import re
import fairseq
import pandas as pd

import argparse
import os
import glob

def make_compact(inputs_embeds, attention_mask):
    inputs_embeds_list, attention_mask_list =  [], []
    length_list = []
    for input_emb, att_mask in zip(inputs_embeds, attention_mask):
        input_emb_compact = input_emb[att_mask]
        att_mask_compact = att_mask[att_mask]
        assert torch.all(att_mask_compact)
        assert len(att_mask_compact) == len(input_emb_compact), f'{len(att_mask_compact)} == {len(input_emb_compact)}'
        inputs_embeds_list.append(input_emb_compact)
        attention_mask_list.append(att_mask_compact)
        length_list.append(len(att_mask_compact))
    inputs_embeds_ = torch.nn.utils.rnn.pad_sequence(inputs_embeds_list, batch_first=True, padding_value=0)
    attention_mask_ = fairseq.data.data_utils.collate_tokens(attention_mask_list, pad_idx=False, left_pad=False)
    return inputs_embeds_, attention_mask_

def get_task_dataset(split, data_dir, n_shot, scenarios, prompt='=>', example_order='prepend', random=False, seed=0):
    task_cfg = TaskConfig()
    task_cfg.data = data_dir
    task_cfg.gpt_path = 'models/gpt2'
    task_cfg._name = 'wavprompt_evaluation'
    task_cfg.labels = 'ltr'
    task_cfg.autoregressive = True
    task_cfg.normalize = False
    task_cfg.eval_wer = True
    task_cfg.scenarios = scenarios
    task_cfg.n_shot = n_shot
    task = TaskClass(task_cfg)

    data_cfg = edict({'sample_rate': task_cfg.sample_rate, 'enable_padding': True, 'normalize': False, 'labels': 'ltr'})
    task.load_dataset(split=split, task_cfg=data_cfg, scenarios=scenarios, prompt=prompt, example_order=example_order, random=random, seed=seed)
    dataset = task.datasets[split]
    return task_cfg, task, dataset

def get_model(ckpt_path, device):
    ckpt_path = glob.glob(ckpt_path)[0]
    models, saved_cfg = checkpoint_utils.load_model_ensemble([ckpt_path])
    model = models[0]
    model = model.eval()
    model = model.to(device)
    return model

def forward_gpt2(model, input_ids, inputs_embeds, attention_mask, l, echo, labels):
    position_ids = attention_mask.long().cumsum(-1) - 1
    position_ids.masked_fill_(attention_mask == 0, 1)
    
    gpt2 = model.decoder.decoder

    logits = gpt2.forward(input_ids=None, inputs_embeds=inputs_embeds, attention_mask=attention_mask, position_ids=position_ids, return_dict=True).logits.detach().cpu()

    if not echo:
        assert l == 1, f'l = {l}'

    if l == 0:
        probs = torch.softmax(logits[:,-l-1:], dim=2).cpu()
    else:
        probs = torch.softmax(logits[:,-l-1:-l], dim=2).cpu()
    
    return probs[:, :, labels]

def fewshot_learning(model, task, dataset, device, seed=0, n_shot=0, max_batch=None, test_ids=None, n_results=100, generation_options={}, content_free_inputs=["N/A", "", "[MASK]"]):
    model.decoder.pad_idx = task.gpt_tokenizer.eos_token_id

    tokenizer = task.gpt_tokenizer
    hyps = []
    refs = []
    txts = []
    full_hyps = []
    results = []
    sub_idx_actual = []
    scenario2id = dataset.scenario2id
    n_tok_to_pred = 1
    examples = None # uninitialized example embedding
    labels = torch.LongTensor([task.gpt_tokenizer.encode(f' {s}') for s in scenario2id.keys()]).view(-1).to(device)
    itr = task.get_batch_iterator(
        dataset=task.dataset(split),
        max_tokens=1280000,
        max_sentences=1,
        max_positions=utils.resolve_max_positions(
            task.max_positions(), model.max_positions()
        ),
        ignore_invalid_inputs=True,
        required_batch_size_multiple=1,
        seed=seed,
        num_shards=1,
        shard_id=0,
        num_workers=2,
        data_buffer_size=0,
    ).next_epoch_itr(shuffle=True)


    probs_list = []
    targets_list = []
    count = 0
    with torch.no_grad():
        for i, sample in tqdm(enumerate(itr)):
            if max_batch is not None and count > max_batch:
                print(f'{max_batch} batch is enough')
                break

            sample = utils.move_to_cuda(sample)
            net_input = sample['net_input']
            if n_shot == 0:
                assert examples is None
            if examples is not None:
                net_input['examples'] = examples
            encoder_out = model.encoder( **net_input )

            examples = encoder_out.get('examples', None) # initialize example embedding after first forward

            input_ids = net_input['prev_output_tokens'] # just use if to convert  to cuda
            transcripts_ids = net_input['gpt2_text']

            inputs_embeds, attention_mask = model.decoder.get_inputs_embeds_for_generateion_slurp(
                prev_output_tokens=None, encoder_out=encoder_out, **generation_options
            )
            assert len(inputs_embeds) == 1, f'{inputs_embeds.shape}'
            if inputs_embeds.size(1) > 900:
                continue
            probs = forward_gpt2(model, input_ids, inputs_embeds, attention_mask, l=0, echo=True, labels=labels).squeeze(1)

            probs_list.append(probs)
            targets_1tok = sample['target'][:, 0].clone()
            targets = targets_1tok.clone()
            for i, l in enumerate(labels):
                targets_1tok[targets_1tok == l] = i
            targets_list.append(targets_1tok.detach().cpu())
            count += 1

        if len(probs_list) > 0:

            predprobs = torch.cat(probs_list)
            predprobs = predprobs / predprobs.sum(1, keepdim=True)
            groundtruth = torch.cat(targets_list)
            
            p_cf = get_p_content_free(model, tokenizer, input_ids, encoder_out, content_free_inputs, generation_options, labels)
            
            truths_np = groundtruth.numpy()
            preds_np = predprobs.numpy()
            cnt = Counter(truths_np)
            n_chop = min(cnt.values())
            unique_labels = np.unique(truths_np)
            distr = [str(len(truths_np[truths_np==l][:n_chop])) for l in unique_labels]
            truths_list = np.concatenate([truths_np[truths_np==l][:n_chop] for l in unique_labels])
            pred_list = np.concatenate([preds_np[truths_np==l][:n_chop] for l in unique_labels])

            acc = eval_accuracy(pred_list, truths_list, mode=None)
            acc_cf = eval_accuracy(pred_list, truths_list, mode='diagonal_W', p_cf=p_cf.numpy())

            print(f'acc: {acc_cf} ({acc}) n_chop:{n_chop}')

            return predprobs, groundtruth, p_cf, acc, acc_cf
        
        return None

def get_p_content_free(model, tokenizer, input_ids, encoder_out, content_free_inputs, generation_options, labels):
    """Query model with content free input, return its prediction probability for each label"""
    n_tok_to_pred = 1
    input_ids = torch.zeros(1).to(input_ids)
    probs_list = []
    padding_mask = encoder_out['encoder_padding_mask'][0]
    
    prompt = encoder_out['gpt2_input_tokens'].clone()
    examples_pt = encoder_out.get('examples', None)
    if examples_pt is not None:
        print('tgt', examples_pt['tgt'].shape, 'prt', examples_pt['prt'].shape, 'txt', examples_pt['txt'].shape, 'src', examples_pt['encoder_out'][0].shape)
    else:
        print('examples_pt is none')
    for cfi in content_free_inputs:
        label_probs = []
        for label in labels:
            cftok = tokenizer.encode(cfi, return_tensors='pt').to(input_ids)
            cfemb = model.decoder.decoder.transformer.wte(cftok)

            encoder_out['encoder_out'] = [cfemb]
            encoder_out['encoder_padding_mask'] = [torch.zeros(cfemb.shape[:2]).to(padding_mask)]
            
            encoder_out['gpt2_text'] = cftok # transcripts
            encoder_out['gpt2_input_tokens'] = torch.cat([prompt, label.view(1, -1)], dim=1) # prompt
            
            inputs_embeds, attention_mask = model.decoder.get_inputs_embeds_for_generateion_slurp(
                prev_output_tokens=None, encoder_out=encoder_out, **generation_options
            )
#             try:
            probs = forward_gpt2(model, input_ids, inputs_embeds, attention_mask, l=n_tok_to_pred, echo=True, labels=label).squeeze(1)
            
#             except:
#                 continue
            label_probs.append(probs)
        label_probs = torch.cat(label_probs).unsqueeze(0)
        probs_list.append(label_probs)
    all_probs = torch.cat(probs_list, dim=0)
    all_probs = all_probs.mean(dim=0)
    all_probs = all_probs / all_probs.sum()

    return all_probs

def eval_accuracy(all_label_probs, test_labels, mode=None, p_cf=None):
    # evaluate the accuracy with and without contextual calibration
    num_classes = all_label_probs.shape[1]
    if p_cf is None:
        # do not calibrate
        W = np.identity(num_classes)
        b = np.zeros([num_classes, 1])
    else:
        # calibrate
        if mode == "diagonal_W":
            W = np.linalg.inv(np.identity(num_classes) * p_cf)
            b = np.zeros([num_classes, 1])
        elif mode == "identity_W":
            W = np.identity(num_classes)
            b = -1 * np.expand_dims(p_cf, axis=-1)
        else:
            assert False

    correctness_list = []
    assert len(all_label_probs) == len(test_labels)
    for label_probs, true_label in zip(all_label_probs, test_labels):
        label_probs = label_probs / np.sum(label_probs) # normalize to 1

        calibrate_label_probs = np.matmul(W, np.expand_dims(label_probs, axis=-1)) + b

        ans_label = np.argmax(calibrate_label_probs)
        if ans_label == true_label:
            correctness_list.append(1)
        else:
            correctness_list.append(0)
    return np.mean(correctness_list)


def do_few_shot(all_scenarios, n_shots, ckpt_paths, all_exp, example_order, max_batch, seeds=None):
    random = seeds is not None
    if not random:
        seeds = [None]
    
    full_results = {}
    device = torch.device('cuda')
    

    n_results = 0

    datasets = {}

    generation_options_dict = {
        key: all_generation_options_dict[key] for key in all_exp
    }

    for scenario_pair in combinations(all_scenarios, 2):
        scenarios = '_'.join(scenario_pair)

        print('*****************scenarios', scenarios)
        for cki, ckpt_path in enumerate(ckpt_paths):
            train_model_tag = ckpt_path.split("/")[1]
            load_model = False
            output_dir = f'{output_folder}/{scenarios}'
            os.makedirs(output_dir, exist_ok=True)

            for n_shot in n_shots:
                for expname, generation_options in generation_options_dict.items():
                    for seed in seeds:
                        output_path = f'{output_folder}/{scenarios}/{train_model_tag}__{scenarios}__{n_shot}_{expname}_{seed}.pk'
                        if os.path.isfile(output_path):
#                             print(f'skip {output_path}')
                            continue
                        load_model = True
            if not load_model:
#                 print(f'skip model {ckpt_path}...')
                continue
            model = get_model(ckpt_path, device)

            for n_shot in n_shots:
                for expname, generation_options in generation_options_dict.items():
                    for seed in seeds:
                        output_path = f'{output_folder}/{scenarios}/{train_model_tag}__{scenarios}__{n_shot}_{expname}_{seed}.pk'
                        if expname == 'noexembtxt' or expname == 'txt' or expname ==  'noexembnotxt':
                            if cki != 0:
#                                 print(f'skip {output_path}')
                                joblib.dump([], output_path)
                                continue
                        if os.path.isfile(output_path):
                            continue
                        print('----------------------------------------------------experiment', ckpt_path, n_shot, expname)

                        if (scenarios, n_shot) in datasets:
                            task_cfg, task, dataset = datasets[(scenarios, n_shot, seed)]
                        else:
                            task_cfg, task, dataset = get_task_dataset(
                                split, data_dir, n_shot, scenarios=scenarios, prompt=prompt, example_order=example_order,
                                seed=seed, random=random
                            )
                            datasets[(scenarios, n_shot, seed)] = task_cfg, task, dataset

                        result = fewshot_learning(
                            model, task, dataset, device=device, seed=seed, n_shot=n_shot,
                            max_batch=max_batch, n_results=n_results, generation_options=generation_options
                        )
                        full_results[f'{ckpt_path}_{scenarios}_{n_shot}'] = result
                        joblib.dump(result, output_path)


def print_results(all_scenarios, ckpt_paths, n_shots, generation_options_dict, seeds, ckpt_path_variable):
    scenarios = '_'.join(all_scenarios)
    

    for expname, generation_options in generation_options_dict.items():
        print(f'exp: {expname}')
        result_table = []
        for cki, ckpt_path in enumerate(ckpt_paths):
            train_model_tag = ckpt_path.split("/")[1]
            result_column = []
            for n_shot in n_shots:
                results = []
                for seed in seeds:
                    output_path = f'{output_folder}/{scenarios}/{train_model_tag}__{scenarios}__{n_shot}_{expname}_{seed}.pk'
                    result = joblib.load(output_path)
                    if result:
                        predprobs, groundtruth, p_cf, acc, acc_cf = result
                        truths_np = groundtruth.numpy()
                        preds_np = predprobs.numpy()
                        cnt = Counter(truths_np)
                        n_chop = min(cnt.values())
                        if n_chop < 10:
                            continue
                        unique_labels = np.unique(truths_np)
                        distr = [str(len(truths_np[truths_np==l][:n_chop])) for l in unique_labels]
                        truths_list = np.concatenate([truths_np[truths_np==l][:n_chop] for l in unique_labels])
                        pred_list = np.concatenate([preds_np[truths_np==l][:n_chop] for l in unique_labels])
                        
                        acc = eval_accuracy(pred_list, truths_list, mode=None)
                        acc_cf = eval_accuracy(pred_list, truths_list, mode='diagonal_W', p_cf=p_cf.numpy())
                        
                        results.append([acc, acc_cf, n_chop])

                if len(results) > 0:
                    acc, acc_cf, nresult = np.array(results).mean(axis=0)
                    result_column.append(f'{acc_cf*100:.2f}% ({acc*100:.2f}) (Nr:{nresult})')
                else:
                    result_column.append(None)
            result_table.append(result_column)
        df = pd.DataFrame(result_table)
        df.columns = n_shots
        df = df.T
        df.columns = ckpt_path_variable
        print(df.to_csv())

all_generation_options_dict = {
    'base': { # no example audio/txt embedding and not input audio/txt embedding, basically random guess
        'exp': 'base',
        'use_txt': False,
        'use_example_encoder_out': True,
        'use_encoder_out': True,

    },
    'txt': { # example txt embedding and have input txt embedding
        'exp': 'txt',
        'use_txt': True, 
        'use_example_encoder_out': True,
        'use_encoder_out': True,
    },
    'noexemb': { # no example audio/txt embedding and have input audio embedding
        'exp': 'noexemb',
        'use_txt': False, 
        'use_example_encoder_out': False,
        'use_encoder_out': True,
    },
    'noexembtxt': { # no example audio/txt embedding and have input txt embedding
        'exp': 'noexembtxt',
        'use_txt': True,
        'use_example_encoder_out': False,
        'use_encoder_out': False,
        
    },
    'noexembnotxt': { # no example audio/txt embedding and not input audio/txt embedding, basically random guess
        'exp': 'noexembnotxt',
        'use_txt': False,
        'use_example_encoder_out': False,
        'use_encoder_out': False,
        
    },
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest-dir', required=True)
    parser.add_argument('--output-folder', required=True)
    parser.add_argument('--ckpt-path-template', required=True)
    parser.add_argument('--ckpt-path-variable', metavar='N', type=int, nargs='+', required=True)
    parser.add_argument('--split', required=True)
    parser.add_argument('--prompt', required=True)
    parser.add_argument('--suffix', required=True)
    parser.add_argument('--max-batch', type=int, default=None, nargs='?')
    parser.add_argument('--all-scenarios', metavar='N', type=str, nargs='+', required=True)
    parser.add_argument('--exp', metavar='N', type=str, nargs='+', default=['base'])
    parser.add_argument('--example-order', type=str, default='prepend')
    args = parser.parse_args()
    
    data_dir = args.manifest_dir
    split = args.split
    all_scenarios = args.all_scenarios
    prompt = args.prompt
    output_folder = args.output_folder
    ckpt_paths = [args.ckpt_path_template.format(v=v) for v in args.ckpt_path_variable]
    rfs = [v for v in args.ckpt_path_variable]
    max_batch = args.max_batch
    scenarios_list = ['_'.join(all_scenarios)]
    all_exp = args.exp
    suffix = args.suffix
    example_order = args.example_order
    
    print('data_dir:', data_dir)
    print('split:', split)
    print('all_scenarios:', all_scenarios)
    print('prompt:', prompt)
    print('output_folder:', output_folder)
    print('ckpt_paths:', ckpt_paths)
    print('max_batch:', max_batch)
    print('all_exp:', all_exp)

    full_results = {}
    device = torch.device('cuda')
    n_shots = [0, 1, 2, 3, 4, 6, 8, 10,]

    seeds = [0, 1, 2, 3, 4]
    n_results = 0
    
    datasets = {}

    generation_options_dict = {
        key: all_generation_options_dict[key] for key in all_exp
    }

    do_few_shot(all_scenarios, n_shots, ckpt_paths, all_exp, example_order, max_batch, seeds)
    print_results(all_scenarios, ckpt_paths, n_shots, generation_options_dict, seeds, args.ckpt_path_variable)

                    
