import os
import numpy as np
import torch
import json
import itertools
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
import soundfile as sf
import argparse
import os
import glob
import random

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

def decode(toks):
    s = task.gpt_tokenizer.decode(toks.detach().cpu().int().numpy())
    return s

def generate(model, task, dataset, model_key, scenarios, device, max_batch):

    os.makedirs(output_dir, exist_ok=True)
    tokenizer = task.gpt_tokenizer
    n_sample = min(len(dataset), max_batch)
    print(f'write to: {output_dir}/{scenarios}.ltr...')
    if os.path.isfile(f'{output_dir}/{scenarios}.ltr') and os.path.isfile(f'{output_dir}/{scenarios}.tsv'):
        print(f'found existing {output_dir}/{scenarios}.ltr...')
        line_count = 0
        with open(f'{output_dir}/{scenarios}.ltr', 'r') as f:
            for l in f:
                line_count += 1
        if line_count >= n_sample:
            print(f'Already generated hypothesis. {line_count} >= {n_sample}. Skip...')
            return
        else:
            print(f'{line_count} == {n_sample}')
    total_sample = 0
    with torch.no_grad():
        with open(f'{output_dir}/{scenarios}.ltr', 'w') as fltr, \
            open(f'{output_dir}/{scenarios}.tsv', 'w') as ftsv:
            root_dir = dataset.dataset.root_dir
            ftsv.write(root_dir + '\n')
            indicies = list(range(len(dataset)))
            random.shuffle(indicies)
            for i in tqdm(indicies, total=n_sample):
                sample = dataset.collater([dataset[i]])
                sample = utils.move_to_cuda(sample)
                target = tokenizer.decode(sample['target'][0]).strip().strip('.')
                prompt = 'What is the scenario?'
                try:
                    gen_out = model.generate(sample)
                    gen_out = utils.strip_pad(gen_out[0], task.target_dictionary.pad())
                except:
                    continue
                
                
                hyp = decode(gen_out).split('\n')[0].strip('.').strip()
                fltr.write('\t'.join([target, prompt, hyp]) + '\n')
                length = sf.info(os.path.join(root_dir, dataset.dataset.fnames[i])).frames
                ftsv.write(f'{dataset.dataset.fnames[i]}\t{length}\t{target}\n')
                total_sample += 1
                if total_sample >= n_sample:
                    break

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest-dir', required=True)
    parser.add_argument('--ckpt-path-template', required=True)
    parser.add_argument('--ckpt-path-variable', metavar='N', type=int, nargs='+', required=True)
    parser.add_argument('--split', required=True)
    parser.add_argument('--all-scenarios', metavar='N', type=str, nargs='+', required=True)
    parser.add_argument('--max-batch', type=int, default=1000, nargs='?')
    parser.add_argument('--transcript-folder', required=True)
    
    args = parser.parse_args()
    
    data_dir = args.manifest_dir
    split = args.split
    all_scenarios = args.all_scenarios
    max_batch = args.max_batch
    output_dir = args.transcript_folder

    ckpt_paths = [args.ckpt_path_template.format(v=v) for v in args.ckpt_path_variable]
    rfs = [v for v in args.ckpt_path_variable]
    scenarios_list = ['_'.join(all_scenarios)]

    device = torch.device('cuda')

    print('[wav2gpt2_generate.py] data_dir:', data_dir, 'max_batch', max_batch, 'output_dir', output_dir)
    
    
    for scenario_pair in combinations(all_scenarios, 2):
        scenarios = '_'.join(scenario_pair)
        print('*****************scenarios', scenarios)
        for cki, ckpt_path in enumerate(ckpt_paths):
            model_key = ckpt_path.split("/")[1]
            if os.path.isfile(f'{output_dir}/{scenarios}.ltr') and os.path.isfile(f'{output_dir}/{scenarios}.tsv'):
                print(f'found existing {output_dir}/{scenarios}.ltr...')
                line_count = 0
                with open(f'{output_dir}/{scenarios}.ltr', 'r') as f:
                    for l in f:
                        line_count += 1
                if line_count >= max_batch:
                    print(f'Already generated hypothesis. {line_count} >= {max_batch}. Skip...')
                    continue 
                else:
                    # print(f'{line_count} == {max_batch}')
                    pass
            task_cfg, task, dataset = get_task_dataset(
                split, data_dir, n_shot=0, scenarios=scenarios, prompt='', example_order='prepend',
                seed=0, random=True
            )
            
            
            model = get_model(ckpt_path, device)
            generate(model, task, dataset, model_key, scenarios, device, max_batch)