# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import numpy as np
from . import BaseWrapperDataset, data_utils


class AddTargetDatasetWavPromptEvaluation(BaseWrapperDataset):
    def __init__(
        self,
        dataset,
        labels,
        pad,
        eos,
        batch_targets,
        process_label=None,
        add_to_input=False,
        n_shot=0,
        scenario2id=None,
        scenarios=None,
        n_repeat=1,
        use_hyp=False,
        seed=0,
        **unused
    ):
        super().__init__(dataset)
        self.labels = labels
        self.batch_targets = batch_targets
        self.pad = pad
        self.eos = eos
        self.process_label = process_label
        self.add_to_input = add_to_input

        self.n_shot = n_shot
        self.n_repeat = n_repeat
        self.examples = []
        self.scenario2id = scenario2id

        self.text_index = 2 if not use_hyp else 5
        print('AddTargetDatasetSLURPZRandom use_hyp', 
            use_hyp, self.text_index, scenarios, 
            'self.pad, self.eos', 
            self.pad, self.eos, 
            'nshot', self.n_shot, 'n_repeat', n_repeat,
            'seed', seed
        )

        assert self.pad == 50256 and self.eos == 50256

        chop_to = min([len(v) for v in scenario2id.values()])

        all_ids = sum([v[:chop_to] for v in scenario2id.values()], [])

        if n_shot > 0 and n_repeat > 0:
            np.random.seed(seed)
            example_ids = np.random.choice(all_ids, n_shot)
            example_ids = example_ids.reshape(1,-1).repeat(n_repeat, axis=0).ravel()

            # print('example_ids', example_ids, 'scenario2id', [(s, scenario2id[s][:n_shot]) for s in scenarios])
            for i in example_ids:
                it = self.dataset[i]
                it['label'] = self.get_label(i)
                self.examples.append(it)

            self.examples_pt = {}
            self.examples_pt = self.dataset.collater(self.examples)

            idx = set(range(self.n_shot))

            tgt = [s["label"][0] for s in self.examples if s["id"] in example_ids] # e.g.: "Answer: 1996"
            prt = [s["label"][1] for s in self.examples if s["id"] in example_ids] # e.g.: "Question: What did the speaker say?"
            txt = [s["label"][self.text_index] for s in self.examples if s["id"] in example_ids] # paragraph

            self.examples_pt['tgt'] = data_utils.collate_tokens(tgt, pad_idx=self.pad, left_pad=False)
            self.examples_pt['prt'] = data_utils.collate_tokens(prt, pad_idx=self.pad, left_pad=False)
            self.examples_pt['txt'] = data_utils.collate_tokens(txt, pad_idx=self.pad, left_pad=False)
            self.examples_pt['src'] = self.examples_pt['net_input']['source']

    def get_label(self, index):
        return self.labels[index]

    def __getitem__(self, index):
        item = self.dataset[index]
        item["label"] = self.get_label(index)

        return item

    def size(self, index):
        sz = self.dataset.size(index)
        own_sz = len(self.get_label(index))
        return (sz, own_sz)

    def collater(self, samples):
        collated = self.dataset.collater(samples)
        if len(collated) == 0:
            return collated
        indices = set(collated["id"].tolist())
        target = [s["label"][0] for s in samples if s["id"] in indices] # e.g.: "Answer: 1996"
        prompt = [s["label"][3] for s in samples if s["id"] in indices] # e.g.: "Question: What did the speaker say? Answer: "
        qstext = [s["label"][self.text_index] for s in samples if s["id"] in indices]

        attention_mask_raw = [torch.LongTensor([1] * len(t)) for t in target]

        if self.batch_targets:
            collated["target_lengths"] = torch.LongTensor([len(t) for t in target])
            target = data_utils.collate_tokens(target, pad_idx=self.pad, left_pad=False)
            collated["ntokens"] = collated["target_lengths"].sum().item()
        else:
            collated["ntokens"] = sum([len(t) for t in target])

        prev_output_tokens = target.clone()

        attention_mask = data_utils.collate_tokens(attention_mask_raw, pad_idx=0, left_pad=False)
        assert attention_mask.shape == target.shape, f'attention_mask {attention_mask.shape} != target {target.shape}'

        collated["target"] = target.long()            
        collated["net_input"]["prev_output_tokens"] = prev_output_tokens.long()
        collated["net_input"]["attention_mask"] = attention_mask
        
        collated["net_input"]["gpt2_input_tokens"] = data_utils.collate_tokens(prompt, pad_idx=self.pad, left_pad=False) # prompt
        collated["net_input"]["gpt2_text"] = data_utils.collate_tokens(qstext, pad_idx=self.pad, left_pad=False) # transcription

        if self.n_shot > 0 and self.n_repeat > 0:
            collated["net_input"]['examples'] = self.examples_pt

        collated["net_input"]["id"] = torch.LongTensor([s["id"] for s in samples if s["id"] in indices])
        collated["ntokens"] += target.size(0)

        return collated
