# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import logging
import os
import sys
import torch
import joblib

from argparse import Namespace
from dataclasses import dataclass, field
from typing import Optional, Any
from omegaconf import MISSING, II, OmegaConf

from fairseq.data import (
    Dictionary, 
    FileAudioLabelDataset,
    encoders,
)
from fairseq.dataclass import FairseqDataclass
from fairseq.dataclass.configs import GenerationConfig
from fairseq.data import AddTargetDatasetWavPromptEvaluation

from . import FairseqTask, register_task
from .. import utils
from ..logging import metrics

import pickle
import numpy as np
from fairseq.models.transformer_lm import TransformerLanguageModel
from transformers import GPT2Tokenizer, GPT2Config

logger = logging.getLogger(__name__)


class LabelEncoder(object):
    def __init__(self, dictionary):
        self.dictionary = dictionary

    def __call__(self, label):
        return self.dictionary.encode_line(
            label, append_eos=False, add_if_not_exist=False
        )



@dataclass
class WavPromptEvaluationConfig(FairseqDataclass):
    data: str = field(default=MISSING, metadata={"help": "path to data directory"})
    labels: Optional[str] = field(
        default=None,
        metadata={"help": "extension of the label file to load, used for fine-tuning"},
    )
    sample_rate: int = field(
        default=16_000,
        metadata={
            "help": "target sample rate. audio files will be up/down sampled to this rate"
        },
    )
    normalize: bool = field(
        default=False,
        metadata={"help": "if set, normalizes input to have 0 mean and unit variance"},
    )
    enable_padding: bool = field(
        default=False, metadata={"help": "pad shorter samples instead of cropping"}
    )
    max_sample_size: Optional[int] = field(
        default=None, metadata={"help": "max sample size to crop to for batching"}
    )
    min_sample_size: Optional[int] = field(
        default=None, metadata={"help": "min sample size to skip small examples"}
    )

    # Options for reporting WER metrics during validation. Only applicable to
    # Seq2Seq models during fine-tuning
    eval_wer: bool = field(
        default=False, metadata={"help": "compute WER for Seq2Seq models"}
    )
    eval_wer_config: GenerationConfig = field(
        default_factory=lambda: GenerationConfig(),
        metadata={"help": "beam search config for evaluating wer during training"},
    )
    eval_wer_tokenizer: Any = field(
        default=None,
        metadata={"help": "tokenizer config for evaluating wer during training"},
    )
    eval_wer_post_process: str = field(
        default="letter",
        metadata={
            "help": "remove BPE tokens before scoring (can be sentencepiece, letter, and more)"
        },
    )
    autoregressive: bool = field(
        default=False,
        metadata={
            "help": "required for autoregressive decoders (like seq2seq models); "
            "adds 'prev_output_tokens' to input and appends eos to target"
        },
    )
    debug: bool = field(
        default=False,
    )

    gpt_path: str = field(
        default="",
        metadata={"help": "path of bart model"},
    )

    gpt_type: str = field(
        default="gpt2",
        metadata={"help": "path of bart model"},
    )

    scenarios: str = field(
        default='news_calendar',
        metadata={"help": "path of bart model"},
    )

    n_shot: int = field(
        default=2,
        metadata={"help": "number of examples"},
    )

    n_repeat: int = field(
        default=1,
        metadata={"help": "number of examples"},
    )

@register_task("wavprompt_evaluation", dataclass=WavPromptEvaluationConfig)
class WavPromptEvaluation(FairseqTask):
    def __init__(
        self,
        cfg: WavPromptEvaluationConfig,
    ):
        super().__init__(cfg)
        if cfg.eval_wer:
            assert cfg.labels is not None, "eval_wer can only be set during fine-tuning"
        self.blank_symbol = "<s>"

        if os.path.isfile(f'{cfg.gpt_path}/{cfg.gpt_type}_tok.pk'):
            self.gpt_tokenizer = joblib.load(f'{cfg.gpt_path}/{cfg.gpt_type}_tok.pk')
        else:
            self.gpt_tokenizer = GPT2Tokenizer.from_pretrained(pretrained_model_name_or_path=cfg.gpt_type, cache_dir=cfg.gpt_path)
            joblib.dump(self.gpt_tokenizer, f'{cfg.gpt_path}/{cfg.gpt_type}_tok.pk')
        
        if os.path.isfile(f'{cfg.gpt_path}/{cfg.gpt_type}_config.pk'):
            self.gpt_config = joblib.load(f'{cfg.gpt_path}/{cfg.gpt_type}_config.pk')
        else:
            self.gpt_config = GPT2Config.from_pretrained(pretrained_model_name_or_path=cfg.gpt_type, cache_dir=cfg.gpt_path)
            joblib.dump(self.gpt_config, f'{cfg.gpt_path}/{cfg.gpt_type}_config.pk')

        tgt_dict = self.load_target_dictionary(self.gpt_tokenizer)
        self.state.merge_state_dict({'target_dictionary': tgt_dict})
        
    @classmethod
    def setup_task(cls, cfg: WavPromptEvaluationConfig, **kwargs):
        """Setup the task (e.g., load dictionaries).
        Args:
            cfg (WavPromptEvaluationConfig): configuration of this task
        """

        return cls(cfg)

    def load_target_dictionary(self, tokenizer):
        tgt_dict = Dictionary() # 0: '<s>', 1: '<pad>', 2: '</s>', 3: '<unk>'

        ## clear the pre-defined special symbols in fairseq
        tgt_dict.symbols = []
        tgt_dict.count = []
        tgt_dict.indices = {}
        for idx, sym in tokenizer.decoder.items():
            # symbol = sym if idx not in special_tokens else special_tokens[idx]
            tgt_dict.add_symbol(sym)

        assert len(tgt_dict) == len(tokenizer.decoder) == 50257, "error when initializing word2index dict!"

        tgt_dict.bos_index, tgt_dict.unk_index, tgt_dict.pad_index, tgt_dict.eos_index = 50256, 50256, 50256, 50256
        return tgt_dict
    
    def encode(self, sentence):
        # 'hello,  world' => '<s> hello , Ġ Ġworld </s>'
        tokens = self.gpt_tokenizer.tokenize(sentence)

        ## only append [EOS]
        if len(tokens) > self.gpt_config.max_position_embeddings - 1:
            tokens = tokens[: self.gpt_config.max_position_embeddings - 1]
        tokens = [self.target_dictionary.bos_word] + tokens + [self.target_dictionary.eos_word]
        bpe_sentence = ' '.join(tokens)
        return bpe_sentence

        
    def load_dataset(self, split: str, task_cfg: FairseqDataclass=None, use_hyp=False, scenarios=None, prompt='=>', seed=None, random=False, **kwargs):
        data_path = self.cfg.data
        task_cfg = task_cfg or self.cfg

        scenarios = (scenarios or getattr(task_cfg, 'scenarios', None)) or self.cfg.scenarios
        if scenarios is not None:
            scenarios = scenarios.split('_')
        n_shot = getattr(task_cfg, 'n_shot', None) or self.cfg.n_shot
        n_repeat = getattr(task_cfg, 'n_repeat', None) or self.cfg.n_repeat

        example_order = kwargs.get('example_order', 'prepend')

        # upgrade old task
        if isinstance(task_cfg, Namespace):
            if not hasattr(task_cfg, "autoregressive"):
                task_cfg.autoregressive = not task_cfg.criterion == 'ctc'

        manifest = os.path.join(data_path, "{}.tsv".format(split))
        self.datasets[split] = FileAudioLabelDataset(
            manifest,
            sample_rate=task_cfg.get('sample_rate', self.cfg.sample_rate),
            max_sample_size=self.cfg.max_sample_size,
            min_sample_size=self.cfg.min_sample_size,
            pad=self.target_dictionary.pad_index, # task_cfg.labels is not None or task_cfg.enable_padding,
            eos=self.target_dictionary.eos_index,
            normalize=task_cfg.normalize,
            scenarios=scenarios
        )

        
        if task_cfg.labels:
            label_path = os.path.join(data_path, f"{split}.{task_cfg.labels}")
            skipped_indices = getattr(self.datasets[split], 'skipped_indices', set())
            
            labels = []
            

            with open(label_path, "r") as f:
                for i, line in enumerate(f):
                    if i in skipped_indices:
                        continue
                    answer, question, text, *hyp = line.strip('\n').split('\t')

                    assert len(hyp) <= 1
                    # print(hyp)
                    hyp = [h.strip(".").replace("#", "\t").replace("*", "\n") for h in hyp]
                    entry = tuple([
                        self.gpt_tokenizer.encode(f' {answer}\n', return_tensors='pt')[0], # answer
                        self.gpt_tokenizer.encode(prompt, return_tensors='pt')[0], # e.g., This is a scenario of news
                        self.gpt_tokenizer.encode(f'{text.strip(".")}', return_tensors='pt')[0].long(), # text
                        self.gpt_tokenizer.encode(prompt, return_tensors='pt')[0],
                        answer,
                        self.gpt_tokenizer.encode(f'{hyp[0]}', return_tensors='pt')[0] if len(hyp) > 0 else torch.LongTensor([]),
                    ])
                    labels.append(entry)

            
            from collections import Counter, defaultdict
            scenario2id = defaultdict(list)
            for ii, (_, __, ___, ____, answer, *hyp) in enumerate(labels):
                scenario2id[answer].append(ii)

            assert len(labels) == len(self.datasets[split]), (
                    f"labels length ({len(labels)}) and dataset length "
                    f"({len(self.datasets[split])}) do not match")

            self.datasets[split] = AddTargetDatasetWavPromptEvaluation(
                self.datasets[split],
                labels,
                pad=self.target_dictionary.pad(),
                eos=self.target_dictionary.eos(),
                batch_targets=True,
                process_label=None,
                add_to_input=task_cfg.get('autoregressive', False),
                n_shot=n_shot,
                scenario2id=scenario2id,
                scenarios=scenarios,
                n_repeat=n_repeat,
                use_hyp=use_hyp,
                example_order=example_order,
                random=random,
                seed=seed
            )

    @property
    def source_dictionary(self):
        return None

    @property
    def target_dictionary(self):
        """Return the :class:`~fairseq.data.Dictionary` for the language
        model."""
        return self.state.target_dictionary

    def max_positions(self):
        """Maximum input length supported by the encoder."""
        return (sys.maxsize, sys.maxsize)

    def filter_indices_by_size(
        self,
        indices,
        dataset,
        max_positions=None,
        ignore_invalid_inputs=False,
    ):
        # we do not need to filter by size in this task as dataloaders take care of this
        return indices

    def valid_step(self, sample, model, criterion):
        loss, sample_size, logging_output = super().valid_step(sample, model, criterion)

        if self.cfg.eval_wer and self.cfg.autoregressive:
            metrics = self._inference_with_wer(self.sequence_generator, sample, model)
            logging_output["_num_char_errors"] = metrics["num_char_errors"]
            logging_output["_num_chars"] = metrics["num_chars"]
            logging_output["_num_word_errors"] = metrics["num_word_errors"]
            logging_output["_num_words"] = metrics["num_words"]
        return loss, sample_size, logging_output

    def build_model(self, model_cfg: FairseqDataclass):
        model = super().build_model(model_cfg)
        if self.cfg.eval_wer and self.cfg.autoregressive:
            self.sequence_generator = self.build_generator(
                [model],
                self.cfg.eval_wer_config,
            )
            if self.cfg.eval_wer_tokenizer:
                self.tokenizer = encoders.build_tokenizer(self.cfg.eval_wer_tokenizer)
            else:
                self.tokenizer = None
        return model

    def _inference_with_wer(self, generator, sample, model):
        import editdistance

        def decode(toks):
            s = self.gpt_tokenizer.decode(toks.detach().cpu().int().numpy())
            return s

        num_word_errors, num_char_errors = 0, 0
        num_chars, num_words = 0, 0
        gen_out = self.inference_step(generator, [model], sample, None)
        for i in range(len(gen_out)):
            hyp = decode(gen_out[i][0]["tokens"])
            ref = decode(
                utils.strip_pad(sample["target"][i], self.target_dictionary.pad()),
            )
            # print(gen_out[i][0]["tokens"])
            # print(sample["target"][i])
            print('test', hyp, ' || ', ref)
            num_char_errors += editdistance.eval(hyp, ref)
            num_chars += len(ref)
            hyp_words = hyp.split()
            ref_words = ref.split()
            num_word_errors += editdistance.eval(hyp_words, ref_words)
            num_words += len(ref_words)

        return {
            "num_char_errors": num_char_errors,
            "num_chars": num_chars,
            "num_word_errors": num_word_errors,
            "num_words": num_words,
        }


    def reduce_metrics(self, logging_outputs, criterion):
        super().reduce_metrics(logging_outputs, criterion)

        zero = torch.scalar_tensor(0.0)
        num_char_errors = sum(
            log.get("_num_char_errors", zero) for log in logging_outputs
        )
        num_chars = sum(log.get("_num_chars", zero) for log in logging_outputs)
        num_word_errors = sum(
            log.get("_num_word_errors", zero) for log in logging_outputs
        )
        num_words = sum(log.get("_num_words", zero) for log in logging_outputs)
        metrics.log_scalar("_num_char_errors", num_char_errors)
        metrics.log_scalar("_num_chars", num_chars)
        metrics.log_scalar("_num_word_errors", num_word_errors)
        metrics.log_scalar("_num_words", num_words)
        if num_chars > 0:
            metrics.log_derived(
                "uer",
                lambda meters: meters["_num_char_errors"].sum
                * 100.0
                / meters["_num_chars"].sum
                if meters["_num_chars"].sum > 0
                else float("nan"),
            )
        if num_words > 0:
            metrics.log_derived(
                "wer",
                lambda meters: meters["_num_word_errors"].sum
                * 100.0
                / meters["_num_words"].sum
                if meters["_num_words"].sum > 0
                else float("nan"),
            )
