from fairseq.models.fairseq_decoder import FairseqDecoder
from json import encoder
import pickle
import contextlib
from numpy.core.fromnumeric import argsort
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules import padding
from fairseq import hub_utils, file_utils,checkpoint_utils
from fairseq.dataclass.utils import (
    convert_namespace_to_omegaconf,
    overwrite_args_by_name,
)
import soundfile as sf
from fairseq.optim import adam
from torch.optim import AdamW

import numpy as np
import argparse
from dataclasses import dataclass, field
from fairseq.dataclass import FairseqDataclass
from omegaconf import MISSING, II, open_dict
from fairseq.tasks import FairseqTask

from fairseq.models import (
    BaseFairseqModel,
    FairseqEncoder,
    FairseqEncoderDecoderModel,
    FairseqIncrementalDecoder,
    register_model,
)
from fairseq.models.wav2vec.wav2vec2 import MASKING_DISTRIBUTION_CHOICES
from fairseq.modules import LayerNorm, PositionalEmbedding, TransformerDecoderLayer
from typing import Optional, Any, Tuple
from argparse import Namespace
from fairseq import checkpoint_utils, tasks, utils
import os
from fairseq.models.transformer import TransformerDecoder

from transformers import GPT2LMHeadModel, GPT2Config, GPT2Tokenizer, GPTNeoForCausalLM, GPTNeoConfig
from transformers.modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
    SequenceClassifierOutputWithPast,
    TokenClassifierOutput,
)
from transformers.modeling_utils import (
    Conv1D,
    PreTrainedModel,
    SequenceSummary,
    find_pruneable_heads_and_indices,
    prune_conv1d_layer
)
from types import MethodType
from transformers.utils import logging
logger = logging.get_logger(__name__)

def prepare_inputs_for_generation(self, input_ids, past=None, **kwargs):
    token_type_ids = kwargs.get("token_type_ids", None)
    # only last token for inputs_ids if past is defined in kwargs
    if past:
        input_ids = input_ids[:, -1].unsqueeze(-1)
        if token_type_ids is not None:
            token_type_ids = token_type_ids[:, -1].unsqueeze(-1)

    attention_mask = kwargs.get("attention_mask", None)
    position_ids = kwargs.get("position_ids", None)

    if attention_mask is not None and position_ids is None:
        # create position_ids on the fly for batch generation
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)
        if past:
            position_ids = position_ids[:, -1].unsqueeze(-1)
    else:
        position_ids = None

    return {
        "input_ids": input_ids,
        "past_key_values": past,
        "use_cache": kwargs.get("use_cache"),
        "position_ids": position_ids,
        "attention_mask": attention_mask,
        "token_type_ids": token_type_ids,
        "encoder_hidden_states": kwargs.get('encoder_hidden_states', None),
        "encoder_attention_mask": kwargs.get('encoder_attention_mask', None),
        "inputs_embeds": kwargs.get('inputs_embeds', None)
    }


@dataclass
class WavPromptConfig(FairseqDataclass):
    w2v_path: str = field(
        default=MISSING, metadata={"help": "path to wav2vec 2.0 model"}
    )
    no_pretrained_weights: bool = field(
        default=False, metadata={"help": "if true, does not load pretrained weights"}
    )
    dropout_input: float = field(
        default=0.0,
        metadata={"help": "dropout to apply to the input (after feat extr)"},
    )
    final_dropout: float = field(
        default=0.0,
        metadata={"help": "dropout after transformer and before final projection"},
    )
    dropout: float = field(
        default=0.0, metadata={"help": "dropout probability inside wav2vec 2.0 model"}
    )
    attention_dropout: float = field(
        default=0.0,
        metadata={
            "help": "dropout probability for attention weights inside wav2vec 2.0 model"
        },
    )
    activation_dropout: float = field(
        default=0.0,
        metadata={
            "help": "dropout probability after activation in FFN inside wav2vec 2.0 model"
        },
    )

    # masking
    apply_mask: bool = field(
        default=False, metadata={"help": "apply masking during fine-tuning"}
    )
    mask_length: int = field(
        default=10, metadata={"help": "repeat the mask indices multiple times"}
    )
    mask_prob: float = field(
        default=0.5,
        metadata={
            "help": "probability of replacing a token with mask (normalized by length)"
        },
    )
    mask_selection: MASKING_DISTRIBUTION_CHOICES = field(
        default="static", metadata={"help": "how to choose masks"}
    )
    mask_other: float = field(
        default=0,
        metadata={
            "help": "secondary mask argument (used for more complex distributions), "
            "see help in compute_mask_indices"
        },
    )
    no_mask_overlap: bool = field(
        default=False, metadata={"help": "whether to allow masks to overlap"}
    )

    # channel masking
    mask_channel_length: int = field(
        default=10, metadata={"help": "length of the mask for features (channels)"}
    )
    mask_channel_prob: float = field(
        default=0.0, metadata={"help": "probability of replacing a feature with 0"}
    )
    mask_channel_selection: MASKING_DISTRIBUTION_CHOICES = field(
        default="static",
        metadata={"help": "how to choose mask length for channel masking"},
    )
    mask_channel_other: float = field(
        default=0,
        metadata={
            "help": "secondary mask argument (used for more complex distributions), "
            "see help in compute_mask_indicesh"
        },
    )
    no_mask_channel_overlap: bool = field(
        default=False, metadata={"help": "whether to allow channel masks to overlap"}
    )
    freeze_finetune_updates: int = field(
        default=0, metadata={"help": "dont finetune wav2vec for this many updates"}
    )
    feature_grad_mult: float = field(
        default=0.0, metadata={"help": "reset feature grad mult in wav2vec 2.0 to this"}
    )
    layerdrop: float = field(
        default=0.0, metadata={"help": "probability of dropping a layer in wav2vec 2.0"}
    )
    normalize: bool = II("task.normalize")
    data: str = II("task.data")
    # this holds the loaded wav2vec args
    w2v_args: Any = None

    fix_extractor: bool = False

    autoregressive: bool = II("task.autoregressive")

    gpt_path: str = field(
        default="",
        metadata={"help": "path of bart model"},
    )
    gpt_type: str = field(
        default="gpt2",
        metadata={"help": "path of bart model"},
    )

    fix_encoder: bool = False
    fix_decoder: bool = False
    fix_conv_encoder: bool = False

    prompt: str = 'what did the speaker say?'

    reduction_factor: int = 1

    decoder_embed_dim: int = 768

    diff_loss_scale: float = 1.

    temp: str = field(
        default='(2.0, 0.5, 0.999995)',
        metadata={
            "help": "temperature for latent variable sampling with gumbel softmax. should be a tuple of 3 values (start, end, decay)"
        },
    )

    n_token: int = 0

import warnings
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

import torch
import torch.distributed as dist
from torch import nn

from transformers.file_utils import ModelOutput
from transformers.generation_beam_search import BeamScorer, BeamSearchScorer
from transformers.generation_logits_process import (
    EncoderNoRepeatNGramLogitsProcessor,
    ForcedBOSTokenLogitsProcessor,
    ForcedEOSTokenLogitsProcessor,
    HammingDiversityLogitsProcessor,
    InfNanRemoveLogitsProcessor,
    LogitsProcessorList,
    MinLengthLogitsProcessor,
    NoBadWordsLogitsProcessor,
    NoRepeatNGramLogitsProcessor,
    PrefixConstrainedLogitsProcessor,
    RepetitionPenaltyLogitsProcessor,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
)
from transformers.generation_stopping_criteria import (
    MaxLengthCriteria,
    MaxNewTokensCriteria,
    MaxTimeCriteria,
    StoppingCriteriaList,
    validate_stopping_criteria,
)
from transformers.utils import logging
logger = logging.get_logger(__name__)

from transformers.generation_utils import (
    GreedySearchOutput,
    SampleOutput,
    BeamSearchOutput,
    BeamSampleOutput
)

def greedy_search(
    self,
    input_ids: torch.LongTensor,
    logits_processor: Optional[LogitsProcessorList] = None,
    stopping_criteria: Optional[StoppingCriteriaList] = None,
    max_length: Optional[int] = None,
    pad_token_id: Optional[int] = None,
    eos_token_id: Optional[int] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    output_scores: Optional[bool] = None,
    return_dict_in_generate: Optional[bool] = None,
    synced_gpus: Optional[bool] = None,
    **model_kwargs,
) -> Union[GreedySearchOutput, torch.LongTensor]:

    # init values
    logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
    stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
    if max_length is not None:
        warnings.warn(
            "`max_length` is deprecated in this function, use `stopping_criteria=StoppingCriteriaList(MaxLengthCriteria(max_length=max_length))` instead.",
            UserWarning,
        )
        stopping_criteria = validate_stopping_criteria(stopping_criteria, max_length)
    pad_token_id = pad_token_id if pad_token_id is not None else self.config.pad_token_id
    eos_token_id = eos_token_id if eos_token_id is not None else self.config.eos_token_id
    output_scores = output_scores if output_scores is not None else self.config.output_scores
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    return_dict_in_generate = (
        return_dict_in_generate if return_dict_in_generate is not None else self.config.return_dict_in_generate
    )

    # init attention / hidden states / scores tuples
    scores = () if (return_dict_in_generate and output_scores) else None
    decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
    cross_attentions = () if (return_dict_in_generate and output_attentions) else None
    decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

    # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
    
    if return_dict_in_generate and self.config.is_encoder_decoder:
        encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
        encoder_hidden_states = (
            model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
        )

    # keep track of which sequences are already finished
    unfinished_sequences = input_ids.new(input_ids.shape[0]).fill_(1)
    cur_len = input_ids.shape[-1]

    this_peer_finished = False  # used by synced_gpus only
    count = 0
    while True:
        model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

        if count > 0:
            model_inputs['inputs_embeds'] = None
        if count == 0:
            
            if 'inputs_embeds' in model_kwargs and model_kwargs['inputs_embeds'] is not None:
                model_inputs['input_ids'] = None
                model_inputs['inputs_embeds'] = model_kwargs['inputs_embeds']

        # forward pass to get next token
        outputs = self(
            **model_inputs,
            return_dict=True,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        next_token_logits = outputs.logits[:, -1, :]

        # pre-process distribution
        next_tokens_scores = logits_processor(input_ids, next_token_logits)

        # argmax
        next_tokens = torch.argmax(next_tokens_scores, dim=-1)

        # finished sentences should have their next token be a padding token
        if eos_token_id is not None:
            assert pad_token_id is not None, "If eos_token_id is defined, make sure that pad_token_id is defined."
            next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

        # update generated ids, model inputs, and length for next step
        input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
        model_kwargs = self._update_model_kwargs_for_generation(
            outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
        )
        cur_len = cur_len + 1

        # if eos_token was found in one sentence, set sentence to finished
        if eos_token_id is not None:
            unfinished_sequences = unfinished_sequences.mul((next_tokens != eos_token_id).long())

        # stop when each sentence is finished, or if we exceed the maximum length
        if unfinished_sequences.max() == 0 or stopping_criteria(input_ids, scores):
            if not synced_gpus:
                break
            else:
                this_peer_finished = True

        count += 1
    return input_ids

@register_model("wavprompt", dataclass=WavPromptConfig)
class WavPrompt(FairseqEncoderDecoderModel):
    def __init__(self, encoder, decoder):
        super().__init__(encoder, decoder)

    @classmethod
    def build_model(cls, cfg:WavPromptConfig, task: FairseqTask):
        """Build a new model instance."""
        
        assert cfg.autoregressive, "Please set task.autoregressive=true for seq2seq asr models"
        decoder = cls.load_gpt_decoder(cfg, task.target_dictionary)
        encoder = cls.load_wav2vec_encoder(cfg, decoder.decoder)
        
        model = WavPrompt(encoder, decoder)
        return model

    def set_num_updates(self, num_updates):
        # self.wav2vec_encoder.set_num_updates(num_updates)
        # self.bart_decoder.set_num_updates(num_updates)
        for m in self.modules():
            if hasattr(m, "set_num_updates") and m != self:
                m.set_num_updates(num_updates)
        self.num_updates = num_updates
    
    @classmethod
    def load_wav2vec_encoder(cls, cfg, gpt2):
        print('diff_loss_scale', cfg.diff_loss_scale)
        model = Wav2VecEncoder(cfg, gpt2)
        if cfg.fix_encoder:
            print('fix w2v encoder')
            for parameter in model.parameters():
                parameter.requires_grad = False
        elif cfg.fix_conv_encoder:
            print('fix w2v conv feature extractor of the encoder')
            for parameter in model.w2v_model.feature_extractor.parameters():
                parameter.requires_grad = False
        else:
            print('do not fix w2v encoder')
        
        return model
   
    @classmethod
    def load_gpt_decoder(cls, cfg, dictionary):
        '''
        return: fairseq.models.TransformerDecoder
        '''
        decoder = GPTDecoder(cfg, dictionary)
        if cfg.fix_decoder:
            print('decoder fixed')
            matched = False
            assert len(decoder.missing_keys) == 0
            for n, parameter in decoder.decoder.named_parameters():
                # if 'wpe' in n or 'wte' in n: # fix pretrained parameters  
                #     matched = True
                #     print(n, 'requires grad')
                # else:
                parameter.requires_grad = False

            # assert matched
            print('decoder key matched:', matched)
        else:
            print('decoder free')

        return decoder

    def get_targets(self, sample, net_output):
        """Get targets from either the sample or the net's output."""
        return sample["target"]

    def max_positions(self):
        """Maximum output length supported by the decoder."""
        return self.decoder.max_positions()

    def max_decoder_positions(self):
        """Maximum output length supported by the decoder."""
        return self.decoder.max_positions()

    def forward(self, **kwargs):
        encoder_out = self.encoder(tbc=False, **kwargs)
        # decoder_out = self.decoder(encoder_out=encoder_out, prev_output_tokens = kwargs['prev_output_tokens'])
        decoder_out = self.decoder(encoder_out=encoder_out, **kwargs)
        return decoder_out
    
    def generate(self, sample, bad_words_ids=None, max_length=200):
        net_input = sample["net_input"]
        encoder_out = self.encoder(tbc=False, **net_input)

        inputs_embeds, attention_mask = self.decoder.get_inputs_embeds_for_generation(encoder_out=encoder_out, **net_input)

        if inputs_embeds.size(1) >= 800:
            print('input length exceed generate limit')
            return None
        input_ids = net_input['prev_output_tokens']

        gpt2 = self.decoder.decoder
        eos_token_id = gpt2.config.eos_token_id
        pad_token_id = eos_token_id

        logits_processor = gpt2._get_logits_processor(
            repetition_penalty=None,
            no_repeat_ngram_size=None,
            encoder_no_repeat_ngram_size=None,
            encoder_input_ids=None,
            bad_words_ids=bad_words_ids,
            min_length=None,
            max_length=gpt2.config.max_length,
            eos_token_id=eos_token_id,
            forced_bos_token_id=None,
            forced_eos_token_id=None,
            prefix_allowed_tokens_fn=None,
            num_beams=gpt2.config.num_beams,
            num_beam_groups=gpt2.config.num_beam_groups,
            diversity_penalty=None,
            remove_invalid_values=None,
        )

        
        cur_len = 0
        input_ids = torch.LongTensor([]).view(len(input_ids), 0).to(net_input['prev_output_tokens'])
        stopping_criteria = gpt2._get_stopping_criteria(
            max_length=max_length, 
            max_time=None, 
            max_new_tokens=None, 
            start_length=cur_len
        )

        # print('inputs_embeds', inputs_embeds.shape, input_ids.shape)
        greedy_output = greedy_search(
            gpt2,
            input_ids,
            logits_processor=logits_processor,
            stopping_criteria=stopping_criteria,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            output_scores=None,
            return_dict_in_generate=None,
            synced_gpus=None,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask
        )
        # print('greedy_output', greedy_output.shape, inputs_embeds.shape)
        return greedy_output
    
    def generate_with_inputs_embeds(self, input_ids, inputs_embeds, attention_mask, max_length=200, bad_words_ids=None):
        gpt2 = self.decoder.decoder
        eos_token_id = gpt2.config.eos_token_id
        pad_token_id = eos_token_id

        logits_processor = gpt2._get_logits_processor(
            repetition_penalty=None,
            no_repeat_ngram_size=None,
            encoder_no_repeat_ngram_size=None,
            encoder_input_ids=None,
            bad_words_ids=bad_words_ids,
            min_length=None,
            max_length=gpt2.config.max_length,
            eos_token_id=eos_token_id,
            forced_bos_token_id=None,
            forced_eos_token_id=None,
            prefix_allowed_tokens_fn=None,
            num_beams=gpt2.config.num_beams,
            num_beam_groups=gpt2.config.num_beam_groups,
            diversity_penalty=None,
            remove_invalid_values=None,
        )

        
        cur_len = 0
        input_ids = torch.LongTensor([]).view(len(input_ids), 0).to(input_ids)
        stopping_criteria = gpt2._get_stopping_criteria(
            max_length=max_length, 
            max_time=None, 
            max_new_tokens=None, 
            start_length=cur_len
        )

        # print('inputs_embeds', inputs_embeds.shape, input_ids.shape)
        greedy_output = greedy_search(
            gpt2,
            input_ids,
            logits_processor=logits_processor,
            stopping_criteria=stopping_criteria,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            output_scores=None,
            return_dict_in_generate=None,
            synced_gpus=None,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask
        )
        # print('greedy_output', greedy_output.shape, inputs_embeds.shape)
        return greedy_output


class Wav2VecEncoder(FairseqEncoder):
    def __init__(self, cfg: WavPromptConfig, gpt2, tgt_dict=None, ):
        self.apply_mask = cfg.apply_mask

        arg_overrides = {
            "dropout": cfg.dropout,
            "activation_dropout": cfg.activation_dropout,
            "dropout_input": cfg.dropout_input,
            "attention_dropout": cfg.attention_dropout,
            "mask_length": cfg.mask_length,
            "mask_prob": cfg.mask_prob,
            "mask_selection": cfg.mask_selection,
            "mask_other": cfg.mask_other,
            "no_mask_overlap": cfg.no_mask_overlap,
            "mask_channel_length": cfg.mask_channel_length,
            "mask_channel_prob": cfg.mask_channel_prob,
            "mask_channel_selection": cfg.mask_channel_selection,
            "mask_channel_other": cfg.mask_channel_other,
            "no_mask_channel_overlap": cfg.no_mask_channel_overlap,
            "encoder_layerdrop": cfg.layerdrop,
            "feature_grad_mult": cfg.feature_grad_mult,
        }

        if cfg.w2v_args is None:
            state = checkpoint_utils.load_checkpoint_to_cpu(cfg.w2v_path, arg_overrides)
            w2v_args = state.get("cfg", None)
            if w2v_args is None:
                w2v_args = convert_namespace_to_omegaconf(state["args"])
            cfg.w2v_args = w2v_args
        else:
            state = None
            w2v_args = cfg.w2v_args
            if isinstance(w2v_args, Namespace):
                cfg.w2v_args = w2v_args = convert_namespace_to_omegaconf(w2v_args)

        assert cfg.normalize == w2v_args.task.normalize, (
            "Fine-tuning works best when data normalization is the same. "
            "Please check that --normalize is set or unset for both pre-training and here"
        )

        w2v_args.task.data = cfg.data
        task = tasks.setup_task(w2v_args.task)
        model = task.build_model(w2v_args.model)

        if state is not None and not cfg.no_pretrained_weights:
            model.load_state_dict(state["model"], strict=True)

        model.remove_pretraining_modules()

        super().__init__(task.source_dictionary)

        d = w2v_args.model.encoder_embed_dim

        self.w2v_model = model

        self.final_dropout = nn.Dropout(cfg.final_dropout)
        self.freeze_finetune_updates = cfg.freeze_finetune_updates
        self.num_updates = 0

        
        print('decoder emb dim', getattr(cfg, "decoder_embed_dim", d))
        if tgt_dict is not None:
            print('projection layer using dictionary', d, len(tgt_dict))
            self.proj = Linear(d, len(tgt_dict))
        elif getattr(cfg, "decoder_embed_dim", d) != d:
            print('projection layer', d, cfg.decoder_embed_dim)
            self.proj = Linear(d, cfg.decoder_embed_dim)
        else:
            print('no projection layer')
            self.proj = None
        # self.proj = Linear(d, cfg.decoder_embed_dim)

        print('decoder emb dim', getattr(cfg, "decoder_embed_dim", d), self.proj)

        self.dim = cfg.decoder_embed_dim

        self.register_buffer("embed", gpt2.transformer.wte.weight.T) # C (hidden_dim) x V (n_emb)
        self.embed.requires_grad = False

        C, V = self.embed.shape
        # self.proj = Linear(d, V)
        self.n_embed = V

        self.reduction_factor = cfg.reduction_factor

        assert not self.embed.requires_grad
        if self.reduction_factor != 1:
            # self.reduce_t = nn.Conv1d(cfg.decoder_embed_dim, cfg.decoder_embed_dim, kernel_size=self.reduction_factor, stride=self.reduction_factor)
            self.reduce_t = nn.Conv1d(d, d, kernel_size=self.reduction_factor, stride=self.reduction_factor)

        self.diff_loss_scale = cfg.diff_loss_scale

        self.n_token = cfg.n_token
        
        self.max_temp, self.min_temp, self.temp_decay = eval(cfg.temp)
        self.curr_temp = self.max_temp

        self.ln = nn.LayerNorm(V)

        print('n_token', self.n_token, 'self.min_temp, self.max_temp, self.temp_decay', self.min_temp, self.max_temp, self.temp_decay)

    def set_num_updates(self, num_updates):
        """Set the number of parameters updates."""
        super().set_num_updates(num_updates)
        self.num_updates = num_updates

        self.curr_temp = max(
            self.max_temp * self.temp_decay ** num_updates, self.min_temp
        )

    def downsampling(self, x, padding_mask):
        if self.reduction_factor != 1:
            x = x.transpose(1, 2) # B x T x C => B x C x T
            x = self.reduce_t(x) # T x C x T => B x C x (T//F)
            x = x.transpose(1, 2) # B x C x (T//F) => B x (T//F) x C
        
            subsampled_padding_mask = padding_mask[:,::self.reduction_factor]
            T_m = subsampled_padding_mask.shape[1]

            T_x = x.shape[1]
            padding_mask = torch.ones(x.shape[0], x.shape[1]).to(padding_mask) # B x (T//F)
        
            T = min(T_x, T_m)
            padding_mask[:, :T] = subsampled_padding_mask[:, :T] # B x (T//F)
        return x, padding_mask

    def chopping(self, x, padding_mask):
        if self.n_token >= 1:
            x, padding_mask = x[:, :self.n_token, :], padding_mask[:, :self.n_token]
        return x, padding_mask

    def sample_gumbel(self, logits, eps=1e-10):
        U = torch.rand(logits.shape).to(logits)
        return -torch.log(-torch.log(U + eps) + eps)
    
    def gumbel_softmax_sample(self, logits, temperature, dim):
        noise = self.sample_gumbel(logits)
        print('any noise is nan', torch.any(noise.isnan()), torch.any(logits.isnan()),)
        y = logits + noise
        print('torch.any(y.isnan())', torch.any(y.isnan()), temperature, torch.any((y/temperature).isnan()), y.max(), y.min())
        return F.softmax(y / temperature, dim=dim)

    def gumbel_softmax(self, logits, temperature, dim=-1, hard=False):
        y = self.gumbel_softmax_sample(logits, temperature, dim)
        return y
        # if not hard:
        #     return y.view(-1, latent_dim * categorical_dim)

        # shape = y.size()
        # _, ind = y.max(dim=-1)
        # y_hard = torch.zeros_like(y).view(-1, shape[-1])
        # y_hard.scatter_(1, ind.view(-1, 1), 1)
        # y_hard = y_hard.view(*shape)
        # # Set gradients w.r.t. y_hard gradients w.r.t. y
        # y_hard = (y_hard - y).detach() + y
        # return y_hard.view(-1, latent_dim * categorical_dim)

    def extract_encoder_features(self, source, padding_mask, tbc=False):
        assert not tbc
        w2v_args = {
            "source": source,
            "padding_mask": padding_mask,
            "mask": self.apply_mask and self.training,
        }

        ft = self.freeze_finetune_updates <= self.num_updates

        with torch.no_grad() if not ft else contextlib.ExitStack():
            x, padding_mask = self.w2v_model.extract_features(**w2v_args)

            if tbc:
                assert False
                # B x T x C -> T x B x C

        x = self.final_dropout(x)
        # print(x.isinf)

        x, padding_mask = self.downsampling(x, padding_mask)

        x, padding_mask = self.chopping(x, padding_mask)

        if self.proj:
            x = self.proj(x) # B x T x C

        return x, padding_mask

    def forward(self, source, padding_mask, tbc=False, **kwargs):
        examples = kwargs.get('examples', None)
        assert not tbc
        
        
        if examples is not None:
            assert not self.training
            example_x = examples.get('encoder_out', None)
            example_padding_mask = examples.get('padding_mask', None)
            if example_x is None and example_padding_mask is None:
                example_source = examples['net_input']['source']
                example_padding_mask = examples['net_input']['padding_mask']
                example_x, example_padding_mask = self.extract_encoder_features(example_source, example_padding_mask)

                examples['encoder_out'] = [example_x]
                examples['padding_mask'] = [example_padding_mask]
            
        x, padding_mask = self.extract_encoder_features(source, padding_mask)
        # print('x', x.shape)

        return {
            "encoder_out": [x],  # B x T x C
            "encoder_padding_mask": [padding_mask],  # B x T
            'gpt2_input_tokens': kwargs.get('gpt2_input_tokens'),
            'examples': examples,
            'attention_mask': [kwargs.get('attention_mask')],
            'gpt2_text': kwargs.get('gpt2_text'),
            'prompt': [kwargs.get('prompt')],
            'prompt_mask': [kwargs.get('prompt_mask')],
        }


    def forward_slurp_gender(self, source, padding_mask, tbc=False, **kwargs):
        examples = kwargs.get('examples', None)
        assert not tbc

        if examples is not None:
            assert not self.training
            tidx = examples.get('tidx', None)

            assert tidx is not None
            # extract feature content
            example_source = examples['net_input']['source']
            example_padding_mask = examples['net_input']['padding_mask']
            example_x_content, example_padding_mask_content = self.extract_encoder_features(example_source, example_padding_mask)

            # extract feature gender
            example_x_gender, example_padding_mask_gender = example_x_content[:,:0,:], example_padding_mask_content[:,:0] # zero dim tensor
            if hasattr(self, 'gender_encoder') and self.gender_encoder is not None:
                example_x_gender, example_padding_mask_gender = self.gender_encoder.extract_encoder_features(
                    example_source, example_padding_mask
                )

            examples['encoder_out'] = [example_x_content]
            examples['padding_mask'] = [example_padding_mask_content]
            examples['encoder_out_gender'] = [example_x_gender]
            examples['padding_mask_gender'] = [example_padding_mask_gender]
            
        x_content, padding_mask_content = self.extract_encoder_features(source, padding_mask)

        x_gender, padding_mask_gender = x_content[:,:0,:], padding_mask_content[:,:0] # zero dim tensor
        if hasattr(self, 'gender_encoder') and self.gender_encoder is not None:
            x_gender, padding_mask_gender = self.gender_encoder.extract_encoder_features(
                source, padding_mask
            )

        return {
            "encoder_out": [x_content],  # B x T x C
            "padding_mask": [padding_mask_content],  # B x T
            "encoder_out_gender": [x_gender],  # B x T x C
            "padding_mask_gender": [padding_mask_gender],  # B x T

            "tidx": kwargs.get('tidx'),
            'gpt2_input_tokens': kwargs.get('gpt2_input_tokens'),
            'examples': examples,
            'attention_mask': [kwargs.get('attention_mask')],
            'gpt2_text': kwargs.get('gpt2_text'),
            'tgen_attention_mask': kwargs.get('tgen_attention_mask'),
            'tgen': kwargs.get('tgen'),
            'cnt': kwargs.get('cnt'),
            'qprt': kwargs.get('qprt')
        }

    def reorder_encoder_out(self, encoder_out, new_order):
        ## since the shape is BTC, the index_select operation should be on the 0 dimension
        if encoder_out["encoder_out"] is not None:
            encoder_out["encoder_out"] = [encoder_out["encoder_out"][0].index_select(
                0, new_order
            )]
        if encoder_out["encoder_padding_mask"] is not None:
            encoder_out["encoder_padding_mask"] = [encoder_out[
                "encoder_padding_mask"
            ][0].index_select(0, new_order)]
        if encoder_out["attention_mask"] is not None:
            encoder_out["attention_mask"] = [encoder_out[
                "attention_mask"
            ][0].index_select(0, new_order)]
        
        if encoder_out["diff"] is not None:
            encoder_out["diff"] = [encoder_out[
                "diff"
            ][0].index_select(0, new_order)]
        return encoder_out

    def max_positions(self):
        """Maximum input length supported by the encoder."""
        return None

    def upgrade_state_dict_named(self, state_dict, name):
        return state_dict


class GPTDecoder(FairseqIncrementalDecoder):
    def __init__(self, cfg: WavPromptConfig, dictionary=None, pre_train=True):
        super().__init__(dictionary)
        gpt_path = cfg.gpt_path
        gpt_type = cfg.gpt_type

        if gpt_type in ['EleutherAI/gpt-neo-1.3B', 'EleutherAI/gpt-neo-2.7B']:
            GPT_MODEL = GPTNeoForCausalLM
            GPT_CONFIG = GPTNeoConfig
            config = GPT_CONFIG.from_pretrained(gpt_type, cache_dir=gpt_path)
            config.n_positions = config.max_position_embeddings
        elif gpt_type in ['gpt2', 'gpt2-large', 'gpt2-xl']:
            GPT_MODEL = GPT2LMHeadModel
            GPT_CONFIG = GPT2Config
            config = GPT_CONFIG.from_pretrained(gpt_type, cache_dir=gpt_path)
        else:
            raise ValueError(f'{gpt_type} not implemented')

        gpt_lm = GPT_MODEL(config)
        self.tokenizer = GPT2Tokenizer.from_pretrained(gpt_type, cache_dir=gpt_path)

        gpt_lm.prepare_inputs_for_generation = MethodType(prepare_inputs_for_generation, gpt_lm)

        self.prompt = cfg.prompt
        if pre_train:
            print(f'loading "{gpt_type}" from pretrained: {gpt_path}')
            # orig_gpt_model = GPT2LMHeadModel.from_pretrained(gpt_type, cache_dir=gpt_path)
            orig_gpt_model = GPT_MODEL.from_pretrained(gpt_type, cache_dir=gpt_path)
            refer_state_dict = orig_gpt_model.state_dict()
            missing_keys, unexpected_keys = gpt_lm.load_state_dict(refer_state_dict, strict=False)
            print('missing_keys', missing_keys, 'unexpected_keys', unexpected_keys)
        
        assert not gpt_lm.transformer.config.add_cross_attention
        self.decoder = gpt_lm
        self.missing_keys = set(missing_keys)

    def pair(self, A):
        part1 = A[::2]
        part2 = A[1::2]
        A = torch.cat([part1, part2], dim=1)
        return A
    
    def concat_pair(self, hidden_states_content, attention_mask_content, hidden_states_gender, attention_mask_gender, without_gender_info):
        if not without_gender_info:
            hidden_states = torch.cat([hidden_states_gender, hidden_states_content], dim=1)
            encoder_attention_mask = torch.cat([attention_mask_gender, attention_mask_content], dim=1)
        else:
            hidden_states = hidden_states_content
            encoder_attention_mask = attention_mask_content

        hidden_states = self.pair(hidden_states) # (2B) x T x H => B x (2T) x H
        encoder_attention_mask = self.pair(encoder_attention_mask)
        return hidden_states, encoder_attention_mask

    
    def pair_encoder_out(self, encoder_out, without_gender_info=False):
        hidden_states_content = encoder_out['encoder_out'][0].contiguous()
        attention_mask_content = ~encoder_out['padding_mask'][0]
        assert 'BoolTensor' in encoder_out['padding_mask'][0].type(), encoder_out['padding_mask'][0].type()

        hidden_states_gender = encoder_out['encoder_out_gender'][0].contiguous()
        attention_mask_gender = ~encoder_out['padding_mask_gender'][0]
        cnt_ids = encoder_out['cnt']
        if cnt_ids.numel() > 0:
            cnt = self.decoder.transformer.wte(cnt_ids.long()).to(hidden_states_content)
            cnt_att_mask = cnt_ids != self.pad_idx
            hidden_states_gender = torch.cat([hidden_states_gender, cnt], dim=1)
            attention_mask_gender = torch.cat([attention_mask_gender, cnt_att_mask], dim=1)
        tidx = encoder_out['tidx']

        hidden_states, encoder_attention_mask = self.concat_pair(
            hidden_states_content, attention_mask_content, hidden_states_gender, attention_mask_gender,
            without_gender_info
        )

        prompt_ids = None
        batch_size = hidden_states.size(0)
        if 'gpt2_input_tokens' in encoder_out:
            prompt_ids = encoder_out['gpt2_input_tokens']
            prompt_ids = prompt_ids[tidx]
            assert batch_size == prompt_ids.size(0), f'batch size: {batch_size} prompt_ids: {prompt_ids.size(0)}'

        qprompt_ids = encoder_out['qprt']
        qprompt_ids = qprompt_ids[tidx]
        assert batch_size == qprompt_ids.size(0), f'batch size: {batch_size} prompt_ids: {qprompt_ids.size(0)}'

        return hidden_states, encoder_attention_mask, prompt_ids, tidx, qprompt_ids

    def get_inputs_embeds_for_generateion_slurp_gender(
        self, prev_output_tokens, encoder_out=None, incremental_state=None, with_example=True, exp=None, **unused
    ):
        without_txt_gender_info = exp =='txt_nogender'
        without_audio_gender_info = False
        encoder_hidden_states, encoder_attention_mask, prompt_ids, tidx, qprompt_ids = self.pair_encoder_out(encoder_out, without_audio_gender_info)

        ttxt = self.decoder.transformer.wte(encoder_out['gpt2_text'])
        tgen = self.decoder.transformer.wte(encoder_out['tgen'])
        prompt_embedding = self.decoder.transformer.wte(prompt_ids.long()).to(encoder_hidden_states) # B x T x H
        qprompt_embedding = self.decoder.transformer.wte(qprompt_ids.long()).to(encoder_hidden_states) # B x T x H

        ttxt_att_mask = encoder_out['gpt2_text'] != self.pad_idx
        tgen_att_mask = encoder_out['tgen_attention_mask'].bool()
        prompt_att_mask = prompt_ids != self.pad_idx
        qprompt_att_mask = qprompt_ids != self.pad_idx
        examples_pt = encoder_out.get('examples', None)


        if encoder_out['cnt'].numel() > 0:
            ecnt_ids = encoder_out['cnt']
            ecnt = self.decoder.transformer.wte(ecnt_ids.long()).to(encoder_hidden_states)
            ecnt_att_mask = ecnt_ids != self.pad_idx
            tgen = torch.cat([tgen, ecnt], dim=1)
            tgen_att_mask = torch.cat([tgen_att_mask, ecnt_att_mask], dim=1)

        ttxt, ttxt_att_mask = self.concat_pair(
            hidden_states_content=ttxt, attention_mask_content=ttxt_att_mask, 
            hidden_states_gender=tgen, attention_mask_gender=tgen_att_mask, 
            without_gender_info=without_txt_gender_info
        )

        if examples_pt is not None and with_example:
            batch_size = encoder_hidden_states.size(0)

            ex_hidden_states, ex_att_mask, _, extidx, exqprt = self.pair_encoder_out(examples_pt, without_audio_gender_info)
            
            tgt = self.decoder.transformer.wte(examples_pt['tgt'][extidx].long()).to(encoder_hidden_states)  # NSHOT x TTGT x H
            prt = self.decoder.transformer.wte(examples_pt['prt'][extidx].long()).to(encoder_hidden_states)  # NSHOT x TPRT x H
            qprt = self.decoder.transformer.wte(examples_pt['qprt'][extidx].long()).to(encoder_hidden_states)  # NSHOT x TPRT x H
            gen = self.decoder.transformer.wte(examples_pt['gen'].long()).to(encoder_hidden_states)
            txt = self.decoder.transformer.wte(examples_pt['txt'].long()).to(encoder_hidden_states)  # NSHOT x TTXT x H      

            tgt_att_mask = examples_pt['tgt'][extidx] != self.pad_idx # NSHOT x TTGT
            prt_att_mask = examples_pt['prt'][extidx] != self.pad_idx # NSHOT x TPRT
            qprt_att_mask = examples_pt['qprt'][extidx] != self.pad_idx # NSHOT x TPRT
            gen_att_mask = examples_pt['gen'] != self.pad_idx
            txt_att_mask = examples_pt['txt'] != self.pad_idx # NSHOT x TPRT

            if examples_pt['cnt'].numel() > 0:
                cnt = self.decoder.transformer.wte(examples_pt['cnt'].long()).to(encoder_hidden_states)
                cnt_att_mask = examples_pt['cnt'] != self.pad_idx
                gen = torch.cat([gen, cnt], dim=1)
                gen_att_mask = torch.cat([gen_att_mask, cnt_att_mask], dim=1)

            txt, txt_att_mask = self.concat_pair(
                hidden_states_content=txt, attention_mask_content=txt_att_mask, 
                hidden_states_gender=gen, attention_mask_gender=gen_att_mask,
                without_gender_info=without_txt_gender_info
            )
            assert 'BoolTensor' in tgt_att_mask.type()
            assert 'BoolTensor' in prt_att_mask.type()
            assert 'BoolTensor' in txt_att_mask.type()

            if exp == 'txt': # txt: example txt embedding and have input txt embedding
                example_att_mask = torch.cat([qprt_att_mask, txt_att_mask, prt_att_mask, tgt_att_mask], dim=1)  # NSHOT x (TSRC + TPRT + TTGT)
                example_embedding = torch.cat([qprt, txt, prt, tgt], dim=1) # NSHOT x (TSRC + TPRT + TTGT) x H

                example_embedding = example_embedding.view(-1, example_embedding.size(-1)).unsqueeze(0).expand(batch_size, -1, -1) # B x (NSHOT x (TSRC + TPRT + TTGT)) x H
                example_att_mask = example_att_mask.view(-1).unsqueeze(0).expand(batch_size, -1) # B x (NSHOT x (TSRC + TPRT + TTGT))

                encoder_state_with_prompt = torch.cat([example_embedding, qprompt_embedding, ttxt, prompt_embedding], dim=1)
                encoder_attention_mask_with_prompt = torch.cat([example_att_mask, qprompt_att_mask, ttxt_att_mask, prompt_att_mask], dim=1)
            elif exp == 'txt_nogender': # txt: example txt embedding and have input txt embedding
                # print('use_txt_without gender')
                example_att_mask = torch.cat([qprt_att_mask, txt_att_mask, prt_att_mask, tgt_att_mask], dim=1)  # NSHOT x (TSRC + TPRT + TTGT)
                example_embedding = torch.cat([qprt, txt, prt, tgt], dim=1) # NSHOT x (TSRC + TPRT + TTGT) x H

                example_embedding = example_embedding.view(-1, example_embedding.size(-1)).unsqueeze(0).expand(batch_size, -1, -1) # B x (NSHOT x (TSRC + TPRT + TTGT)) x H
                example_att_mask = example_att_mask.view(-1).unsqueeze(0).expand(batch_size, -1) # B x (NSHOT x (TSRC + TPRT + TTGT))

                encoder_state_with_prompt = torch.cat([example_embedding, qprompt_embedding, ttxt, prompt_embedding], dim=1)
                encoder_attention_mask_with_prompt = torch.cat([example_att_mask, qprompt_att_mask, ttxt_att_mask, prompt_att_mask], dim=1)
            elif exp == 'base': # base
                # print('ex_att_mask.shape, prt_att_mask.shape, tgt_att_mask.shape', ex_att_mask.shape, prt_att_mask.shape, tgt_att_mask.shape, extidx, tgt.shape)
                example_att_mask = torch.cat([qprt_att_mask, ex_att_mask, prt_att_mask, tgt_att_mask], dim=1)  # NSHOT x (TSRC + TPRT + TTGT)
                example_embedding = torch.cat([qprt, ex_hidden_states, prt, tgt], dim=1) # NSHOT x (TSRC + TPRT + TTGT) x H
                example_embedding = example_embedding.view(-1, example_embedding.size(-1)).unsqueeze(0).expand(batch_size, -1, -1) # B x (NSHOT x (TSRC + TPRT + TTGT)) x H
                example_att_mask = example_att_mask.view(-1).unsqueeze(0).expand(batch_size, -1) # B x (NSHOT x (TSRC + TPRT + TTGT))

                assert 'BoolTensor' in example_att_mask.type()
                assert 'BoolTensor' in encoder_attention_mask.type()
                assert 'BoolTensor' in prompt_att_mask.type()

                encoder_state_with_prompt = torch.cat([example_embedding, qprompt_embedding, encoder_hidden_states, prompt_embedding], dim=1)
                encoder_attention_mask_with_prompt = torch.cat([example_att_mask, qprompt_att_mask, encoder_attention_mask, prompt_att_mask], dim=1)
            else:
                raise ValueError(f'exp: {exp} option Not implemented')
        else: 
            if exp == 'txt':
                encoder_state_with_prompt = torch.cat([qprompt_embedding, ttxt, prompt_embedding], dim=1)
                encoder_attention_mask_with_prompt = torch.cat([qprompt_att_mask, ttxt_att_mask, prompt_att_mask], dim=1)
            else:
                encoder_state_with_prompt = torch.cat([qprompt_embedding, encoder_hidden_states, prompt_embedding], dim=1)
                encoder_attention_mask_with_prompt = torch.cat([qprompt_att_mask, encoder_attention_mask, prompt_att_mask], dim=1)

        return encoder_state_with_prompt, encoder_attention_mask_with_prompt


    def get_inputs_embeds_for_generateion_slurp(
        self, prev_output_tokens, encoder_out=None, incremental_state=None, with_example=True, use_example_encoder_out=True, use_encoder_out=True, use_txt=False, exp=None, **unused
    ):

        encoder_hidden_states = encoder_out['encoder_out'][0].contiguous()
        assert 'BoolTensor' in encoder_out['encoder_padding_mask'][0].type()
        encoder_attention_mask = ~encoder_out['encoder_padding_mask'][0]
        if encoder_out.get('gpt2_input_tokens') is not None:
            # print('gpt2_input_tokens', encoder_out['gpt2_input_tokens'])
            batch_size = encoder_out['encoder_out'][0].size(0)
            prompt_ids = encoder_out.get('gpt2_input_tokens')
            # print('get prompt from dataset', prompt_ids)
            prompt_embedding = self.decoder.transformer.wte(prompt_ids.long()).to(encoder_hidden_states) # B x T x H
            prompt_padding_mask = prompt_ids == self.pad_idx
            examples_pt = encoder_out.get('examples', None)
            if examples_pt is not None and with_example:
                # print('examples_pt.keys()', examples_pt.keys())
                src = examples_pt['encoder_out'][0] # NSHOT x TSRC x H
                src_padding_mask = examples_pt['padding_mask'][0] # NSHOT x TSRC
                tgt = self.decoder.transformer.wte(examples_pt['tgt'].long()).to(encoder_hidden_states)  # NSHOT x TTGT x H
                prt = self.decoder.transformer.wte(examples_pt['prt'].long()).to(encoder_hidden_states)  # NSHOT x TPRT x H
                txt = self.decoder.transformer.wte(examples_pt['txt'].long()).to(encoder_hidden_states)  # NSHOT x TTXT x H
                
                tgt_padding_mask = examples_pt['tgt'] == self.pad_idx # NSHOT x TTGT
                prt_padding_mask = examples_pt['prt'] == self.pad_idx # NSHOT x TPRT
                txt_padding_mask = examples_pt['txt'] == self.pad_idx # NSHOT x TPRT
                assert 'BoolTensor' in tgt_padding_mask.type()
                assert 'BoolTensor' in prt_padding_mask.type()
                assert 'BoolTensor' in txt_padding_mask.type()
                # print('[src_padding_mask, prt_padding_mask, tgt_padding_mask]', src_padding_mask.shape, prt_padding_mask.shape, tgt_padding_mask.shape)
                # print('not use_example_encoder_out', use_example_encoder_out)
                if exp == 'txt': # txt: example txt embedding and have input txt embedding
                    # print('use_txt')
                    example_padding_mask = torch.cat([txt_padding_mask, prt_padding_mask, tgt_padding_mask], dim=1)  # NSHOT x (TSRC + TPRT + TTGT)
                    example_embedding = torch.cat([txt, prt, tgt], dim=1) # NSHOT x (TSRC + TPRT + TTGT) x H

                    example_embedding = example_embedding.view(-1, example_embedding.size(-1)).unsqueeze(0).expand(batch_size, -1, -1) # B x (NSHOT x (TSRC + TPRT + TTGT)) x H
                    example_padding_mask = example_padding_mask.view(-1).unsqueeze(0).expand(batch_size, -1) # B x (NSHOT x (TSRC + TPRT + TTGT))
                    prompt_embedding = prompt_embedding.expand(batch_size, -1, -1)
                    prompt_padding_mask = prompt_padding_mask.expand(batch_size, -1)
                    ttxt = self.decoder.transformer.wte(encoder_out['gpt2_text'])
                    ttxt_padding_mask = encoder_out['gpt2_text'] == self.pad_idx
                    encoder_state_with_prompt = torch.cat([example_embedding, ttxt, prompt_embedding], dim=1)

                    assert 'BoolTensor' in example_padding_mask.type()
                    assert 'BoolTensor' in ttxt_padding_mask.type()
                    assert 'BoolTensor' in prompt_padding_mask.type()

                    encoder_attention_mask_with_prompt = torch.cat([~example_padding_mask, ~ttxt_padding_mask, ~prompt_padding_mask], dim=1)
                elif exp == 'noexemb': # no example audio/txt embedding and have input audio embedding
                    example_padding_mask = torch.cat([prt_padding_mask, tgt_padding_mask], dim=1)  # NSHOT x (TSRC + TPRT + TTGT)
                    example_embedding = torch.cat([prt, tgt], dim=1) # NSHOT x (TSRC + TPRT + TTGT) x H
                    example_embedding = example_embedding.view(-1, example_embedding.size(-1)).unsqueeze(0).expand(batch_size, -1, -1) # B x (NSHOT x (TSRC + TPRT + TTGT)) x H
                    example_padding_mask = example_padding_mask.view(-1).unsqueeze(0).expand(batch_size, -1) # B x (NSHOT x (TSRC + TPRT + TTGT))
                    prompt_embedding = prompt_embedding.expand(batch_size, -1, -1)
                    prompt_padding_mask = prompt_padding_mask.expand(batch_size, -1)
                    ttxt = self.decoder.transformer.wte(encoder_out['gpt2_text'])
                    ttxt_padding_mask = encoder_out['gpt2_text'] == self.pad_idx
                    encoder_state_with_prompt = torch.cat([example_embedding, encoder_hidden_states, prompt_embedding], dim=1)
                    
                    encoder_attention_mask_with_prompt = torch.cat([~example_padding_mask, encoder_attention_mask, ~prompt_padding_mask], dim=1)
                elif exp == 'noexembtxt': # noexembtxt: no example audio/txt embedding and have input txt embedding
                    example_padding_mask = torch.cat([prt_padding_mask, tgt_padding_mask], dim=1)  # NSHOT x (TSRC + TPRT + TTGT)
                    example_embedding = torch.cat([prt, tgt], dim=1) # NSHOT x (TSRC + TPRT + TTGT) x H
                    
                    example_embedding = example_embedding.view(-1, example_embedding.size(-1)).unsqueeze(0).expand(batch_size, -1, -1) # B x (NSHOT x (TSRC + TPRT + TTGT)) x H
                    example_padding_mask = example_padding_mask.view(-1).unsqueeze(0).expand(batch_size, -1) # B x (NSHOT x (TSRC + TPRT + TTGT))
                    prompt_embedding = prompt_embedding.expand(batch_size, -1, -1)
                    prompt_padding_mask = prompt_padding_mask.expand(batch_size, -1)
                    ttxt = self.decoder.transformer.wte(encoder_out['gpt2_text'])
                    ttxt_padding_mask = encoder_out['gpt2_text'] == self.pad_idx
                    encoder_state_with_prompt = torch.cat([example_embedding, ttxt, prompt_embedding], dim=1)

                    encoder_attention_mask_with_prompt = torch.cat([~example_padding_mask, ~ttxt_padding_mask, ~prompt_padding_mask], dim=1)
                elif exp == 'noexembnotxt': # noexembnotxt: no example audio/txt embedding and not input audio/txt embedding, basically random guess
                    example_padding_mask = torch.cat([prt_padding_mask, tgt_padding_mask], dim=1)  # NSHOT x (TSRC + TPRT + TTGT)
                    example_embedding = torch.cat([prt, tgt], dim=1) # NSHOT x (TSRC + TPRT + TTGT) x H
                    
                    example_embedding = example_embedding.view(-1, example_embedding.size(-1)).unsqueeze(0).expand(batch_size, -1, -1) # B x (NSHOT x (TSRC + TPRT + TTGT)) x H
                    example_padding_mask = example_padding_mask.view(-1).unsqueeze(0).expand(batch_size, -1) # B x (NSHOT x (TSRC + TPRT + TTGT))
                    prompt_embedding = prompt_embedding.expand(batch_size, -1, -1)
                    prompt_padding_mask = prompt_padding_mask.expand(batch_size, -1)
                    ttxt = self.decoder.transformer.wte(encoder_out['gpt2_text'])
                    ttxt_padding_mask = encoder_out['gpt2_text'] == self.pad_idx
                    encoder_state_with_prompt = torch.cat([example_embedding, prompt_embedding], dim=1)
                    
                    encoder_attention_mask_with_prompt = torch.cat([~example_padding_mask, ~prompt_padding_mask], dim=1)
                
                
                elif exp == 'base': # base
                    assert 'BoolTensor' in src_padding_mask.type()
                    assert 'BoolTensor' in prt_padding_mask.type()
                    assert 'BoolTensor' in tgt_padding_mask.type()
                    example_padding_mask = torch.cat([src_padding_mask, prt_padding_mask, tgt_padding_mask], dim=1)  # NSHOT x (TSRC + TPRT + TTGT)
                    # print('[src, prt, tgt]', src.shape, prt.shape, tgt.shape)
                    example_embedding = torch.cat([src, prt, tgt], dim=1) # NSHOT x (TSRC + TPRT + TTGT) x H
                    # print('example_embedding.shape', example_embedding.shape)
                    example_embedding = example_embedding.view(-1, example_embedding.size(-1)).unsqueeze(0).expand(batch_size, -1, -1) # B x (NSHOT x (TSRC + TPRT + TTGT)) x H
                    # print('example_embedding.shape 2', example_embedding.shape)
                    example_padding_mask = example_padding_mask.view(-1).unsqueeze(0).expand(batch_size, -1) # B x (NSHOT x (TSRC + TPRT + TTGT))

                    
                    prompt_embedding = prompt_embedding.expand(batch_size, -1, -1)
                    prompt_padding_mask = prompt_padding_mask.expand(batch_size, -1)
                    assert 'BoolTensor' in example_padding_mask.type()
                    assert 'BoolTensor' in encoder_attention_mask.type()
                    assert 'BoolTensor' in prompt_padding_mask.type()

                    # print('example_embedding, encoder_hidden_states, prompt_embedding', example_embedding.shape, encoder_hidden_states.shape, prompt_embedding.shape)
                    encoder_state_with_prompt = torch.cat([example_embedding, encoder_hidden_states, prompt_embedding], dim=1)
                    # print('~example_padding_mask, encoder_attention_mask, ~prompt_padding_mask', example_padding_mask.shape, encoder_attention_mask.shape, prompt_padding_mask.shape)
                    encoder_attention_mask_with_prompt = torch.cat([~example_padding_mask, encoder_attention_mask, ~prompt_padding_mask], dim=1)
                else:
                    raise ValueError(f'exp: {exp} use_example_encoder_out {use_example_encoder_out} use_encoder_out {use_encoder_out} use_txt {use_txt} option Not implemented')
            else: 
                if use_txt or exp == 'txt':
                    # print("encoder_out['gpt2_text']", encoder_out['gpt2_text'])
                    ttxt = self.decoder.transformer.wte(encoder_out['gpt2_text'])
                    ttxt_padding_mask = encoder_out['gpt2_text'] == self.pad_idx
                    encoder_state_with_prompt = torch.cat([ttxt, prompt_embedding], dim=1)
                    encoder_attention_mask_with_prompt = torch.cat([~ttxt_padding_mask, ~prompt_padding_mask], dim=1)
                else:
                    encoder_state_with_prompt = torch.cat([encoder_hidden_states, prompt_embedding], dim=1)
                    encoder_attention_mask_with_prompt = torch.cat([encoder_attention_mask, ~prompt_padding_mask], dim=1)

        return encoder_state_with_prompt, encoder_attention_mask_with_prompt

    def get_inputs_embeds_for_generation(
        self, prev_output_tokens, encoder_out=None, incremental_state=None, **unused
    ):
        attention_mask = prev_output_tokens != self.tokenizer.eos_token_id
        batch_size, seq_len = prev_output_tokens.shape
        # print('batch_size, seq_len', batch_size, seq_len)
        encoder_hidden_states = encoder_out['encoder_out'][0].contiguous()
        encoder_attention_mask = ~encoder_out['encoder_padding_mask'][0]

        prompt_ids, prompt_mask = encoder_out['prompt'][0], encoder_out['prompt_mask'][0] # prompt mask is attention mask
        if prompt_ids is not None:
            assert prompt_mask is not None
            prompt_ids = prompt_ids.to(prev_output_tokens)
            prompt_mask = prompt_mask.to(encoder_attention_mask)
        else:
            prompt_ids = [self.tokenizer.encode(self.prompt) for i in range(batch_size)]
            prompt_ids = torch.LongTensor(prompt_ids).to(prev_output_tokens)
            prompt_mask = torch.ones(prompt_ids.shape).to(encoder_attention_mask)
        
        prompt_embedding = self.decoder.transformer.wte(prompt_ids).to(encoder_hidden_states) # B x T x H
        inputs_embeds = self.decoder.transformer.wte(prev_output_tokens)

        inputs_embeds = torch.cat([encoder_hidden_states, prompt_embedding], dim=1)
        attention_mask = torch.cat([encoder_attention_mask, prompt_mask], dim=1)
        
        return inputs_embeds, attention_mask


    def forward(
        self, prev_output_tokens, encoder_out=None, incremental_state=None, **unused
    ):
        """
        Args:
            prev_output_tokens (LongTensor): previous decoder outputs of shape
                `(batch, tgt_len)`, for teacher forcing
            encoder_out (Tensor, optional): output from the encoder, used for
                encoder-side attention
            incremental_state (dict): dictionary used for storing state during
                :ref:`Incremental decoding`
        Returns:
            tuple:
                - the decoder's output of shape `(batch, tgt_len, vocab)`
                - a dictionary with any model-specific outputs
        """
        
        if incremental_state:
            past = self.get_incremental_state("past")
        else:
            past = None

        batch_size, seq_len = prev_output_tokens.shape

        audio_embeds = encoder_out['encoder_out'][0].contiguous()
        audio_attention_mask = ~encoder_out['encoder_padding_mask'][0]

        prompt_ids, prompt_mask = encoder_out['prompt'][0], encoder_out['prompt_mask'][0] # prompt mask is attention mask
        if prompt_ids is not None:
            assert prompt_mask is not None
            prompt_ids = prompt_ids.to(prev_output_tokens)
            prompt_mask = prompt_mask.to(audio_attention_mask)
        else:
            prompt_ids = [self.tokenizer.encode(self.prompt) for i in range(batch_size)]
            prompt_ids = torch.LongTensor(prompt_ids).to(prev_output_tokens)
            prompt_mask = torch.ones(prompt_ids.shape).to(audio_attention_mask)
        
        prompt_embedding = self.decoder.transformer.wte(prompt_ids).to(audio_embeds) # B x T x H

        transcription_embeds = self.decoder.transformer.wte(prev_output_tokens)
        transcription_attension_mask = encoder_out['attention_mask'][0]

        inputs_embeds = torch.cat([audio_embeds, prompt_embedding, transcription_embeds], dim=1)
        attention_mask = torch.cat([audio_attention_mask, prompt_mask, transcription_attension_mask], dim=1)

        if audio_embeds.size(1) >= 1024:
            print(f'{audio_embeds.shape} exceed length limit')
            return None

        transformer_outputs = self.decoder.transformer(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            past_key_values=past
        )

        hidden_states = transformer_outputs[0]
        lm_logits_ = self.decoder.lm_head(hidden_states)
        lm_logits = lm_logits_[:,-seq_len-1:-1, :] # do not contain current label info

        if incremental_state:
            self.set_incremental_state(incremental_state, "past", transformer_outputs[1])

        return lm_logits, {
            # "diff": diff,
            "tuple": transformer_outputs[1:],
            "attn": None,
            "hidden states":transformer_outputs[0],    
        }


    def max_positions(self):
        """Maximum output length supported by the decoder."""
        return self.decoder.config.n_positions - 1

    def buffered_future_mask(self, tensor):
        
        return self.decoder.buffered_future_mask

    def upgrade_state_dict_named(self, state_dict, name):
        return state_dict


def Embedding(num_embeddings, embedding_dim, padding_idx):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.normal_(m.weight, mean=0, std=embedding_dim ** -0.5)
    nn.init.constant_(m.weight[padding_idx], 0)
    return m


def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0.0)
    return m