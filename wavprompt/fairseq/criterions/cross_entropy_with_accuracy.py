import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.dataclass import FairseqDataclass
from omegaconf import II


@dataclass
class CrossEntropyCriterionConfig(FairseqDataclass):
    sentence_avg: bool = II("optimization.sentence_avg")

@register_criterion("cross_entropy_with_acc", dataclass=CrossEntropyCriterionConfig)
class CrossEntropyCriterionACC(FairseqCriterion):
    def __init__(self, task, sentence_avg, report_accuracy = True, debug = False):
        super().__init__(task)
        self.sentence_avg = sentence_avg
        self.report_accuracy = report_accuracy
        self.ignore_prefix_size = 0
        self.debug = debug
        print('CrossEntropyCriterionACC self.padding_idx', self.padding_idx)
        assert self.padding_idx == 50256 or self.padding_idx == 1

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.
        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(**sample["net_input"])

        if net_output is None:
            sample_size = (
                sample["target"].size(0) if self.sentence_avg else sample["ntokens"]
            )
            logging_output = {
                "loss": 0,
                "ntokens": sample["ntokens"],
                "nsentences": sample["target"].size(0),
                "sample_size": sample_size,
            }
            return None, sample_size, logging_output

        loss, diff_loss = self.compute_loss(model, net_output, sample, reduce=reduce)

        sample_size = (
            sample["target"].size(0) if self.sentence_avg else sample["ntokens"]
        )
        logging_output = {
            "total_loss": loss.data + diff_loss.data if diff_loss else loss.data,
            "loss": loss.data,
            "diff_loss": diff_loss.data if diff_loss else 0,
            "ntokens": sample["ntokens"],
            "nsentences": sample["target"].size(0),
            "sample_size": sample_size,
        }
        if self.report_accuracy:
            n_correct, total = self.compute_accuracy(model, net_output, sample)
            logging_output["n_correct"] = utils.item(n_correct.data)
            logging_output["total"] = utils.item(total.data)
        return loss+diff_loss if diff_loss else loss, sample_size, logging_output

    def compute_accuracy(self, model, net_output, sample):
        lprobs, target = self.get_lprobs_and_target(model, net_output, sample)
        if self.debug:
            print("pred: ", torch.argmax(lprobs, dim = -1))
            print("target: ", target)
            print()
        mask = target.ne(self.padding_idx)
        n_correct = torch.sum(
            lprobs.argmax(1).masked_select(mask).eq(target.masked_select(mask))
        )
        total = torch.sum(mask)
        return n_correct, total

    def get_lprobs_and_target(self, model, net_output, sample):
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        target = model.get_targets(sample, net_output)
        if self.ignore_prefix_size > 0:
            if getattr(lprobs, "batch_first", False):
                lprobs = lprobs[:, self.ignore_prefix_size :, :].contiguous()
                target = target[:, self.ignore_prefix_size :].contiguous()
            else:
                lprobs = lprobs[self.ignore_prefix_size :, :, :].contiguous()
                target = target[self.ignore_prefix_size :, :].contiguous()
        return lprobs.view(-1, lprobs.size(-1)), target.view(-1)


    def compute_loss(self, model, net_output, sample, reduce=True):
        
        diff_loss = None
        if self.padding_idx == 50256:
            batch_target = model.get_targets(sample, net_output)
            mask = batch_target != self.padding_idx
            stop_idx = torch.sum(mask, dim=1)
            mask = mask.scatter_(1, stop_idx.unsqueeze(1), True)
            batch_lprobs = model.get_normalized_probs(net_output, log_probs=True)

            target = batch_target[mask]
            lprobs = batch_lprobs[mask]

            loss = F.nll_loss(
                lprobs,
                target,
                reduction="sum" if reduce else "none",
            )
        else:
            lprobs = model.get_normalized_probs(net_output, log_probs=True)
            lprobs = lprobs.view(-1, lprobs.size(-1))
            target = model.get_targets(sample, net_output).view(-1)

            loss = F.nll_loss(
                lprobs,
                target,
                ignore_index=self.padding_idx,
                reduction="sum" if reduce else "none",
            )
        
        if net_output[1].get('diff', None):
            diff_loss = net_output[1]['diff']
            
        return loss, diff_loss

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        diff_loss_sum = sum(log.get("diff_loss", 0) for log in logging_outputs)
        total_loss_sum = sum(log.get("total_loss", 0) for log in logging_outputs)
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

        # we divide by log(2) to convert the loss from base e to base 2
        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "diff_loss", diff_loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "total_loss", total_loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        if sample_size != ntokens:
            metrics.log_scalar(
                "nll_loss", loss_sum / ntokens / math.log(2), ntokens, round=3
            )
            metrics.log_derived(
                "ppl", lambda meters: utils.get_perplexity(meters["nll_loss"].avg)
            )
        else:
            metrics.log_derived(
                "ppl", lambda meters: utils.get_perplexity(meters["loss"].avg)
            )
            
        total = utils.item(sum(log.get("total", 0) for log in logging_outputs))
        if total > 0:
            metrics.log_scalar("total", total)
            n_correct = utils.item(
                sum(log.get("n_correct", 0) for log in logging_outputs)
            )
            metrics.log_scalar("n_correct", n_correct)
            metrics.log_derived(
                "accuracy",
                lambda meters: round(
                    meters["n_correct"].sum * 100.0 / meters["total"].sum, 3
                )
                if meters["total"].sum > 0
                else float("nan"),
            )

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True