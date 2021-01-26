# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math

import torch
import torch.nn.functional as F
import pdb
from fairseq import utils

from . import FairseqCriterion, register_criterion


@register_criterion('masked_lm')
class MaskedLmLoss(FairseqCriterion):
    """
    Implementation for the loss used in masked language model (MLM) training.
    """

    def __init__(self, args, task):
        super().__init__(args, task)

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.
        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        # compute MLM loss
        pdb.set_trace()
        if self.args.two_inputs:
            bpe_masked_tokens = sample['target']['bpe'].ne(self.padding_idx)
            phoneme_masked_tokens = sample['target']['phoneme'].ne(
                self.padding_idx)
            sample_size = phoneme_masked_tokens.int().sum().item()
            # (Rare case) When all tokens are masked, the model results in empty
            # tensor and gives CUDA error.
            if sample_size == 0:
                phoneme_masked_tokens = None
            logitps, logitbs = model(**sample['net_input'], masked_tokens=phoneme_masked_tokens,
                                     bpe_masked_tokens=bpe_masked_tokens)
            targets = model.get_targets(sample, [logitps])
            targets_p = targets['phoneme']
            targets_b = targets['bpe']

            if sample_size != 0:
                targets_p = targets_p[phoneme_masked_tokens]
                targets_b = targets_b[bpe_masked_tokens]

            pdb.set_trace()
            loss_p = F.nll_loss(
                F.log_softmax(
                    logitps.view(-1, logitps.size(-1)),
                    dim=-1,
                    dtype=torch.float32,
                ),
                targets_p.view(-1),
                reduction='sum',
                ignore_index=self.padding_idx,
            )

            loss_b = F.nll_loss(
                F.log_softmax(
                    logitbs.view(-1, logitbs.size(-1)),
                    dim=-1,
                    dtype=torch.float32,
                ),
                targets_b.view(-1),
                reduction='sum',
                ignore_index=self.padding_idx,
            )

            loss = loss_b + loss_p
            logging_output = {
                'loss_p': utils.item(loss_p.data) if reduce else loss_p.data,
                'nll_loss_p': utils.item(loss_p.data) if reduce else loss_p.data,
                'loss_b': utils.item(loss_b.data) if reduce else loss_b.data,
                'nll_loss_b': utils.item(loss_b.data) if reduce else loss_b.data,
                'loss': utils.item(loss.data) if reduce else loss.data,
                'nll_loss': utils.item(loss.data) if reduce else loss.data,
                'ntokens': sample['ntokens'],
                'nsentences': sample['nsentences'],
                'sample_size': sample_size,
            }

        else:
            masked_tokens = sample['target'].ne(self.padding_idx)
            sample_size = masked_tokens.int().sum().item()

            # (Rare case) When all tokens are masked, the model results in empty
            # tensor and gives CUDA error.
            if sample_size == 0:
                masked_tokens = None
            # pdb.set_trace()

            logits = model(**sample['net_input'],
                           masked_tokens=masked_tokens)
            targets = model.get_targets(sample, [logits])

            if sample_size != 0:
                targets = targets[masked_tokens]

            loss = F.nll_loss(
                F.log_softmax(
                    logits.view(-1, logits.size(-1)),
                    dim=-1,
                    dtype=torch.float32,
                ),
                targets.view(-1),
                reduction='sum',
                ignore_index=self.padding_idx,
            )

            logging_output = {
                'loss': utils.item(loss.data) if reduce else loss.data,
                'nll_loss': utils.item(loss.data) if reduce else loss.data,
                'ntokens': sample['ntokens'],
                'nsentences': sample['nsentences'],
                'sample_size': sample_size,
            }
        return loss, sample_size, logging_output

    @staticmethod
    def aggregate_logging_outputs(logging_outputs):
        """Aggregate logging outputs from data parallel training."""
        loss = sum(log.get('loss', 0) for log in logging_outputs)
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        nsentences = sum(log.get('nsentences', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)

        agg_output = {
            'loss': loss / sample_size / math.log(2),
            'nll_loss': sum(log.get('nll_loss', 0) for log in logging_outputs) / sample_size / math.log(2) if ntokens > 0 else 0.,
            'ntokens': ntokens,
            'nsentences': nsentences,
            'sample_size': sample_size,
        }
        return agg_output
