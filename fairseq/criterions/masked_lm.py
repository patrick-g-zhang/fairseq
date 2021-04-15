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


def dur_loss(dur_pred, dur_gt, input):

    nonpadding = (input != 0).float()

    # 对targets 取对数
    targets = torch.log(dur_gt.float() + 1.0)
    loss = torch.nn.MSELoss(reduction="none")(dur_pred, targets.float())
    loss = (loss * nonpadding).sum() / nonpadding.sum()

    return loss


def pitch_loss(p_pred, pitch, uv):
    assert p_pred[..., 0].shape == pitch.shape
    assert p_pred[..., 0].shape == uv.shape
    nonpadding = (pitch != -200).float().reshape(-1)
    uv_loss = (F.binary_cross_entropy_with_logits(
        p_pred[:, :, 1].reshape(-1), uv.reshape(-1), reduction='none') * nonpadding).sum() \
        / nonpadding.sum()
    nonpadding = (pitch != -200).float() * (uv == 0).float()
    nonpadding = nonpadding.reshape(-1)

    f0_loss = (F.l1_loss(
        p_pred[:, :, 0].reshape(-1), pitch.reshape(-1), reduction='none') * nonpadding).sum() \
        / nonpadding.sum()
    return uv_loss, f0_loss


def phoneme_pitch_loss(p_pred, pitch):
    assert p_pred[..., 0].shape == pitch.shape

    nonpadding = (pitch != -200).float().reshape(-1)
    uv_loss = None

    pitch_loss = (F.l1_loss(
        p_pred[:, :, 0].reshape(-1), pitch.reshape(-1), reduction='none') * nonpadding).sum() \
        / nonpadding.sum()
    return uv_loss, pitch_loss


def energy_loss(energy_pred, energy):
    nonpadding = (energy != 0).float()
    loss = (F.mse_loss(energy_pred, energy, reduction='none')
            * nonpadding).sum() / nonpadding.sum()
    return loss


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

        if self.args.two_inputs:
            # 多个输入
            bpe_masked_tokens = sample['target']['bpe'].ne(self.padding_idx)
            phoneme_masked_tokens = sample['target']['phoneme'].ne(
                self.padding_idx)
            sample_size = phoneme_masked_tokens.int().sum().item()
            # (Rare case) When all tokens are masked, the model results in empty
            # tensor and gives CUDA error.
            if sample_size == 0:
                phoneme_masked_tokens = None

            if self.args.prosody_predict:
                # 如果是做韵律预测 要会有多个输出
                logitps, logitbs, dur_pred, pitch_pred, energy_pred = model(**sample['net_input'], masked_tokens=phoneme_masked_tokens,
                                                                            bpe_masked_tokens=bpe_masked_tokens)

            else:
                logitps, logitbs = model(**sample['net_input'], masked_tokens=phoneme_masked_tokens,
                                         bpe_masked_tokens=bpe_masked_tokens)

            targets = model.get_targets(sample, [logitps])
            targets_p = targets['phoneme']
            targets_b = targets['bpe']

            if sample_size != 0:
                targets_p = targets_p[phoneme_masked_tokens]
                targets_b = targets_b[bpe_masked_tokens]

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

            if self.args.prosody_predict:
                # 增加额外的loss
                # energy loss
                energy = sample['target']['energy']
                loss_energy = energy_loss(
                    energy_pred, energy) * self.args.prosody_loss_coeff
                loss += loss_energy
                logging_output['loss_energy'] = utils.item(
                    loss_energy.data) if reduce else loss_energy.data

                # dur loss
                dur_gt = sample['target']['dur_gt']
                loss_dur = dur_loss(
                    dur_pred, dur_gt, sample['net_input']['src_tokens']['phoneme']) * self.args.prosody_loss_coeff
                loss += loss_dur
                logging_output['loss_dur'] = utils.item(
                    loss_dur.data) if reduce else loss_dur.data

                # pitch loss

                f0 = sample['target']['f0']

                # 如果不是phoneme prosody 那么需要保存
                if not self.args.phoneme_prosody:
                    uv = sample['target'].get('uv', None)
                    loss_uv, loss_f0 = pitch_loss(pitch_pred, f0, uv)
                    loss_uv = loss_uv * self.args.prosody_loss_coeff
                    loss += loss_uv
                    logging_output['loss_uv'] = utils.item(
                        loss_uv.data) if reduce else loss_uv.data

                else:
                    _, loss_f0 = phoneme_pitch_loss(pitch_pred, f0)

                loss_f0 = loss_f0 * self.args.prosody_loss_coeff
                loss += loss_f0
                logging_output['loss_f0'] = utils.item(
                    loss_f0.data) if reduce else loss_f0.data

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
        loss_p = sum(log.get('loss_p', 0) for log in logging_outputs)
        loss_b = sum(log.get('loss_b', 0) for log in logging_outputs)
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        nsentences = sum(log.get('nsentences', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)

        agg_output = {
            'loss': loss / sample_size / math.log(2),
            'loss_p': loss_p / sample_size / math.log(2),
            'loss_b': loss_b / sample_size / math.log(2),
            'nll_loss': sum(log.get('nll_loss', 0) for log in logging_outputs) / sample_size / math.log(2) if ntokens > 0 else 0.,
            'nll_loss_p': sum(log.get('nll_loss_p', 0) for log in logging_outputs) / sample_size / math.log(2) if ntokens > 0 else 0.,
            'nll_loss_b': sum(log.get('nll_loss_b', 0) for log in logging_outputs) / sample_size / math.log(2) if ntokens > 0 else 0.,
            'ntokens': ntokens,
            'nsentences': nsentences,
            'sample_size': sample_size,
        }

        if logging_outputs[0].get('loss_energy', 0) > 0:
            # 需要输出韵律相关的特征
            loss_energy = sum(log.get('loss_energy', 0)
                              for log in logging_outputs)
            agg_output['loss_energy'] = loss_energy / sample_size / math.log(2)

            loss_dur = sum(log.get('loss_dur', 0) for log in logging_outputs)
            agg_output['loss_dur'] = loss_dur / sample_size / math.log(2)

            loss_f0 = sum(log.get('loss_f0', 0) for log in logging_outputs)
            agg_output['loss_f0'] = loss_f0 / sample_size / math.log(2)

            loss_uv = sum(log.get('loss_uv', 0) for log in logging_outputs)
            agg_output['loss_uv'] = loss_uv / sample_size / math.log(2)

        return agg_output
