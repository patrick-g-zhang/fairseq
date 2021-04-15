# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from fairseq.data import data_utils

from . import BaseWrapperDataset


class PadDataset(BaseWrapperDataset):

    def __init__(self, dataset, pad_idx, left_pad):
        super().__init__(dataset)
        self.pad_idx = pad_idx
        self.left_pad = left_pad

    def collater(self, samples):
        return data_utils.collate_tokens(samples, self.pad_idx, left_pad=self.left_pad)


class DictPadDataset(BaseWrapperDataset):
    def __init__(self, dataset, pad_idx, left_pad, prosody_predict=False, phoneme_prosody=False, is_target=False):
        super().__init__(dataset)
        self.phoneme_pad_idx = pad_idx
        self.bpe_pad_idx = pad_idx
        self.left_pad = left_pad
        self.is_target = is_target
        self.prosody_predict = prosody_predict
        self.phoneme_prosody = phoneme_prosody
        print(self.prosody_predict)

    def collater(self, samples):
        phonemes = [sample['phoneme'] for sample in samples]
        phonemes = data_utils.collate_tokens(
            phonemes, self.phoneme_pad_idx, left_pad=self.left_pad)
        bpes = [sample['bpe'] for sample in samples]
        bpes = data_utils.collate_tokens(
            bpes, self.bpe_pad_idx, left_pad=self.left_pad)

        items = {'phoneme': phonemes, 'bpe': bpes}

        if self.is_target:
            if self.prosody_predict:
                f0 = data_utils.collate_tokens(
                    [s['f0'] for s in samples], pad_idx=-200)
                items['f0'] = f0
                energy = data_utils.collate_tokens(
                    [s['energy'] for s in samples], pad_idx=0)
                items['energy'] = energy
                dur_gt = data_utils.collate_tokens(
                    [s['dur_gt'] for s in samples], pad_idx=0)
                items['dur_gt'] = dur_gt
                if not self.phoneme_prosody:
                    uv = data_utils.collate_tokens(
                        [s['uv'] for s in samples], pad_idx=0)
                    items['uv'] = uv
            return items

        phoneme2bpes = [sample['phoneme2bpe'] for sample in samples]
        phoneme2bpes = data_utils.collate_tokens(
            phoneme2bpes, self.phoneme_pad_idx)

        items['phoneme2bpe'] = phoneme2bpes
        if self.prosody_predict:
            spk_id = data_utils.collate_tokens(
                [s['spk_id'] for s in samples], pad_idx=0)
            items['spk_id'] = spk_id
            if not self.phoneme_prosody:
                mel2ph = data_utils.collate_tokens(
                    [s['mel2ph'] for s in samples], pad_idx=0)
                items['mel2ph'] = mel2ph
        return items


class LeftPadDataset(PadDataset):

    def __init__(self, dataset, pad_idx):
        super().__init__(dataset, pad_idx, left_pad=True)


class RightPadDataset(PadDataset):

    def __init__(self, dataset, pad_idx):
        super().__init__(dataset, pad_idx, left_pad=False)
