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
    def __init__(self, dataset, pad_idx, left_pad):
        super().__init__(dataset)
        self.phoneme_pad_idx = pad_idx
        self.bpe_pad_idx = pad_idx
        self.left_pad = left_pad

    def collater(self, samples):
        phonemes = [sample['phoneme'] for sample in samples]
        phonemes = data_utils.collate_tokens(
            phonemes, self.phoneme_pad_idx, left_pad=self.left_pad)
        bpes = [sample['bpe'] for sample in samples]
        bpes = data_utils.collate_tokens(
            bpes, self.bpe_pad_idx, left_pad=self.left_pad)
        phoneme2bpes = [sample['phoneme2bpe'] for sample in samples]
        phoneme2bpes = data_utils.collate_tokens(
            phoneme2bpes, self.phoneme_pad_idx)

        items = {'phoneme': phonemes, 'bpe': bpes,
                 'phoneme2bpe': phoneme2bpes}
        return items


class LeftPadDataset(PadDataset):

    def __init__(self, dataset, pad_idx):
        super().__init__(dataset, pad_idx, left_pad=True)


class RightPadDataset(PadDataset):

    def __init__(self, dataset, pad_idx):
        super().__init__(dataset, pad_idx, left_pad=False)
