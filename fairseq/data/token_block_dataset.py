# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch
import pdb
from fairseq.data import FairseqDataset, plasma_utils


class TokenBlockDataset(FairseqDataset):
    """Break a Dataset of tokens into blocks.

    Args:
        dataset (~torch.utils.data.Dataset): dataset to break into blocks
        sizes (List[int]): sentence lengths (required for 'complete' and 'eos')
        block_size (int): maximum block size (ignored in 'eos' break mode)
        break_mode (str, optional): Mode used for breaking tokens. Values can
            be one of:
            - 'none': break tokens into equally sized blocks (up to block_size)
            - 'complete': break tokens into blocks (up to block_size) such that
                blocks contains complete sentences, although block_size may be
                exceeded if some sentences exceed block_size
            - 'complete_doc': similar to 'complete' mode, but do not
                cross document boundaries
            - 'eos': each block contains one sentence (block_size is ignored)
        include_targets (bool, optional): return next tokens as targets
            (default: False).
        document_sep_len (int, optional): document separator size (required for
            'complete_doc' break mode). Typically 1 if the sentences have eos
            and 0 otherwise.
    """

    def __init__(
        self,
        dataset,
        sizes,
        block_size,
        break_mode=None,
        document_sep_len=1,
        two_inputs=False,
        prosody_predict=False,
        phoneme_prosody=False,
    ):
        try:
            from fairseq.data.token_block_utils_fast import (
                _get_slice_indices_fast,
                _get_block_to_dataset_index_fast,
            )
        except ImportError:
            raise ImportError(
                'Please build Cython components with: `pip install --editable .` '
                'or `python setup.py build_ext --inplace`'
            )

        super().__init__()
        self.dataset = dataset
        self.two_inputs = two_inputs
        self.prosody_predict = prosody_predict
        self.phoneme_prosody = phoneme_prosody

        assert len(dataset) == len(sizes)
        assert len(dataset) > 0

        if isinstance(sizes, list):
            sizes = np.array(sizes, dtype=np.int64)
        else:
            if torch.is_tensor(sizes):
                sizes = sizes.numpy()
            sizes = sizes.astype(np.int64)

        break_mode = break_mode if break_mode is not None else 'none'

        slice_indices = _get_slice_indices_fast(
            sizes, break_mode, block_size, document_sep_len)
        self._sizes = slice_indices[:, 1] - slice_indices[:, 0]

        # build index mapping block indices to the underlying dataset indices

        block_to_dataset_index = _get_block_to_dataset_index_fast(
            sizes,
            slice_indices,
        )
        self._slice_indices = plasma_utils.PlasmaArray(slice_indices)
        self._sizes = plasma_utils.PlasmaArray(self._sizes)
        self._block_to_dataset_index = plasma_utils.PlasmaArray(
            block_to_dataset_index)

    @property
    def slice_indices(self):
        return self._slice_indices.array

    @property
    def sizes(self):
        return self._sizes.array

    @property
    def block_to_dataset_index(self):
        return self._block_to_dataset_index.array

    def attr(self, attr: str, index: int):
        start_ds_idx, _, _ = self.block_to_dataset_index[index]
        return self.dataset.attr(attr, start_ds_idx)

    def __getitem__(self, index):
        start_ds_idx, start_offset, end_ds_idx = self.block_to_dataset_index[index]
        if self.two_inputs:
            if self.prosody_predict:
                # 需要把多句话合并成一句话
                # phoneme level韵律
                if self.phoneme_prosody:

                    f0_buffer = []
                    energy_buffer = []
                    mel2ph_buffer = []

                    spk_buffer = []
                    phoneme_buffer = []
                    bpe_buffer = []
                    phoneme2bpe_buffer = []

                    prev_bpe = 0
                    prev_ph = 0

                    for idx in range(start_ds_idx, end_ds_idx + 1):
                        phoneme_ids = self.dataset[idx]['phoneme_ids']
                        phoneme_buffer.append(phoneme_ids)

                        bpe_buffer.append(self.dataset[idx]['bpe_ids'])
                        phoneme2bpe_buffer.append(
                            torch.IntTensor(self.dataset[idx]['phoneme2bpe']) + prev_bpe)
                        prev_bpe += self.dataset[idx]['bpe_ids'].size(0)

                        mel2ph = torch.LongTensor(self.dataset[idx]['mel2ph'])
                        mel2ph_buffer.append(mel2ph + prev_ph)

                        f0 = torch.FloatTensor(self.dataset[idx]['f0'])
                        energy = torch.FloatTensor(self.dataset[idx]['energy'])
                        assert self.dataset[idx].get('uv', None) is None

                        T_t = phoneme_ids.size(0)
                        # # of frames for each phoneme
                        dur_gt = mel2ph.new_zeros(
                            T_t + 1).scatter_add(0, mel2ph, torch.ones_like(mel2ph))

                        dur_gt = dur_gt[1:].float()

                        phoneme_pitch = f0.new_zeros(
                            T_t + 1).scatter_add(0, mel2ph, f0)
                        phoneme_pitch = phoneme_pitch[1:]

                        phoneme_pitch[dur_gt > 0] = phoneme_pitch[dur_gt >
                                                                  0] / dur_gt[dur_gt > 0]
                        phoneme_pitch[dur_gt <= 0] = -200
                        f0 = phoneme_pitch

                        # energy merge to phoneme level
                        phoneme_energy = energy.new_zeros(
                            T_t + 1).scatter_add(0, mel2ph, energy)
                        phoneme_energy = phoneme_energy[1:]

                        phoneme_energy[dur_gt > 0] = phoneme_energy[dur_gt >
                                                                    0] / dur_gt[dur_gt > 0]
                        phoneme_energy[dur_gt <= 0] = 0
                        energy = phoneme_energy

                        f0_buffer.append(f0)
                        energy_buffer.append(energy)

                        prev_ph += T_t

                        # 为了方便spk id 可以进行升到和 phoneme数据量一致
                        spk_buffer.append(
                            torch.LongTensor([self.dataset[idx]['spk_id']] * self.dataset[idx]['phoneme_ids'].size(0)))

                    phoneme_buffer = torch.cat(phoneme_buffer)
                    bpe_buffer = torch.cat(bpe_buffer)
                    phoneme2bpe_buffer = torch.cat(phoneme2bpe_buffer)
                    f0_buffer = torch.cat(f0_buffer)
                    energy_buffer = torch.cat(energy_buffer)
                    mel2ph_buffer = torch.cat(mel2ph_buffer)

                    # 这个地方需要注意 我讲spk upsample到 phoneme层级
                    spk_buffer = torch.cat(spk_buffer)
                    assert phoneme_buffer.size(0) == phoneme2bpe_buffer.size(0)
                    assert f0_buffer.size(0) == phoneme_buffer.size(0)

                    slice_s, slice_e = self.slice_indices[index]
                    length = slice_e - slice_s
                    s, e = start_offset, start_offset + length
                    assert s == 0
                    assert e == phoneme_buffer.size(0)
                    item = {
                        'phoneme': phoneme_buffer,
                        'bpe': bpe_buffer,
                        'phoneme2bpe': phoneme2bpe_buffer,
                        'f0': f0_buffer,
                        'energy': energy_buffer,
                        'mel2ph': mel2ph_buffer,
                        'spk_id': spk_buffer
                    }
                else:
                    f0_buffer = []
                    energy_buffer = []
                    mel2ph_buffer = []
                    uv_buffer = []
                    spk_buffer = []
                    phoneme_buffer = []
                    bpe_buffer = []
                    phoneme2bpe_buffer = []

                    prev_bpe = 0
                    prev_ph = 0

                    for idx in range(start_ds_idx, end_ds_idx + 1):
                        phoneme_buffer.append(self.dataset[idx]['phoneme_ids'])
                        bpe_buffer.append(self.dataset[idx]['bpe_ids'])
                        phoneme2bpe_buffer.append(
                            torch.IntTensor(self.dataset[idx]['phoneme2bpe']) + prev_bpe)
                        prev_bpe += self.dataset[idx]['bpe_ids'].size(0)

                        f0_buffer.append(self.dataset[idx]['f0'])
                        energy_buffer.append(self.dataset[idx]['energy'])
                        uv_buffer.append(self.dataset[idx]['uv'])

                        mel2ph_buffer.append(
                            torch.LongTensor(self.dataset[idx]['mel2ph']) + prev_ph)
                        prev_ph += self.dataset[idx]['phoneme_ids'].size(0)

                        # 为了方便spk id 可以进行升到和 phoneme数据量一致
                        spk_buffer.append(
                            torch.LongTensor([self.dataset[idx]['spk_id']] * self.dataset[idx]['phoneme_ids'].size(0)))

                    phoneme_buffer = torch.cat(phoneme_buffer)
                    bpe_buffer = torch.cat(bpe_buffer)
                    phoneme2bpe_buffer = torch.cat(phoneme2bpe_buffer)
                    f0_buffer = torch.cat(f0_buffer)
                    energy_buffer = torch.cat(energy_buffer)
                    uv_buffer = torch.cat(uv_buffer)
                    mel2ph_buffer = torch.cat(mel2ph_buffer)

                    # 这个地方需要注意 我讲spk upsample到 phoneme层级
                    spk_buffer = torch.cat(spk_buffer)
                    assert phoneme_buffer.size(0) == phoneme2bpe_buffer.size(0)
                    assert f0_buffer.size(0) == mel2ph_buffer.size(0)

                    slice_s, slice_e = self.slice_indices[index]
                    length = slice_e - slice_s
                    s, e = start_offset, start_offset + length
                    assert s == 0
                    assert e == phoneme_buffer.size(0)
                    item = {
                        'phoneme': phoneme_buffer,
                        'bpe': bpe_buffer,
                        'phoneme2bpe': phoneme2bpe_buffer,
                        'f0': f0_buffer,
                        'energy': energy_buffer,
                        'uv': uv_buffer,
                        'mel2ph': mel2ph_buffer,
                        'spk_id': spk_buffer
                    }

            else:
                phoneme_buffer = []
                bpe_buffer = []
                phoneme2bpe_buffer = []
                prev_bpe = 0
                for idx in range(start_ds_idx, end_ds_idx + 1):
                    phoneme_buffer.append(self.dataset[idx]['phoneme_ids'])
                    bpe_buffer.append(self.dataset[idx]['bpe_ids'])
                    phoneme2bpe_buffer.append(
                        torch.IntTensor(self.dataset[idx]['phoneme2bpe']) + prev_bpe)
                    prev_bpe += self.dataset[idx]['bpe_ids'].size(0)

                phoneme_buffer = torch.cat(phoneme_buffer)
                bpe_buffer = torch.cat(bpe_buffer)
                phoneme2bpe_buffer = torch.cat(phoneme2bpe_buffer)
                slice_s, slice_e = self.slice_indices[index]
                length = slice_e - slice_s
                s, e = start_offset, start_offset + length
                assert s == 0
                assert e == phoneme_buffer.size(0)
                item = {
                    'phoneme': phoneme_buffer,
                    'bpe': bpe_buffer,
                    'phoneme2bpe': phoneme2bpe_buffer,

                }

        else:

            buffer = torch.cat(
                [self.dataset[idx]
                    for idx in range(start_ds_idx, end_ds_idx + 1)]
            )

            slice_s, slice_e = self.slice_indices[index]
            length = slice_e - slice_s
            s, e = start_offset, start_offset + length
            item = buffer[s:e]

        return item

    def __len__(self):
        return len(self.slice_indices)

    @property
    def supports_prefetch(self):
        return getattr(self.dataset, "supports_prefetch", False)

    def prefetch(self, indices):
        self.dataset.prefetch(
            {
                ds_idx
                for index in indices
                for start_ds_idx, _, end_ds_idx in [self.block_to_dataset_index[index]]
                for ds_idx in range(start_ds_idx, end_ds_idx + 1)
            }
        )
