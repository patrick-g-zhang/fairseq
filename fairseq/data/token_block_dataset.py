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
        pdb.set_trace()
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
