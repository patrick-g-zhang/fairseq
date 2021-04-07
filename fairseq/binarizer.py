# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from collections import Counter
import os
import pdb
from fairseq.tokenizer import tokenize_line
import pickle
import numpy as np


def safe_readline(f):
    pos = f.tell()
    while True:
        try:
            return f.readline()
        except UnicodeDecodeError:
            pos -= 1
            f.seek(pos)  # search where this character begins


class IndexedDataset:
    def __init__(self, path):
        super().__init__()
        self.path = path
        self.data_file = None
        self.data_offsets = np.load(
            f"{path}.idx", allow_pickle=True).item()['offsets']
        self.data_file = open(f"{path}.data", 'rb', buffering=0)

    def check_index(self, i):
        if i < 0 or i >= len(self.data_offsets) - 1:
            raise IndexError('index out of range')

    def __del__(self):
        if self.data_file:
            self.data_file.close()

    def __getitem__(self, i):
        self.check_index(i)
        self.data_file.seek(self.data_offsets[i])
        b = self.data_file.read(
            self.data_offsets[i + 1] - self.data_offsets[i])
        item = pickle.loads(b)
        return item

    def __len__(self):
        return len(self.data_offsets) - 1


class Binarizer:
    @staticmethod
    def binarize(filename, dict, consumer, tokenize=tokenize_line, append_eos=True, reverse_order=False,
                 offset=0, end=-1):
        nseq, ntok = 0, 0
        replaced = Counter()

        def replaced_consumer(word, idx):
            if idx == dict.unk_index and word != dict.unk_word:
                replaced.update([word])

        with open(filename, 'r', encoding='utf-8') as f:
            f.seek(offset)
            # next(f) breaks f.tell(), hence readline() must be used
            line = safe_readline(f)
            while line:
                if end > 0 and f.tell() > end:
                    break
                ids = dict.encode_line(
                    line=line,
                    line_tokenizer=tokenize,
                    add_if_not_exist=False,
                    consumer=replaced_consumer,
                    append_eos=append_eos,
                    reverse_order=reverse_order,
                )
                nseq += 1
                ntok += len(ids)
                consumer(ids)
                line = f.readline()
        return {'nseq': nseq, 'nunk': sum(replaced.values()), 'ntok': ntok, 'replaced': replaced}

    @staticmethod
    def binarize_two(filename, dictp, dictb, consumer, tokenize=tokenize_line, append_eos=True, reverse_order=False,
                     offset=0, end=-1):
        nseq, ntok = 0, 0

        with open(filename, 'r', encoding='utf-8') as f:
            f.seek(offset)
            # next(f) breaks f.tell(), hence readline() must be used
            line = safe_readline(f)

            while line:
                if end > 0 and f.tell() > end:
                    break
                line = line.strip()
                line1, line2 = line.split('$')
                line1 = line1.strip()  # bpe sequence
                line2 = line2.strip()  # phoneme sequence
                # pdb.set_trace()
                phoneme_ids = dictp.encode_line(
                    line=line2,
                    line_tokenizer=tokenize,
                    add_if_not_exist=False,
                    append_eos=append_eos,
                    reverse_order=reverse_order,
                )

                bpe_ids = dictb.encode_line(
                    line=line1,
                    line_tokenizer=tokenize,
                    add_if_not_exist=False,
                    append_eos=append_eos,
                    reverse_order=reverse_order,
                )
                phoneme2bpe = []
                newphoneme = []

                for bpe_idx, bpe_seq in enumerate(line1.split(" ")):
                    phoneme2bpe.extend(
                        [bpe_idx + 1] * (len(bpe_seq.split("+"))))
                    newphoneme.extend(
                        bpe_seq.split("+"))
                assert len(phoneme2bpe) == phoneme_ids.size(0)
                item = {
                    'phoneme_ids': phoneme_ids,
                    'bpe_ids': bpe_ids,
                    'phoneme2bpe': phoneme2bpe
                }

                nseq += 1
                ntok += len(phoneme_ids)
                consumer(item)
                line = f.readline()
        return {'nseq': nseq, 'ntok': ntok, }

    @staticmethod
    def binarize_two_index_dataset(filename, dictp, dictb, consumer, tokenize=tokenize_line, append_eos=True, reverse_order=False,
                                   offset=0, end=-1):
        nseq, ntok = 0, 0
        indexed_bs = IndexedDataset(filename)

        if offset > 0 and end == 0:
            end = len(indexed_bs)

        for index in range(offset, end):
            pdb.set_trace()
            out_item = indexed_bs[index]

            line2 = out_item['rline']
            line1 = out_item['phoneme_bpe_tokens']

            line1 = line1.strip()  # bpe sequence
            line2 = line2.strip()  # phoneme sequence
            # pdb.set_trace()
            phoneme_ids = dictp.encode_line(
                line=line2,
                line_tokenizer=tokenize,
                add_if_not_exist=False,
                append_eos=append_eos,
                reverse_order=reverse_order,
            )

            bpe_ids = dictb.encode_line(
                line=line1,
                line_tokenizer=tokenize,
                add_if_not_exist=False,
                append_eos=append_eos,
                reverse_order=reverse_order,
            )
            phoneme2bpe = []
            newphoneme = []

            for bpe_idx, bpe_seq in enumerate(line1.split(" ")):
                phoneme2bpe.extend(
                    [bpe_idx + 1] * (len(bpe_seq.split("+"))))
                newphoneme.extend(
                    bpe_seq.split("+"))
            assert len(phoneme2bpe) == phoneme_ids.size(0)
            item = {
                'phoneme_ids': phoneme_ids,
                'bpe_ids': bpe_ids,
                'phoneme2bpe': phoneme2bpe,
                'mel2ph': out_item['mel2ph'],
                'spk_id': out_item['spk_id'],
                'f0': out_item['f0'],
                'uv': out_item['uv'],
                'pitch': out_item['pitch'],
            }

            nseq += 1
            ntok += len(phoneme_ids)
            consumer(item)

        return {'nseq': nseq, 'ntok': ntok, }

    @staticmethod
    def binarize_alignments(filename, alignment_parser, consumer, offset=0, end=-1):
        nseq = 0

        with open(filename, 'r') as f:
            f.seek(offset)
            line = safe_readline(f)
            while line:
                if end > 0 and f.tell() > end:
                    break
                ids = alignment_parser(line)
                nseq += 1
                consumer(ids)
                line = f.readline()
        return {'nseq': nseq}

    @staticmethod
    def find_offsets(filename, num_chunks):
        with open(filename, 'r', encoding='utf-8') as f:
            size = os.fstat(f.fileno()).st_size
            chunk_size = size // num_chunks
            offsets = [0 for _ in range(num_chunks + 1)]
            for i in range(1, num_chunks):
                f.seek(chunk_size * i)
                safe_readline(f)
                offsets[i] = f.tell()
            return offsets

    @staticmethod
    def find_indexdataset_offsets(filename, num_chunks):
        indexed_bs = IndexedDataset(filename)
        size = len(indexed_bs)
        chunk_size = size // num_chunks
        offsets = [0 for _ in range(num_chunks + 1)]
        for i in range(1, num_chunks):
            offsets[i] = int(chunk_size * i)
        return offsets
