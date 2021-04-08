#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import contextlib
import sys

from collections import Counter
from multiprocessing import Pool

from fairseq.data.encoders.phoneme_bpe import get_encoder
import re
import pdb
import pickle
import numpy as np
import torch


def process_f0(f0, f0_mean, f0_std):
    f0_ = (f0 - f0_mean) / f0_std
    f0_[f0 == 0] = np.interp(np.where(f0 == 0)[0],
                             np.where(f0 > 0)[0], f0_[f0 > 0])
    uv = (torch.FloatTensor(f0) == 0).float()
    f0 = f0_
    f0 = torch.FloatTensor(f0)
    return f0, uv


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


class IndexedDatasetBuilder:
    def __init__(self, path):
        self.path = path
        self.out_file = open(f"{path}.data", 'wb')
        self.byte_offsets = [0]

    def add_item(self, item):
        s = pickle.dumps(item)
        bytes = self.out_file.write(s)
        self.byte_offsets.append(self.byte_offsets[-1] + bytes)

    def finalize(self):
        self.out_file.close()
        np.save(open(f"{self.path}.idx", 'wb'), {'offsets': self.byte_offsets})

    def merge_file_(self, another_file):
        index = IndexedDataset(another_file)

        begin = self.byte_offsets[-1]
        for offset in index.data_offsets[1:]:
            self.byte_offsets.append(begin + offset)

        with open(f"{another_file}.data", 'rb') as f:
            while True:
                data = f.read(1024)
                if data:
                    self.out_file.write(data)
                else:
                    break


def main():
    """
    Helper script to encode raw text with the GPT-2 BPE using multiple processes.

    The encoder.json and vocab.bpe files can be obtained here:
    - https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/encoder.json
    - https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/vocab.bpe
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--vocab-bpe",
        type=str,
        help='path to vocab.bpe',
    )

    # 有些时候，我们不想要中间那个词分隔符
    parser.add_argument(
        "--no-word-sep",
        action="store_true",
        help='no word seperator',
    )

    # 输入原来是每一行一个phoneme sequence
    # 但是这里需要更多一点的信息，所以我们考虑将原来的dictionary转换成string铺在每一行
    parser.add_argument(
        "--inputs",
        nargs="+",
        default=['-'],
        help="input files to filter/encode",
    )
    parser.add_argument(
        "--outputs",
        nargs="+",
        default=['-'],
        help="path to save encoded outputs",
    )
    parser.add_argument(
        "--keep-empty",
        action="store_true",
        help="keep empty lines",
    )
    parser.add_argument("--workers", type=int, default=20)
    args = parser.parse_args()

    assert len(args.inputs) == len(args.outputs), \
        "number of input and output paths should match"

    with contextlib.ExitStack() as stack:
        indexed_bs = IndexedDataset(args.inputs[0])
        spks_mv = np.load(
            f'{args.inputs[0].split(".")[0]}_f0s.pkl', allow_pickle=True)

        # self.indexed_bs = IndexedDataset(
        # f'{self.data_dir}/{self.prefix}')
        builder = IndexedDatasetBuilder(args.outputs[0])

        encoder = MultiprocessingEncoder(args, spks_mv)

        # multiprocess
        pool = Pool(args.workers, initializer=encoder.initializer)
        encoded_lines = pool.imap(encoder.encode_lines, indexed_bs, 100)
        # encoder.initializer()
        # encoded_lines = []
        # for item in indexed_bs:
        # encoded_line = encoder.encode_lines(item, spks_mv)
        # encoded_lines.append(encoded_line)

        stats = Counter()

        for i, (filt, out_item) in enumerate(encoded_lines, start=1):
            if filt == "PASS":
                builder.add_item(out_item)
            else:
                stats["num_filtered_" + filt] += 1
            if i % 10000 == 0:
                print("processed {} lines".format(i), file=sys.stderr)

        builder.finalize()
        for k, v in stats.most_common():
            print("[{}] filtered {} lines".format(k, v), file=sys.stderr)


class MultiprocessingEncoder(object):

    def __init__(self, args, spks_mv):
        self.args = args
        self.spks_mv = spks_mv

    def initializer(self):
        global bpe
        bpe = get_encoder(self.args.vocab_bpe)

    def encode(self, line):
        global bpe
        ids = bpe.process_line(line, no_word_sep=self.args.no_word_sep)
        return list(map(str, ids))

    def decode(self, tokens):
        global bpe
        return bpe.decode(tokens)

    def encode_lines(self, item):
        """
        Encode a set of lines. All lines will be encoded together.
        """
        out_item = {}
        mel2ph = item['mel2ph']
        out_item['mel2ph'] = mel2ph
        spk_id = item['spk_id']
        out_item['spk_id'] = spk_id
        f0, uv = process_f0(
            item["f0"], self.spks_mv[spk_id][0], self.spks_mv[spk_id][1])
        out_item['f0'] = f0
        out_item['uv'] = uv
        energy = item["energy"]
        out_item['energy'] = energy
        ph = item['phone']

        # remove repeat "|"
        rline = re.sub("(\|\s)+", r"\1", ph)
        line = re.sub('<UNK>', '', rline)           # Delete pattern abc
        line = re.sub('<EOS>', '', line)           # Delete pattern abc
        line = line.strip()
        if self.args.no_word_sep:
            # no word sep
            rline = re.sub('\| ', '', rline)
        phoneme_bpe_tokens = self.encode(line)
        phoneme_bpe_tokens.insert(0, '<s>')
        phoneme_bpe_tokens.append('</s>')

        out_item['phoneme_bpe_tokens'] = " ".join(phoneme_bpe_tokens)
        out_item['rline'] = rline
        if not sum(map(lambda x: len(x.split("+")),
                       phoneme_bpe_tokens)) == len(rline.split(" ")):
            new_phonemes = []
            for phoneme in phoneme_bpe_tokens:
                new_phonemes.extend(phoneme.split("+"))
            old_phonemes = rline.split(" ")
            for ph_idx in range(min(len(new_phonemes), len(old_phonemes))):
                print(new_phonemes[ph_idx], old_phonemes[ph_idx])

        return ["PASS", out_item]

    def decode_lines(self, lines):
        dec_lines = []
        for line in lines:
            tokens = map(int, line.strip().split())
            dec_lines.append(self.decode(tokens))
        return ["PASS", dec_lines]


if __name__ == "__main__":
    main()
