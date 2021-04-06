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

    pdb.set_trace()
    assert len(args.inputs) == len(args.outputs), \
        "number of input and output paths should match"

    with contextlib.ExitStack() as stack:
        pdb.set_trace()
        inputs = [
            # stack.enter_context(open(input, "r", encoding="utf-8"))
            stack.enter_context(open(input, "rb"))
            if input != "-" else sys.stdin
            for input in args.inputs
        ]

        # self.indexed_bs = IndexedDataset(
        # f'{self.data_dir}/{self.prefix}')
        outputs = [
            stack.enter_context(open(output, "w", encoding="utf-8"))
            if output != "-" else sys.stdout
            for output in args.outputs
        ]

        encoder = MultiprocessingEncoder(args)

        # multiprocess
        # pool = Pool(args.workers, initializer=encoder.initializer)
        # encoded_lines = pool.imap(encoder.encode_lines, zip(*inputs), 100)
        encoder.initializer()
        encoded_lines = []
        for encoded_line in zip(*inputs):
            encoded_line = encoder.encode_lines(encoded_line)
            encoded_lines.append(encoded_line)

        stats = Counter()
        for i, (filt, enc_lines) in enumerate(encoded_lines, start=1):
            if filt == "PASS":
                for enc_line, output_h in zip(enc_lines, outputs):
                    print(enc_line, file=output_h)
            else:
                stats["num_filtered_" + filt] += 1
            if i % 10000 == 0:
                print("processed {} lines".format(i), file=sys.stderr)

        for k, v in stats.most_common():
            print("[{}] filtered {} lines".format(k, v), file=sys.stderr)


class MultiprocessingEncoder(object):

    def __init__(self, args):
        self.args = args

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

    def encode_lines(self, lines):
        """
        Encode a set of lines. All lines will be encoded together.
        """
        enc_lines = []
        for rline in lines:
            pdb.set_trace()
            rline = rline.strip()
            pickle.load(rline)
            # remove repeat "|"
            rline = re.sub("(\|\s)+", r"\1", rline)
            # Delete pattern abc
            rline = re.sub('<UNK> \|', '<UNK>', rline)
            # Delete pattern abc
            rline = re.sub('\| <EOS>', '<EOS>', rline)
            line = re.sub('<UNK>', '', rline)           # Delete pattern abc
            line = re.sub('<EOS>', '', line)           # Delete pattern abc
            line = line.strip()
            if self.args.no_word_sep:
                # no word sep
                rline = re.sub('\| ', '', rline)
            phoneme_bpe_tokens = self.encode(line)
            phoneme_bpe_tokens.insert(0, '<s>')
            phoneme_bpe_tokens.append('</s>')

            if not sum(map(lambda x: len(x.split("+")),
                           phoneme_bpe_tokens)) == len(rline.split(" ")):
                new_phonemes = []
                for phoneme in phoneme_bpe_tokens:
                    new_phonemes.extend(phoneme.split("+"))
                old_phonemes = rline.split(" ")
                for ph_idx in range(min(len(new_phonemes), len(old_phonemes))):
                    print(new_phonemes[ph_idx], old_phonemes[ph_idx])

            encoded_one_line = " ".join(phoneme_bpe_tokens) + ' $ ' + rline
            enc_lines.append(encoded_one_line)
        return ["PASS", enc_lines]

    def decode_lines(self, lines):
        dec_lines = []
        for line in lines:
            tokens = map(int, line.strip().split())
            dec_lines.append(self.decode(tokens))
        return ["PASS", dec_lines]


if __name__ == "__main__":
    main()
