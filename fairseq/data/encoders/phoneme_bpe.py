"""
Byte pair encoding utilities from GPT-2.

Original source: https://github.com/openai/gpt-2/blob/master/src/encoder.py
Original license: MIT
"""

import json
import pdb

import sys
import os
import inspect
import codecs
import io
import argparse
import re
import warnings
import random
from multiprocessing import Pool, cpu_count
# hack for python2/3 compatibility
from io import open


def get_encoder(vocab_bpe_path):
    with codecs.open(vocab_bpe_path, encoding='utf-8') as bpefile:
        return BPE(bpefile)


class BPE(object):

    def __init__(self, codes, merges=-1, vocab=None, glossaries=None):

        codes.seek(0)
        offset = 1

        # check version information
        firstline = codes.readline()
        if firstline.startswith('#version:'):
            self.version = tuple([int(x) for x in re.sub(
                r'(\.0+)*$', '', firstline.split()[-1]).split(".")])
            offset += 1
        else:
            self.version = (0, 1)
            codes.seek(0)

        self.bpe_codes = [tuple(item.strip('\r\n ').split(' ')) for (n, item) in enumerate(
            codes.read().rstrip('\n').split('\n')) if (n < merges or merges == -1)]

        for i, item in enumerate(self.bpe_codes):
            if len(item) != 2:
                sys.stderr.write('Error: invalid line {0} in BPE codes file: {1}\n'.format(
                    i + offset, ' '.join(item)))
                sys.stderr.write(
                    'The line should exist of exactly two subword units, separated by whitespace\n')
                sys.exit(1)

        # some hacking to deal with duplicates (only consider first instance)
        self.bpe_codes = dict(
            [(code, i) for (i, code) in reversed(list(enumerate(self.bpe_codes)))])

        self.bpe_codes_reverse = dict(
            [(pair[0] + pair[1], pair) for pair, i in self.bpe_codes.items()])

        self.cache = {}

    def process_line(self, line, dropout=0):
        """segment line, dealing with leading and trailing whitespace"""
        tokens = [word.strip() for word in line.split('|')]
        segments = self.segment_tokens(tokens, dropout)
        return segments

    def segment_tokens(self, tokens, dropout=0):
        """segment a sequence of tokens with BPE encoding"""
        output = []
        for word in tokens:
            # eliminate double spaces
            if not word:
                continue
            new_word = [out for out in encode(word,
                                              self.bpe_codes,
                                              self.bpe_codes_reverse,
                                              self.cache,
                                              dropout)]

            for item in new_word:
                output.append(item)
                if item.endswith('</w>'):
                    output.append('|')
            if output[-1] == "|":
                return output[:-1]
        return output


def encode(orig, bpe_codes, bpe_codes_reverse, cache, dropout=0):
    """Encode word based on list of BPE merge operations, which are applied consecutively
    """
    if not dropout and orig in cache:
        return cache[orig]

    if len(orig) == 1:
        return orig

    worig = orig + '</w>'
    word = worig.split(" ")

    while len(word) > 1:

        # get list of symbol pairs; optionally apply dropout
        pairs = [(bpe_codes[pair], i, pair) for (i, pair) in enumerate(zip(word, word[1:])) if (
            not dropout or random.random() > dropout) and pair in bpe_codes]

        if not pairs:
            break

        # get first merge operation in list of BPE codes
        bigram = min(pairs)[2]

        # find start position of all pairs that we want to merge
        positions = [i for (rank, i, pair) in pairs if pair == bigram]

        i = 0
        new_word = []
        bigram = '+'.join(bigram)
        for j in positions:
            # merges are invalid if they start before current position. This can happen if there are overlapping pairs: (x x x -> xx x)
            if j < i:
                continue
            new_word.extend(word[i:j])  # all symbols before merged pair
            new_word.append(bigram)  # merged pair
            i = j + 2  # continue after merged pair
        new_word.extend(word[i:])  # add all symbols until end of word
        word = new_word

    word = tuple(word)

    cache[orig] = word
    return word


def recursive_split(segment, bpe_codes, vocab, separator, final=False):
    """Recursively split segment into smaller units (by reversing BPE merges)
    until all units are either in-vocabulary, or cannot be split futher."""

    try:
        if final:
            left, right = bpe_codes[segment + '</w>']
            right = right[:-4]
        else:
            left, right = bpe_codes[segment]
    except:
        #sys.stderr.write('cannot split {0} further.\n'.format(segment))
        yield segment
        return

    if left + separator in vocab:
        yield left
    else:
        for item in recursive_split(left, bpe_codes, vocab, separator, False):
            yield item

    if (final and right in vocab) or (not final and right + separator in vocab):
        yield right
    else:
        for item in recursive_split(right, bpe_codes, vocab, separator, final):
            yield item
