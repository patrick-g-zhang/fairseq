# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import pdb
import numpy as np

from fairseq.data import (
    data_utils,
    Dictionary,
    PhonemeDictionary,
    IdDataset,
    MaskTokensDataset,
    NestedDictionaryDataset,
    NumelDataset,
    DictNumelDataset,
    NumSamplesDataset,
    PadDataset,
    DictPadDataset,
    PrependTokenDataset,
    SortDataset,
    TokenBlockDataset,
    BPEMaskTokensDataset
)
from fairseq.tasks import FairseqTask, register_task
from fairseq.data.encoders.utils import get_whole_word_mask


@register_task('masked_lm')
class MaskedLMTask(FairseqTask):
    """Task for training masked language models (e.g., BERT, RoBERTa)."""

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        parser.add_argument('data', help='colon separated path to data directories list, \
                            will be iterated upon during epochs in round-robin manner')
        parser.add_argument('--sample-break-mode', default='complete',
                            choices=['none', 'complete',
                                     'complete_doc', 'eos'],
                            help='If omitted or "none", fills each sample with tokens-per-sample '
                                 'tokens. If set to "complete", splits samples only at the end '
                                 'of sentence, but may include multiple sentences per sample. '
                                 '"complete_doc" is similar but respects doc boundaries. '
                                 'If set to "eos", includes only one sentence per sample.')
        parser.add_argument('--tokens-per-sample', default=512, type=int,
                            help='max number of total tokens over all segments '
                                 'per sample for BERT dataset')
        parser.add_argument('--mask-prob', default=0.15, type=float,
                            help='probability of replacing a token with mask')
        parser.add_argument('--continuous-mask', default=1, type=int,
                            help='continuously mask k words')
        parser.add_argument('--leave-unmasked-prob', default=0.1, type=float,
                            help='probability that a masked token is unmasked')
        parser.add_argument('--random-token-prob', default=0.1, type=float,
                            help='probability of replacing a token with a random token')
        parser.add_argument('--freq-weighted-replacement', action='store_true',
                            help='sample random replacement words based on word frequencies')
        parser.add_argument('--mask-whole-words', default=False, action='store_true',
                            help='mask whole words; you may also want to set --bpe')
        parser.add_argument('--no-pad-prepend-token', default=True, action='store_false',
                            help='pad prependtoken or not')
        parser.add_argument('--phoneme-dict', default=False, action='store_true',
                            help='pad prependtoken or not')

    def __init__(self, args, dictionary):
        super().__init__(args)
        if args.two_inputs:
            self.phoneme_dictionary, self.bpe_dictionary = dictionary
            self.phoneme_mask_idx = self.phoneme_dictionary.add_symbol(
                '<mask>')
            self.bpe_mask_idx = self.bpe_dictionary.add_symbol('<mask>')
        else:
            self.dictionary = dictionary
            self.mask_idx = dictionary.add_symbol('<mask>')
        self.seed = args.seed

        # add mask token

    @classmethod
    def setup_task(cls, args, **kwargs):
        paths = args.data.split(':')
        assert len(paths) > 0
        if args.phoneme_dict:
            dictionary = PhonemeDictionary.load(
                os.path.join(paths[0], 'dict.txt'))
            print('| dictionary: {} types'.format(len(dictionary)))

        elif args.two_inputs:
            phoneme_dictionary = PhonemeDictionary.load(
                os.path.join(paths[0], 'dict.p.txt'))
            bpe_dictionary = Dictionary.load(
                os.path.join(paths[0], 'dict.b.txt'))
            print('| phoneme dictionary: {} types'.format(
                len(phoneme_dictionary)))
            print('| bpe dictionary: {} types'.format(len(bpe_dictionary)))
            dictionary = (phoneme_dictionary, bpe_dictionary)

        else:
            dictionary = Dictionary.load(os.path.join(paths[0], 'dict.txt'))
            print('| dictionary: {} types'.format(len(dictionary)))

        return cls(args, dictionary)

    def load_dataset(self, split, epoch=0, combine=False, **kwargs):
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """
        paths = self.args.data.split(':')
        assert len(paths) > 0
        data_path = paths[epoch % len(paths)]
        split_path = os.path.join(data_path, split)
        if self.args.two_inputs:
            dataset = data_utils.load_two_indexed_datasets(
                path=split_path,
                dictionary_p=self.phoneme_dictionary,
                dictionary_b=self.bpe_dictionary,
                dataset_impl=self.args.dataset_impl,
                combine=combine,
            )
        else:
            dataset = data_utils.load_indexed_dataset(
                split_path,
                self.source_dictionary,
                self.args.dataset_impl,
                combine=combine,
            )

        if dataset is None:
            raise FileNotFoundError(
                'Dataset not found: {} ({})'.format(split, split_path))

        # create continuous blocks of tokens
        dataset = TokenBlockDataset(
            dataset,
            dataset.sizes,
            self.args.tokens_per_sample - 1,  # one less for <s>
            break_mode=self.args.sample_break_mode,
            two_inputs=self.args.two_inputs,
        )
        print('| loaded {} blocks from: {}'.format(len(dataset), split_path))

        # prepend beginning-of-sentence token (<s>, equiv. to [CLS] in BERT)
        if self.args.no_pad_prepend_token:
            dataset = PrependTokenDataset(
                dataset, self.source_dictionary.bos())

        # create masked input and targets
        # pdb.set_trace()
        # mask_whole_words = get_whole_word_mask(self.args, self.source_dictionary) \
        #     if self.args.mask_whole_words else None

        if not self.args.two_inputs:
            src_dataset, tgt_dataset = MaskTokensDataset.apply_mask(
                dataset,
                self.source_dictionary,
                pad_idx=self.source_dictionary.pad(),
                mask_idx=self.mask_idx,
                seed=self.args.seed,
                mask_prob=self.args.mask_prob,
                leave_unmasked_prob=self.args.leave_unmasked_prob,
                random_token_prob=self.args.random_token_prob,
                freq_weighted_replacement=self.args.freq_weighted_replacement,
                mask_whole_words=self.args.mask_whole_words,
                continuous_mask=self.args.continuous_mask,
            )
        else:
            src_dataset, tgt_dataset = BPEMaskTokensDataset.apply_mask(
                dataset=dataset,
                vocab_p=self.phoneme_dictionary,
                vocab_b=self.bpe_dictionary,
                phoneme_pad_idx=self.phoneme_dictionary.pad(),
                bpe_pad_idx=self.bpe_dictionary.pad(),
                phoneme_mask_idx=self.phoneme_mask_idx,
                bpe_mask_idx=self.bpe_mask_idx,
                seed=self.args.seed,
                mask_prob=self.args.mask_prob,
                leave_unmasked_prob=self.args.leave_unmasked_prob,
                random_token_prob=self.args.random_token_prob,
                freq_weighted_replacement=self.args.freq_weighted_replacement,
                mask_whole_words=self.args.mask_whole_words,
            )
        pdb.set_trace()
        with data_utils.numpy_seed(self.args.seed + epoch):
            shuffle = np.random.permutation(len(src_dataset))

        if not self.args.two_inputs:
            self.datasets[split] = SortDataset(
                NestedDictionaryDataset(
                    {
                        'id': IdDataset(),
                        'net_input': {
                            'src_tokens': PadDataset(
                                src_dataset,
                                pad_idx=self.source_dictionary.pad(),
                                left_pad=False,
                            ),
                            'src_lengths': NumelDataset(src_dataset, reduce=False),
                        },
                        'target': PadDataset(
                            tgt_dataset,
                            pad_idx=self.source_dictionary.pad(),
                            left_pad=False,
                        ),
                        'nsentences': NumSamplesDataset(),
                        'ntokens': NumelDataset(src_dataset, reduce=True),
                    },
                    sizes=[src_dataset.sizes],
                ),
                sort_order=[
                    shuffle,
                    src_dataset.sizes,
                ],
            )
        else:
            self.datasets[split] = SortDataset(
                NestedDictionaryDataset(
                    {
                        'id': IdDataset(),
                        'net_input': {
                            'src_tokens': DictPadDataset(
                                src_dataset,
                                pad_idx=self.phoneme_dictionary.pad(),
                                left_pad=False,
                            ),
                            'src_lengths': DictNumelDataset(src_dataset, reduce=False),
                        },
                        'target': DictPadDataset(
                            tgt_dataset,
                            pad_idx=self.phoneme_dictionary.pad(),
                            left_pad=False,
                            is_target=True,
                        ),
                        'nsentences': NumSamplesDataset(),
                        'ntokens': DictNumelDataset(src_dataset, reduce=True),
                    },
                    sizes=[src_dataset.sizes],
                ),
                sort_order=[
                    shuffle,
                    src_dataset.sizes,
                ],
            )

    def build_dataset_for_inference(self, src_tokens, src_lengths, sort=True):
        src_dataset = PadDataset(
            TokenBlockDataset(
                src_tokens,
                src_lengths,
                self.args.tokens_per_sample - 1,  # one less for <s>
                pad=self.source_dictionary.pad(),
                eos=self.source_dictionary.eos(),
                break_mode='eos',
            ),
            pad_idx=self.source_dictionary.pad(),
            left_pad=False,
        )
        src_dataset = PrependTokenDataset(
            src_dataset, self.source_dictionary.bos())
        src_dataset = NestedDictionaryDataset(
            {
                'id': IdDataset(),
                'net_input': {
                    'src_tokens': src_dataset,
                    'src_lengths': NumelDataset(src_dataset, reduce=False),
                },
            },
            sizes=src_lengths,
        )
        if sort:
            src_dataset = SortDataset(src_dataset, sort_order=[src_lengths])
        return src_dataset

    @property
    def source_dictionary(self):
        return self.dictionary

    @property
    def target_dictionary(self):
        return self.dictionary
