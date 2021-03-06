#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Data pre-processing: build vocabularies and binarize training data.
"""

from collections import Counter
from itertools import zip_longest

from fairseq import options, tasks, utils
from fairseq.data import indexed_dataset
from fairseq.binarizer import Binarizer
from multiprocessing import Pool

import os
import shutil
import pdb


def main(args):
    utils.import_user_module(args)

    print(args)

    os.makedirs(args.destdir, exist_ok=True)
    target = not args.only_source

    task = tasks.get_task(args.task)

    def train_path(lang):
        return "{}{}".format(args.trainpref, ("." + lang) if lang else "")

    def file_name(prefix, lang):
        fname = prefix
        if lang is not None:
            fname += ".{lang}".format(lang=lang)
        return fname

    def dest_path(prefix, lang):
        return os.path.join(args.destdir, file_name(prefix, lang))

    def dict_path(lang):
        return dest_path("dict", lang) + ".txt"

    def build_dictionary(filenames, src=False, tgt=False):
        assert src ^ tgt
        return task.build_dictionary(
            filenames,
            workers=args.workers,
            threshold=args.thresholdsrc if src else args.thresholdtgt,
            nwords=args.nwordssrc if src else args.nwordstgt,
            padding_factor=args.padding_factor,
        )

    if not args.srcdict and os.path.exists(dict_path(args.source_lang)):
        raise FileExistsError(dict_path(args.source_lang))
    if target and not args.tgtdict and os.path.exists(dict_path(args.target_lang)):
        raise FileExistsError(dict_path(args.target_lang))

    if args.joined_dictionary:

        assert not args.srcdict or not args.tgtdict, \
            "cannot use both --srcdict and --tgtdict with --joined-dictionary"

        if args.srcdict:
            src_dict = task.load_dictionary(args.srcdict)
        elif args.tgtdict:
            src_dict = task.load_dictionary(args.tgtdict)
        else:
            assert args.trainpref, "--trainpref must be set if --srcdict is not specified"
            src_dict = build_dictionary(
                {train_path(lang) for lang in [args.source_lang, args.target_lang]}, src=True
            )
        tgt_dict = src_dict
    else:
        # for masked ml
        if args.srcdict:
            if args.two_inputs:
                src_dict_p, src_dict_b = task.load_two_dictionary(
                    args.srcdict, args.tgtdict)
                src_dict_p.save(dict_path('p'))
                src_dict_b.save(dict_path('b'))
            else:
                src_dict = task.load_dictionary(args.srcdict)
                src_dict.save(dict_path(args.lang))
        else:
            assert args.trainpref, "--trainpref must be set if --srcdict is not specified"
            src_dict = build_dictionary(
                [train_path(args.source_lang)], src=True)
            src_dict.save(dict_path(args.source_lang))

    def make_binary_dataset(vocab, input_prefix, output_prefix, lang, num_workers, vocabb=None):
        print("| [{}] Phoneme Dictionary: {} types".format(
            lang, len(vocab) - 1))
        if vocabb is not None:
            print("| [{}] BPE Dictionary: {} types".format(
                lang, len(vocabb) - 1))
        n_seq_tok = [0, 0]

        def merge_result(worker_result):
            n_seq_tok[0] += worker_result["nseq"]
            n_seq_tok[1] += worker_result["ntok"]

        input_file = "{}{}".format(
            input_prefix, ("." + lang) if lang is not None else ""
        )
        # the files will be cut for different parts for processing
        # !!!! pay attention for my dataset
        # I will use dataset with multiple input 现在的输入不是文本而是indexed dataset
        if args.indexed_dataset:
            # the input will be indexed dataset 不是纯文本
            offsets = Binarizer.find_indexdataset_offsets(
                input_file, num_workers)
        else:
            # 输入是普通文本
            offsets = Binarizer.find_offsets(input_file, num_workers)
        pool = None

        if num_workers > 1:
            pool = Pool(processes=num_workers - 1)
            for worker_id in range(1, num_workers):
                prefix = "{}{}".format(output_prefix, worker_id)
                pool.apply_async(
                    binarize,
                    (
                        args,
                        input_file,
                        vocab,
                        prefix,
                        lang,
                        offsets[worker_id],
                        offsets[worker_id + 1],
                        vocabb,
                    ),
                    callback=merge_result
                )
                # pdb.set_trace()
                # binarize(args,
                #          input_file,
                #          vocab,
                #          prefix,
                #          lang,
                #          offsets[worker_id],
                #          offsets[worker_id + 1],
                #          vocabb,
                #          ),
            pool.close()

        ds = indexed_dataset.make_builder(dataset_dest_file(args, output_prefix, lang, "bin"),
                                          impl=args.dataset_impl, vocab_size=len(vocab))
        if not args.two_inputs:
            merge_result(
                Binarizer.binarize(
                    input_file, vocab, lambda t: ds.add_item(t),
                    offset=0, end=offsets[1], append_eos=False
                )
            )
        else:
            # indexed dataset as input
            if args.indexed_dataset:
                merge_result(
                    Binarizer.binarize_two_index_dataset(
                        input_file, vocab, vocabb, lambda t: ds.add_item(t),
                        offset=0, end=offsets[1], append_eos=False
                    )
                )

            else:
                merge_result(
                    Binarizer.binarize_two(
                        input_file, vocab, vocabb, lambda t: ds.add_item(t),
                        offset=0, end=offsets[1], append_eos=False
                    )
                )

        if num_workers > 1:
            pool.join()
            for worker_id in range(1, num_workers):
                # pdb.set_trace()
                prefix = "{}{}".format(output_prefix, worker_id)
                temp_file_path = dataset_dest_prefix(args, prefix, lang)
                ds.merge_file_(temp_file_path)
                os.remove(indexed_dataset.data_file_path(temp_file_path))
                os.remove(indexed_dataset.index_file_path(temp_file_path))

        ds.finalize(dataset_dest_file(args, output_prefix, lang, "idx"))

        print(
            "| [{}] {}: {} sents, {} tokens".format(
                lang,
                input_file,
                n_seq_tok[0],
                n_seq_tok[1],
                vocab.unk_word,
            )
        )

    def make_dataset(vocab, input_prefix, output_prefix, lang, num_workers=1, vocabb=None):
        if args.dataset_impl == "raw":
            # Copy original text file to destination folder
            output_text_file = dest_path(
                output_prefix +
                ".{}-{}".format(args.source_lang, args.target_lang),
                lang,
            )
            shutil.copyfile(file_name(input_prefix, lang), output_text_file)
        else:
            make_binary_dataset(vocab, input_prefix,
                                output_prefix, lang, num_workers, vocabb=vocabb)

    def make_all(lang, vocab, vocabb=None):
        if args.trainpref:
            make_dataset(vocab, args.trainpref, "train",
                         lang, num_workers=args.workers, vocabb=vocabb)
        if args.validpref:
            for k, validpref in enumerate(args.validpref.split(",")):
                outprefix = "valid{}".format(k) if k > 0 else "valid"
                make_dataset(vocab, validpref, outprefix,
                             lang, num_workers=args.workers, vocabb=vocabb)
        if args.testpref:
            for k, testpref in enumerate(args.testpref.split(",")):
                outprefix = "test{}".format(k) if k > 0 else "test"
                make_dataset(vocab, testpref, outprefix,
                             lang, num_workers=args.workers, vocabb=vocabb)

    if args.two_inputs:
        make_all(args.source_lang, src_dict_p, src_dict_b)
    else:
        make_all(args.source_lang, src_dict)

    print("| Wrote preprocessed data to {}".format(args.destdir))

    if args.alignfile:
        assert args.trainpref, "--trainpref must be set if --alignfile is specified"
        src_file_name = train_path(args.source_lang)
        tgt_file_name = train_path(args.target_lang)
        freq_map = {}
        with open(args.alignfile, "r", encoding='utf-8') as align_file:
            with open(src_file_name, "r", encoding='utf-8') as src_file:
                with open(tgt_file_name, "r", encoding='utf-8') as tgt_file:
                    for a, s, t in zip_longest(align_file, src_file, tgt_file):
                        si = src_dict.encode_line(
                            s, add_if_not_exist=False, append_eos=False)
                        ti = tgt_dict.encode_line(
                            t, add_if_not_exist=False, append_eos=False)
                        ai = list(
                            map(lambda x: tuple(x.split("-")), a.split()))
                        for sai, tai in ai:
                            srcidx = si[int(sai)]
                            tgtidx = ti[int(tai)]
                            if srcidx != src_dict.unk() and tgtidx != tgt_dict.unk():
                                assert srcidx != src_dict.pad()
                                assert srcidx != src_dict.eos()
                                assert tgtidx != tgt_dict.pad()
                                assert tgtidx != tgt_dict.eos()

                                if srcidx not in freq_map:
                                    freq_map[srcidx] = {}
                                if tgtidx not in freq_map[srcidx]:
                                    freq_map[srcidx][tgtidx] = 1
                                else:
                                    freq_map[srcidx][tgtidx] += 1

        align_dict = {}
        for srcidx in freq_map.keys():
            align_dict[srcidx] = max(
                freq_map[srcidx], key=freq_map[srcidx].get)

        with open(
                os.path.join(
                    args.destdir,
                    "alignment.{}-{}.txt".format(args.source_lang,
                                                 args.target_lang),
                ),
                "w", encoding='utf-8'
        ) as f:
            for k, v in align_dict.items():
                print("{} {}".format(src_dict[k], tgt_dict[v]), file=f)


def binarize(args, filename, vocab, output_prefix, lang, offset, end, vocabb=None, append_eos=False):

    # ds is MMapIndexedDatasetBuilder for indexed Dataset builder
    ds = indexed_dataset.make_builder(dataset_dest_file(args, output_prefix, lang, "bin"),
                                      impl=args.dataset_impl, vocab_size=len(vocab))

    def consumer(tensor):
        ds.add_item(tensor)

    # append eos is done here
    if not args.two_inputs:
        res = Binarizer.binarize(filename, vocab, consumer, append_eos=append_eos,
                                 offset=offset, end=end)
    else:
        # indexed dataset 作为输入,额外的任务输入
        if args.indexed_dataset:
            res = Binarizer.binarize_two_index_dataset(
                filename, vocab, vocabb, consumer, append_eos=append_eos, offset=offset, end=end)
        else:
            res = Binarizer.binarize_two(
                filename, vocab, vocabb, consumer, append_eos=append_eos, offset=offset, end=end)
    print(dataset_dest_file(args, output_prefix, lang, "idx"))
    ds.finalize(dataset_dest_file(args, output_prefix, lang, "idx"))
    return res


def binarize_alignments(args, filename, parse_alignment, output_prefix, offset, end):
    ds = indexed_dataset.make_builder(dataset_dest_file(args, output_prefix, None, "bin"),
                                      impl=args.dataset_impl, vocab_size=None)

    def consumer(tensor):
        ds.add_item(tensor)

    res = Binarizer.binarize_alignments(filename, parse_alignment, consumer, offset=offset,
                                        end=end)
    ds.finalize(dataset_dest_file(args, output_prefix, None, "idx"))
    return res


def dataset_dest_prefix(args, output_prefix, lang):
    base = "{}/{}".format(args.destdir, output_prefix)
    if lang is not None:
        lang_part = ".{}-{}.{}".format(args.source_lang,
                                       args.target_lang, lang)
    elif args.only_source:
        lang_part = ""
    else:
        lang_part = ".{}-{}".format(args.source_lang, args.target_lang)

    return "{}{}".format(base, lang_part)


def dataset_dest_file(args, output_prefix, lang, extension):
    base = dataset_dest_prefix(args, output_prefix, lang)
    return "{}.{}".format(base, extension)


def get_offsets(input_file, num_workers):
    return Binarizer.find_offsets(input_file, num_workers)


def cli_main():
    parser = options.get_preprocessing_parser()
    args = parser.parse_args()
    main(args)


if __name__ == "__main__":
    cli_main()
