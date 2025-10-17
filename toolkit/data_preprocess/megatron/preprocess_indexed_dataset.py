# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
#
# Modified tokenizer calls and added to handle fixed-length data
# ============================================================================
"""Processing large data for pretraining."""
import time
import gzip
import glob
import argparse
import math
import json
import os
import sys
import multiprocessing
import numpy as np
import nltk
from nltk.tokenize.punkt import PunktLanguageVars

# pylint: disable=W0611
from mindformers.dataset.blended_datasets.indexed_dataset import IndexedDatasetBuilder
from mindformers.models import build_tokenizer

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             os.path.pardir)))


# https://stackoverflow.com/questions/33139531/preserve-empty-lines-with-nltks-punkt-tokenizer
class CustomLanguageVars(PunktLanguageVars):

    _period_context_fmt = r"""
        \S*                          # some word material
        %(SentEndChars)s             # a potential sentence ending
        \s*                       #  <-- THIS is what I changed
        (?=(?P<after_tok>
            %(NonWord)s              # either other punctuation
            |
            (?P<next_tok>\S+)     #  <-- Normally you would have \s+ here
        ))"""


class IdentitySplitter:
    @staticmethod
    def tokenize(*text):
        return text


class Encoder:
    """Encoder"""
    def __init__(self, args):
        self.args = args
        self.tokenizer = get_tokenizer(args)

    def initializer(self):
        """initializer"""
        # Use Encoder class as a container for global data
        if self.args.split_sentences:
            library = os.path.join("tokenizers", "punkt", f"{self.args.lang}.pickle")
            url = f"nltk:{library}"
            splitter = nltk.load(url)
            if self.args.keep_newlines:
                # pylint: disable=W0212
                Encoder.splitter = nltk.tokenize.punkt.PunktSentenceTokenizer(train_text=splitter._params,
                                                                              lang_vars=CustomLanguageVars())
            else:
                Encoder.splitter = splitter

        else:
            Encoder.splitter = IdentitySplitter()

    def split(self, json_line):
        """split sentences"""
        data = json.loads(json_line)
        output = {}
        for key in self.args.json_keys:
            text = data[key]
            max_len = 1000000
            tokens_list = [Encoder.splitter.tokenize(text[i:i + max_len]) for i in range(0, len(text), max_len)]
            output[key] = [
                tokens
                for partial in tokens_list
                for tokens in partial
            ]
        return json.dumps(output), len(json_line)

    def encode(self, json_line):
        """encode data"""
        data = json.loads(json_line)
        ids = {}
        lens = {}
        for key in self.args.json_keys:
            text = data[key]
            if isinstance(text, list):
                sentences = text
            else:
                sentences = [text]
            doc_ids = []
            sentence_lens = []
            for sentence in sentences:
                sentence_ids = self.tokenizer(sentence)
                if sentence_ids:
                    doc_ids.extend(sentence_ids['input_ids'])
                    sentence_lens.append(len(sentence_ids['input_ids']))
            if doc_ids and self.args.append_eod:
                doc_ids.append(self.tokenizer.eos_token_id)
                sentence_lens[-1] += 1
            ids[key] = doc_ids
            lens[key] = sentence_lens
        return ids, lens, len(json_line)


class Partition:
    """Partition and processing json file"""
    def __init__(self, args, workers):
        self.args = args
        self.workers = workers

    def print_processing_stats(self, count, proc_start, total_bytes_processed):
        """print processing stats"""
        if count % self.args.log_interval == 0:
            current = time.time()
            elapsed = current - proc_start
            mbs = total_bytes_processed / elapsed / 1024 / 1024
            print(f"Processed {count} documents",
                  f"({count/elapsed} docs/s, {mbs} MB/s).",
                  file=sys.stderr)

    def split_sentences(self, file_name):
        """split_sentence"""
        input_file_name, output_file_name = file_name
        print("Opening", input_file_name)
        fin = open(input_file_name, 'r', encoding='utf-8')
        fout = open(output_file_name, 'w')

        encoder = Encoder(self.args)
        pool = multiprocessing.Pool(self.workers, initializer=encoder.initializer)
        split_docs = pool.imap(encoder.split, fin, 32)

        proc_start = time.time()
        total_bytes_processed = 0
        for i, (doc, bytes_processed) in enumerate(split_docs, start=1):
            total_bytes_processed += bytes_processed
            fout.write(doc + "\n")
            self.print_processing_stats(i, proc_start, total_bytes_processed)

        fin.close()
        fout.close()


    def process_json_file(self, file_name):
        """Processing json file"""
        input_file_name, output_prefix = file_name
        print("Opening", input_file_name)
        fin = open(input_file_name, 'r', encoding='utf-8')

        startup_start = time.time()
        encoder = Encoder(self.args)
        pool = multiprocessing.Pool(self.workers, initializer=encoder.initializer)
        encoded_docs = pool.imap(encoder.encode, fin, 32)

        level = "document"
        if self.args.split_sentences:
            level = "sentence"

        output_bin_files = {}
        output_idx_files = {}
        builders = {}

        for key in self.args.json_keys:
            output_bin_files[key] = "{}_{}_{}.bin".format(output_prefix,
                                                          key, level)
            output_idx_files[key] = "{}_{}_{}.idx".format(output_prefix,
                                                          key, level)
            builders[key] = IndexedDatasetBuilder(
                output_bin_files[key],
                dtype=np.int32,
            )

        startup_end = time.time()
        proc_start = time.time()
        total_bytes_processed = 0
        print("Time to startup:", startup_end - startup_start)
        for i, (doc, sentence_lens, bytes_processed) in enumerate(encoded_docs, start=1):
            total_bytes_processed += bytes_processed
            for key in doc.keys():
                builders[key].add_document(np.array(doc[key], dtype=np.int32), sentence_lens[key])
            self.print_processing_stats(i, proc_start, total_bytes_processed)

        fin.close()
        builders[key].finalize(output_idx_files[key])


def get_tokenizer(args):
    """Get tokenizer according to args."""
    if args.tokenizer_type == 'HuggingFaceTokenizer':
        return build_tokenizer(use_legacy=False,
                               pretrained_model_dir=args.tokenizer_dir,
                               trust_remote_code=args.trust_remote_code)
    if args.tokenizer_type == 'AutoRegister':
        tokenizer_config = {
            'type': args.auto_register.split('.')[-1],
            'vocab_file': args.vocab_file,
            'merges_file': args.merges_file,
            'tokenizer_file': args.tokenizer_file,
            'eos_token': args.eos_token,
            'add_bos_token': args.add_bos_token,
            'add_eos_token': args.add_eos_token,
            'auto_register': args.auto_register
        }
        return build_tokenizer(tokenizer_config)
    raise NotImplementedError(f"Currently, `--tokenizer_type` does not support {args.tokenizer_type}.")


def get_args():
    """get arguments"""
    parser = argparse.ArgumentParser()
    group = parser.add_argument_group(title='input data')
    group.add_argument('--input', type=str, required=True,
                       help='Path to input JSON')
    group.add_argument('--json-keys', nargs='+', default=['text'],
                       help='space separate listed of keys to extract from json')
    group.add_argument('--split-sentences', action='store_true',
                       help='Split documents into sentences.')
    group.add_argument('--keep-newlines', action='store_true',
                       help='Keep newlines between sentences when splitting.')

    group = parser.add_argument_group(title='tokenizer')
    group.add_argument('--tokenizer-type', type=str, required=True,
                       choices=['AutoRegister', 'HuggingFaceTokenizer'],
                       help='The tokenizer of the corresponding model.')
    group.add_argument('--vocab-file', type=str, default=None,
                       help='Path to the vocab file or tokenizer.model')
    group.add_argument('--merges-file', type=str, default=None,
                       help='Path to the BPE merge file (if necessary).')
    group.add_argument('--tokenizer-file', type=str, default=None,
                       help='The path of tokenizer.json')
    group.add_argument('--tokenizer-dir', type=str, default=None,
                       help='The directory of HuggingFace tokenizer.')
    group.add_argument('--trust-remote-code', action='store_true',
                       help='Whether or not to allow for custom models defined on the Hub in their own modeling files.')
    group.add_argument('--eos_token', type=str, default='</s>')
    group.add_argument('--add_bos_token', type=str, default=False)
    group.add_argument('--add_eos_token', type=str, default=False)
    group.add_argument('--vocab-size', default=786,
                       help='size of vocab for use with NullTokenizer')
    group.add_argument('--append-eod', action='store_true',
                       help='Append an <eod> token to the end of a document.')
    group.add_argument('--lang', type=str, default='english',
                       help='Language to use for NLTK-powered sentence splitting.')
    group.add_argument('--register_path', type=str, default=None,
                       help='the register path of outer tokenizer.')
    group.add_argument('--auto_register', type=str, default=None,
                       help='the import order of tokenizer according to register_path. '
                            'For example, auto_register="qwen2_tokenizer.Qwen2Tokenizer" '
                            'with register_path="research/qwen2".')

    group = parser.add_argument_group(title='output data')
    group.add_argument('--output-prefix', type=str, required=True,
                       help='Path to binary output file without suffix')
    group = parser.add_argument_group(title='runtime')
    group.add_argument('--workers', type=int, default=1,
                       help=('Number of worker processes to launch.'
                             'A good default for fast pre-processing '
                             'is: (workers * partitions) = available CPU cores.'))
    group.add_argument('--partitions', type=int, default=1,
                       help='Number of file partitions')
    group.add_argument('--log-interval', type=int, default=1000,
                       help='Interval between progress updates')
    group.add_argument('--keep-sequential-samples', action='store_true',
                       help='Ensure ordering of samples in .jsonl files is preserved when using partitions>1.')
    args = parser.parse_args()
    args.keep_empty = False
    if args.tokenizer_type == 'AutoRegister':
        if not args.register_path or not args.auto_register:
            raise ValueError('When tokenizer_type = "AutoRegister", register_path and auto_register should be set.')
        os.environ["REGISTER_PATH"] = args.register_path
        if args.register_path not in sys.path:
            sys.path.append(args.register_path)
        print(f'The tokenizer {args.auto_register} in the path {args.register_path} will be applied.')
    elif args.tokenizer_type == 'HuggingFaceTokenizer':
        if not args.tokenizer_dir:
            raise ValueError('When tokenizer_type = "HuggingFaceTokenizer", tokenizer_dir should be set.')
        print(f'The directory {args.tokenizer_dir} will be applied to build HuggingFace tokenizer.')

    # some default/dummy values for the tokenizer
    args.rank = 1
    args.make_vocab_size_divisible_by = 128
    args.tensor_model_parallel_size = 1
    args.vocab_extra_ids = 0

    return args


def get_file_name(args, file_id):
    """Get file name"""
    file_name, extension = os.path.splitext(args.input)
    input_file_name = file_name + "_" + str(file_id) + extension
    sentence_split_file = file_name + "_ss_" + str(file_id) + extension
    output_prefix = args.output_prefix + "_" + str(file_id)
    file_names = {
        'partition': input_file_name,
        'sentence_split': sentence_split_file,
        'output_prefix': output_prefix}
    return file_names


def check_files_exist(in_ss_out_names, key, num_partitions):
    """check files exist"""
    for i in range(num_partitions):
        if not os.path.exists(in_ss_out_names[i][key]):
            return False
    return True


def partition_file(args):
    """partition file"""
    in_ss_out_names = []
    if args.partitions == 1:
        file_name, extension = os.path.splitext(args.input)
        sentence_split_file = file_name + "_ss" + extension
        file_names = {
            'partition': args.input,
            'sentence_split': sentence_split_file,
            'output_prefix': args.output_prefix}
        in_ss_out_names.append(file_names)
    else:
        in_file_names = glob.glob(args.input)
        # Count total number of lines across .jsonl files
        if args.keep_sequential_samples:
            total_sample_count = 0
            for filename in in_file_names:
                fc = 0
                with open(filename, "r") as fin:
                    for fc, _ in enumerate(fin):
                        pass
                total_sample_count += (fc + 1)
            partition_size = math.ceil(total_sample_count / args.partitions)
        # create .jsonl partition files
        for idx in range(args.partitions):
            in_ss_out_name = get_file_name(args, idx)
            in_ss_out_names.append(in_ss_out_name)
        # check to see if partitions were already created
        partitions_present = check_files_exist(in_ss_out_names, 'partition', args.partitions)
        # check to see if partitions with split sentences already created
        split_sentences_present = check_files_exist(in_ss_out_names, 'sentence_split', args.partitions)
        if not partitions_present and not split_sentences_present:
            # populate .jsonl partition files from parent files
            partitioned_input_files = []
            for idx in range(args.partitions):
                partitioned_input_file = open(in_ss_out_names[idx]['partition'], 'w')
                partitioned_input_files.append(partitioned_input_file)
            index = 0
            if args.keep_sequential_samples:
                line_count = 0
            for in_file_name in in_file_names:
                # support for gzip files
                if in_file_name.endswith(".gz"):
                    fin = gzip.open(in_file_name, 'rt')
                else:
                    fin = open(in_file_name, 'r', encoding='utf-8')
                for line in fin:
                    partitioned_input_files[index].write(line)
                    if args.keep_sequential_samples:
                        line_count += 1
                        if line_count % partition_size == 0:
                            index += 1
                    else:
                        index = (index + 1) % args.partitions
                fin.close()
            for idx in range(args.partitions):
                partitioned_input_files[idx].close()

    assert args.workers % args.partitions == 0
    partition = Partition(args, args.workers // args.partitions)

    return in_ss_out_names, partition


def main():
    args = get_args()

    if args.split_sentences:
        raise Exception("nltk library required for sentence splitting is not available.")

    in_ss_out_names, partition = partition_file(args)

    # check to see if paritions with split sentences already created
    split_sentences_present = check_files_exist(in_ss_out_names, 'sentence_split', args.partitions)

    # split sentences in partition files
    if args.split_sentences and not split_sentences_present:
        processes = []
        for name in in_ss_out_names:
            p = multiprocessing.Process(target=partition.split_sentences,
                                        args=((name['partition'], name['sentence_split']),))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()
        if args.partitions == 1:
            return


    # encode partition files in parallel
    processes = []
    input_key = 'sentence_split' if args.split_sentences else 'partition'
    for name in in_ss_out_names:
        p = multiprocessing.Process(target=partition.process_json_file,
                                    args=((name[input_key], name['output_prefix']),))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    if args.partitions == 1:
        return

    # merge bin/idx partitions
    level = "document"
    if args.split_sentences:
        level = "sentence"

    output_bin_files = {}
    output_idx_files = {}
    builders = {}

    for key in args.json_keys:
        output_bin_files[key] = "{}_{}_{}.bin".format(args.output_prefix,
                                                      key, level)
        output_idx_files[key] = "{}_{}_{}.idx".format(args.output_prefix,
                                                      key, level)
        builders[key] = IndexedDatasetBuilder(
            output_bin_files[key],
            dtype=np.int32,
        )

        for name in in_ss_out_names:
            parition_output_prefix = name['output_prefix']
            full_partition_output_prefix = "{}_{}_{}".format(parition_output_prefix,
                                                             key, level)
            builders[key].add_index(full_partition_output_prefix)
        builders[key].finalize(output_idx_files[key])


if __name__ == '__main__':

    main()
