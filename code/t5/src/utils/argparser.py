from __future__ import absolute_import
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Creating and Searching index files')

    parser.add_argument('-c', '--checkpoint', action='store', type=str, default='t5-base',
                        help='Path to the folder with saved chackpoint of the target model')

    # parser.add_argument('-i', '--index', action='store', type=str, default=None,
    #                     help='Path to index file')
    # parser.add_argument('-p', '--path', action='store', type=str, default=None,
    #                     help='Path to data')
    # parser.add_argument('-d', '--dataset', action='store', type=str, required=True,
    #                     choices=['mlqa_dev', 'mlqa_test', 'wiki'], help='Dataset for indexing')
    # parser.add_argument('-e', '--eval-dataset', action='store', type=str, required=False, default=None,
    #                     choices=['dev', 'test'], help='Dataset for evaluation with answers')
    # parser.add_argument('-l', '--language', action='store', type=str, required=True,
    #                     choices=['en', 'es', 'de','multi'], help='Context language')
    # parser.add_argument('-a', '--analyzer', action='store', type=str, default=None,
    #                     choices=['en', 'es', 'de','standard'], help='Select analyzer')
    # parser.add_argument('-q', '--query', action='store', type=str, default=None,
    #                     help='Query data')
    # parser.add_argument('-c', '--create', action='store_true',
    #                     help='Create new index')
    # parser.add_argument('-m', '--metric', action='store', type=str,
    #                     choices=['dist', 'hit@k', 'qa_f1', 'review'], help='Compute metric')
    # parser.add_argument('-r', '--run', action='store', type=str,
    #                     choices=['reader', 'retriever'], help='Run interactively')
    # parser.add_argument('-s', '--ram_size', action='store', type=int, default=2048,
    #                     help='Ram size for indexing')

    # parser.add_argument('--progress_bar', action='store_true',
    #                     help='Show progress bar while indexing TODO')

    # parser.add_argument('--dry', action='store_true',
    #                     help='Test run TODO')
    # parser.add_argument('--test', action='store_true',
    #                     help='Test run TODO')

    return parser.parse_args()