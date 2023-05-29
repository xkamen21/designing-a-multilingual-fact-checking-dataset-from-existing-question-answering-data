# Copyright (c) Daniel Kamenicky.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import absolute_import
import pandas as pd
from simpletransformers.t5 import T5Model, T5Args
import logging
import torch
import json
from sklearn.utils import shuffle

LANGUAGES = ["ar", "bn", "fi", "ja", "ko", "ru", "te"]

FILE_NAMES = {'tsv': {'train': ["qa2d_train.tsv"],
                      'dev': ["qa2d_dev.tsv"]},
              'jsonl': {'train': ["nq_train_annotated_data.jsonl",
                                  "ambig_train_annotated_data.jsonl"],
                        'dev': ["nq_dev_annotated_data.jsonl",
                                "ambig_dev_annotated_data.jsonl"]}
             }

FILE_PATH = "/mnt/minerva1/nlp/projects/multiopen_FC/ZPJa/data/T5/mul_t5_park_data/"


class MTrainer():
    # def __init__(self, model_name="google/mt5-base"):
    def __init__(self, model_name="/mnt/minerva1/nlp/projects/multiopen_FC/ZPJa/models/T5/mul_base/run3/fine_tune/run1/best_model"):
        self.model_name = model_name
        self.max_seq_length = 256
        self.train_batch_size = 5 #12
        self.eval_batch_size = 5 #12
        self.num_train_epochs = 10

        self.__set_hyperparameters()
        self.model = self.__set_model()
        self.__set_logging()

    def __set_model(self):
        return T5Model("mt5", self.model_name, args=self.model_args)


    def __set_logging(self):
        logging.basicConfig(level=logging.INFO)
        transformers_logger = logging.getLogger("transformers")
        transformers_logger.setLevel(logging.WARNING)


    def __set_hyperparameters(self):
        self.model_args = T5Args()
        self.model_args.max_seq_length = self.max_seq_length #256
        self.model_args.train_batch_size = self.train_batch_size #5
        self.model_args.eval_batch_size = self.eval_batch_size #5
        self.model_args.num_train_epochs = self.num_train_epochs #2
        self.model_args.evaluate_during_training = True
        self.model_args.evaluate_during_training_steps = 10000
        self.model_args.use_multiprocessing = False
        self.model_args.fp16 = False
        self.model_args.save_steps = -1
        self.model_args.save_eval_checkpoints = False
        self.model_args.no_cache = True
        self.model_args.reprocess_input_data = True
        self.model_args.overwrite_output_dir = True
        self.model_args.preprocess_inputs = False
        self.model_args.num_return_sequences = 1
        # self.model_args.wandb_project = "mt5_base_finetuning3"#"mt5_base_run1"
        self.model_args.output_dir = "/mnt/minerva1/nlp/projects/multiopen_FC/ZPJa/models/T5/mul_base/run3/fine_tune/run2/model"
        self.model_args.best_model_dir = "/mnt/minerva1/nlp/projects/multiopen_FC/ZPJa/models/T5/mul_base/run3/fine_tune/run2/best_model"
        self.model_args.gradient_accumulation_steps = 3
        self.model_args.max_length = 256

    def read_tsv_file(self):
        self.train_data_dict = {}
        self.validation_data_dict = {}
        for lang in self.lang_list:
            self.train_data_dict[lang] = pd.read_table(self.train_dict[lang][0])
            self.validation_data_dict[lang] = pd.read_table(self.validation_dict[lang][0])


    def read_jsonl_file(self):
        self.train_data_dict = {}
        self.validation_data_dict = {}
        for lang in self.lang_list:
            with open(self.train_dict[lang][0], 'r') as f:
                data = [json.loads(line) for line in f]
            data = pd.DataFrame(data)
            
            with open(self.train_dict[lang][1], 'r') as f:
                data2 = [json.loads(line) for line in f]
            data2 = pd.DataFrame(data2)
            
            self.train_data_dict[lang] = pd.concat([data, data2]).reset_index(drop=True)

            with open(self.validation_dict[lang][0], 'r') as f:
                data = [json.loads(line) for line in f]
            data = pd.DataFrame(data)

            with open(self.validation_dict[lang][1], 'r') as f:
                data2 = [json.loads(line) for line in f]
            data2 = pd.DataFrame(data2)
            
            self.validation_data_dict[lang] = pd.concat([data, data2]).reset_index(drop=True)


    def get_file_paths(self):
        train_dict = {}
        validation_dict = {}
        
        # if user wants to train model for one specific language
        if not self.one_language:
            lang_list = LANGUAGES
        else:
            lang_list = [self.one_language]
        
        for lang in lang_list:
            train_list = []
            validation_list = []
            for file in FILE_NAMES[self.file_type]["train"]:
                train_list.append(FILE_PATH + lang + '/' + file)
            for file in FILE_NAMES[self.file_type]["dev"]:
                validation_list.append(FILE_PATH + lang + '/' + file)
            train_dict[lang] = train_list
            validation_dict[lang] = validation_list

        return train_dict, validation_dict


    def preprocess_data(self):

        if self.fine_tune:
            for lang in list(self.validation_data_dict.keys()):
                question = self.validation_data_dict[lang]['question'].astype(str).tolist() * 2
                answer = self.validation_data_dict[lang]['pos_answer'].astype(str).tolist() + self.validation_data_dict[lang]['neg_answer'].astype(str).tolist()
                target = self.validation_data_dict[lang]['pos_claim'].astype(str).tolist() + self.validation_data_dict[lang]['neg_claim'].astype(str).tolist()
                
                new_df = pd.DataFrame()
                new_df['q'] = question
                new_df['a'] = answer
                new_df['t'] = target
                new_df['input_text'] = 'language: ' + lang + ' question: ' + new_df['q'] + ' answer: ' + new_df['a']
                new_df['prefix'] = ""
                new_df['target_text'] = new_df['t']

                self.validation_data_dict[lang] = new_df

                self.validation_data_dict[lang] = self.validation_data_dict[lang].drop('a', axis=1)
                self.validation_data_dict[lang] = self.validation_data_dict[lang].drop('q', axis=1)
                self.validation_data_dict[lang] = self.validation_data_dict[lang].drop('t', axis=1)
                self.validation_data_dict[lang] = self.validation_data_dict[lang].reindex(columns=['prefix', 'input_text', 'target_text'])


                question = self.train_data_dict[lang]['question'].astype(str).tolist() * 2
                answer = self.train_data_dict[lang]['pos_answer'].astype(str).tolist() + self.train_data_dict[lang]['neg_answer'].astype(str).tolist()
                target = self.train_data_dict[lang]['pos_claim'].astype(str).tolist() + self.train_data_dict[lang]['neg_claim'].astype(str).tolist()
                
                new_df = pd.DataFrame()
                new_df['q'] = question
                new_df['a'] = answer
                new_df['t'] = target
                new_df['input_text'] = 'language: ' + lang + ' question: ' + new_df['q'] + ' answer: ' + new_df['a']
                new_df['prefix'] = ""
                new_df['target_text'] = new_df['t']

                self.train_data_dict[lang] = new_df

                self.train_data_dict[lang] = self.train_data_dict[lang].drop('a', axis=1)
                self.train_data_dict[lang] = self.train_data_dict[lang].drop('q', axis=1)
                self.train_data_dict[lang] = self.train_data_dict[lang].drop('t', axis=1)
                self.train_data_dict[lang] = self.train_data_dict[lang].reindex(columns=['prefix', 'input_text', 'target_text'])

        else:
            for lang in list(self.validation_data_dict.keys()):
                self.validation_data_dict[lang]['prefix'] = ""
                self.validation_data_dict[lang]['input_text'] = 'language: ' + str(lang) + ' question: ' + self.validation_data_dict[lang]['question'].astype(str) + ' answer: ' + self.validation_data_dict[lang]['answer'].astype(str)
                self.validation_data_dict[lang]['target_text'] = self.validation_data_dict[lang]['turker_answer'].astype(str)
                self.validation_data_dict[lang] = self.validation_data_dict[lang].drop('example_uid', axis=1)
                self.validation_data_dict[lang] = self.validation_data_dict[lang].drop('rule-based', axis=1)
                self.validation_data_dict[lang] = self.validation_data_dict[lang].drop('dataset', axis=1)
                self.validation_data_dict[lang] = self.validation_data_dict[lang].drop('question', axis=1)
                self.validation_data_dict[lang] = self.validation_data_dict[lang].drop('answer', axis=1)
                self.validation_data_dict[lang] = self.validation_data_dict[lang].drop('turker_answer', axis=1)
                self.validation_data_dict[lang] = self.validation_data_dict[lang].reindex(columns=['prefix', 'input_text', 'target_text'])

                self.train_data_dict[lang]['prefix'] = ""
                self.train_data_dict[lang]['input_text'] = 'language: ' + str(lang) + ' question: ' + self.train_data_dict[lang]['question'].astype(str) + ' answer: ' + self.train_data_dict[lang]['answer'].astype(str)
                self.train_data_dict[lang]['target_text'] = self.train_data_dict[lang]['turker_answer'].astype(str)
                self.train_data_dict[lang] = self.train_data_dict[lang].drop('example_uid', axis=1)
                self.train_data_dict[lang] = self.train_data_dict[lang].drop('rule-based', axis=1)
                self.train_data_dict[lang] = self.train_data_dict[lang].drop('dataset', axis=1)
                self.train_data_dict[lang] = self.train_data_dict[lang].drop('question', axis=1)
                self.train_data_dict[lang] = self.train_data_dict[lang].drop('answer', axis=1)
                self.train_data_dict[lang] = self.train_data_dict[lang].drop('turker_answer', axis=1)
                self.train_data_dict[lang] = self.train_data_dict[lang].reindex(columns=['prefix', 'input_text', 'target_text'])


    def prepare_data(self):
        if self.one_language:
            self.lang_list = [self.one_language]
        else:
            self.lang_list = LANGUAGES

        if self.fine_tune:
            self.file_type='jsonl'
        else:
            self.file_type='tsv'
        
        self.train_dict, self.validation_dict = self.get_file_paths()

        if self.fine_tune:
            self.read_jsonl_file()
        else:
            self.read_tsv_file()

        self.preprocess_data()
        
        self.train_df = pd.concat(self.train_data_dict.values())
        self.eval_df = pd.concat(self.validation_data_dict.values())
        
        self.train_df = shuffle(self.train_df)
        self.eval_df = shuffle(self.eval_df)

        self.train_df = self.train_df.reset_index(drop=True)
        self.eval_df = self.eval_df.reset_index(drop=True)



    def train_model(self, one_language=None, fine_tune=False):
        self.one_language = one_language
        self.fine_tune = fine_tune

        self.prepare_data()

        self.model.train_model(self.train_df, eval_data=self.eval_df)


if __name__=="__main__":
    trainer = MTrainer()
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        with torch.cuda.device(1):
            # trainer.train_model(fine_tune=False)
            trainer.train_model(fine_tune=True)
            # trainer.train_model(fine_tune=False, one_language='ru')