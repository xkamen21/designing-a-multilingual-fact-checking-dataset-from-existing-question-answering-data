# Copyright (c) Daniel Kamenicky.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import absolute_import
import pandas as pd
import json
from datasets import DatasetDict, Dataset

class Preprocessor():
    def __init__(self, tokenizer = None):
        self.tokenizer = tokenizer


    def read_tsv_file(self):
        self.train_data = pd.read_table(self.train_path[0])
        self.validation_data = pd.read_table(self.validation_path[0])


    def read_jsonl_file(self):
        with open(self.train_path[0], 'r') as f:
            data = [json.loads(line) for line in f]
        data = pd.DataFrame(data)
        
        with open(self.train_path[1], 'r') as f:
            data2 = [json.loads(line) for line in f]
        data2 = pd.DataFrame(data2)
        
        self.train_data = pd.concat([data, data2]).reset_index(drop=True)

        with open(self.validation_path[0], 'r') as f:
            data = [json.loads(line) for line in f]
        data = pd.DataFrame(data)

        with open(self.validation_path[1], 'r') as f:
            data2 = [json.loads(line) for line in f]
        data2 = pd.DataFrame(data2)
        
        self.validation_data = pd.concat([data, data2]).reset_index(drop=True)


    def create_datasets(self):
        if not self.fine_tune:
            validation_data = {
                'question': self.validation_data['question'].astype(str).tolist(),
                'answer': self.validation_data['answer'].astype(str).tolist(),
                'target': list(self.validation_data['turker_answer'].astype(str)),
                'idx': list(range(1, self.validation_data.__len__()+1))
            }

            train_data = {
                'question': self.train_data['question'].astype(str).tolist(),
                'answer': self.train_data['answer'].astype(str).tolist(),
                'target': list(self.train_data['turker_answer'].astype(str)),
                'idx': list(range(1, self.train_data.__len__()+1))
            }
        else:
            validation_data = {
                'question': self.validation_data['question'].astype(str).tolist() + self.validation_data['question'].astype(str).tolist(),
                'answer': self.validation_data['pos_answer'].astype(str).tolist() + self.validation_data['neg_answer'].astype(str).tolist(),
                'target': self.validation_data['pos_claim'].astype(str).tolist() + self.validation_data['neg_claim'].astype(str).tolist(),
                'idx': list(range(1, (self.validation_data.__len__()*2)+1))
            }

            train_data = {
                'question': self.train_data['question'].astype(str).tolist() + self.train_data['question'].astype(str).tolist(), 
                'answer': self.train_data['pos_answer'].astype(str).tolist() + self.train_data['neg_answer'].astype(str).tolist(),
                'target': self.train_data['pos_claim'].astype(str).tolist() + self.train_data['neg_claim'].astype(str).tolist(),
                'idx': list(range(1, (self.train_data.__len__()*2)+1))
            }

        train_dataset = Dataset.from_dict(train_data)
        validation_dataset = Dataset.from_dict(validation_data)
        self.datasets = DatasetDict({'validation': validation_dataset, 'train': train_dataset})


    def preprocess_input(self, example):
        example['input_text'] = 'question: %s  answer: %s' % (example['question'], example['answer'])
        example['target_text'] = '%s' % example['target']
        
        return example


    def convert_to_features(self, example):
        input_encodings = self.tokenizer.batch_encode_plus(example['input_text'], padding='max_length', max_length=512)
        target_encodings = self.tokenizer.batch_encode_plus(example['target_text'], padding='max_length', max_length=128)

        encodings = {
            'input_ids': input_encodings['input_ids'], 
            'attention_mask': input_encodings['attention_mask'],
            'labels': target_encodings['input_ids'],
            'decoder_attention_mask': target_encodings['attention_mask']
        }
        
        return encodings

    
    def tokenize_datasets(self):
        self.tokenized_datasets = self.datasets.map(self.preprocess_input)
        self.tokenized_datasets = self.tokenized_datasets.map(self.convert_to_features, batched=True)


    def postprocess_datasets(self):
        self.tokenized_datasets = self.tokenized_datasets.remove_columns(["question", "answer", "idx", "target", "input_text", "target_text"])
        self.tokenized_datasets.set_format("torch")


    def prepare_datasets(self, train_path, validation_path, is_tsv, fine_tune=False):
        self.train_path = train_path
        self.validation_path = validation_path
        self.is_tsv = is_tsv
        self.fine_tune = fine_tune

        print("Loading data")
        if self.is_tsv:
            self.read_tsv_file()
        else:
            self.read_jsonl_file()

        print("Creating datasets")
        self.create_datasets()

        print("Tokenizing datasets")
        self.tokenize_datasets()

        print("Postprocessing datasets")
        self.postprocess_datasets()

        return self.tokenized_datasets['train'], self.tokenized_datasets['validation']
