# Copyright (c) Daniel Kamenicky.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
 
from __future__ import absolute_import
from tqdm import tqdm
import pandas as pd
import torch
import logging
from .postprocess_mGEN import mGENPostprocess
from .match_score import MatchScore
from ..translator.translator import Translator
from simpletransformers.t5 import T5Model, T5Args
from transformers import T5ForConditionalGeneration, T5Tokenizer
import json, csv
import argparse


class CDataset():
    def __init__(self):
        self.argument_parser()
        self.args = self.parser.parse_args()

        self.mdpr_files = {'train': {'valid': self.args.mdpr_train_possitive,
                                     'invalid': self.args.mdpr_train_negative},
                           'dev': {'valid': self.args.mdpr_dev_possitive,
                                   'invalid': self.args.mdpr_dev_negative}}

        self.mgen_files = {'train': {'valid': self.args.mgen_train_possitive,
                                     'invalid': self.args.mgen_train_negative},
                           'dev': {'valid': self.args.mgen_dev_possitive,
                                   'invalid': self.args.mgen_dev_negative}}

        self.multilingual=self.args.multilingual
        if self.multilingual:
            self.__init_model(model_path=self.args.model_path)
        else:
            self.__init_model(model_path=self.args.model_path)

        self.translator = Translator()
        self.mgen_postprocessor = mGENPostprocess()


    def argument_parser(self):
        self.parser = argparse.ArgumentParser(description='posprocess mDPR parser')
        self.parser.add_argument('-mp','--model_path', type=str, help='model path')
        self.parser.add_argument('-mul','--multilingual', type=bool, help='process multilingual T5 model (mT5)')
        self.parser.add_argument('-mdpr_tp','--mdpr_train_possitive', type=str, help='possitive train data from mdpr')
        self.parser.add_argument('-mdpr_tn','--mdpr_train_negative', type=str, help='negative train data from mdpr')
        self.parser.add_argument('-mdpr_dp','--mdpr_dev_possitive', type=str, help='possitive dev data from mdpr')
        self.parser.add_argument('-mdpr_dn','--mdpr_dev_negative', type=str, help='negative dev data from mdpr')
        self.parser.add_argument('-mgen_tp','--mgen_train_possitive', type=str, help='possitive train data from mgen')
        self.parser.add_argument('-mgen_tn','--mgen_train_negative', type=str, help='negative train data from mgen')
        self.parser.add_argument('-mgen_dp','--mgen_dev_possitive', type=str, help='possitive dev data from mgen')
        self.parser.add_argument('-mgen_dn','--mgen_dev_negative', type=str, help='negative dev data from mgen')
        self.parser.add_argument('-sp','--save_path', type=str, help='path for final datasets')
        self.parser.add_argument('-sr','--show_results', type=str, help='show results during the run')


    def __init_model(self, model_path):
        if self.multilingual:
            self.__set_hyperparameters()
            self.model = T5Model("mt5", model_path, args=self.model_args)
            self.__set_logging()
        else:
            self.tokenizer = T5Tokenizer.from_pretrained('t5-base')
            self.model = T5ForConditionalGeneration.from_pretrained(model_path)


    def __set_logging(self):
        logging.basicConfig(level=logging.INFO)
        transformers_logger = logging.getLogger("transformers")
        transformers_logger.setLevel(logging.WARNING)


    def __set_hyperparameters(self):
        self.model_args = T5Args()
        self.model_args.max_length = 256
        self.model_args.length_penalty = 1
        self.model_args.num_beams = 5
    

    def set_label_and_split(self, data_type, file_type):
        self.splits = []
        self.labels = []

        self.splits.extend([file_type]*self.data_valid.shape[0])
        self.splits.extend([file_type]*self.data_invalid.shape[0])

        self.labels.extend(['support']*self.data_valid.shape[0])
        self.labels.extend(['refute']*self.data_invalid.shape[0])


    def insert_data(self, own_id, id, claim, evidence, lang, question, answer, label, split):
        item = {'id': own_id,
                'original_id': id,
                'lang': lang,
                'claim': claim,
                'label': label,
                'evidence': evidence,
                'question': question,
                'answer': answer,
                'split': split
                }

        self.dataset.append(item)


    def save_dataset(self):
        path = self.args.save_path
        if self.multilingual:
            path += "mt5/"
        else:
            path += "t5/"
        
        self.save_statistics(path=path)

        if self.splits[0] == 'dev':
            path += "dev_dataset.json"
        else:
            path += "train_dataset.json"
        
        with open(path, 'w') as f:
            json.dump(self.dataset, f)


    def save_statistics(self, path):
        stats = {"size": self.num_all_examples,
                 "supports_count": self.num_valid_claim,
                 "refute_count": self.num_invalid_claim,
                 "refute_invalid_evidence": self.num_invalid_calim_invalid_evidence,
                 "refute_valid_evidence": self.num_invalid_calim_valid_evidence,
                 "langs_distribution": [self.lang_dict]}

        if self.splits[0] == 'dev':
            path += "dev_stats.json"
        else:
            path += "train_stats.json"

        with open(path, 'w') as f:
            json.dump(stats, f)



    def match_valid_evidence_with_invalid_answer(self):
        self.data_invalid = pd.merge(self.data_invalid, self.data_valid[['q_id', 'ctxs']], on='q_id', how='left')
        
        self.num_invalid_calim_invalid_evidence = int(self.data_invalid['ctxs_y'].isna().sum())

        self.data_invalid['ctxs_y'] = self.data_invalid['ctxs_y'].fillna(self.data_invalid['ctxs_x'])
        self.data_invalid = self.data_invalid.drop('ctxs_x', axis=1).rename(columns={'ctxs_y': 'ctxs'})
        
        self.data = pd.concat([self.data_valid, self.data_invalid]).reset_index(drop=True)
        self.predictions = self.predictions_valid + self.predictions_invalid

        self.num_all_examples = self.predictions.__len__()
        self.num_invalid_calim_valid_evidence = self.predictions_invalid.__len__() - self.num_invalid_calim_invalid_evidence
        self.num_invalid_claim = self.predictions_invalid.__len__()
        self.num_valid_claim = self.predictions_valid.__len__()


    def mT5_dataset(self):
        self.lang_dict = {}

        if self.data.shape[0] != len(self.predictions):
            print('Predictions and targets are different length')
            exit()
        
        # create input_ids for generator
        input_list = []
        print("Loadidng input ids for model...")
        for i in tqdm(range(self.data.shape[0])):
            lang = str(self.data['lang'][i])

            lang = lang.replace(" ", "")

            question = str(self.data['question'][i][:-5])
            answer = str(self.predictions[i])
            input_text = 'language: ' + lang + ' question: ' + question + ' answer: ' + answer
            input_list.append(input_text)
        
        print("Generating outputs...")
        batched_input_list = []
        claims = []
        for i in range(input_list.__len__()):
            batched_input_list.append(input_list[i])
            if (batched_input_list.__len__() == 100) or (i == input_list.__len__()-1):
                claims.extend(self.model.predict(batched_input_list))
                batched_input_list = []

        print("Saving results to file...")        
        for i in tqdm(range(self.data.shape[0])):
            own_id = str(self.index)
            id = str(self.data['q_id'][i])
            lang = str(self.data['lang'][i])
            
            lang = lang.replace(" ", "")

            if lang not in list(self.lang_dict.keys()):
                self.lang_dict[lang] = 1
            else:
                self.lang_dict[lang] += 1

            question = str(self.data['question'][i][:-5])
            answer = str(self.predictions[i])
            evidence = str(self.data['ctxs'][i][0]['text'])
            claim = str(claims[i])
            label = self.labels[i]
            split = self.splits[i]

            self.insert_data(own_id=own_id, 
                             id=id,
                             lang=lang,
                             claim=claim,
                             evidence=evidence,
                             question=question,
                             answer=answer,
                             label=label,
                             split=split)
            self.index += 1


    def T5_dataset(self):
        batch_size = 8
        input_list = []
        self.lang_dict = {}

        print("Generating outputs...")
        pbar = tqdm(total=int(self.data.shape[0]/batch_size))
        for i in range(self.data.shape[0]):
            lang = str(self.data['lang'][i])
            question = str(self.data['question'][i][:-5])
            answer = str(self.predictions[i])        

            lang = lang.replace(" ", "")

            translated_question = self.translator.translate(question, lang, 'en')
            translated_answer = self.translator.translate(answer, lang, 'en')
            
            input_text = 'question: ' + str(translated_question) + ' answer: ' + str(translated_answer)

            input_list.append(input_text)

            if ((i+1)%batch_size == 0) or ((i+1) == self.data.shape[0]):
                # generate and save results
                input_ids = self.tokenizer(input_list, padding='max_length', max_length=512, truncation=True, return_tensors="pt").input_ids

                outputs = self.model.generate(input_ids=input_ids,
                                            max_length=256,
                                            num_beams=5,
                                            no_repeat_ngram_size = 3,
                                            early_stopping=True)
                
                claims = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
                input_list = []
                
                for j in range(claims.__len__()):
                    # calculate index for last 8 items
                    k = i-(batch_size-1)+j

                    own_id = str(self.index)
                    id = str(self.data['q_id'][k])
                    lang = str(self.data['lang'][k])
        
                    lang = lang.replace(" ", "")

                    if lang not in list(self.lang_dict.keys()):
                        self.lang_dict[lang] = 1
                    else:
                        self.lang_dict[lang] += 1
                    
                    question = str(self.data['question'][k][:-5])
                    answer = str(self.predictions[k])
                    evidence = str(self.data['ctxs'][k][0]['text'])
                    eng_claim = str(claims[j])
                    label = self.labels[k]
                    split = self.splits[k]

                    claim = self.translator.translate(eng_claim, 'en', lang)

                    self.insert_data(own_id=own_id, 
                                     id=id,
                                     lang=lang,
                                     claim=claim,
                                     evidence=evidence,
                                     question=question,
                                     answer=answer,
                                     label=label,
                                     split=split)
            
                    self.index += 1
                pbar.update(1)

        pbar.close()


    def process(self):

        for file_type in ['dev', 'train']:
            self.dataset = []
            self.index = 0

            print('Loading data...')
            data_type = 'valid'
            self.data_valid, self.predictions_valid = self.mgen_postprocessor.process(mDPR_path=self.mdpr_files[file_type][data_type],
                                                                                      mGEN_path=self.mgen_files[file_type][data_type],
                                                                                      data_type=data_type,
                                                                                      show_results=self.args.show_results)

            print('Valid data loaded.')
            data_type = 'invalid'
            self.data_invalid, self.predictions_invalid = self.mgen_postprocessor.process(mDPR_path=self.mdpr_files[file_type][data_type],
                                                                                          mGEN_path=self.mgen_files[file_type][data_type],
                                                                                          data_type=data_type,
                                                                                          show_results=self.args.show_results)
            
            print('Invalid data loaded.')
            self.set_label_and_split(data_type=data_type, file_type=file_type)

            print('Matching invalid answers with valid evidence...')
            self.match_valid_evidence_with_invalid_answer()
            
            if self.multilingual:
                self.mT5_dataset()
            else:
                self.T5_dataset()
            
            self.save_dataset()
            


if __name__=='__main__':
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        with torch.cuda.device(0):
            own_dataset = CDataset()
            own_dataset.process()
