# Copyright (c) Daniel Kamenicky.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
 
from __future__ import absolute_import
import pandas as pd
import json
import argparse
import os

class mDPRPostprocess():
    def __init__(self):
        self.argument_parser()
        self.args = self.parser.parse_args()

    
    def argument_parser(self):
        self.parser = argparse.ArgumentParser(description='posprocess mDPR parser')
        self.parser.add_argument('-it','--input_train', type=str, help='input path to train data')
        self.parser.add_argument('-id','--input_dev', type=str, help='input path to evaluation data')
        self.parser.add_argument('-otp','--output_train_possitive', type=str, help='output for possitive train docs')
        self.parser.add_argument('-otn','--output_train_negative', type=str, help='output for negative train docs')
        self.parser.add_argument('-odp','--output_dev_possitive', type=str, help='output for possitive dev docs')
        self.parser.add_argument('-odn','--output_dev_negative', type=str, help='output for negative dev docs')

    def stats_positive_and_negative_docs(self):
        all_positive = 0
        all_negative  = 0
        count_all = 0
        dict = {}
        for example in self.data_list['ctxs']:
            positive = 0
            negative  = 0
            for i in range(example.__len__()):
                if example[i]['has_answer']:
                    positive += 1
                else:
                    negative += 1
            if positive:
                all_positive += 1
                if positive not in dict.keys():
                    dict[positive] = 1
                else:
                    dict[positive] += 1
            else:
                all_negative += 1
            if positive+negative==self.num_docs:
                count_all += 1
        print('Number of examples with at least one positive question: ', all_positive)
        print('Number of examples with no positive question: ', all_negative)
        sorted_dict = {k: dict[k] for k in sorted(dict.keys())}
        print('Dict with number of positive docs per question: ', sorted_dict)
        print('Dict with exact ' + str(self.num_docs) + ' docs: ', count_all)


    def create_possitive(self):
        for example in self.data.to_dict(orient='records'):
            docs_list = []
            index_list = []
            for i in range(example['ctxs'].__len__()):
                if docs_list.__len__()==self.num_docs:
                    break
                if example['ctxs'][i]['has_answer']:
                    docs_list.append(example['ctxs'][i])
                    index_list.append(i)
            
            if docs_list:
                i = 0
                while docs_list.__len__()<self.num_docs or i>example['ctxs'].__len__()-1:
                    if not i in index_list:
                        docs_list.append(example['ctxs'][i])
                        index_list.append(i)
                    i+=1
                example['ctxs'] = docs_list
                example['q_id'] = str(example['q_id'])
                self.data_list.append(example)
        
        self.data_list = pd.DataFrame(self.data_list)

        if self.train_set:
            with open(self.args.output_train_possitive, 'w') as f:
                json.dump(self.data_list.to_dict(orient='records'), f, indent=4)
        else:
            with open(self.args.output_dev_possitive, 'w') as f:
                json.dump(self.data_list.to_dict(orient='records'), f, indent=4)


    def create_negative(self):

        for example in self.data.to_dict(orient='records'):
            docs_list = []
            index_list = []
            all_bad=True
            for i in range(example['ctxs'].__len__()):
                if docs_list.__len__()==self.num_docs:
                    break
                if not example['ctxs'][i]['has_answer']:
                    docs_list.append(example['ctxs'][i])
                    index_list.append(i)
            
            if docs_list:
                i = 0
                while docs_list.__len__()<self.num_docs or i>example['ctxs'].__len__()-1:
                    if not i in index_list:
                        docs_list.append(example['ctxs'][i])
                        index_list.append(i)
                        all_bad=False
                    i-=1
                example['ctxs'] = docs_list
                example['q_id'] = str(example['q_id'])
                if all_bad:
                    self.data_list.append(example)
        
        self.data_list = pd.DataFrame(self.data_list)

        if self.train_set:
            with open(self.args.output_train_negative, 'w') as f:
                json.dump(self.data_list.to_dict(orient='records'), f, indent=4)
        else:
            with open(self.args.output_dev_negative, 'w') as f:
                json.dump(self.data_list.to_dict(orient='records'), f, indent=4)



    def process(self, train_set=True):
        self.train_set=train_set
        self.num_docs=15

        print("Reading output data...")
        
        if self.train_set:
            self.data = pd.read_json(self.args.input_train)
        else:
            self.data = pd.read_json(self.args.input_dev)
        
        self.data_list = []
        print("Creating valid dataset...")
        self.create_possitive()

        print("Calculating Postprocess statistics...")
        self.stats_positive_and_negative_docs()
        print()

        self.data_list = []
        print("Creating invalid dataset...")
        self.create_negative()
        
        print("Calculating Postprocess statistics...")
        self.stats_positive_and_negative_docs()


if __name__=="__main__":
    postprocessor = mDPRPostprocess()
    postprocessor.process(train_set=True)
    postprocessor.process(train_set=False)