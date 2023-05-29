# Copyright (c) Daniel Kamenicky.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
 
from __future__ import absolute_import
import pandas as pd
from collections import Counter
import re, string
from typing import Dict, List
from .match_score import MatchScore


class mGENPostprocess():
    def __init__(self):
        self.match_score = MatchScore()
        self.doc_number = 5

    
    def init_vars(self):
        self.self.data_list = []


    def normalize_answer(self, s):
        def remove_articles(text):
            return re.sub(r"\b(a|an|the)\b", " ", text)

        def white_space_fix(text):
            return " ".join(text.split())

        def remove_punc(text):
            exclude = set(string.punctuation)
            return "".join(ch for ch in text if ch not in exclude)

        def lower(text):
            return text.lower()

        return white_space_fix(remove_articles(remove_punc(lower(s))))


    def f1_score(self, prediction, ground_truth):
        prediction_tokens = self.normalize_answer(prediction).split()
        ground_truth_tokens = self.normalize_answer(ground_truth).split()
        common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
        num_same = sum(common.values())
        if num_same == 0:
            return 0
        precision = 1.0 * num_same / len(prediction_tokens)
        recall = 1.0 * num_same / len(ground_truth_tokens)
        f1 = (2 * precision * recall) / (precision + recall)
        return f1


    def exact_match_score(self, prediction, ground_truth):
        return self.normalize_answer(prediction) == self.normalize_answer(ground_truth)


    def calculate_exact_match(self, output_lns: List[str], reference_lns: List[str]) -> Dict:
        assert len(output_lns) == len(reference_lns)
        em = 0
        for hypo, pred in zip(output_lns, reference_lns):
            em += self.exact_match_score(hypo, pred)
        if len(output_lns) > 0:
            em /= len(output_lns)
        return {"em": em}
    

    def preprocess_base_on_f1(self):
        f1_scores = []
        index_list = []
        
        if self.show_results:
            print('Number of examples before f1 correction: ', self.predictions.__len__())
        
        if self.data_type == 'invalid':
            for i in range(self.predictions.__len__()):
                f1 = self.f1_score(self.predictions[i], self.targets[i])
                if f1==1:
                    index_list.append(i)
                f1_scores.append(f1)
        elif self.data_type == 'valid':
            for i in range(self.predictions.__len__()):
                f1 = self.f1_score(self.predictions[i], self.targets[i])
                f1_scores.append(f1)
        else:
            print('Error: invalid type of file')

        if self.show_results:
            print("F1 score: ", sum(f1_scores)/f1_scores.__len__())
            print("Exact match score: ", self.calculate_exact_match(self.predictions, self.targets)['em'])
            print()
        
        for i in index_list[::-1]:
            del self.predictions[i]
            del self.targets[i]
            self.data = self.data.drop(index=i)

        if self.show_results:
            print('Number of examples after f1 correction: ', self.predictions.__len__())

        self.data = self.data.reset_index(drop=True)

        f1_scores = []
        for i in range(self.predictions.__len__()):
            f1 = self.f1_score(self.predictions[i], self.targets[i])
            f1_scores.append(f1)

        if self.show_results:
            print("F1 score: ", sum(f1_scores)/f1_scores.__len__())
            print("Exact match score: ", self.calculate_exact_match(self.predictions, self.targets)['em'])
            print()


    def preprocess_based_on_document(self):
        counter = 0
        valid_index_list = []
        invalid_index_list = []
        doc_list = []
        
        if self.show_results:
            print('Number of examples before document correction: ', self.predictions.__len__())
        
        for i in range(self.predictions.__len__()):
            for j in range(self.doc_number):
                result = self.match_score.has_answer(answers=[self.predictions[i]], text=self.data['ctxs'][i][j]['text'])
                if result:
                    valid_index_list.append(i)
                    doc_list.append(j)
                    counter+=1
                    break
            if not result:
                invalid_index_list.append(i)

        pd.options.mode.chained_assignment = None

        for i in range(valid_index_list.__len__()):
            self.data['ctxs'][valid_index_list[i]] = [self.data['ctxs'][valid_index_list[i]][doc_list[i]]]
        
        pd.options.mode.chained_assignment = 'warn'

        for i in invalid_index_list[::-1]:
            del self.predictions[i]
            del self.targets[i]
            self.data = self.data.drop(index=i)     
        
        self.data = self.data.reset_index(drop=True)
        if self.show_results:
            print('Number of examples after document correction: ', self.predictions.__len__())
            print()
            f1_scores = []
            for i in range(self.predictions.__len__()):
                f1 = self.f1_score(self.predictions[i], self.targets[i])
                f1_scores.append(f1)

            print("F1 score: ", sum(f1_scores)/f1_scores.__len__())
            print("Exact match score: ", self.calculate_exact_match(self.predictions, self.targets)['em'])


    def process(self, mDPR_path=None, mGEN_path=None, data_type=None, show_results=True):
        if not mDPR_path:
            print('Did not get any path to mDPR results...')
            return
        elif not mGEN_path:
            print('Did not get any path to mGEN results...')
            return
        elif not data_type:
            print('Did not get type of files...')
            return
        
        self.data_type = data_type
        self.show_results = show_results

        self.data = pd.read_json(mDPR_path)
        with open(mGEN_path, 'r') as f:
            self.predictions = f.read().split('\n')
        
        self.targets = []
        
        for item in self.data['answers']:
            self.targets.append(item[0])
        
        # delete empty last line of text file
        del self.predictions[self.predictions.__len__()-1]

        self.preprocess_base_on_f1()

        self.preprocess_based_on_document()

        return self.data, self.predictions

if __name__=='__main__':
    postprocess = mGENPostprocess()
    postprocess.process()