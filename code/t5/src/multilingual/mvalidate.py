# Copyright (c) Daniel Kamenicky.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import absolute_import
import logging
import sacrebleu # zkusit
import pandas as pd
from simpletransformers.t5 import T5Model, T5Args
import torch
from .mtrainer import LANGUAGES
import json, os
from ..utils.mvalidator import MValidator

class MVladidate():
    # def __init__(self, model_path="/mnt/minerva1/nlp/projects/multiopen_FC/ZPJa/models/T5/mul_base/run2/normal/model/checkpoint-84994-epoch-1",
    # def __init__(self, model_path="/mnt/minerva1/nlp/projects/multiopen_FC/ZPJa/models/T5/mul_base/run2/normal/best_model",
    #                    data_path="/mnt/minerva1/nlp/projects/multiopen_FC/ZPJa/data/T5/results/mul/normal/"):
    # def __init__(self, model_path="/mnt/minerva1/nlp/projects/multiopen_FC/ZPJa/models/T5/mul_base/run3/fine_tune/run2/best_model",
    #                    data_path="/mnt/minerva1/nlp/projects/multiopen_FC/ZPJa/data/T5/results/mul/fine_tune/run2/"):
    def __init__(self, model_path="/mnt/minerva1/nlp/projects/multiopen_FC/ZPJa/models/T5/mul_base/run3/fine_tune/run2/model/checkpoint-6330-epoch-10",
                       data_path="/mnt/minerva1/nlp/projects/multiopen_FC/ZPJa/data/T5/results/mul/fine_tune/run2/"):
        self.data_path = data_path
        self.model_path = model_path
        self.__set_hyperparameters()
        self.model = self.__set_model()
        
        self.__set_logging()

        self.validator = MValidator()

    
    def __set_model(self):
        return T5Model("mt5", self.model_path, args=self.model_args)


    def __set_logging(self):
        logging.basicConfig(level=logging.INFO)
        transformers_logger = logging.getLogger("transformers")
        transformers_logger.setLevel(logging.WARNING)


    def __set_hyperparameters(self):
        self.model_args = T5Args()
        self.model_args.max_length = 512
        self.model_args.length_penalty = 1
        self.model_args.num_beams = 5

    def save_results(self, lang):
        results = {
            "model": self.model_name,
            "model_path": self.model_path,
            "questions": self.questions[lang],
            "answers": self.answers[lang],
            "languages": self.lang_list[lang],
            "targets": self.targets[lang],
            "outputs": self.outputs[lang]
            }
        
        with open(self.dir_path + lang + '_results.json', 'w') as f:
            json.dump(results, f)

    def prepare_data_structures(self):
        self.lang_list = {}
        self.targets = {}
        self.questions = {}
        self.answers = {}
        self.validate_list = {}
        self.outputs = {}
        for lang in LANGUAGES:
            self.lang_list[lang] = []
            self.targets[lang] = []
            self.questions[lang] = []
            self.answers[lang] = []
            self.validate_list[lang] = []
            self.outputs[lang] = []

        self.model_name = self.model_path.split('/')
        self.model_name = self.model_name[len(self.model_name)-1]

        self.dir_path = self.data_path + self.model_name + "/"

    def read_jsonl_file(self, file):
        with open(file, 'r') as f:
            data = [json.loads(line) for line in f]
        data = pd.DataFrame(data)

        return data        
        

    def load_train_dev_set(self):
        for lang in LANGUAGES:
            self.load_path = self.dir_path + lang + '_results.json'
            if os.path.exists(self.load_path):
                with open(self.load_path) as f:
                    data = json.load(f)
                self.lang_list[lang] = data['languages']
                self.targets[lang] = data['targets']
                self.questions[lang] = data['questions']
                self.answers[lang] = data['answers']
                self.outputs[lang] = data['outputs']
                loaded = True
            else:
                df = pd.read_table("/mnt/minerva1/nlp/projects/multiopen_FC/ZPJa/data/T5/mul_t5_park_data/" + lang + "/qa2d_dev.tsv")
                for i in range(df.shape[0]):
                    self.questions[lang].append(str(df['question'][i]))
                    self.answers[lang].append(str(df['answer'][i]))
                    self.validate_list[lang].append("language: " + str(lang) + " question: " + str(df['question'][i]) + " answer: " + str(df['answer'][i]))
                    self.lang_list[lang].append(lang)
                    self.targets[lang].append(str(df["turker_answer"][i]))
                loaded = False
        return loaded

    def load_fine_tune_dev_set(self):
        for lang in LANGUAGES:
            self.load_path = self.dir_path + lang + '_results.json'
            if os.path.exists(self.load_path):
                with open(self.load_path) as f:
                    data = json.load(f)
                self.lang_list[lang] = data['languages']
                self.targets[lang] = data['targets']
                self.questions[lang] = data['questions']
                self.answers[lang] = data['answers']
                self.outputs[lang] = data['outputs']
                loaded = True
            else:
                df1 = self.read_jsonl_file("/mnt/minerva1/nlp/projects/multiopen_FC/ZPJa/data/T5/mul_t5_park_data/" + lang + "/ambig_dev_annotated_data.jsonl")
                df2 = self.read_jsonl_file("/mnt/minerva1/nlp/projects/multiopen_FC/ZPJa/data/T5/mul_t5_park_data/" + lang + "/nq_dev_annotated_data.jsonl")
                for i in range(df1.shape[0]):
                    for ans, clm in [('pos_answer', 'pos_claim'), ('neg_answer', 'neg_claim')]:
                        self.questions[lang].append(str(df1['question'][i]))
                        self.answers[lang].append(str(df1[ans][i]))
                        self.validate_list[lang].append("language: " + str(lang) + " question: " + str(df1['question'][i]) + " answer: " + str(df1[ans][i]))
                        self.lang_list[lang].append(lang)
                        self.targets[lang].append(str(df1[clm][i]))
                for i in range(df2.shape[0]):
                    for ans, clm in [('pos_answer', 'pos_claim'), ('neg_answer', 'neg_claim')]:
                        self.questions[lang].append(str(df2['question'][i]))
                        self.answers[lang].append(str(df2[ans][i]))
                        self.validate_list[lang].append("language: " + str(lang) + " question: " + str(df2['question'][i]) + " answer: " + str(df2[ans][i]))
                        self.lang_list[lang].append(lang)
                        self.targets[lang].append(str(df2[clm][i]))
                loaded = False
        return loaded


    def load_data(self):
        self.prepare_data_structures()

        if not os.path.exists(self.dir_path):
            os.makedirs(self.dir_path)
        
        if self.fine_tune:
            loaded = self.load_fine_tune_dev_set()
        else:
            loaded = self.load_train_dev_set()
        

        return loaded


    def generate(self):
        if not self.load_data():
            for lang in LANGUAGES:
                self.outputs[lang] = []
                print("Generating outputs for language: " + str(lang))
                input_text = []
                for i in range(len(self.validate_list[lang])):
                    input_text.append(self.validate_list[lang][i])
                    if (i+1)%1000 == 0: 
                        self.outputs[lang].extend(self.model.predict(input_text))
                        input_text = []
                self.outputs[lang].extend(self.model.predict(input_text))
                self.save_results(lang)


    def validate(self, fine_tune=False):
        self.fine_tune = fine_tune
        self.generate()

        print("Validating outputs for all languages")
        self.validator.get_scores(generated_texts=[item for sublist in self.outputs.values() for item in sublist], 
                                  reference_texts=[item for sublist in self.targets.values() for item in sublist], 
                                  lang_codes=[item for sublist in self.lang_list.values() for item in sublist])
        for lang in LANGUAGES:
            print("Validating outputs for language: " + str(lang))
            self.validator.get_scores(generated_texts=self.outputs[lang], reference_texts=self.targets[lang], lang_codes=self.lang_list[lang])


if __name__=="__main__":
    validate  = MVladidate()
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        with torch.cuda.device(0):
            # validate.validate(fine_tune=False)
            validate.validate(fine_tune=True)
