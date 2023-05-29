# Copyright (c) Daniel Kamenicky.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import absolute_import
import numpy as np
import pandas as pd
import torch
import json
import os
from transformers import T5ForConditionalGeneration, T5Tokenizer, DataCollatorWithPadding, AdamW, get_scheduler
from accelerate import Accelerator
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from .preprocessor import Preprocessor
from ..utils.validator import Validator
from collections import deque


class Trainer():
    # def __init__(self, checkpoint='t5-base'):
    def __init__(self, checkpoint='/mnt/minerva1/nlp/projects/multiopen_FC/ZPJa/models/T5/base/run/fine_tuned/own-t5-base200'):
        self.checkpoint = checkpoint
        self.model = self.__set_model()
        self.tokenizer = self.__set_tokenizer()
        self.data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
        self.optimizer = AdamW(self.model.parameters(), lr=1e-5, no_deprecation_warning=True)

        self.epochs = 100
        self.model_index = 1
        self.model_saving_name = 'own-t5-base'
        # self.metric_save_path = '/mnt/minerva1/nlp/projects/multiopen_FC/ZPJa/outputs/T5/model_scores/base/run/'
        self.metric_save_path = '/mnt/minerva1/nlp/projects/multiopen_FC/ZPJa/outputs/T5/model_scores/base/run/fine_tuned/'
        # self.model_save_path = '/mnt/minerva1/nlp/projects/multiopen_FC/ZPJa/models/T5/base/run/'
        self.model_save_path = '/mnt/minerva1/nlp/projects/multiopen_FC/ZPJa/models/T5/base/run/fine_tuned/'


        self.train_buffer = deque(maxlen=20)
        self.train_loss_list = []
        self.eval_loss_list = []
        self.eval_loss = 0
        self.best_train_loss = 100000.0
        self.best_model_state_dict = {}

        self.preprocessor = Preprocessor(tokenizer = self.tokenizer)
        self.validator = Validator()


    def __set_model(self):
        print("Loading model")
        # pouzivat config file, gradient_checkpointing
        return T5ForConditionalGeneration.from_pretrained(self.checkpoint)

    def __set_tokenizer(self):
        print("Loading tokenizer")
        return T5Tokenizer.from_pretrained('t5-base')
    
    def save_results(self, r1, r2, rl, ms):
        path = self.metric_save_path + "results_" + self.model_saving_name + str(self.model_index) + ".json"
        results = {
            "epoch": self.model_index,
            "rouge-1": r1,
            "rouge-2": r2,
            "rouge-l": rl,
            "match_score": ms,
            "train_loss_evolution": self.train_loss_list,
            "eval_loss": self.eval_loss
            }
        with open(path, 'w') as f:
            json.dump(results, f)
    
    def save_checkpoint_loss(self):
        path = self.model_save_path + "checkpoints/checkpoint_loss" + str(self.model_index) + ".json"
        results = {
            "train_loss": self.best_train_loss
            }
        with open(path, 'w') as f:
            json.dump(results, f)

    def save_model(self):
        if not self.fine_tune or (self.model_index%50 == 0):
            path = self.model_save_path + self.model_saving_name + str(self.model_index)
            if not os.path.exists(path):
                os.makedirs(path)
            self.model.save_pretrained(path)

            # saving best checkpoint from the epoch
            path2 = self.model_save_path + "checkpoints/" + "checkpoint" + str(self.model_index) + ".pt"
            torch.save(self.best_model_state_dict, path2)
            self.save_checkpoint_loss()            

    def train(self):
        self.model.train()
        self.optimizer.zero_grad()
        counter = 1
        for batch in self.train_dataloader:
            with self.accelerator.accumulate(self.model):
                batch['labels'][batch['labels'] == self.tokenizer.pad_token_id] = -100
                # batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model(**batch)
                loss = outputs.loss
                self.train_buffer.append(loss.item())
                if counter%50 == 0:
                    print('Train Loss for last 50 batches: ', sum(self.train_buffer)/20)
                    self.train_loss_list.append(sum(self.train_buffer)/20)
                counter += 1
                if self.best_train_loss > loss.item():
                    self.best_train_loss = loss.item()
                    self.best_model_state_dict = self.model.state_dict()

                self.accelerator.backward(loss)
                # cliper
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

                self.optimizer.step()
                self.lr_scheduler.step()
                self.optimizer.zero_grad()
            
            self.progress_bar.update(1)
        
        self.save_model()


    def validate(self):
        outs = []
        targets = []
        self.model.eval()
        for batch in self.eval_dataloader:
            batch = {k: v.to(self.device) for k, v in batch.items()}
            with torch.no_grad():
                loss = self.model(**batch).loss
                outputs = self.model.generate(input_ids=batch['input_ids'], 
                                              attention_mask=batch['attention_mask'],
                                              max_length=128,
                                              num_beams=5,
                                              no_repeat_ngram_size = 3,
                                              early_stopping=True)
            self.eval_loss_list.append(loss.item())
            outs.extend([self.tokenizer.decode(ids, skip_special_tokens=True) for ids in outputs])
            targets.extend([self.tokenizer.decode(trgt, skip_special_tokens=True) for trgt in list(batch['labels'])])
        
        if outs.__len__() != targets.__len__():
            print('ERROR: Predictions and targets are different length')
            exit()
        else:
            self.eval_loss = sum(self.eval_loss_list)/self.eval_loss_list.__len__()
            r1, r2, rl, ms = self.validator.get_scores(outs, targets)
            self.save_results(r1, r2, rl, ms)


    def process(self, fine_tune=False):
        self.fine_tune = fine_tune
        if not self.fine_tune:
            self.train_dataset, self.validation_dataset = self.preprocessor.prepare_datasets(train_path = ["/mnt/minerva1/nlp/projects/multiopen_FC/ZPJa/data/T5/t5_park_data/qa2d_train.tsv"], 
                                                                                            validation_path = ["/mnt/minerva1/nlp/projects/multiopen_FC/ZPJa/data/T5/t5_park_data/qa2d_dev.tsv"], 
                                                                                            is_tsv=True,
                                                                                            fine_tune=False)
        else:
            self.train_dataset, self.validation_dataset = self.preprocessor.prepare_datasets(train_path = ["/mnt/minerva1/nlp/projects/multiopen_FC/ZPJa/data/T5/t5_park_data/nq_train_annotated_data.jsonl",
                                                                                                           "/mnt/minerva1/nlp/projects/multiopen_FC/ZPJa/data/T5/t5_park_data/ambig_train_annotated_data.jsonl"], 
                                                                                            validation_path = ["/mnt/minerva1/nlp/projects/multiopen_FC/ZPJa/data/T5/t5_park_data/nq_dev_annotated_data.jsonl",
                                                                                                               "/mnt/minerva1/nlp/projects/multiopen_FC/ZPJa/data/T5/t5_park_data/ambig_dev_annotated_data.jsonl"], 
                                                                                            is_tsv=False,
                                                                                            fine_tune=True)
        self.train_dataloader = DataLoader(
            self.train_dataset, shuffle=True, batch_size=8, collate_fn=self.data_collator
        )
        self.eval_dataloader = DataLoader(
            self.validation_dataset, shuffle=True, batch_size=8, collate_fn=self.data_collator
        )

        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        self.num_training_steps = self.epochs * len(self.train_dataloader)
        
        self.lr_scheduler = get_scheduler(
            "linear",
            optimizer=self.optimizer,
            num_warmup_steps=0,
            num_training_steps=self.num_training_steps,
        )

        self.progress_bar = tqdm(range(self.num_training_steps))

        self.accelerator = Accelerator(mixed_precision='no', gradient_accumulation_steps=3)
        self.model, self.optimizer, self.train_dataloader, self.eval_dataloader = self.accelerator.prepare(
            self.model, self.optimizer, self.train_dataloader, self.eval_dataloader
        )
        
        for epoch in range(self.epochs):
            self.train_loss_list.clear()
            self.eval_loss_list.clear()
            self.train_buffer.clear()
            self.best_model_state_dict = {}
            self.best_train_loss = 100000.0
            self.train()
            self.validate()
            self.model_index += 1
                

if __name__=='__main__':
    trainer = Trainer()
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        with torch.cuda.device(1):
            # trainer.process(fine_tune=False)
            trainer.process(fine_tune=True)
    else:
        print('CUDA is not available.')
