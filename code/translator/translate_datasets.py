# Copyright (c) Daniel Kamenicky.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import absolute_import
import pandas as pd
import json
import torch
import warnings
from tqdm.auto import tqdm
from .translator import Translator

warnings.filterwarnings("ignore", category=UserWarning)

def read_tsv_file(path):
    return pd.read_table(path)


def read_jsonl_file(path):
    with open(path, 'r') as f:
        data = [json.loads(line) for line in f]
    return pd.DataFrame(data)


def save_tsv_file(df, name):
    df.to_csv(name, sep='\t', index=False)


def save_jsonl_file(df, name):
    df.to_json(name, orient='records', lines=True)


def translate_jsonl_file(data, src_lang, tgt_lang, save_file):
    translated_dataframe = pd.DataFrame({})
    # progress_bar = tqdm(range(data.shape[0]))
    for i in tqdm(range(data.shape[0])):
        question = translator.translate(data["question"][i], src_lang, tgt_lang)
        pos_answer = translator.translate(data["pos_answer"][i], src_lang, tgt_lang)
        pos_claim = translator.translate(data["pos_claim"][i], src_lang, tgt_lang)
        neg_answer = translator.translate(data["neg_answer"][i], src_lang, tgt_lang)
        neg_claim = translator.translate(data["neg_claim"][i], src_lang, tgt_lang)
        
        new_row = {"question": question, 
                "pos_answer": pos_answer, 
                "pos_claim": pos_claim,
                "neg_answer": neg_answer,
                "neg_claim": neg_claim}
        
        new_df = pd.DataFrame([new_row])

        translated_dataframe = pd.concat([translated_dataframe, new_df], ignore_index=True)

    #     progress_bar.update(1)

    # progress_bar.close()
    save_jsonl_file(translated_dataframe, save_file)


def translate_tsv_file(data, src_lang, tgt_lang, save_file):
    translated_dataframe = pd.DataFrame({})
    # progress_bar = tqdm(range(data.shape[0]))
    for i in tqdm(range(data.shape[0])):
        question = translator.translate(data["question"][i], src_lang, tgt_lang)
        answer = translator.translate(data["answer"][i], src_lang, tgt_lang)
        label = translator.translate(data["turker_answer"][i], src_lang, tgt_lang)

        new_row = {"dataset": data["dataset"][i], 
                "example_uid": data["example_uid"][i], 
                "question": question,
                "answer": answer,
                "turker_answer": label,
                "rule-based": data["rule-based"][i]}
        
        new_df = pd.DataFrame([new_row])

        translated_dataframe = pd.concat([translated_dataframe, new_df], ignore_index=True)
        
    #     progress_bar.update(1)
    
    # progress_bar.close()
    save_tsv_file(translated_dataframe, save_file)


def translate_datasets():
    src_lang = "en"
    target_languages = ["ar", "bn", "fi", "ja", "ko", "ru", "te"]
    dataset_names = [("ambig_dev_annotated_data.jsonl", False), 
                     ("ambig_train_annotated_data.jsonl", False), 
                     ("nq_dev_annotated_data.jsonl", False), 
                     ("nq_train_annotated_data.jsonl", False),
                     ("qa2d_dev.tsv", True), # 'question', 'answer', 'turker_answer'
                     ("qa2d_train.tsv", True)] # 'question', 'answer', 'turker_answer'
    global save_path
    save_path = "/mnt/minerva1/nlp/projects/multiopen_FC/ZPJa/data/T5/mul_t5_park_data/"
    path = "/mnt/minerva1/nlp/projects/multiopen_FC/ZPJa/data/T5/t5_park_data/"
    
    global translator
    translator = Translator()
    
    for tgt_lang in target_languages:
        for file, is_tsv in dataset_names:
            file_path = path + file
            save_file = save_path + tgt_lang + "/" + file
            print("Processing file: ", file, " to target language: ", tgt_lang)
            if is_tsv:
                data = read_tsv_file(file_path)
                translate_tsv_file(data, src_lang, tgt_lang, save_file)
            else:
                data = read_jsonl_file(file_path)
                translate_jsonl_file(data, src_lang, tgt_lang, save_file)
            print("Language done: ", tgt_lang)
        
        print("All files were translated to the language: ", tgt_lang)
    

if __name__=="__main__":
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        with torch.cuda.device(0):
            translate_datasets()
    else:
        print('CUDA is not available.')