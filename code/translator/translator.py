# Copyright (c) Daniel Kamenicky.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
 
from __future__ import absolute_import
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, M2M100ForConditionalGeneration, M2M100Tokenizer
from argparse import Namespace
from typing import Union, List, Dict
import torch


TRANSLATORS = {
    # korea
    "ko-en": "facebook/m2m100_1.2B",
    "en-ko": "facebook/m2m100_1.2B",
    # arabic
    "ar-en": "Helsinki-NLP/opus-mt-ar-en",
    "en-ar": "Helsinki-NLP/opus-mt-en-ar",
    # bengali
    "bn-en": "Helsinki-NLP/opus-mt-bn-en",
    "en-bn": "Helsinki-NLP/opus-mt-en-mul",
    # finnish
    "fi-en": "Helsinki-NLP/opus-mt-fi-en",
    "en-fi": "Helsinki-NLP/opus-mt-en-fi",
    # japanese
    "ja-en": "Helsinki-NLP/opus-mt-ja-en",
    "en-ja": "Helsinki-NLP/opus-mt-en-mul",
    # russian 
    "ru-en": "Helsinki-NLP/opus-mt-ru-en",
    "en-ru": "Helsinki-NLP/opus-mt-en-ru",
    # telugu
    "te-en": "Helsinki-NLP/opus-mt-mul-en",
    "en-te": "Helsinki-NLP/opus-mt-en-mul"
    
}

LANGUAGE = {
    "ar": "ara",
    "bn": "ben",
    "fi": "fin",
    "ja": "jpn",
    "ko": "kor",
    "ru": "rus",
    "te": "tel"
}


class Translator():
    def __init__(self, device='cpu') -> None:
        # self.device = device 
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        # translators
        self.translators = {}
    
    def init_translator(self, translator):
        model_name = TRANSLATORS[translator]
        if 'ko' in translator:
            model = M2M100ForConditionalGeneration.from_pretrained(model_name).to(self.device)
            tokenizer = M2M100Tokenizer.from_pretrained(model_name)
        else:
            model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device)
            tokenizer = AutoTokenizer.from_pretrained(model_name)
        decode = tokenizer.decode

        return Namespace(model=model,
                         generate=model.generate,
                         model_name=model_name,
                         tokenizer=tokenizer,
                         decode=decode)


    def translate(self, src_text, src_lang, dst_lang):
        if dst_lang == 'ko' or src_lang == 'ko':
            text = str(src_text)
        elif not (dst_lang == 'en' or dst_lang == 'ru' or dst_lang == 'fi'):
            text = ">>" + LANGUAGE[dst_lang] + "<<" + str(src_text)
        else:
            text = str(src_text)

        translator = src_lang + '-' + dst_lang
        if translator not in self.translators:
            self.translators[translator] = self.init_translator(translator)

        try:
            if 'ko' in translator:
                self.translators[translator].tokenizer.src_lang = src_lang
                input_ids = self.translators[translator].tokenizer(text, return_tensors="pt").to(self.device)

                translated_ids = self.translators[translator].generate(**input_ids, forced_bos_token_id=self.translators[translator].tokenizer.get_lang_id(dst_lang))
                translations = self.translators[translator].tokenizer.decode(translated_ids[0], skip_special_tokens=True)
            else:
                tokenizer_input = self.translators[translator].tokenizer(text, return_tensors="pt").to(self.device) #, max_length=512, truncation=True).to(self.device)

                translated = self.translators[translator].generate(**tokenizer_input)
                translations = self.translators[translator].decode(translated[0], skip_special_tokens=True)
        except Exception as e:
            raise e
        return translations


if __name__=='__main__':
    translator = Translator()
    # src = "en"
    # dst = "ru"
    # text = "Hello, today is a rainy day."
    # translation = translator.translate(text, src, dst)
    # print(translation)
    # back_translation = translator.translate(translation, dst, src)
    # print(back_translation)
