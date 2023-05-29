from __future__ import absolute_import
# from sklearn.metrics import f1_score
# from rouge import Rouge 
# from .eval_utils import exact_match_score
from transformers import MT5Tokenizer
import evaluate

import nltk
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer


NORM_LANG = {'ar': 'arabic', 'fi': 'finnish', 'ru': 'russian'}

class MValidator():
    def __init__(self, tokenizer_model='google/mt5-base'):
        self.tokenizer = MT5Tokenizer.from_pretrained(tokenizer_model)
        nltk.download('stopwords', quiet=True)


    def normalize(self, text, lang):
        # Remove punctuation
        text = ''.join([char for char in text if char.isalpha() or char.isspace()])

        if lang in list(NORM_LANG.keys()):
            stop_words = set(stopwords.words(NORM_LANG[lang]))
            stemmer = SnowballStemmer(NORM_LANG[lang])
            
            # Convert text to lowercase
            text = text.lower()

            # Remove stop words
            text = ' '.join([word for word in text.split() if word not in stop_words])

            # Stem words
            text = ' '.join([stemmer.stem(word) for word in text.split()])

        return text


    def get_rouge_scores(self):
        rouge = evaluate.load('rouge')
        results = rouge.compute(predictions=self.generated_texts,
                                references=self.reference_texts,
                                tokenizer=self.tokenizer.tokenize)

        print("ROUGE-1 F1 score for batch: ", results['rouge1'])
        print("ROUGE-2 F1 score for batch: ", results['rouge2'])
        print("ROUGE-L F1 score for batch: ", results['rougeL'])
        return results['rouge1'], results['rouge2'], results['rougeL']


    def exact_match_score(self, sentence1, sentence2, lang):
        """
        Computes the exact match score between a generated text and a reference text.
        Returns 1 if the generated text matches the reference text exactly, and 0 otherwise.
        """
        result = int(self.normalize(sentence1, lang) == self.normalize(sentence2, lang))
        if lang in list(NORM_LANG.keys()):
            self.normalize_match_score.append(result)
        return result

    def calculate_match_score(self):
        self.normalize_match_score = []
        result = list(map(self.exact_match_score, self.generated_texts, self.reference_texts, self.lang_codes))
        print("Match Score: ", sum(result)/result.__len__())
        if self.normalize_match_score.__len__():
            self.normalize_result = sum(self.normalize_match_score)/self.normalize_match_score.__len__()
            print("Match score with normalization: ", self.normalize_result)
        else:
            self.normalize_result = -1
            print("Match score with normalization: ", "No results from languages that can be normalized")
        return sum(result)/result.__len__()

    def get_scores(self, generated_texts, reference_texts, lang_codes):
        self.generated_texts = generated_texts
        self.reference_texts = reference_texts
        self.lang_codes = lang_codes
        
        r1, r2, rl = self.get_rouge_scores()
        ms = self.calculate_match_score()
        
        return r1, r2, rl, ms, self.normalize_result


if __name__=='__main__':
    mvalidator = MValidator()

    generated = ["Я учусь программированию и мне нравится создавать новые проекты."]
    reference = ["Моя кошка любит есть рыбу и спать на солнце."]

    # r1, r2, rl, ms = mvalidator.get_scores(generated_texts=generated, reference_texts=reference)
    mvalidator.get_rouge_scores()
    # mvalidator.normalize()