from __future__ import absolute_import
from rouge import Rouge 
from .eval_utils import exact_match_score
# import nltk


class Validator():
    def __init__(self):
        pass


    def get_rouge_scores(self):
        rouge_1_f1_score = rouge_2_f1_score = rouge_l_f1_score = 0
        rouge = Rouge()
        scores = rouge.get_scores(self.generated_texts, self.reference_texts)
        for score in scores:
            rouge_1_f1_score += score['rouge-1']['f']
            rouge_2_f1_score += score['rouge-2']['f']
            rouge_l_f1_score += score['rouge-l']['f']
        
        rouge_1_f1_score = rouge_1_f1_score/scores.__len__()
        rouge_2_f1_score = rouge_2_f1_score/scores.__len__()
        rouge_l_f1_score = rouge_l_f1_score/scores.__len__()

        print("ROUGE-1 F1 score for batch: ", rouge_1_f1_score)
        print("ROUGE-2 F1 score for batch: ", rouge_2_f1_score)
        print("ROUGE-L F1 score for batch: ", rouge_l_f1_score)
        return rouge_1_f1_score, rouge_2_f1_score, rouge_l_f1_score
    

    def exact_match_score(self, sentence1, sentence2):
        """
        Computes the exact match score between a generated text and a reference text.
        Returns 1 if the generated text matches the reference text exactly, and 0 otherwise.
        """
        result = int(exact_match_score(sentence1, sentence2))        
        return result


    def calculate_match_score(self):
        result = list(map(self.exact_match_score, self.generated_texts, self.reference_texts))
        print("Match Score: ", sum(result)/result.__len__())
        return sum(result)/result.__len__()


    def get_scores(self, generated_texts, reference_texts):
        self.generated_texts = generated_texts
        self.reference_texts = reference_texts

        r1, r2, rl = self.get_rouge_scores()
        ms = self.calculate_match_score()
        return r1, r2, rl, ms


if __name__=='__main__':
    validator = Validator()

    generated = ["Я учусь программированию и мне нравится создавать новые проекты."]
    reference = ["Моя кошка любит есть рыбу и спать на солнце."]

    r1, r2, rl, ms = validator.get_scores(generated_texts=generated, reference_texts=reference)