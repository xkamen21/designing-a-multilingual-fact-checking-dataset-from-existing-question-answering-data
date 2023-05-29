from __future__ import absolute_import
import pandas as pd
import json
import os
import matplotlib.pyplot as plt

ROUGE1 = "rouge-1"
ROUGE2 = "rouge-2"
ROUGEL = "rouge-l"
MATCHSCORE = "match_score"
TRAINLOSS = "train_loss_evolution"

# code for printing plots for rouge, EM and train loss for English model

def read_jsonl_file(path):
    with open(path, 'r') as f:
        data = [json.loads(line) for line in f]
    return pd.DataFrame(data)


def prepare_data(fine_tune=False):
    if fine_tune:
        path_base = "/mnt/minerva1/nlp/projects/multiopen_FC/ZPJa/outputs/T5/model_scores/base/run4/fine_tuned/"
    else:
        path_base = "/mnt/minerva1/nlp/projects/multiopen_FC/ZPJa/outputs/T5/model_scores/base/run4/"
    list_dir = os.listdir(path_base)

    for file in list_dir:
        path = path_base + file
        data = read_jsonl_file(path)
        list_rouge_1.append(data[ROUGE1][0])
        list_rouge_2.append(data[ROUGE2][0])
        list_rouge_l.append(data[ROUGEL][0])
        list_match_score.append(data[MATCHSCORE][0])
        list_train_loss.extend(data[TRAINLOSS][0])


def plot_data(fine_tune=False):
    for key in dict_of_lists.keys():
        fig, ax = plt.subplots()

        # Plot the data
        ax.plot(dict_of_lists[key])

        # Set the x and y axis labels
        ax.set_xlabel('Checkpoints')
        ax.set_ylabel('Values')

        # Set the title of the plot
        ax.set_title(key)

        # Save the plot as a PNG file
        if fine_tune:
            plt.savefig('/mnt/minerva1/nlp/projects/multiopen_FC/ZPJa/outputs/T5/model_scores_graph/base/run4/fine_tuned/' + key +'.png')
        else:
            plt.savefig('/mnt/minerva1/nlp/projects/multiopen_FC/ZPJa/outputs/T5/model_scores_graph/base/run4/' + key +'.png')


def parse(fine_tune=False):
    global list_rouge_1
    global list_rouge_2
    global list_rouge_l
    global list_match_score
    global list_train_loss
    global dict_of_lists

    list_rouge_1 = []    
    list_rouge_2 = []    
    list_rouge_l = []    
    list_match_score = []
    list_train_loss = []
    dict_of_lists = {"rouge_1": list_rouge_1, "rouge_2": list_rouge_2, "rouge_l": list_rouge_l, "match_score": list_match_score, "train_loss": list_train_loss}


    prepare_data(fine_tune=fine_tune)
    if not fine_tune:
        original_list = dict_of_lists["train_loss"]

        averages_list = []
        for i in range(int(dict_of_lists["train_loss"].__len__()/101)):
            start_index = i * 101
            end_index = start_index + 101
            sub_list = original_list[start_index:end_index]
            avg = sum(sub_list) / len(sub_list)
            averages_list.append(avg)

        dict_of_lists["train_loss"] = averages_list

    plot_data(fine_tune=fine_tune)


if __name__=='__main__':
    # parse(fine_tune=False)
    parse(fine_tune=True)