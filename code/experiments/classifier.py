# Copyright (c) Daniel Kamenicky.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import absolute_import
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB

import torch
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, classification_report
from sklearn.metrics import cohen_kappa_score


def load_data(is_mt5=True):
    if is_mt5:
        path = 'mt5'
        dev_data = pd.read_json("/data/results/dataset/mt5/dev_dataset.json")
        train_data = pd.read_json("/data/results/dataset/mt5/train_dataset.json")
    else:
        path = 't5'
        dev_data = pd.read_json("/data/results/dataset/t5/dev_dataset.json")
        train_data = pd.read_json("/data/results/dataset/t5/train_dataset.json")
    
    return train_data, dev_data, path


def preprocess_data(dev_data, train_data):
    dev_data = dev_data.sample(frac=1, random_state=420).reset_index(drop=True)
    train_data = train_data.sample(frac=1, random_state=420).reset_index(drop=True)
    
    dev_data['label'] = dev_data['label'].replace({'refute': 0, 'support': 1})
    train_data['label'] = train_data['label'].replace({'refute': 0, 'support': 1})
    
    y_test = dev_data['label'].values
    y_train = train_data['label'].values
    
    vectorizer = TfidfVectorizer(
        sublinear_tf=True, max_df=0.5, min_df=5, stop_words=None
    )
    
    X_train = vectorizer.fit_transform(list(train_data['claim']))
    X_test = vectorizer.transform(list(dev_data['claim']))

    target_names = ['refute', 'support']

    return X_train, X_test, y_train, y_test, target_names


def process():
    for model_type in [True, False]:    
        train_data, dev_data, path = load_data(is_mt5=model_type)

        if model_type:
            print("Evaluating mT5 dataset...")
        else:
            print("Evaluating T5 dataset...")

        X_train, X_test, y_train, y_test, target_names = preprocess_data(dev_data=dev_data, train_data=train_data)

        model = LogisticRegression(random_state=0, max_iter=1000)
        base_dir = '/data/results/experiments'
        
        if not os.path.exists(base_dir):
            os.makedirs(base_dir)
        
        path = base_dir + '/' + path
        if not os.path.exists(path):
            os.makedirs(path)

        model.fit(X_train, y_train)
        pred = model.predict(X_test)

        print(model.__class__,__name__)
        print(classification_report(y_test,pred, target_names=target_names))

        fig, ax = plt.subplots(figsize=(10, 5))
        ConfusionMatrixDisplay.from_predictions(y_test, pred, ax=ax)
        ax.xaxis.set_ticklabels(target_names)
        ax.yaxis.set_ticklabels(target_names)
        _ = ax.set_title(
            f"Confusion Matrix for {model.__class__.__name__}"
        )
        plt.savefig(path + '/' + model.__class__.__name__ + ".png")


if __name__=="__main__":
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        with torch.cuda.device(0):
            process()