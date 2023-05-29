# Designing a Multilingual Fact-Checking Dataset from Existing Question-Answering Data
## Daniel Kamenicky

Please check the latest version of the code on [github](https://github.com/xkamen21/designing-a-multilingual-fact-checking-dataset-from-existing-question-answering-data) as there are still absolute paths in this code.

## RUN
To run the evaluation for the results described in the thesis pleas do the following steps:
1. You need to download [t5 pretrained models](https://nextcloud.fit.vutbr.cz/s/o6G6oiMs5rYiXcW) into /data/models/t5/
2. go to [code/CORA/](code/CORA/) folder. Pleas follow the steps in the CORA [README](code/CORA/README.md).
3. Run the dataset convertor module.
```sh
python -m code.utils.create_datasets.py
```
4. Run the evaluation with the TF-IDF logistic regression. 
```sh
python -m code.experiments.classifier.py
```

## Requirements
- it is recommended to have:
  - `python==3.8.10` (must be lower than `3.10`)
- the mDPR part was run with: 
  - `transformers==3.0.2`
  - `torch==1.7.0`
  - `faiss==1.7.2`
- the mGEN part was run with:
  - `transformers==4.26.1`
  - `torch==1.13.1`
  - `faiss==1.7.2`
- other parts was runned with libraries with the newest verisons (to date 05.17.2023)


