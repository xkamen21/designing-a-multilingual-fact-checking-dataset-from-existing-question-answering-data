# CORA
This code was taken from the official paper and was used in this work.

 This is the original implementation of the following paper: 
 Akari Asai, Xinyan Yu, Jungo Kasai and Hannaneh Hajishirzi. [One Question Answering Model for Many Languages with Cross-lingual Dense Passage Retrieval](https://arxiv.org/abs/2107.11976). *NeurIPS*. 2021. 

If you want to find out more information about the implementation please refer to the official paper.

## How to reproduce thesis results
Run the follow scripts with which you can easily set up and run evaluation on the XOR QA full dev set.

Steps:
- [download.sh](download.sh)
- [run_mDPR.sh](run_mDPR.sh)
- run the code [postprocess_mDPR.py](../utils/postprocess_mDPR.py)
- [run_convert.sh](run_convert.sh)
- [run_mGEN.sh](run_mGEN.sh)


The scripts will
1. download trained mDPR, mGEN and encoded Wikipedia embeddings,
2. run the whole pipeline on the evaluation and training XOR-TyDi QA sets, and 
3. calculate the QA scores.

## Installation

### Dependencies 
- Python 3.8.10
- [PyTorch](https://pytorch.org/) (currently tested on version 1.7.0 for mDPR and 1.13.1 for mGEN)
- [Transformers](https://github.com/huggingface/transformers) (version 3.0.2 for mDPR, version 4.26.1 for mGEN )
- Faiss (tested on version 1.7.2)


### Trained models
You can download trained models by running the commands below:
```sh
mkdir models
wget https://nlp.cs.washington.edu/xorqa/cora/models/all_w100.tsv
wget https://nlp.cs.washington.edu/xorqa/cora/models/mGEN_model.zip
wget https://nlp.cs.washington.edu/xorqa/cora/models/mDPR_biencoder_best.cpt
unzip mGEN_model.zip
mkdir embeddings
cd embeddings
for i in 0 1 2 3 4 5 6 7;
do 
  wget https://nlp.cs.washington.edu/xorqa/cora/models/wikipedia_split/wiki_emb_en_$i 
done
for i in 0 1 2 3 4 5 6 7;
do 
  wget https://nlp.cs.washington.edu/xorqa/cora/models/wikipedia_split/wiki_emb_others_$i  
done
cd ../..
```

## Evaluation

1. Run mDPR on the input data

```sh
python dense_retriever.py \
    --model_file ../models/mDPR_biencoder_best.cpt \
    --ctx_file ../models/all_w100.tsv \
    --qa_file ../data/xor_dev_full_v1_1.jsonl \
    --encoded_ctx_file "../models/embeddings/wiki_emb_*" \
    --out_file xor_dev_dpr_retrieval_results.json \
    --n-docs 20 --validation_workers 1 --batch_size 256 --add_lang
```

2. Postprocess retrieved passages from mDPR.

```sh
cd ../..
python -m code.utils.postprocess_mDPR.py
```

3. Convert the retrieved results into mGEN input format

```sh
cd mGEN
python3 convert_dpr_retrieval_results_to_seq2seq.py \
    --dev_fp ../mDPR/xor_dev_dpr_retrieval_results.json \
    --output_dir xorqa_dev_final_retriever_results \
    --top_n 15 \
    --add_lang \
    --xor_engspan_train data/xor_train_retrieve_eng_span.jsonl \
    --xor_full_train data/xor_train_full.jsonl \
    --xor_full_dev data/xor_dev_full_v1_1.jsonl
```

4. Run mGEN
```sh
CUDA_VISIBLE_DEVICES=0 python eval_mgen.py \
    --model_name_or_path \
    --evaluation_set xorqa_dev_final_retriever_results/val.source \
    --gold_data_path xorqa_dev_final_retriever_results/gold_para_qa_data_dev.tsv \
    --predictions_path xor_dev_final_results.txt \
    --gold_data_mode qa \
    --model_type mt5 \
    --max_length 20 \
    --eval_batch_size 4
cd ..
```