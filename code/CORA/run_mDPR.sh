# DEV SET
CUDA_VISIBLE_DEVICES=0 python dense_retriever.py \
    --model_file ../../data/models/mDPR_biencoder_best.cpt \
    --ctx_file ../../data/models/all_w100.tsv \
    --qa_file  ../../data/xor_dev_full_v1_1.jsonl \
    --encoded_ctx_file "../../data/models/embeddings/wiki_emb_*" \
    --out_file ../../data/outputs/mDPR/dev/xor_dpr_retrieval_results.json \
    --n-docs 100 --validation_workers 1 --batch_size 256 --add_lang

# TRAIN SET
CUDA_VISIBLE_DEVICES=0 python dense_retriever.py \
    --model_file ../../data/models/mDPR_biencoder_best.cpt \
    --ctx_file ../../data/models/all_w100.tsv \
    --qa_file  ../../data/xor_train_full.jsonl \
    --encoded_ctx_file "../../data/models/embeddings/wiki_emb_*" \
    --out_file ../../data/outputs/mDPR/train/xor_dpr_retrieval_results.json \
    --n-docs 100 --validation_workers 1 --batch_size 256 --add_lang
