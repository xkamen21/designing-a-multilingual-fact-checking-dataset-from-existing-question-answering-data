cd mGEN
CUDA_VISIBLE_DEVICES=0

# DEV SET VALID_DOCS
python convert_dpr_retrieval_results_to_seq2seq.py \
    --dev_fp ../../data/outputs/mDPR/dev/examples_with_possitive_doc.json \
    --output_dir ../../data/outputs/converted/dev/possitive/ \
    --top_n 15 \
    --add_lang \
    --xor_engspan_train ../../data/xor_train_retrieve_eng_span.jsonl \
    --xor_full_train ../../data/xor_train_full.jsonl \
    --xor_full_dev ../../data/xor_dev_full_v1_1.jsonl

# DEV SET INVALID_DOCS
python convert_dpr_retrieval_results_to_seq2seq.py \
    --dev_fp ../../data/outputs/mDPR/dev/examples_with_negative_doc.json \
    --output_dir ../../data/outputs/converted/dev/negative/ \
    --top_n 15 \
    --add_lang \
    --xor_engspan_train ../../data/xor_train_retrieve_eng_span.jsonl \
    --xor_full_train ../../data/xor_train_full.jsonl \
    --xor_full_dev ../../data/xor_dev_full_v1_1.jsonl

# TRAIN SET VALID_DOCS
python convert_dpr_retrieval_results_to_seq2seq.py \
    --dev_fp ../../data/outputs/mDPR/train/examples_with_possitive_doc.json \
    --output_dir ../../data/outputs/converted/train/possitive/ \
    --top_n 15 \
    --add_lang \
    --xor_engspan_train ../../data/xor_train_retrieve_eng_span.jsonl \
    --xor_full_train ../../data/xor_train_full.jsonl \
    --xor_full_dev ../../data/xor_dev_full_v1_1.jsonl

# TRAIN SET INVALID_DOCS
python convert_dpr_retrieval_results_to_seq2seq.py \
    --dev_fp ../../data/outputs/mDPR/train/examples_with_negative_doc.json \
    --output_dir ../../data/outputs/converted/train/negative/ \
    --top_n 15 \
    --add_lang \
    --xor_engspan_train ../../data/xor_train_retrieve_eng_span.jsonl \
    --xor_full_train ../../data/xor_train_full.jsonl \
    --xor_full_dev ../../data/xor_dev_full_v1_1.jsonl