cd mGEN

# DEV SET INVALID DOCS DK02
CUDA_VISIBLE_DEVICES=1 python eval_mgen.py \
    --model_name_or_path ../../data/models/mGEN_model/ \
    --evaluation_set ../../data/outputs/converted/dev/negative/val.source \
    --gold_data_path ../../data/outputs/converted/dev/negative/gold_para_qa_data_dev.tsv \
    --predictions_path ../../data/outputs/mGEN/dev/negative/xor_dev_final_results.txt \
    --gold_data_mode qa \
    --model_type mt5 \
    --max_length 20 \
    --eval_batch_size 1

# DEV SET VALID DOCS
python eval_mgen.py \
    --model_name_or_path ../../data/models/mGEN_model/ \
    --evaluation_set ../../data/outputs/converted/dev/possitive/val.source \
    --gold_data_path ../../data/outputs/converted/dev/possitive/gold_para_qa_data_dev.tsv \
    --predictions_path ../../data/outputs/mGEN/dev/possitive/xor_dev_final_results.txt \
    --gold_data_mode qa \
    --model_type mt5 \
    --max_length 20 \
    --eval_batch_size 1

# TRAIN SET INVALID DOCS DK01
python eval_mgen.py \
    --model_name_or_path ../../data/models/mGEN_model/ \
    --evaluation_set ../../data/outputs/converted/train/negative/val.source \
    --gold_data_path ../../data/outputs/converted/train/negative/gold_para_qa_data_dev.tsv \
    --predictions_path ../../data/outputs/mGEN/train/negative/xor_dev_final_results.txt \
    --gold_data_mode qa \
    --model_type mt5 \
    --max_length 20 \
    --eval_batch_size 1

# TRAIN SET VALID DOCS
python eval_mgen.py \
    --model_name_or_path ../../data/models/mGEN_model/ \
    --evaluation_set ../../data/outputs/converted/train/possitive/val.source \
    --gold_data_path ../../data/outputs/converted/train/possitive/gold_para_qa_data_dev.tsv \
    --predictions_path ../../data/outputs/mGEN/train/possitive/xor_dev_final_results.txt \
    --gold_data_mode qa \
    --model_type mt5 \
    --max_length 20 \
    --eval_batch_size 1