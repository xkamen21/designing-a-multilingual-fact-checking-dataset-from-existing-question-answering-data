# download models
mkdir ../../data/
mkdir ../../data/models/
mkdir ../../data/outputs/
mkdir ../../data/outputs/converted
mkdir ../../data/outputs/converted/train
mkdir ../../data/outputs/converted/train/possitive
mkdir ../../data/outputs/converted/train/negative
mkdir ../../data/outputs/converted/dev
mkdir ../../data/outputs/converted/dev/possitive
mkdir ../../data/outputs/converted/dev/negative
mkdir ../../data/outputs/mDPR
mkdir ../../data/outputs/mDPR/train
mkdir ../../data/outputs/mDPR/dev
mkdir ../../data/outputs/mGEN
mkdir ../../data/outputs/mGEN/train
mkdir ../../data/outputs/mGEN/train/possitive
mkdir ../../data/outputs/mGEN/train/negative
mkdir ../../data/outputs/mGEN/dev
mkdir ../../data/outputs/mGEN/dev/possitive
mkdir ../../data/outputs/mGEN/dev/negative
mkdir ../../data/results
mkdir ../../data/results/dataset
mkdir ../../data/results/experiments

cd ../../data/models/
wget https://nlp.cs.washington.edu/xorqa/cora/models/all_w100.tsv
wget https://nlp.cs.washington.edu/xorqa/cora/models/mDPR_biencoder_best.cpt
wget https://nlp.cs.washington.edu/xorqa/cora/models/mGEN_model.zip
unzip mGEN_model.zip
rm -rf mGEN_model.zip

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

# download eval data
cd ../../
wget https://nlp.cs.washington.edu/xorqa/XORQA_site/data/xor_dev_full_v1_1.jsonl
wget https://nlp.cs.washington.edu/xorqa/XORQA_site/XORQA_site/data/xor_train_full.jsonl
wget https://nlp.cs.washington.edu/xorqa/XORQA_site/XORQA_site/data/xor_train_retrieve_eng_span.jsonl
