python3 -m code.utils.create_dataset \
        -mp /data/models/t5/t5/ \
        -mul false \
        -mdpr_tp /data/outputs/mDPR/train/examples_with_possitive_doc.json \
        -mdpr_tn /data/outputs/mDPR/train/examples_with_negative_doc.json \
        -mdpr_dp /data/outputs/mDPR/dev/examples_with_possitive_doc.json \
        -mdpr_dn /data/outputs/mDPR/dev/examples_with_negative_doc.json \
        -mgen_tp /data/outputs/mGEN/train/possitive/xor_dev_final_results.txt \
        -mgen_tn /data/outputs/mGEN/train/negative/xor_dev_final_results.txt \
        -mgen_dp /data/outputs/mGEN/dev/possitive/xor_dev_final_results.txt \
        -mgen_dn /data/outputs/mGEN/dev/negative/xor_dev_final_results.txt \
        -sp /data/results/dataset/ \
        -sr false

python3 -m code.utils.create_dataset \
        -mp /data/models/t5/mt5/ \
        -mul true \
        -mdpr_tp /data/outputs/mDPR/train/examples_with_possitive_doc.json \
        -mdpr_tn /data/outputs/mDPR/train/examples_with_negative_doc.json \
        -mdpr_dp /data/outputs/mDPR/dev/examples_with_possitive_doc.json \
        -mdpr_dn /data/outputs/mDPR/dev/examples_with_negative_doc.json \
        -mgen_tp /data/outputs/mGEN/train/possitive/xor_dev_final_results.txt \
        -mgen_tn /data/outputs/mGEN/train/negative/xor_dev_final_results.txt \
        -mgen_dp /data/outputs/mGEN/dev/possitive/xor_dev_final_results.txt \
        -mgen_dn /data/outputs/mGEN/dev/negative/xor_dev_final_results.txt \
        -sp /data/results/dataset/ \
        -sr false