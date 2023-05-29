python3 -m code.utils.postprocess_mDPR \
        -it /data/outputs/mDPR/train/xor_dpr_retrieval_results.json \
        -id /data/outputs/mDPR/dev/xor_dpr_retrieval_results.json \
        -otp /data/outputs/mDPR/train/examples_with_possitive_doc.json \
        -otn /data/outputs/mDPR/train/examples_with_negative_doc.json \
        -odp /data/outputs/mDPR/dev/examples_with_possitive_doc.json \
        -odn /data/outputs/mDPR/dev/examples_with_negative_doc.json \