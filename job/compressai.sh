#!/bin/bash 
export HOME=/ghome/gaocs
cd /ghome/gaocs/FCM-NQ/coding/CompressAI
# pip install -e . -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install -e . 

python run_batch.py \
    --task "$1" \
    --arch "$2" \
    --lambda_value "$3" \
    --epochs "$4" \
    --save_period "$5" \
    --learning_rate "$6" \
    --batch_size "$7" \
    --patch_size "$8" \
    --quant_type "$9" \
    --samples "${10}" \
    --bit_depth "${11}" \
    --pretrained_model "${12}" 
