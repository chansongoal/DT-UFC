#!/bin/bash 
export HOME=/ghome/gaocs
cd /ghome/gaocs/FCM-NQ/coding/CompressAI; #pip install -e .

python cross_eval_batch.py 
# python cross_eval_batch.py \
#     --lambda_value "$1" \
#     --epochs "$2" \
#     --save_period "$3" \
#     --learning_rate "$4" \
#     --batch_size "$5" \
#     --patch_size "$6" \
#     --bit_depth "$7"
