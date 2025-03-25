#!/bin/bash 

export HOME=/ghome/gaocs
cd /ghome/gaocs/FCM-NQ/machines/llama3/transformers; pip install -e .
pip install safetensors==0.5.0 huggingface-hub==0.27.0

export PATH=$HOME/.local/bin:$PATH

cd /ghome/gaocs/FCM-NQ/machines/llama3
# python llama3.py >/gdata1/gaocs/Data_FCM_NQ/llama3/csr/hyperprior/cross_eval_log/train_seg_trunl-71.5_trunh47.75_kmeans10_bitdepth8_0.003.txt 2>&1

python llama3.py \
    --arch "$1" \
    --trun_low "$2" \
    --trun_high "$3" \
    --quant_type "$4" \
    --samples "$5" \
    --bit_depth "$6" \
    --train_task "$7" \
    --learning_rate "$8" \
    --epochs "$9" \
    --batch_size "${10}" \
    --patch_size "${11}" \
    --lambda_value_all "${@:12}" >/gdata1/gaocs/Data_FCM_NQ/llama3/csr/${1}/eval_log/trained_${7}_csr_eval_${1}_trunl${2}_trunh${3}_${4}${5}_bitdepth${6}_epochs${9}_batch${10}_full.txt
