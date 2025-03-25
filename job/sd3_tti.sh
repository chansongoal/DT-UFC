#!/bin/bash 
export HOME=/ghome/gaocs
cd /ghome/gaocs/FCM-NQ/machines/sd3/diffusers; pip install -e .
# cd /ghome/gaocs/FCM-NQ/machines/sd3; python sd3.py >/gdata1/gaocs/FCM_LM_Train_Data/sd3/tti/extract_training_data_10000_rtx3090.txt 2>&1
cd /ghome/gaocs/FCM-NQ/machines/sd3
python sd3.py \
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
    --lambda_value_all "${@:12}" >/gdata1/gaocs/Data_FCM_NQ/sd3/tti/${1}/eval_log/trained_${7}_tti_eval_${1}_trunl${2}_trunh${3}_${4}${5}_bitdepth${6}_epochs${9}_full.txt
    # --lambda_value_all "${@:12}" >/gdata1/gaocs/Data_FCM_NQ/sd3/tti/quantization/tti_eval_quantization.txt
