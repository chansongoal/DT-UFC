#!/bin/bash 
cd /ghome/gaocs/FCM-NQ/machines/dinov2
# python seg.py >/gdata1/gaocs/Data_FCM_NQ/dinov2/seg/quantization/seg_eval_quantization.txt

python seg.py \
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
    --lambda_value_all "${@:12}" >/gdata1/gaocs/Data_FCM_NQ/dinov2/seg/${1}/eval_log/trained_${7}_seg_eval_${1}_trunl${2}_trunh${3}_${4}${5}_bitdepth${6}_epochs${9}_full.txt
