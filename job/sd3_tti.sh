#!/bin/bash 
export HOME=/ghome/gaocs
cd /ghome/gaocs/FCM-UFC/machines/sd3/diffusers; pip install -e .

cd /ghome/gaocs/FCM-UFC/machines/sd3
python sd3.py \
    --arch "$1" \
    --train_task "$2" \
    --transform_type "$3" \
    --samples "$4" \
    --bit_depth "$5" \
    --learning_rate "$6" \
    --epochs "$7" \
    --batch_size "${8}" \
    --patch_size "${9}" \
    --lambda_value_all "${@:10}" >/gdata1/gaocs/Data_DTUFC/accuracy_log/${1}/trained_${2}/${3}${4}_bitdepth${5}/sd3_tti/${1}_trained_${2}_eval_tti.txt

