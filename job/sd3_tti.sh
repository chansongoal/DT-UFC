#!/bin/bash 
export HOME=/ghome/gaocs
cd /ghome/gaocs/FCM-NQ/machines/sd3/diffusers; pip install -e .
cd /ghome/gaocs/FCM-NQ/machines/sd3; python sd3.py >/gdata1/gaocs/Data_FCM_NQ/sd3/tti/evaluate_tti_3090_0.02.txt 2>&1
