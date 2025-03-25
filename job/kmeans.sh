#!/bin/bash 

export HOME=/ghome/gaocs
pip install scikit-learn

cd /ghome/gaocs/FCM-NQ/coding/NQ; python nonlinear_quant.py >/gdata1/gaocs/Data_FCM_NQ/sd3/tti/sd3_tti_kmenas.txt 2>&1
