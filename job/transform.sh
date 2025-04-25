#!/bin/bash 

export HOME=/ghome/gaocs
pip install scikit-learn

cd /ghome/gaocs/FCM-UFC/coding/transform; python nonlinear_transform.py >/gdata1/gaocs/Data_DTUFC/sd3/tti/sd3_tti_kmenas10_bitdepth8.txt 2>&1
