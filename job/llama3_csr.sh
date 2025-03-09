#!/bin/bash 

pip list
export HOME=/ghome/gaocs
cd /ghome/gaocs/FCM-NQ/machines/llama3/transformers; pip install -e .
pip install safetensors==0.5.0 huggingface-hub==0.27.0

export PATH=$HOME/.local/bin:$PATH
pip list

cd /ghome/gaocs/FCM-NQ/machines/llama3; python llama3.py >/gdata1/gaocs/FCM_LM_Train_Data/llama3/csr/test_3090_500.txt 2>&1
