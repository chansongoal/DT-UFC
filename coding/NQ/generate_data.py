import os
import numpy as np
from typing import Union, List
import nonlinear_quant

def perform_nonlinear_quantization(org_feat_path, rec_feat_path, quant_mapping_path, model_type, task, samples, trun_flag, trun_high, trun_low, quant_type, bit_depth, data_size, source_file=None, crop_flag=True):
    # get quantization points
    if task == 'dpt':
        quantization_mapping_name = []
        for ch in range(4):
            suffix = f"_ch{ch}"
            quantization_mapping_name.append(f"{quant_mapping_path}/quantization_mapping_{task}{suffix}_trunl{trun_low}_trunh{trun_high}_{quant_type}{samples}_bitdepth{bit_depth}.json")
    else:
        quantization_mapping_name = f"{quant_mapping_path}/quantization_mapping_{task}_trunl{trun_low}_trunh{trun_high}_{quant_type}{samples}_bitdepth{bit_depth}.json"
    
    quantization_points = nonlinear_quant.load_quantization_points(quantization_mapping_name)

    if source_file == None: feat_names = os.listdir(org_feat_path)[:data_size]
    else: 
        with open(source_file, 'r') as f:
            feat_names = [line.strip().split(' ', 1)[0][:-4]+'.npy' for line in f if line.strip()]
    print('number samples: ', len(feat_names))
    # feat_names = [f for f in os.listdir(org_feat_path) if f.startswith("arc_")]
    # feat_names = os.listdir(org_feat_path)
    # Load and quantize features
    for feat_name in feat_names:
        org_feat_name = os.path.join(org_feat_path, feat_name)
        org_feat = np.load(org_feat_name)
        N, C, H, W = org_feat.shape
        dtype = org_feat.dtype
        # print(f"{feat_name}: {N}, {C}, {H}, {W}")
        rec_feat_name = os.path.join(rec_feat_path, feat_name)

        if trun_flag:
            trun_feat = nonlinear_quant.truncation(org_feat, trun_low, trun_high)
        else:
            trun_feat = org_feat

        quantized_feat = nonlinear_quant.nonlinear_quantization(trun_feat, quantization_points, bit_depth)

        pack_feat = nonlinear_quant.packing(quantized_feat, model_type)
        if crop_flag == True:
            # cropped_feat = nonlinear_quant.random_crop(pack_feat, (256, 256))   # perform crop after packing, for other tasks
            cropped_feat = nonlinear_quant.random_crop(pack_feat, (64, 4096))   # perform crop after packing, only for csr
            # for n in range(2):
            #     rec_feat_name_iter = f"{rec_feat_name[:-4]}_iter{n}.npy"
            #     cropped_feat = nonlinear_quant.random_crop(pack_feat, (64, 4096))
            #     cropped_feat = cropped_feat.astype(dtype)
            #     np.save(rec_feat_name_iter, cropped_feat)
        else: cropped_feat = pack_feat

        cropped_feat = cropped_feat.astype(dtype)
        np.save(rec_feat_name, cropped_feat)
    

        # dequantized_feat = nonlinear_quant.nonlinear_dequantization(quantized_feat, quantization_points, bit_depth)
        # dequantized_feat = dequantized_feat.astype(dtype)
        # np.save(rec_feat_name, dequantized_feat)


def perform_uniform_normalization(org_feat_path, rec_feat_path, model_type, trun_flag, trun_high, trun_low, bit_depth, data_size, crop_flag):
    feat_names = os.listdir(org_feat_path)[:data_size]
    # Load and quantize features
    for feat_name in feat_names:
        org_feat_name = os.path.join(org_feat_path, feat_name)
        org_feat = np.load(org_feat_name)
        N, C, H, W = org_feat.shape
        dtype = org_feat.dtype; print(dtype)
        # print(f"{feat_name}: {N}, {C}, {H}, {W}")
        rec_feat_name = os.path.join(rec_feat_path, feat_name)

        if trun_flag:
            trun_feat = nonlinear_quant.truncation(org_feat, trun_low, trun_high)
        else:
            trun_feat = org_feat

        quantized_feat = nonlinear_quant.uniform_quantization(trun_feat, trun_low, trun_high, bit_depth)

        pack_feat = nonlinear_quant.packing(quantized_feat, model_type)
        if crop_flag == True:
            cropped_feat = nonlinear_quant.random_crop(pack_feat, (256, 256))   # perform crop after packing
        else: cropped_feat = pack_feat

        cropped_feat = cropped_feat.astype(dtype)
        np.save(rec_feat_name, cropped_feat)

        # dequantized_feat = nonlinear_quant.uniform_dequantization(quantized_feat, trun_low, trun_high, bit_depth)
        # np.save(rec_feat_name, dequantized_feat)


if __name__ == "__main__":
    # model_type = 'dinov2'; task = 'seg'; max_v = 105.95; min_v = -506.97; trun_high = 105.95; trun_low = -506.97
    model_type = 'llama3'; task = 'csr'; max_v = 47.75; min_v = -71.50; trun_high = 47.75; trun_low = -71.50
    # model_type = 'sd3'; task = 'tti'; max_v = 4.46; min_v = -5.79; trun_high = 4.46; trun_low = -5.79
    # config = 'train'; source_file = None
    # config = 'test'; source_file = '/gdata1/gaocs/FCM_LM_Test_Dataset/{model_type}/{task}/source/captions_val2017_select500.txt'
    config = 'test'; source_file = f'/gdata1/gaocs/FCM_LM_Test_Dataset/{model_type}/{task}/source/arc_challenge_test_longest500_shape.txt'

    # org_feat_path = f'/gdata1/gaocs/FCM_LM_Train_Data/{model_type}/{task}/org_feat/{config}'
    org_feat_path = f'/gdata1/gaocs/FCM_LM_Test_Dataset/{model_type}/{task}/feature' 
    quant_mapping_path = f'/gdata1/gaocs/Data_FCM_NQ/{model_type}/{task}/quantization_mapping'
    
    # quant_type = 'uniform'; samples = 0
    quant_type = 'kmeans'; samples = 10
    bit_depths = [8]
    trun_flag = False
    if trun_flag==False:
        trun_high = max_v; trun_low = min_v

    data_size = 10000
    for bit_depth in bit_depths:
        print(model_type, task, trun_flag, quant_type, samples, max_v, min_v, trun_high, trun_low, bit_depth)

        # perform kmeans quantization to generate training data
        rec_feat_path = f'/gdata1/gaocs/FCM_LM_Train_Data/{model_type}/{task}/{quant_type}{samples}_bitdepth{bit_depth}/crop_hgt64_wdt4096/{config}'
        if not os.path.exists(rec_feat_path): os.makedirs(rec_feat_path)
        perform_nonlinear_quantization(org_feat_path, rec_feat_path, quant_mapping_path, model_type, task, samples, trun_flag, trun_high, trun_low, quant_type, bit_depth, data_size, source_file, True)
        
        # # perform linear quantization to generate training data
        # rec_feat_path = f'/gdata1/gaocs/FCM_LM_Train_Data/{model_type}/{task}/{quant_type}{samples}_bitdepth{bit_depth}/crop_hgt256_wdt256/test'
        # if not os.path.exists(rec_feat_path): os.makedirs(rec_feat_path)
        # perform_uniform_normalization(org_feat_path, rec_feat_path, model_type, trun_flag, trun_high, trun_low, bit_depth, data_size, True)

        # # perform kmeans quantization to generate test data 
        # org_feat_path = f'/gdata1/gaocs/FCM_LM_Test_Dataset/{model_type}/{task}/feature' 
        # rec_feat_path = f'/gdata1/gaocs/Data_FCM_NQ/{model_type}/{task}/quantization/trunl{trun_low}_trunh{trun_high}_{quant_type}{samples}/bitdepth{bit_depth}'
        # os.makedirs(rec_feat_path, exist_ok=True)
        # perform_nonlinear_quantization(org_feat_path, rec_feat_path, quant_mapping_path, model_type, task, samples, trun_flag, trun_high, trun_low, quant_type, bit_depth, data_size, source_file, False)