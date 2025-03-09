import os
import re
import matplotlib.pyplot as plt
import time
import argparse

os.environ['MKL_THREADING_LAYER'] = 'GNU'

import argparse


def generate_cross_eval_commands(data_root_train, data_root_test, model_type_train, model_type_test, task_train, task_test, 
                                 trun_flag_train, trun_high_train, trun_low_train,trun_flag_test, trun_high_test, trun_low_test, 
                                 quant_type, samples, bit_depth, quant_points_name, 
                                 arch, model_name, lambda_value, epochs, learning_rate, batch_size, patch_size):
    if isinstance(trun_low_train, list):
        trun_low_train = '[' + ','.join(map(str, trun_low_train)) + ']'
        trun_high_train = '[' + ','.join(map(str, trun_high_train)) + ']'
    if isinstance(trun_low_test, list):
        trun_low_test = '[' + ','.join(map(str, trun_low_test)) + ']'
        trun_high_test = '[' + ','.join(map(str, trun_high_test)) + ']'

    eval_command = (
        f"python -m compressai.utils.eval_model checkpoint "
        f"{data_root_test}/{model_type_test}/{task_test}/feature_test "
        f"-a {arch} --cuda -v "
        # modify the output path accordingly, use the epoch information
        f"-d {data_root_test}/{model_type_test}/{task_test}/{model_name}/decoded/trained_{task_train}_trunl{trun_low_test}_trunh{trun_high_test}_{quant_type}{samples}_bitdepth{bit_depth}/lambda{lambda_value}_epochs{epochs}_lr{learning_rate}_bs{batch_size}_patch{patch_size.replace(' ', '-')} "
        # load specified checkpoint accordingly, use the epoch information
        f"--per-image -p {data_root_train}/{model_type_train}/{task_train}/{model_name}/training_models/trunl{trun_low_train}_trunh{trun_high_train}_{quant_type}{samples}_bitdepth{bit_depth}/"
        f"{arch}_lambda{lambda_value}_epochs{epochs}_lr{learning_rate}_bs{batch_size}_patch{patch_size.replace(' ', '-')}_checkpoint_best.pth.tar "  
        f"--model_type={model_type_test} --task={task_test} --trun_flag={trun_flag_test} --trun_low={trun_low_test} --trun_high={trun_high_test} --quant_type={quant_type} --qsamples={samples} --bit_depth={bit_depth} --quant_points_name={quant_points_name} "
        f">{data_root_test}/{model_type_test}/{task_test}/{model_name}/encoding_log/trained_{task_train}_trunl{trun_low_test}_trunh{trun_high_test}_{quant_type}{samples}_bitdepth{bit_depth}/"
        f"compress_cross_{model_name}_lambda{lambda_value}_epochs{epochs}_lr{learning_rate}_bs{batch_size}_patch{patch_size.replace(' ', '-')}.txt 2>&1"
    )

    return eval_command

def hyperprior_evaluate_pipeline(data_root_train, data_root_test, model_type_train, model_type_test, task_train, task_test, 
                                 trun_flag_train, trun_high_train, trun_low_train, trun_flag_test, trun_high_test, trun_low_test, 
                                 quant_type, samples, bit_depth, quant_points_name, 
                                 lambda_value, epochs, learning_rate, batch_size, patch_size):
    print(f"evaluate <{task_test}> task using models trained on <{task_train}> task")
    arch = 'bmshj2018-hyperprior'
    model_name = arch.split('-')[-1]; print(f"\n{model_name}")

    eval_cmd = generate_cross_eval_commands(data_root_train, data_root_test, model_type_train, model_type_test, task_train, task_test, 
                                            trun_flag_train, trun_high_train, trun_low_train, trun_flag_test, trun_high_test, trun_low_test, 
                                            quant_type, samples, bit_depth, quant_points_name, 
                                            arch, model_name, lambda_value, epochs, learning_rate, batch_size, patch_size)

    time_start = time.time()
    print("Eval Command:")
    print(eval_cmd)
    os.system(eval_cmd)
    print('encoding time: ', time.time() - time_start)

def argument_parsing():
    parser = argparse.ArgumentParser(description="Hyperprior Evaluation Pipeline")
    parser.add_argument('--lambda_value', type=float, default=0.01, help='lambda')
    parser.add_argument('--epochs', type=int, default=800, help='epochs')
    parser.add_argument('--save_period', type=int, default=20, help='save_period')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='learning_rate')
    parser.add_argument('--batch_size', type=int, default=16, help='batch_size')
    parser.add_argument('--patch_size', type=int, default=256, help='patch_size')
    parser.add_argument('--bit_depth', type=int, default=8, help='bit_depth')
    
    args = parser.parse_args()
    
    return args

if __name__ == "__main__":
    # model_type = 'llama3'; task = 'csr'
    # max_v = 47.75; min_v = -78; trun_high = 5; trun_low = -5
    # lambda_value_all = [0.01405, 0.0142, 0.015, 0.07, 10]
    # epochs = 200; learning_rate = "1e-4"; batch_size = 40; patch_size = "64 4096" # height first, width later

    # model_type = 'dinov2'; task = 'cls'
    # max_v = 104.1752; min_v = -552.451; trun_high = 5; trun_low = -5
    # lambda_value_all = [0.001, 0.0017, 0.003, 0.0035, 0.01]
    # epochs = 800; learning_rate = "1e-4"; batch_size = 128; patch_size = "256 256"   # height first, width later

    # model_type = 'dinov2'; task = 'seg'
    # max_v = 103.2168; min_v = -530.9767; trun_high = 5; trun_low = -5
    # lambda_value_all = [0.0005, 0.001, 0.003, 0.007, 0.015]
    # epochs = 500; save_period = 20; learning_rate = "1e-4"; batch_size = 64; patch_size = "256 256"   # height first, width later

    # model_type = 'dinov2'; task = 'dpt'
    # max_v = [3.2777, 5.0291, 25.0456, 102.0307]; min_v = [-2.4246, -26.8908, -323.2952, -504.4310]; trun_high = [1,2,10,10]; trun_low = [-1,-2,-10,-10]
    # lambda_value_all = [0.001, 0.005, 0.02, 0.05, 0.12]
    # epochs = 200; learning_rate = "1e-4"; batch_size = 128; patch_size = "256 256"   # height first, width later
    
    # model_type = 'sd3'; task = 'tti'
    # max_v = 4.668; min_v = -6.176; trun_high = 4.668; trun_low = -6.176
    # lambda_value_all = [0.005, 0.01, 0.02, 0.05, 0.2]
    # epochs = 60; learning_rate = "1e-4"; batch_size = 32; patch_size = "512 512"   # height first, width later

    model_type = 'dinov2'
    model_type_train = model_type
    model_type_test= model_type
    task_train = "seg"
    task_test = "cls"

    # train
    max_v = 103.2168; min_v = -530.9767; trun_high_train = 5; trun_low_train = -5
    trun_flag_train = False
    if trun_flag_train == False: trun_high_train = max_v; trun_low_train = min_v

    # test
    max_v = 104.1752; min_v = -552.451; trun_high_test = 5; trun_low_test = -5
    trun_flag_test = False
    if trun_flag_test == False: trun_high_test = max_v; trun_low_test = min_v

    quant_type = 'kmeans'; samples = 10#; bit_depth = 8

    args = argument_parsing()
    #lambda_value = args.lambda_value
    lambda_value_all = [0.0005, 0.001, 0.003, 0.007, 0.015]
    epochs = args.epochs
    save_period = args.save_period
    learning_rate = args.learning_rate
    batch_size = args.batch_size
    patch_size = f"{args.patch_size} {args.patch_size}"
    bit_depth = args.bit_depth

    data_root = "/gdata1/gaocs/Data_FCM_NQ"
    data_root_train = data_root
    data_root_test = data_root

    quant_points_name = f"/gdata1/gaocs/Data_FCM_NQ/{model_type}/{task_test}/quantization_mapping/quantization_mapping_{task_test}_trunl{trun_low_test}_trunh{trun_high_test}_{quant_type}{samples}_bitdepth{bit_depth}.json"

    # config following parameters here
    for lambda_value in lambda_value_all:
        hyperprior_evaluate_pipeline(data_root_train, data_root_test, model_type_train, model_type_test, task_train, task_test, 
                                    trun_flag_train, trun_high_train, trun_low_train, trun_flag_test, trun_high_test, trun_low_test, 
                                    quant_type, samples, bit_depth, quant_points_name, 
                                    lambda_value, epochs, learning_rate, batch_size, patch_size)