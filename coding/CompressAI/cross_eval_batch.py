import os
import re
import matplotlib.pyplot as plt
import time
import argparse

os.environ['MKL_THREADING_LAYER'] = 'GNU'

import argparse


def generate_cross_eval_commands(data_root_train, data_root_test, dataset_root, source_file, model_type_train, model_type_test, task_train, task_test, 
                                 trun_flag_train, trun_high_train, trun_low_train,trun_flag_test, trun_high_test, trun_low_test, 
                                 quant_type, samples, bit_depth, quant_points_name, 
                                 arch, model_name, lambda_value, epochs, learning_rate, batch_size, patch_size):
    if isinstance(trun_low_train, list):
        trun_low_train = '[' + ','.join(map(str, trun_low_train)) + ']'
        trun_high_train = '[' + ','.join(map(str, trun_high_train)) + ']'
    if isinstance(trun_low_test, list):
        trun_low_test = '[' + ','.join(map(str, trun_low_test)) + ']'
        trun_high_test = '[' + ','.join(map(str, trun_high_test)) + ']'

    training_models_path = f"{data_root_train}/{model_type_train}/{task_train}/{model_name}/training_models/trunl{trun_low_train}_trunh{trun_high_train}_{quant_type}{samples}_bitdepth{bit_depth}"
    os.makedirs(training_models_path, exist_ok=True)
    eval_log_path = f"{data_root_test}/{model_type_test}/{task_test}/{model_name}/encoding_log/trunl{trun_low_test}_trunh{trun_high_test}_{quant_type}{samples}_bitdepth{bit_depth}"
    os.makedirs(eval_log_path, exist_ok=True)

    eval_command = (
        f"python -m compressai.utils.eval_model checkpoint "
        f"{dataset_root}/{model_type_test}/{task_test}/feature "
        f"-a {arch} --cuda -v "
        f"--source_file {dataset_root}/{model_type_test}/{task_test}/source/{source_file} "
        # modify the output path accordingly, use the epoch information
        f"-d {data_root_test}/{model_type_test}/{task_test}/{model_name}/decoded/trunl{trun_low_test}_trunh{trun_high_test}_{quant_type}{samples}_bitdepth{bit_depth}/trained_{task_train}_lambda{lambda_value}_epochs{epochs}_lr{learning_rate}_bs{batch_size}_patch{patch_size.replace(' ', '-')} "
        # load specified checkpoint accordingly, use the epoch information
        f"--per-image -p {training_models_path}/{arch}_lambda{lambda_value}_epochs{epochs}_lr{learning_rate}_bs{batch_size}_patch{patch_size.replace(' ', '-')}_checkpoint_best.pth.tar "  
        f"--model_type={model_type_test} --task={task_test} --trun_flag={trun_flag_test} --trun_low={trun_low_test} --trun_high={trun_high_test} --quant_type={quant_type} --qsamples={samples} --bit_depth={bit_depth} --quant_points_name={quant_points_name} "
        f">{eval_log_path}/trained_{task_train}_compress_cross_{model_name}_lambda{lambda_value}_epochs{epochs}_lr{learning_rate}_bs{batch_size}_patch{patch_size.replace(' ', '-')}.txt 2>&1"
    )

    return eval_command

def hyperprior_evaluate_pipeline(data_root_train, data_root_test, dataset_root, source_file, model_type_train, model_type_test, task_train, task_test, 
                                 trun_flag_train, trun_high_train, trun_low_train, trun_flag_test, trun_high_test, trun_low_test, 
                                 quant_type, samples, bit_depth, quant_points_name, 
                                 lambda_value, epochs, learning_rate, batch_size, patch_size):
    print(f"evaluate <{task_test}> task using models trained on <{task_train}> task")
    arch = 'bmshj2018-hyperprior'
    if arch == 'bmshj2018-hyperprior':
        model_name = 'hyperprior'; print(model_name)
    elif arch == 'elic2022-official':
        model_name = 'elic'; print(model_name)

    eval_cmd = generate_cross_eval_commands(data_root_train, data_root_test, dataset_root, source_file, model_type_train, model_type_test, task_train, task_test, 
                                            trun_flag_train, trun_high_train, trun_low_train, trun_flag_test, trun_high_test, trun_low_test, 
                                            quant_type, samples, bit_depth, quant_points_name, 
                                            arch, model_name, lambda_value, epochs, learning_rate, batch_size, patch_size)

    time_start = time.time()
    print("Eval Command:")
    print(eval_cmd)
    os.system(eval_cmd)
    print(eval_cmd)
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

def get_train_config(task):
    if task == 'seg':
        model_type_train = 'dinov2'; task_train = 'seg'
        trun_flag_train = False; trun_high_train = 105.95; trun_low_train = -506.97
        lambda_value_all = [0.00055, 0.001, 0.003, 0.004, 0.005, 0.007]
        epochs = 200; learning_rate = "0.0001"; batch_size = 128; patch_size = "256 256"   # height first, width later
    elif task == 'csr':
        model_type_train = 'llama3'; task_train = 'csr'
        trun_flag_train = False; trun_high_train = 47.75; trun_low_train = -71.50
        lambda_value_all = [0.0005, 0.0008, 0.0015, 0.002, 0.0025, 0.003, 0.004]
        epochs = 400; learning_rate = "0.0001"; batch_size = 128; patch_size = "64-1024"   # height first, width later
    elif task == 'tti':
        model_type_train = 'sd3'; task_train = 'tti'
        trun_flag_train = False; trun_high_train = 4.46; trun_low_train = -5.79
        lambda_value_all = [0.001, 0.002, 0.004, 0.006, 0.008, 0.01, 0.015, 0.02, 0.05]
        epochs = 600; learning_rate = "0.0001"; batch_size = 32; patch_size = "512-512"   # height first, width later

    return model_type_train, task_train, trun_flag_train, trun_low_train, trun_high_train, lambda_value_all, epochs, learning_rate, batch_size, patch_size

def get_test_config(task):
    if task == 'seg':
        model_type_test = 'dinov2'; task_test = "seg"; trun_flag_test = False; trun_low_test = -506.97; trun_high_test = 105.95
        source_file = 'seg_val_20.txt'
    elif task == 'tti':
        model_type_test = 'sd3'; task_test = "tti"; trun_flag_test = False; trun_low_test = -5.79; trun_high_test = 4.46
        source_file = 'captions_val2017_select100.txt'
    elif task == 'csr':
        model_type_test = 'llama3'; task_test = "csr"; trun_flag_test = False; trun_low_test = -71.50; trun_high_test = 47.75
        source_file = 'arc_challenge_test_longest100_shape.txt'

    return model_type_test, task_test, trun_flag_test, trun_low_test, trun_high_test, source_file

if __name__ == "__main__":
    # args = argument_parsing()
    # #lambda_value = args.lambda_value
    # lambda_value_all = [0.0005, 0.001, 0.003, 0.007, 0.015]
    # epochs = args.epochs
    # save_period = args.save_period
    # learning_rate = args.learning_rate
    # batch_size = args.batch_size
    # patch_size = f"{args.patch_size} {args.patch_size}"
    # bit_depth = args.bit_depth

    # train
    train_task = 'tti'
    model_type_train, task_train, trun_flag_train, trun_low_train, trun_high_train, lambda_value_all, epochs, learning_rate, batch_size, patch_size = get_train_config(train_task)

    # test
    test_task = 'seg'
    model_type_test, task_test, trun_flag_test, trun_low_test, trun_high_test, source_file = get_test_config(test_task)
    
    quant_type = 'kmeans'; samples = 10; bit_depth = 8

    data_root = "/gdata1/gaocs/Data_FCM_NQ"
    data_root_train = data_root
    data_root_test = data_root
    dataset_root = '/gdata1/gaocs/FCM_LM_Test_Dataset'

    quant_points_name = f"/gdata1/gaocs/Data_FCM_NQ/{model_type_test}/{task_test}/quantization_mapping/quantization_mapping_{task_test}_trunl{trun_low_test}_trunh{trun_high_test}_{quant_type}{samples}_bitdepth{bit_depth}.json"

    # config following parameters here
    for lambda_value in lambda_value_all:
        hyperprior_evaluate_pipeline(data_root_train, data_root_test, dataset_root, source_file, model_type_train, model_type_test, task_train, task_test, 
                                    trun_flag_train, trun_high_train, trun_low_train, trun_flag_test, trun_high_test, trun_low_test, 
                                    quant_type, samples, bit_depth, quant_points_name, 
                                    lambda_value, epochs, learning_rate, batch_size, patch_size)